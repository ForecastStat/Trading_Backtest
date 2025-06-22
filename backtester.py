import os
import subprocess
import json
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import time

# --- CONFIGURAZIONE DEL BACKTEST (MODALITÀ TEST RAPIDO 1 MESE) ---
START_DATE_STR = "2015-01-01"
END_DATE_STR = "2015-01-31"    # Testiamo solo 1 mese per velocità
INITIAL_CAPITAL = 100000.0
MAX_DAYS_PER_RUN = 250 # Ora possiamo processare molti più giorni in un singolo run

BACKTEST_DATA_DIR = "data_backtest"
BACKTEST_STATE_FILE = os.path.join(BACKTEST_DATA_DIR, "trading_state.json")
PORTFOLIO_HISTORY_FILE = os.path.join(BACKTEST_DATA_DIR, "portfolio_evolution.json")
EXECUTION_SIGNALS_FILE = os.path.join(BACKTEST_DATA_DIR, "execution_signals.json")

BACKTEST_TICKER_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'WMT', 'JPM', 'PG', 'XOM', 'CVX', 
    'UNH', 'HD', 'BAC', 'KO', 'PFE', 'VZ', 'DIS', 'CSCO', 'PEP', 'INTC',
    'MCD', 'T', 'BA', 'IBM', 'CAT', 'GE', 'MMM', 'HON', 'AXP', 'NKE',
    'GS', 'MRK', 'ORCL', 'UPS', 'LMT', 'COST', 'SBUX', 'SPY', 'QQQ', 'IWM'
]

def initialize_backtest():
    os.makedirs(BACKTEST_DATA_DIR, exist_ok=True)
    if not os.path.exists(BACKTEST_STATE_FILE):
        state = { "capital": INITIAL_CAPITAL, "open_positions": [], "trade_history": [], "last_simulated_date": None }
        with open(BACKTEST_STATE_FILE, 'w') as f: json.dump(state, f, indent=2)
    if not os.path.exists(PORTFOLIO_HISTORY_FILE):
        with open(PORTFOLIO_HISTORY_FILE, 'w') as f: json.dump([], f)

def download_price_data_for_chunk(start_date, end_date):
    print(f"--- Ottimizzazione: Download di tutti i dati di prezzo da {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')} ---")
    try:
        data = yf.download(
            tickers=BACKTEST_TICKER_UNIVERSE,
            start=start_date,
            end=end_date + timedelta(days=5),
            progress=True,
            auto_adjust=True
        )
        price_cache = data['Close']
        print("--- Download completato. Dati in memoria. ---")
        return price_cache
    except Exception as e:
        print(f"ERRORE CRITICO durante il download massivo dei dati: {e}")
        return None

def simulate_broker(simulated_date, state, price_cache):
    try:
        with open(EXECUTION_SIGNALS_FILE, 'r') as f:
            signals = json.load(f).get("signals", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return state

    # CORREZIONE QUI: Rimuoviamo .normalize() e ci assicuriamo che la data sia pulita
    execution_date = (pd.to_datetime(simulated_date) + pd.tseries.offsets.BDay(1)).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for sell_signal in signals.get("sells", []):
        ticker, qty = sell_signal['ticker'], sell_signal['quantity']
        pos_index = next((i for i, p in enumerate(state['open_positions']) if p['ticker'] == ticker), -1)
        if pos_index != -1:
            position = state['open_positions'].pop(pos_index)
            try:
                exec_price = price_cache.loc[execution_date, ticker]
                if pd.isna(exec_price): raise ValueError("Prezzo non disponibile")
                
                sell_value = exec_price * qty
                state['capital'] += sell_value
                # ... (logica cronologia)
            except (KeyError, ValueError):
                state['open_positions'].insert(pos_index, position)

    for buy_signal in signals.get("buys", []):
        ticker, qty = buy_signal['ticker'], buy_signal['quantity_estimated']
        try:
            exec_price = price_cache.loc[execution_date, ticker]
            if pd.isna(exec_price): raise ValueError("Prezzo non disponibile")

            trade_value = exec_price * qty
            if state['capital'] >= trade_value:
                state['capital'] -= trade_value
                state['open_positions'].append({
                    "ticker": ticker, "quantity": qty, "entry": exec_price, 
                    "date": execution_date.isoformat(), "amount_invested": trade_value,
                    # ... (resto dei dati)
                })
        except (KeyError, ValueError):
            pass
            
    return state

def update_portfolio_history(date, state, portfolio_history, price_cache):
    open_positions_value = 0
    # CORREZIONE QUI: Rimuoviamo .normalize() e usiamo un metodo sicuro
    date_normalized = date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    for pos in state['open_positions']:
        try:
            current_price = price_cache.loc[date_normalized, pos['ticker']]
            if not pd.isna(current_price):
                open_positions_value += current_price * pos['quantity']
            else:
                open_positions_value += pos.get('amount_invested', 0)
        except KeyError:
            open_positions_value += pos.get('amount_invested', 0)
            
    total_value = state['capital'] + open_positions_value
    portfolio_history.append({"date": date.strftime('%Y-%m-%d'), "value": round(total_value, 2)})
    print(f"  > Valore Portafoglio: ${total_value:,.2f} (Capitale: ${state['capital']:,.2f}, Posizioni: ${open_positions_value:,.2f})")
    return portfolio_history

def run_backtest_chunk():
    initialize_backtest()
    with open(BACKTEST_STATE_FILE, 'r') as f: state = json.load(f)
    with open(PORTFOLIO_HISTORY_FILE, 'r') as f: portfolio_history = json.load(f)
        
    last_date_str = state.get("last_simulated_date")
    start_date = datetime.strptime(last_date_str, '%Y-%m-%d') + timedelta(days=1) if last_date_str else datetime.strptime(START_DATE_STR, '%Y-%m-%d')
    end_date = datetime.strptime(END_DATE_STR, '%Y-%m-%d')

    if start_date > end_date:
        print("🎉🎉🎉 BACKTEST COMPLETATO! 🎉🎉🎉")
        return

    price_cache = download_price_data_for_chunk(start_date, end_date)
    if price_cache is None: return

    trading_days = pd.bdate_range(start=start_date, end=end_date)
    days_processed = 0
    
    for day in trading_days:
        if days_processed >= MAX_DAYS_PER_RUN:
            print(f"\n--- Raggiunto limite di {MAX_DAYS_PER_RUN} giorni per questa esecuzione. ---")
            break

        day_dt = day.to_pydatetime()
        date_str = day_dt.strftime('%Y-%m-%d')
        
        print(f"\n--- Simulazione {date_str} ({days_processed + 1}/{MAX_DAYS_PER_RUN}) ---")
        os.environ['SIMULATED_DATE'] = date_str
        
        scripts_to_run = ["best_buy.py", "stock_analyzer_2_0.py", "trading_engine_30_0.py"]
        all_scripts_succeeded = True
        for i, script_name in enumerate(scripts_to_run):
            result = subprocess.run(["python", script_name], text=True, check=False)
            if result.returncode != 0:
                print(f"  - ERRORE FATALE in {script_name}.")
                print("    --- INIZIO LOG DI ERRORE SCRIPT ---")
                print(result.stderr)
                print("    --- FINE LOG DI ERRORE SCRIPT ---")
                all_scripts_succeeded = False
                break
        
        if not all_scripts_succeeded:
            portfolio_history = update_portfolio_history(day_dt, state, portfolio_history, price_cache)
            state["last_simulated_date"] = date_str
            days_processed += 1
            continue
        
        state = simulate_broker(day_dt, state, price_cache)
        portfolio_history = update_portfolio_history(day_dt, state, portfolio_history, price_cache)
        
        state["last_simulated_date"] = date_str
        days_processed += 1

    print("\n--- Blocco di simulazione completato. Salvataggio progressi... ---")
    with open(BACKTEST_STATE_FILE, 'w') as f: json.dump(state, f, indent=2)
    with open(PORTFOLIO_HISTORY_FILE, 'w') as f: json.dump(portfolio_history, f, indent=2)
    print("Salvataggio completato.")

if __name__ == "__main__":
    run_backtest_chunk()
