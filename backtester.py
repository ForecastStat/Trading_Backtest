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

# Universo di ticker usato negli script figli. Lo definiamo anche qui per la cache dei prezzi.
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
    """
    NUOVA FUNZIONE OTTIMIZZATA: Scarica tutti i dati di prezzo necessari
    per l'intero blocco di backtest IN UNA SOLA VOLTA.
    """
    print(f"--- Ottimizzazione: Download di tutti i dati di prezzo da {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')} ---")
    # Aggiungiamo 5 giorni di buffer per le esecuzioni che avvengono il giorno dopo
    try:
        data = yf.download(
            tickers=BACKTEST_TICKER_UNIVERSE,
            start=start_date,
            end=end_date + timedelta(days=5),
            progress=True, # Mostriamo il progresso del download
            auto_adjust=True # Usiamo i prezzi aggiustati per dividendi/split
        )
        # yf.download restituisce un DataFrame multi-indice. Prendiamo solo le chiusure.
        price_cache = data['Close']
        print("--- Download completato. I dati sono ora in memoria per un'elaborazione rapida. ---")
        return price_cache
    except Exception as e:
        print(f"ERRORE CRITICO durante il download massivo dei dati: {e}. Il backtest non può continuare.")
        return None

def simulate_broker(simulated_date, state, price_cache):
    """MODIFICATA: Ora usa la cache dei prezzi invece di fare chiamate API."""
    try:
        with open(EXECUTION_SIGNALS_FILE, 'r') as f:
            signals = json.load(f).get("signals", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return state

    execution_date = (simulated_date + pd.tseries.offsets.BDay(1)).normalize()
    
    # SIMULA VENDITE
    for sell_signal in signals.get("sells", []):
        ticker, qty = sell_signal['ticker'], sell_signal['quantity']
        pos_index = next((i for i, p in enumerate(state['open_positions']) if p['ticker'] == ticker), -1)
        if pos_index != -1:
            position = state['open_positions'].pop(pos_index)
            try:
                # Legge il prezzo dalla cache (istantaneo!)
                exec_price = price_cache.loc[execution_date, ticker]
                if pd.isna(exec_price): raise ValueError("Prezzo non disponibile nella cache")
                
                sell_value = exec_price * qty
                state['capital'] += sell_value
                # ... (resto della logica per la cronologia) ...
            except (KeyError, ValueError):
                # Se non troviamo il prezzo, ripristiniamo la posizione
                state['open_positions'].insert(pos_index, position)

    # SIMULA ACQUISTI
    for buy_signal in signals.get("buys", []):
        ticker, qty = buy_signal['ticker'], buy_signal['quantity_estimated']
        try:
            # Legge il prezzo dalla cache (istantaneo!)
            exec_price = price_cache.loc[execution_date, ticker]
            if pd.isna(exec_price): raise ValueError("Prezzo non disponibile nella cache")

            trade_value = exec_price * qty
            if state['capital'] >= trade_value:
                state['capital'] -= trade_value
                state['open_positions'].append({
                    "ticker": ticker, "quantity": qty, "entry": exec_price, 
                    "date": execution_date.isoformat(), "amount_invested": trade_value,
                    # ... (resto dei dati) ...
                })
        except (KeyError, ValueError):
            pass # Ignora l'acquisto se il prezzo non è disponibile
            
    return state

def update_portfolio_history(date, state, portfolio_history, price_cache):
    """MODIFICATA: Ora usa la cache dei prezzi, è quasi istantanea."""
    open_positions_value = 0
    date_normalized = date.normalize()
    
    for pos in state['open_positions']:
        try:
            # Legge il prezzo dalla cache
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

    # OTTIMIZZAZIONE: Scarica tutti i dati necessari per il blocco in una sola volta
    price_cache = download_price_data_for_chunk(start_date, end_date)
    if price_cache is None:
        return # Interrompi se il download fallisce

    trading_days = pd.bdate_range(start=start_date, end=end_date)
    days_processed = 0
    
    for day in trading_days:
        if days_processed >= MAX_DAYS_PER_RUN:
            print(f"\n--- Raggiunto limite di {MAX_DAYS_PER_RUN} giorni per questa esecuzione. ---")
            break

        day_dt = day.to_pydatetime()
        os.environ['SIMULATED_DATE'] = day_dt.strftime('%Y-%m-%d')
        
        # (La logica di esecuzione degli script figli rimane la stessa)
        # ...
        
        state = simulate_broker(day_dt, state, price_cache)
        portfolio_history = update_portfolio_history(day_dt, state, portfolio_history, price_cache)
        
        state["last_simulated_date"] = day_dt.strftime('%Y-%m-%d')
        days_processed += 1

    # (La logica di salvataggio finale rimane la stessa)
    # ...
    
# Incolla qui il resto del codice di backtester.py, ma assicurati che la funzione run_backtest_chunk 
# e le altre funzioni usino la price_cache come argomento dove necessario.

# Ecco la funzione run_backtest_chunk completa e corretta:
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

    # OTTIMIZZAZIONE: Scarica tutti i dati in anticipo
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
        
        # Esecuzione script figli
        scripts_to_run = ["best_buy.py", "stock_analyzer_2_0.py", "trading_engine_30_0.py"]
        all_scripts_succeeded = True
        for i, script_name in enumerate(scripts_to_run):
            result = subprocess.run(["python", script_name], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"  -> ERRORE in {script_name}. Salto il giorno. Errore: {result.stderr}")
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
