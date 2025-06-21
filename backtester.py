# Contenuto completo di backtester.py (assicurati che sia questo)
import os
import subprocess
import json
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import requests
import time

# --- CONFIGURAZIONE DEL BACKTEST ---
#START_DATE_STR = "2010-01-01"
#END_DATE_STR = "2024-12-31"
START_DATE_STR = "2015-01-01"  # Iniziamo dal 1 Gennaio 2015
END_DATE_STR = "2015-12-31"    # Finiamo il 31 Dicembre 2015
INITIAL_CAPITAL = 100000.0
MAX_DAYS_PER_RUN = 90

BACKTEST_DATA_DIR = "data_backtest"
BACKTEST_STATE_FILE = os.path.join(BACKTEST_DATA_DIR, "trading_state.json")
PORTFOLIO_HISTORY_FILE = os.path.join(BACKTEST_DATA_DIR, "portfolio_evolution.json")
EXECUTION_SIGNALS_FILE = os.path.join(BACKTEST_DATA_DIR, "execution_signals.json")

def initialize_backtest():
    os.makedirs(BACKTEST_DATA_DIR, exist_ok=True)
    if not os.path.exists(BACKTEST_STATE_FILE):
        print("File di stato del backtest non trovato. Creazione nuovo stato...")
        state = { "capital": INITIAL_CAPITAL, "open_positions": [], "trade_history": [], "last_simulated_date": None }
        with open(BACKTEST_STATE_FILE, 'w') as f: json.dump(state, f, indent=2)
    
    if not os.path.exists(PORTFOLIO_HISTORY_FILE):
        with open(PORTFOLIO_HISTORY_FILE, 'w') as f: json.dump([], f)

def simulate_broker(simulated_date, state):
    try:
        with open(EXECUTION_SIGNALS_FILE, 'r') as f:
            signals = json.load(f).get("signals", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return state

    execution_date = (simulated_date + pd.tseries.offsets.BDay(1)).to_pydatetime()
    
    for sell_signal in signals.get("sells", []):
        ticker, qty = sell_signal['ticker'], sell_signal['quantity']
        pos_index = next((i for i, p in enumerate(state['open_positions']) if p['ticker'] == ticker), -1)
        if pos_index != -1:
            position = state['open_positions'].pop(pos_index)
            try:
                price_data = yf.download(ticker, start=execution_date, end=execution_date + timedelta(days=2), progress=False, auto_adjust=True)
                if price_data.empty: raise ValueError(f"Nessun dato per {ticker}")
                exec_price = price_data['Open'].iloc[0]
                
                sell_value = exec_price * qty
                state['capital'] += sell_value
                
                position['exit_price'] = exec_price
                position['exit_date'] = execution_date.isoformat()
                position['profit'] = sell_value - position.get('amount_invested', sell_value)
                position['profit_percentage'] = (position['profit'] / position['amount_invested'] * 100) if position.get('amount_invested', 0) > 0 else 0
                entry_dt = datetime.fromisoformat(position['date'])
                position['hold_days'] = (execution_date.replace(tzinfo=None) - entry_dt.replace(tzinfo=None)).days
                position['sell_reason'] = sell_signal.get('reason', 'N/A')
                state['trade_history'].append(position)
            except Exception as e:
                state['open_positions'].insert(pos_index, position)

    for buy_signal in signals.get("buys", []):
        ticker, qty = buy_signal['ticker'], buy_signal['quantity_estimated']
        try:
            price_data = yf.download(ticker, start=execution_date, end=execution_date + timedelta(days=2), progress=False, auto_adjust=True)
            if price_data.empty: raise ValueError(f"Nessun dato per {ticker}")
            exec_price = price_data['Open'].iloc[0]
            trade_value = exec_price * qty
            if state['capital'] >= trade_value:
                state['capital'] -= trade_value
                state['open_positions'].append({
                    "ticker": ticker, "quantity": qty, "entry": exec_price, 
                    "date": execution_date.isoformat(), "amount_invested": trade_value,
                    "stop_loss": buy_signal.get('stop_loss'), "take_profit": buy_signal.get('take_profit'),
                    "method": buy_signal.get("reason_methods", ["Unknown"])[0],
                    "ai_evaluation_details": buy_signal.get("ai_evaluation_details"),
                    "advanced_indicators_at_buy": buy_signal.get("advanced_indicators_at_buy")
                })
        except Exception as e:
            pass
            
    return state

def update_portfolio_history(date, state, portfolio_history):
    open_positions_value = 0
    date_ts = int(date.timestamp())
    
    for pos in state['open_positions']:
        try:
            time.sleep(0.1)
            price_data = yf.download(pos['ticker'], start=date, end=date + timedelta(days=2), progress=False, auto_adjust=True)
            if not price_data.empty:
                open_positions_value += price_data['Close'].iloc[0] * pos['quantity']
            else:
                open_positions_value += pos.get('amount_invested', 0)
        except Exception:
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
        return False

    trading_days = pd.bdate_range(start=start_date, end=end_date)
    days_processed = 0
    
    for day in trading_days:
        if days_processed >= MAX_DAYS_PER_RUN:
            print(f"\n--- Raggiunto limite di {MAX_DAYS_PER_RUN} giorni per questa esecuzione. ---")
            break

        day_dt = day.to_pydatetime()
        date_str = day_dt.strftime('%Y-%m-%d')
        
        print(f"\n--- Simulazione {date_str} ({days_processed + 1}/{MAX_DAYS_PER_RUN} di questo blocco) ---")
        os.environ['SIMULATED_DATE'] = date_str
        
        # Esegui la pipeline passo dopo passo per un debug migliore
        print("  - [1/3] Esecuzione Best Buy Selector...")
        result_bb = subprocess.run(["python", "best_buy.py"], capture_output=True, text=True)
        if result_bb.returncode != 0:
            print(f"  -> ERRORE in best_buy.py. Salto il giorno. Errore: {result_bb.stderr}")
            continue

        print("  - [2/3] Esecuzione Stock Analyzer...")
        result_sa = subprocess.run(["python", "stock_analyzer_2_0.py"], capture_output=True, text=True)
        if result_sa.returncode != 0:
            print(f"  -> ERRORE in stock_analyzer_2_0.py. Salto il giorno. Errore: {result_sa.stderr}")
            continue

        print("  - [3/3] Esecuzione Trading Engine...")
        result_te = subprocess.run(["python", "trading_engine_30_0.py"], capture_output=True, text=True)
        if result_te.returncode != 0:
            print(f"  -> ERRORE in trading_engine_30_0.py. Salto il giorno. Errore: {result_te.stderr}")
            continue
        
        print("  - Simulando il broker...")
        state = simulate_broker(day_dt, state)
        portfolio_history = update_portfolio_history(day_dt, state, portfolio_history)
        
        state["last_simulated_date"] = date_str
        days_processed += 1

    print("\n--- Blocco di simulazione completato. Salvataggio progressi... ---")
    with open(BACKTEST_STATE_FILE, 'w') as f: json.dump(state, f, indent=2)
    with open(PORTFOLIO_HISTORY_FILE, 'w') as f: json.dump(portfolio_history, f, indent=2)
    print("Salvataggio completato.")
    return True

if __name__ == "__main__":
    run_backtest_chunk()
