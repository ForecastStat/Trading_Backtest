# backtest_orchestrator.py

# ==============================================================================
# --- PATCH DI COMPATIBILITÀ PER NUMPY 2.0 ---
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
# ==============================================================================

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
import shutil
import warnings
import time

warnings.filterwarnings('ignore')

# --- IMPORTAZIONE DELLE LOGICHE MODIFICATE ---
try:
    from best_buy_backtest import run_one_time_screening_for_backtest
    from stock_analyzer_backtest import run_analysis_for_date
    from trading_engine_backtest import IntegratedRevolutionaryTradingEngine
except ImportError as e:
    print(f"ERRORE CRITICO: Assicurati che i file rinominati esistano.")
    print(f"Dettaglio errore: {e}")
    exit()

# --- CONFIGURAZIONE DEL BACKTEST ---
START_DATE = '2015-01-01'
END_DATE = '2015-12-31'
INITIAL_CAPITAL = 100000.0

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data_backtest"
AI_LEARNING_DIR = DATA_DIR / "ai_learning"
REPORTS_DIR = DATA_DIR / "reports"
SIGNALS_HISTORY_DIR = DATA_DIR / "signals_history"
AI_DB_FILE = AI_LEARNING_DIR / "performance.db"
ANALYSIS_FILE_PATH = DATA_DIR / "latest_analysis.json"
EXECUTION_SIGNALS_FILE = DATA_DIR / "execution_signals.json"
HISTORICAL_EXECUTION_SIGNALS_FILE = DATA_DIR / "historical_execution_signals.json"

def setup_backtest_environment():
    print("FASE 1: Setup dell'ambiente di backtest...")
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    for directory in [DATA_DIR, AI_LEARNING_DIR, REPORTS_DIR, SIGNALS_HISTORY_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    AI_DB_FILE.touch()
    with open(HISTORICAL_EXECUTION_SIGNALS_FILE, 'w') as f:
        json.dump({"historical_signals": []}, f)
    print("✅ Ambiente di backtest pronto.\n")

def pre_fetch_all_historical_data(tickers):
    print("FASE 2: Download di tutti i dati storici...")
    fetch_start_date = (pd.to_datetime(START_DATE) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end_date = (pd.to_datetime(END_DATE) + timedelta(days=1)).strftime('%Y-%m-%d')
    all_data = {}
    tickers_to_fetch = tickers + ['^GSPC']
    for i, ticker in enumerate(tickers_to_fetch):
        try:
            time.sleep(0.1) 
            print(f"  Scaricando {ticker} ({i+1}/{len(tickers_to_fetch)})...")
            data = yf.download(ticker, start=fetch_start_date, end=fetch_end_date, progress=False, timeout=10)
            if not data.empty:
                all_data[ticker] = data
        except Exception as e:
            print(f"    ERRORE scaricando {ticker}: {e}")
    print(f"\n✅ Dati storici scaricati per {len(all_data)} simboli.")
    return all_data

# ==============================================================================
# --- CORREZIONE CHIAVE: Stato del Portafoglio Globale ---
# Definiamo queste variabili QUI, fuori dal loop, in modo che mantengano il loro valore
# tra una giornata e l'altra del backtest.
# ==============================================================================
capital = INITIAL_CAPITAL
open_positions = []
closed_trades_for_csv = []

def run_daily_simulation_step(current_date, all_historical_data, tickers_to_analyze, sp500_data):
    """
    Questa funzione ora esegue UN SINGOLO GIORNO del backtest,
    modificando le variabili globali del portafoglio.
    """
    # Usiamo 'global' per dire a Python che vogliamo modificare le variabili definite fuori da questa funzione
    global capital, open_positions, closed_trades_for_csv
    
    current_date_str = current_date.strftime('%Y-%m-%d')
    print(f"\n{'='*20} SIMULAZIONE GIORNO: {current_date_str} {'='*20}")
    
    # 3.0: Esecuzione ordini del giorno precedente
    if EXECUTION_SIGNALS_FILE.exists():
        print(f"  - 3.0: Esecuzione ordini preparati il giorno prima...")
        try:
            with open(EXECUTION_SIGNALS_FILE, 'r') as f:
                signals = json.load(f).get('signals', {})
            buy_signals = signals.get('buys', [])
            sell_signals = signals.get('sells', [])

            if sell_signals:
                positions_to_remove = []
                for sell in sell_signals:
                    pos_to_close = next((p for p in open_positions if p['ticker'] == sell['ticker']), None)
                    if pos_to_close:
                        try:
                            exit_price = all_historical_data[sell['ticker']].loc[current_date_str]['Open']
                            exit_value = exit_price * pos_to_close['quantity']
                            capital += exit_value
                            print(f"    -> VENDITA: {pos_to_close['quantity']} di {sell['ticker']} @ ${exit_price:.2f}. Capitale: ${capital:,.2f}")
                            closed_trades_for_csv.append({'entry_price': pos_to_close['entry_price'], 'exit_price': exit_price, 'quantity': pos_to_close['quantity']})
                            positions_to_remove.append(pos_to_close)
                        except KeyError:
                            print(f"    ATTENZIONE: Nessun dato per {sell['ticker']} il {current_date_str}. Vendita non eseguita.")
                open_positions = [p for p in open_positions if p not in positions_to_remove]
            
            if buy_signals:
                for buy in buy_signals:
                    try:
                        entry_price = all_historical_data[buy['ticker']].loc[current_date_str]['Open']
                        trade_value = entry_price * buy['quantity_estimated']
                        if capital >= trade_value:
                            capital -= trade_value
                            open_positions.append({'ticker': buy['ticker'], 'entry_price': entry_price, 'quantity': buy['quantity_estimated'], 'trade_value': trade_value, 'entry_date': current_date})
                            print(f"    -> ACQUISTO: {buy['quantity_estimated']} di {buy['ticker']} @ ${entry_price:.2f}. Capitale: ${capital:,.2f}")
                        else:
                            print(f"    ACQUISTO SALTATO: {buy['ticker']} - Capitale insufficiente.")
                    except KeyError:
                        print(f"    ATTENZIONE: Nessun dato per {buy['ticker']} il {current_date_str}. Acquisto non eseguito.")
            
            os.remove(EXECUTION_SIGNALS_FILE)
        except (json.JSONDecodeError, FileNotFoundError):
             print("    ATTENZIONE: File dei segnali non trovato o corrotto. Si procede.")

    # Stampa lo stato del portafoglio DOPO l'esecuzione degli ordini
    print(f"Stato post-esecuzione: Capitale ${capital:,.2f}, Posizioni Aperte: {len(open_positions)}")
    if open_positions:
        pos_summary = ", ".join([f"{p['ticker']}({p['quantity']})" for p in open_positions])
        print(f"  Dettaglio posizioni: {pos_summary}")

    # 3.1 Stock Analyzer
    print("  - 3.1: Esecuzione Stock Analyzer (offline)...")
    run_analysis_for_date(tickers_to_analyze, all_historical_data, current_date, ANALYSIS_FILE_PATH)

    # 3.2 Trading Engine
    print("  - 3.2: Esecuzione Trading Engine per generare ordini per DOMANI...")
    engine = IntegratedRevolutionaryTradingEngine(capital, open_positions, AI_DB_FILE)
    engine.trade_history = closed_trades_for_csv
    engine.run_integrated_trading_session_for_backtest(ANALYSIS_FILE_PATH, sp500_data, current_date)

def save_backtest_results(final_value):
    print("\n--- FASE 4: SALVATAGGIO DEI RISULTATI DEL BACKTEST ---")
    if not closed_trades_for_csv:
        print("⚠️ Nessun trade è stato chiuso durante il backtest. Il file CSV non verrà creato.")
        return
    df_results = pd.DataFrame(closed_trades_for_csv)
    output_df = df_results[['entry_price', 'exit_price', 'quantity']]
    output_filename = "backtest_results.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"✅ Risultati del backtest salvati in '{output_filename}'.")
    print(f"Numero di trade chiusi: {len(output_df)}")
    print(f"Capitale Iniziale: ${INITIAL_CAPITAL:,.2f}")
    print(f"Valore Finale del Portafoglio: ${final_value:,.2f}")
    profit = final_value - INITIAL_CAPITAL
    profit_pct = (profit / INITIAL_CAPITAL) * 100
    print(f"Profitto/Perdita Totale: ${profit:,.2f} ({profit_pct:.2f}%)")

if __name__ == "__main__":
    setup_backtest_environment()
    
    print("\n--- ESECUZIONE SCRIPT 1: Best Buy (Screening una tantum) ---")
    tickers_for_backtest = run_one_time_screening_for_backtest()
    if not tickers_for_backtest:
        print("ERRORE: Nessun ticker ottenuto dallo screening iniziale.")
        exit()
    
    historical_data_store = pre_fetch_all_historical_data(tickers_for_backtest)
    if not historical_data_store:
        print("ERRORE: Nessun dato storico scaricato.")
        exit()
        
    # --- Loop di simulazione principale ---
    print("\n--- FASE 3: INIZIO SIMULAZIONE DI TRADING GIORNALIERA ---")
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    for date in date_range:
        run_daily_simulation_step(date, historical_data_store, tickers_for_backtest, historical_data_store['^GSPC'])

    # --- Calcolo e salvataggio finale ---
    print("\n--- SIMULAZIONE COMPLETATA ---")
    final_portfolio_value = capital
    for pos in open_positions:
        try:
            last_price = historical_data_store[pos['ticker']]['Close'].iloc[-1]
            final_portfolio_value += pos['quantity'] * last_price
        except:
            final_portfolio_value += pos['trade_value']
    
    save_backtest_results(final_portfolio_value)
    
    print("\n--- BACKTEST COMPLETATO CON SUCCESSO ---")
