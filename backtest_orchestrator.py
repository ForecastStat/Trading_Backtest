# backtest_orchestrator.py (Versione Corretta e Definitiva)

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

# --- IMPORTAZIONE DELLE LOGICHE MODIFICATE DAI TUOI SCRIPT ---
try:
    from best_buy_backtest import run_one_time_screening_for_backtest
    from stock_analyzer_backtest import run_analysis_for_date
    from trading_engine_backtest import TradingEngineForBacktest
except ImportError as e:
    print(f"ERRORE CRITICO: Assicurati che i file 'best_buy_backtest.py', 'stock_analyzer_backtest.py' e 'trading_engine_backtest.py' esistano.")
    print(f"Dettaglio errore: {e}")
    exit()

# --- CONFIGURAZIONE DEL BACKTEST ---
START_DATE = '2015-01-01'
END_DATE = '2015-12-31'
INITIAL_CAPITAL = 100000.0

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data_backtest"
REPORTS_DIR = DATA_DIR / "reports"
AI_DB_FILE = DATA_DIR / "ai_learning" / "performance.db"
ANALYSIS_FILE_PATH = DATA_DIR / "latest_analysis.json"
EXECUTION_SIGNALS_FILE = DATA_DIR / "execution_signals.json"

def setup_backtest_environment():
    print("FASE 1: Setup dell'ambiente di backtest...")
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    # Creiamo le directory necessarie, inclusa la cartella genitore del DB AI
    for directory in [DATA_DIR, REPORTS_DIR, AI_DB_FILE.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    AI_DB_FILE.touch()
    print("✅ Ambiente di backtest pronto.\n")

def pre_fetch_all_historical_data(tickers):
    print("FASE 2: Download di tutti i dati storici...")
    fetch_start_date = (pd.to_datetime(START_DATE) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end_date = (pd.to_datetime(END_DATE) + timedelta(days=1)).strftime('%Y-%m-%d')
    all_data = {}
    tickers_to_fetch = tickers + ['^GSPC']
    for i, ticker in enumerate(tickers_to_fetch):
        try:
            print(f"  Scaricando {ticker} ({i+1}/{len(tickers_to_fetch)})...")
            data = yf.download(ticker, start=fetch_start_date, end=fetch_end_date, progress=False, auto_adjust=True, timeout=10)
            if not data.empty:
                all_data[ticker] = data
            time.sleep(0.1)
        except Exception as e:
            print(f"    ERRORE scaricando {ticker}: {e}")
    print(f"\n✅ Dati storici scaricati per {len(all_data)} simboli.\n")
    return all_data

def save_backtest_results(closed_trades, final_value):
    print("\n--- FASE 4: SALVATAGGIO DEI RISULTATI DEL BACKTEST ---")
    if not closed_trades:
        print("⚠️ Nessun trade è stato chiuso durante il backtest. Il file CSV non verrà creato.")
        return
    df = pd.DataFrame(closed_trades)
    output_df = df[['entry_price', 'exit_price', 'quantity']]
    output_filename = "backtest_results.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"✅ Risultati salvati in '{output_filename}'.")
    print(f"  - Trade Chiusi: {len(output_df)}")
    print(f"  - Capitale Iniziale: ${INITIAL_CAPITAL:,.2f}")
    print(f"  - Valore Finale Portafoglio: ${final_value:,.2f}")
    profit = final_value - INITIAL_CAPITAL
    profit_pct = (profit / INITIAL_CAPITAL) * 100
    print(f"  - Profitto/Perdita Totale: ${profit:,.2f} ({profit_pct:.2f}%)")

if __name__ == "__main__":
    setup_backtest_environment()

    print("--- ESECUZIONE SCRIPT 1: Best Buy (Screening) ---")
    tickers_to_analyze = run_one_time_screening_for_backtest()
    
    historical_data_store = pre_fetch_all_historical_data(tickers_to_analyze)

    # ==============================================================================
    # --- CORREZIONE DEFINITIVA: GESTIONE DELLO STATO ---
    # Le variabili del portafoglio sono definite qui, nel contesto principale,
    # e vengono modificate direttamente all'interno del loop.
    # ==============================================================================
    capital = INITIAL_CAPITAL
    open_positions = []
    closed_trades = []
    
    analyzer = StockAnalyzerForBacktest()

    print("\n--- FASE 3: INIZIO SIMULAZIONE DI TRADING GIORNALIERA ---")
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='B')

    for current_date in date_range:
        current_date_str = current_date.strftime('%Y-%m-%d')
        print(f"\n{'='*20} SIMULAZIONE GIORNO: {current_date_str} {'='*20}")

        # 1. ESECUZIONE ORDINI (INIZIO GIORNATA)
        print(f"  - Stato Inizio Giornata: Capitale ${capital:,.2f}, Posizioni: {len(open_positions)}")
        if EXECUTION_SIGNALS_FILE.exists():
            print("    Eseguo ordini preparati il giorno prima...")
            with open(EXECUTION_SIGNALS_FILE, 'r') as f:
                signals = json.load(f).get('signals', {})
            
            # Vendite
            positions_to_remove = []
            for sell in signals.get('sells', []):
                pos = next((p for p in open_positions if p['ticker'] == sell['ticker']), None)
                if pos:
                    try:
                        exit_price = historical_data_store[sell['ticker']].loc[current_date_str]['Open']
                        exit_value = exit_price * pos['quantity']
                        capital += exit_value
                        closed_trades.append({'entry_price': pos['entry_price'], 'exit_price': exit_price, 'quantity': pos['quantity']})
                        positions_to_remove.append(pos)
                    except KeyError:
                        print(f"    ATTENZIONE: Nessun dato per {sell['ticker']} il {current_date_str}. Vendita non eseguita.")
            open_positions = [p for p in open_positions if p not in positions_to_remove]
            
            # Acquisti
            for buy in signals.get('buys', []):
                try:
                    entry_price = historical_data_store[buy['ticker']].loc[current_date_str]['Open']
                    trade_value = entry_price * buy['quantity_estimated']
                    if capital >= trade_value:
                        capital -= trade_value
                        open_positions.append({'ticker': buy['ticker'], 'entry_price': entry_price, 'quantity': buy['quantity_estimated'], 'trade_value': trade_value, 'entry_date': current_date})
                except KeyError:
                    print(f"    ATTENZIONE: Nessun dato per {buy['ticker']} il {current_date_str}. Acquisto non eseguito.")
            os.remove(EXECUTION_SIGNALS_FILE)

        # 2. ANALISI e GENERAZIONE ORDINI (FINE GIORNATA)
        print("  - Analisi e Generazione ordini per DOMANI...")
        print(f"    Stato Pre-Engine: Capitale ${capital:,.2f}, Posizioni: {len(open_positions)}")
        
        analyzer.run_analysis_for_date(tickers_to_analyze, historical_data_store, current_date, ANALYSIS_FILE_PATH)
        
        engine = TradingEngineForBacktest(capital, open_positions, closed_trades, AI_DB_FILE)
        engine.run_daily_logic(ANALYSIS_FILE_PATH, historical_data_store['^GSPC'].loc[:current_date], current_date)

    # 3. VALUTAZIONE FINALE
    print("\n--- SIMULAZIONE COMPLETATA ---")
    final_portfolio_value = capital
    for pos in open_positions:
        try:
            last_price = historical_data_store[pos['ticker']]['Close'].iloc[-1]
            final_portfolio_value += pos['quantity'] * last_price
        except:
            final_portfolio_value += pos['trade_value']
            
    save_backtest_results(closed_trades, final_portfolio_value)
    print("\n--- BACKTEST COMPLETATO CON SUCCESSO ---")
