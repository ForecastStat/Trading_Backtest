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
import logging # Importa logging

warnings.filterwarnings('ignore')

# --- IMPORTAZIONE DELLE LOGICHE ---
# NOTA: Queste importazioni ora avvengono prima della configurazione del logging,
# che è corretto. La configurazione avverrà solo quando lo script viene eseguito.
try:
    from best_buy_backtest import run_one_time_screening_for_backtest
    from stock_analyzer_backtest import run_analysis_for_date
    from trading_engine_backtest import IntegratedRevolutionaryTradingEngine
except ImportError as e:
    # Usiamo print qui perché il logging non è ancora configurato
    print(f"ERRORE CRITICO: Assicurati che i file necessari esistano.")
    print(f"Dettaglio errore: {e}")
    exit()

# --- CONFIGURAZIONE GLOBALE ---
START_DATE = '2015-01-01'
END_DATE = '2015-06-30'
INITIAL_CAPITAL = 100000.0

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data_backtest"
REPORTS_DIR = DATA_DIR / "reports"
SIGNALS_DIR = BASE_DIR / "data"
AI_LEARNING_DIR = DATA_DIR / "ai_learning"
AI_DB_FILE = AI_LEARNING_DIR / "performance.db"
ANALYSIS_FILE_PATH = DATA_DIR / "latest_analysis.json"
EXECUTION_SIGNALS_FILE = SIGNALS_DIR / "execution_signals.json"
HISTORICAL_EXECUTION_SIGNALS_FILE = SIGNALS_DIR / "historical_execution_signals.json"

def setup_backtest_environment():
    """Versione robusta per GitHub Actions, preserva il DB se esiste."""
    logging.info("FASE 1: Setup dell'ambiente di backtest...")
    for directory in [DATA_DIR, AI_LEARNING_DIR, REPORTS_DIR, SIGNALS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    if AI_DB_FILE.exists():
        logging.info(f"  - Database AI trovato (da cache): {AI_DB_FILE.stat().st_size} bytes.")
    else:
        logging.info(f"  - Database AI non trovato. Verrà creato.")
    if EXECUTION_SIGNALS_FILE.exists():
        os.remove(EXECUTION_SIGNALS_FILE)
    logging.info("✅ Ambiente di backtest pronto.\n")

def pre_fetch_all_historical_data(tickers):
    """FASE 2: Download di tutti i dati storici necessari."""
    logging.info("FASE 2: Download di tutti i dati storici...")
    fetch_start_date = (pd.to_datetime(START_DATE) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end_date = (pd.to_datetime(END_DATE) + timedelta(days=1)).strftime('%Y-%m-%d')
    logging.info(f"  - Periodo download: {fetch_start_date} a {fetch_end_date}")
    all_data = {}
    tickers_to_fetch = tickers + ['^GSPC']
    for i, ticker in enumerate(tickers_to_fetch):
        try:
            time.sleep(0.05)
            data = yf.download(ticker, start=fetch_start_date, end=fetch_end_date, progress=False, timeout=10, auto_adjust=True)
            if not data.empty and len(data) > 100:
                all_data[ticker] = data
        except Exception as e:
            logging.error(f"  - Errore download {ticker}: {e}")
    logging.info(f"✅ Download completato: {len(all_data)}/{len(tickers_to_fetch)} titoli scaricati.")
    return all_data

def execute_signals_for_day(signals, all_historical_data, current_date, capital, open_positions, closed_trades):
    if not signals: return capital, open_positions, closed_trades
    current_date_str = current_date.strftime('%Y-%m-%d')
    logging.info(f"  📈 Esecuzione segnali per {current_date_str}...")
    
    # ESECUZIONE VENDITE
    for sell_signal in signals.get('sells', []):
        ticker = sell_signal.get('ticker')
        matching_position = next((pos for pos in open_positions if pos['ticker'] == ticker), None)
        if not matching_position: continue
        try:
            if current_date_str not in all_historical_data.get(ticker, pd.DataFrame()).index: continue
            exit_price = float(all_historical_data[ticker].loc[current_date_str]['Open'])
            quantity = matching_position['quantity']
            exit_value = exit_price * quantity
            capital += exit_value
            trade_record = {
                'ticker': ticker, 'entry_date': matching_position['entry_date'].strftime('%Y-%m-%d'), 'exit_date': current_date_str,
                'entry_price': matching_position['entry_price'], 'exit_price': exit_price, 'quantity': quantity,
                'entry_value': matching_position['trade_value'], 'exit_value': exit_value,
                'profit': exit_value - matching_position['trade_value'],
                'profit_pct': ((exit_value / matching_position['trade_value']) - 1) * 100 if matching_position['trade_value'] else 0,
                'hold_days': (current_date - matching_position['entry_date']).days,
                'reason': sell_signal.get('reason', 'Strategy Exit')
            }
            closed_trades.append(trade_record)
            open_positions.remove(matching_position)
            logging.info(f"      ✅ VENDUTO: {quantity} {ticker} @ ${exit_price:.2f} (P/L: {trade_record['profit_pct']:+.1f}%)")
        except Exception as e:
            logging.error(f"      ❌ ERRORE VENDITA {ticker}: {e}")
            
    # ESECUZIONE ACQUISTI
    for buy_signal in signals.get('buys', []):
        ticker = buy_signal.get('ticker')
        if not ticker: continue
        try:
            if current_date_str not in all_historical_data.get(ticker, pd.DataFrame()).index: continue
            entry_price = float(all_historical_data[ticker].loc[current_date_str]['Open'])
            quantity = buy_signal.get('quantity_estimated', 0)
            trade_value = entry_price * quantity
            
            if capital >= trade_value and quantity > 0:
                capital -= trade_value
                
                # Creiamo il dizionario della posizione usando TUTTI i nomi delle chiavi
                # necessari, sia per l'orchestratore che per l'engine, per evitare errori.
                new_position = {
                    # Chiavi che la funzione di vendita dell'engine si aspetta:
                    'entry': entry_price,
                    'date': current_date.isoformat(), # Formato stringa standard
                    'quantity': quantity,
                    'amount_invested': trade_value,
                    
                    # Chiavi che il resto dell'orchestratore potrebbe usare (le manteniamo):
                    'ticker': ticker,
                    'entry_price': entry_price,
                    'trade_value': trade_value,
                    'entry_date': current_date, # Oggetto datetime
                    'stop_loss': buy_signal.get('stop_loss'),
                    'take_profit': buy_signal.get('take_profit')
                }
                
                open_positions.append(new_position)
                logging.info(f"      ✅ ACQUISTATO: {quantity} {ticker} @ ${entry_price:.2f}")
            
            
            elif quantity > 0:
                logging.warning(f"      ⚠️ ACQUISTO SALTATO: {ticker} - Capitale insufficiente.")
        except Exception as e:
            logging.error(f"      ❌ ERRORE ACQUISTO {ticker}: {e}")

    return capital, open_positions, closed_trades

def run_backtest_simulation(all_historical_data, tickers_to_analyze):
    """FASE 3: Simulazione di trading giornaliera"""
    logging.info("\n" + "="*80 + "\nFASE 3: INIZIO SIMULAZIONE DI TRADING GIORNALIERA\n" + "="*80)
    capital, open_positions, closed_trades = INITIAL_CAPITAL, [], []
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    sp500_data = all_historical_data.get('^GSPC')

    if sp500_data is None or sp500_data.empty:
        logging.error("❌ ERRORE: Dati S&P 500 non disponibili per la simulazione")
        return closed_trades, capital

    for day_num, current_date in enumerate(date_range, 1):
        current_date_str = current_date.strftime('%Y-%m-%d')
        logging.info(f"\n{'='*20} GIORNO {day_num}/{len(date_range)}: {current_date_str} {'='*20}")
        
        portfolio_value = capital
        for pos in open_positions:
            try:
                current_price_data = all_historical_data[pos['ticker']].loc[current_date_str]['Close']
                current_price = float(current_price_data.iloc[0] if hasattr(current_price_data, 'iloc') else current_price_data)
                portfolio_value += pos['quantity'] * current_price
            except Exception:
                portfolio_value += pos['trade_value']
        
        logging.info(f"💰 Stato Portfolio: Capitale=${capital:,.2f}, Posizioni={len(open_positions)}, Valore Totale≈${portfolio_value:,.2f}")
        
        if EXECUTION_SIGNALS_FILE.exists():
            try:
                with open(EXECUTION_SIGNALS_FILE, 'r') as f:
                    signals = json.load(f).get('signals', {})
                capital, open_positions, closed_trades = execute_signals_for_day(signals, all_historical_data, current_date, capital, open_positions, closed_trades)
                os.remove(EXECUTION_SIGNALS_FILE)
            except Exception as e:
                logging.error(f"    ❌ ERRORE processando segnali: {e}")
        
        logging.info("  🔍 Generazione analisi e segnali per domani...")
        try:
            run_analysis_for_date(tickers_to_analyze, all_historical_data, current_date, ANALYSIS_FILE_PATH)
            
            engine = IntegratedRevolutionaryTradingEngine(capital=capital, open_positions=open_positions, performance_db_path=str(AI_DB_FILE))
            
            # --- INIZIO BLOCCO CRUCIALE CORRETTO ---
            # 1. Passa i trade chiusi all'engine
            engine.trade_history = closed_trades.copy()
            
            # 2. ORDINA all'engine di registrarli nel suo DB.
            #    Questa chiamata è fondamentale e deve avvenire QUI, prima di avviare la sessione di trading.
            if engine.ai_enabled:
                 engine._register_historical_trades_in_ai()
            # --- FINE BLOCCO CRUCIALE CORRETTO ---

            engine.run_integrated_trading_session_for_backtest(
                analysis_data_path=str(ANALYSIS_FILE_PATH),
                sp500_data_full=sp500_data,
                current_backtest_date=current_date
            )
        except Exception as e:
            logging.error(f"    ❌ ERRORE CRITICO nella generazione segnali: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        logging.info(f"  📊 Fine giornata: Capitale=${capital:,.2f}, Posizioni Aperte={len(open_positions)}, Trade Chiusi Totali={len(closed_trades)}")

    logging.info("\n" + "="*80 + "\nSIMULAZIONE COMPLETATA\n" + "="*80)
    final_portfolio_value = capital
    if open_positions:
        logging.info(f"📊 Posizioni aperte da liquidare: {len(open_positions)}")
        for pos in open_positions:
            try:
                last_price = float(all_historical_data[pos['ticker']]['Close'].iloc[-1])
                final_portfolio_value += pos['quantity'] * last_price
                logging.info(f"  - Liquidazione {pos['ticker']}: {pos['quantity']} azioni @ ${last_price:.2f}")
            except Exception:
                final_portfolio_value += pos['trade_value']
                logging.warning(f"  - Liquidazione {pos['ticker']}: prezzo finale non disponibile, uso valore di acquisto.")
    
    logging.info(f"💎 Valore finale totale portafoglio: ${final_portfolio_value:,.2f}")
    return closed_trades, final_portfolio_value

def save_backtest_results(closed_trades, final_value):
    """FASE 4: Salvataggio dei risultati del backtest"""
    logging.info("\n" + "="*80 + "\nFASE 4: SALVATAGGIO DEI RISULTATI DEL BACKTEST\n" + "="*80)
    if not closed_trades:
        logging.warning("⚠️ ATTENZIONE: Nessun trade è stato chiuso durante il backtest.")
        return
    
    pd.DataFrame(closed_trades).to_csv("backtest_results.csv", index=False)
    logging.info("✅ Risultati salvati in 'backtest_results.csv'")
    logging.info(f"📊 STATISTICHE BACKTEST:")
    logging.info(f"  - Periodo: {START_DATE} a {END_DATE}")
    logging.info(f"  - Capitale Iniziale: ${INITIAL_CAPITAL:,.2f}")
    logging.info(f"  - Valore Finale: ${final_value:,.2f}")
    profit = final_value - INITIAL_CAPITAL
    profit_pct = (profit / INITIAL_CAPITAL) * 100
    logging.info(f"  - Profitto/Perdita: ${profit:,.2f} ({profit_pct:+.2f}%)")
    logging.info(f"  - Trade Chiusi: {len(closed_trades)}")
    if closed_trades:
        wins = [t for t in closed_trades if t['profit'] > 0]
        logging.info(f"  - Win Rate: {(len(wins) / len(closed_trades)) * 100:.1f}%")
        logging.info(f"  - Profitto Medio per Trade: ${sum(t['profit'] for t in closed_trades) / len(closed_trades):,.2f}")
        logging.info(f"  - Giorni di Holding Medi: {sum(t['hold_days'] for t in closed_trades) / len(closed_trades):.1f}")
    logging.info("="*80)

if __name__ == "__main__":
    # La configurazione del logging è già stata spostata all'inizio del file
    
    logging.info("🚀 AVVIO BACKTEST ORCHESTRATOR")
    setup_backtest_environment()
    
    tickers_for_backtest = run_one_time_screening_for_backtest()
    if not tickers_for_backtest:
        logging.error("❌ ERRORE CRITICO: Nessun ticker ottenuto dallo screening iniziale.")
        exit(1)
    
    historical_data_store = pre_fetch_all_historical_data(tickers_for_backtest)
    if not historical_data_store or '^GSPC' not in historical_data_store:
        logging.error("❌ ERRORE CRITICO: Dati storici o S&P 500 non disponibili.")
        exit(1)
        
    final_trades, final_portfolio_value = run_backtest_simulation(
        all_historical_data=historical_data_store,
        tickers_to_analyze=tickers_for_backtest
    )
    
    save_backtest_results(final_trades, final_portfolio_value)
    
    logging.info("\n🎉 BACKTEST COMPLETATO CON SUCCESSO!")
    log_filepath = REPORTS_DIR / "backtest_log.txt"
    logging.info(f"   Il log completo di questa esecuzione è stato salvato in: {log_filepath}")
