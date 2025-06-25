# backtest_orchestrator.py

# ==============================================================================
# --- IMPORTAZIONI E CONFIGURAZIONE LOGGING (spostato all'inizio) ---
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
import logging # Importa logging qui

# --- INIZIO BLOCCO MODIFICA PER LOGGING ---
# Definiamo le directory qui così sono globali per tutto lo script
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data_backtest"
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True) # Crea la cartella subito

# Definisci il percorso del file di log
log_filepath = REPORTS_DIR / "backtest_log.txt"

# Configura il logging UNA SOLA VOLTA, all'inizio del file.
# Questo assicura che tutti i messaggi da qualsiasi modulo vengano catturati.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath, mode='w'), # Sovrascrive il file ad ogni esecuzione
        logging.StreamHandler() # Mantiene l'output anche nel terminale
    ]
)
# --- FINE BLOCCO MODIFICA PER LOGGING ---

warnings.filterwarnings('ignore')

# --- IMPORTAZIONE DELLE LOGICHE MODIFICATE ---
try:
    from best_buy_backtest import run_one_time_screening_for_backtest
    from stock_analyzer_backtest import run_analysis_for_date
    from trading_engine_backtest import IntegratedRevolutionaryTradingEngine
except ImportError as e:
    logging.error(f"ERRORE CRITICO: Assicurati che i file 'best_buy_backtest.py', 'stock_analyzer_backtest.py', e 'trading_engine_backtest.py' esistano.")
    logging.error(f"Dettaglio errore: {e}")
    exit()

# --- CONFIGURAZIONE DEL BACKTEST ---
START_DATE = '2015-01-01'
END_DATE = '2015-12-31'
INITIAL_CAPITAL = 100000.0

# Le directory sono già state definite sopra
AI_LEARNING_DIR = DATA_DIR / "ai_learning"
SIGNALS_DIR = BASE_DIR / "data"
SIGNALS_HISTORY_DIR = DATA_DIR / "signals_history"
AI_DB_FILE = AI_LEARNING_DIR / "performance.db"
ANALYSIS_FILE_PATH = DATA_DIR / "latest_analysis.json"
EXECUTION_SIGNALS_FILE = SIGNALS_DIR / "execution_signals.json"
HISTORICAL_EXECUTION_SIGNALS_FILE = SIGNALS_DIR / "historical_execution_signals.json"

def setup_backtest_environment():
    """FASE 1: Setup dell'ambiente di backtest (versione per GitHub Actions)"""
    logging.info("FASE 1: Setup dell'ambiente di backtest...")

    for directory in [DATA_DIR, AI_LEARNING_DIR, REPORTS_DIR, SIGNALS_HISTORY_DIR, SIGNALS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logging.info(f"  - Assicurata esistenza directory: {directory}")

    if ANALYSIS_FILE_PATH.exists():
        os.remove(ANALYSIS_FILE_PATH)
        logging.info(f"  - Rimosso vecchio file di analisi: {ANALYSIS_FILE_PATH}")
    if EXECUTION_SIGNALS_FILE.exists():
        os.remove(EXECUTION_SIGNALS_FILE)
        logging.info(f"  - Rimosso vecchio file di segnali: {EXECUTION_SIGNALS_FILE}")

    if not AI_DB_FILE.exists():
        try:
            conn = sqlite3.connect(AI_DB_FILE)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    unique_trade_id TEXT UNIQUE
                )
            ''')
            conn.close()
            logging.info(f"  - Creato nuovo database AI (non trovato): {AI_DB_FILE}")
        except Exception as e:
            logging.error(f"  - ERRORE nella creazione del DB AI: {e}")
    else:
        logging.info(f"  - Database AI trovato (probabilmente da cache GitHub): {AI_DB_FILE}")
        try:
            db_size = AI_DB_FILE.stat().st_size
            logging.info(f"    -> Dimensione DB: {db_size} bytes.")
        except Exception:
            pass

    with open(HISTORICAL_EXECUTION_SIGNALS_FILE, 'w') as f:
        json.dump({"historical_signals": [], "last_updated": "", "total_signals": 0}, f, indent=2)
    logging.info(f"  - Inizializzato file segnali storici: {HISTORICAL_EXECUTION_SIGNALS_FILE}")
    
    logging.info("✅ Ambiente di backtest pronto.\n")

def pre_fetch_all_historical_data(tickers):
    """FASE 2: Download di tutti i dati storici necessari"""
    logging.info("FASE 2: Download di tutti i dati storici...")
    
    fetch_start_date = (pd.to_datetime(START_DATE) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end_date = (pd.to_datetime(END_DATE) + timedelta(days=1)).strftime('%Y-%m-%d')
    
    logging.info(f"  - Periodo download: {fetch_start_date} a {fetch_end_date}")
    logging.info(f"  - Titoli da scaricare: {len(tickers)} + S&P 500")
    
    all_data = {}
    tickers_to_fetch = tickers + ['^GSPC']
    
    successful_downloads = 0
    for i, ticker in enumerate(tickers_to_fetch):
        try:
            time.sleep(0.1)
            logging.info(f"  Scaricando {ticker} ({i+1}/{len(tickers_to_fetch)})...")
            
            data = yf.download(ticker, start=fetch_start_date, end=fetch_end_date, progress=False, timeout=10, auto_adjust=True)
            
            if not data.empty and len(data) > 100:
                all_data[ticker] = data
                successful_downloads += 1
                logging.info(f"  --> ✅ OK ({len(data)} righe)")
            else:
                logging.warning(f"  --> ❌ Dati insufficienti per {ticker}")
                
        except Exception as e:
            logging.error(f"  --> ❌ Errore download {ticker}: {str(e)[:50]}...")
    
    logging.info(f"\n✅ Download completato: {successful_downloads}/{len(tickers_to_fetch)} titoli scaricati con successo.")
    return all_data

def execute_signals_for_day(signals, all_historical_data, current_date, capital, open_positions, closed_trades):
    """Esegue i segnali di trading per il giorno corrente"""
    if not signals:
        return capital, open_positions, closed_trades
    
    current_date_str = current_date.strftime('%Y-%m-%d')
    logging.info(f"  📈 Esecuzione segnali per {current_date_str}...")
    
    sell_signals = signals.get('sells', [])
    if sell_signals:
        logging.info(f"    🔄 Processando {len(sell_signals)} segnali di vendita...")
        positions_to_remove = []
        
        for sell_signal in sell_signals:
            ticker = sell_signal.get('ticker')
            if not ticker: continue
                
            matching_position = next((pos for pos in open_positions if pos['ticker'] == ticker), None)
            if not matching_position:
                logging.warning(f"      ⚠️ VENDITA SALTATA: {ticker} non trovato in posizioni aperte")
                continue
            
            try:
                if ticker not in all_historical_data:
                    logging.warning(f"      ⚠️ VENDITA SALTATA: Nessun dato per {ticker}")
                    continue
                    
                ticker_data = all_historical_data[ticker]
                if current_date_str not in ticker_data.index:
                    logging.warning(f"      ⚠️ VENDITA SALTATA: Nessun dato per {ticker} il {current_date_str}")
                    continue
                
                exit_price = float(ticker_data.loc[current_date_str]['Open'])
                quantity = matching_position['quantity']
                exit_value = exit_price * quantity
                
                capital += exit_value
                
                trade_record = {
                    'ticker': ticker,
                    'entry_date': matching_position['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': current_date_str,
                    'entry_price': matching_position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'entry_value': matching_position['trade_value'],
                    'exit_value': exit_value,
                    'profit': exit_value - matching_position['trade_value'],
                    'profit_pct': ((exit_value - matching_position['trade_value']) / matching_position['trade_value']) * 100,
                    'hold_days': (current_date - matching_position['entry_date']).days,
                    'reason': sell_signal.get('reason', 'Strategy Exit')
                }
                
                closed_trades.append(trade_record)
                positions_to_remove.append(matching_position)
                
                logging.info(f"      ✅ VENDUTO: {quantity} {ticker} @ ${exit_price:.2f} = ${exit_value:,.2f} (P/L: {trade_record['profit_pct']:+.1f}%)")
                
            except Exception as e:
                logging.error(f"      ❌ ERRORE VENDITA {ticker}: {e}")
        
        for pos in positions_to_remove:
            open_positions.remove(pos)
        
        logging.info(f"    ✅ Vendite completate. Capitale aggiornato: ${capital:,.2f}")
    
    buy_signals = signals.get('buys', [])
    if buy_signals:
        logging.info(f"    🔄 Processando {len(buy_signals)} segnali di acquisto...")
        
        for buy_signal in buy_signals:
            ticker = buy_signal.get('ticker')
            quantity_estimated = buy_signal.get('quantity_estimated', 0)
            
            if not ticker or quantity_estimated <= 0:
                logging.warning(f"      ⚠️ ACQUISTO SALTATO: Dati segnale invalidi per {ticker}")
                continue
            
            try:
                if ticker not in all_historical_data:
                    logging.warning(f"      ⚠️ ACQUISTO SALTATO: Nessun dato per {ticker}")
                    continue
                    
                ticker_data = all_historical_data[ticker]
                if current_date_str not in ticker_data.index:
                    logging.warning(f"      ⚠️ ACQUISTO SALTATO: Nessun dato per {ticker} il {current_date_str}")
                    continue
                
                entry_price = float(ticker_data.loc[current_date_str]['Open'])
                trade_value = entry_price * quantity_estimated
                
                if capital >= trade_value:
                    capital -= trade_value
                    
                    new_position = {
                        'ticker': ticker, 'entry_price': entry_price, 'quantity': quantity_estimated,
                        'trade_value': trade_value, 'entry_date': current_date,
                        'stop_loss': buy_signal.get('stop_loss'), 'take_profit': buy_signal.get('take_profit'),
                        'entry_p': entry_price, 'entry_d': current_date, 'entry_date_str': current_date.strftime('%Y-%m-%d'),
                        'position_id': f"{ticker}_{current_date.strftime('%Y%m%d')}_{int(entry_price*100)}"
                    }
                    open_positions.append(new_position)
                    logging.info(f"      ✅ ACQUISTATO: {quantity_estimated} {ticker} @ ${entry_price:.2f} = ${trade_value:,.2f}")
                else:
                    logging.warning(f"      ⚠️ ACQUISTO SALTATO: {ticker} - Capitale insufficiente (serve ${trade_value:,.2f}, disponibile ${capital:,.2f})")
                    
            except Exception as e:
                logging.error(f"      ❌ ERRORE ACQUISTO {ticker}: {e}")
        
        logging.info(f"    ✅ Acquisti completati. Capitale rimanente: ${capital:,.2f}")
    
    return capital, open_positions, closed_trades

def run_backtest_simulation(all_historical_data, tickers_to_analyze):
    """FASE 3: Simulazione di trading giornaliera"""
    logging.info("\n" + "="*80)
    logging.info("FASE 3: INIZIO SIMULAZIONE DI TRADING GIORNALIERA")
    logging.info("="*80)
    
    capital = INITIAL_CAPITAL
    open_positions = []
    closed_trades = []
    
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    sp500_data = all_historical_data.get('^GSPC')
    
    if sp500_data is None or sp500_data.empty:
        logging.error("❌ ERRORE: Dati S&P 500 non disponibili per la simulazione")
        return closed_trades, capital
    
    logging.info(f"📅 Periodo simulazione: {len(date_range)} giorni lavorativi")
    logging.info(f"💰 Capitale iniziale: ${capital:,.2f}")
    logging.info(f"📊 Titoli nel portafoglio: {len(tickers_to_analyze)}")
    
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
            logging.info("  📋 Esecuzione ordini del giorno precedente...")
            try:
                with open(EXECUTION_SIGNALS_FILE, 'r') as f:
                    signals = json.load(f).get('signals', {})
                if signals:
                    capital, open_positions, closed_trades = execute_signals_for_day(signals, all_historical_data, current_date, capital, open_positions, closed_trades)
                else:
                    logging.info("    ℹ️ Nessun segnale da eseguire")
                os.remove(EXECUTION_SIGNALS_FILE)
            except Exception as e:
                logging.error(f"    ❌ ERRORE processando segnali: {e}")
        else:
            logging.info("  ℹ️ Nessun file di segnali da processare")
        
        logging.info("  🔍 Generazione analisi e segnali per domani...")
        try:
            run_analysis_for_date(tickers_to_analyze, all_historical_data, current_date, ANALYSIS_FILE_PATH)
            
            logging.info(f"    🔧 Inizializzazione trading engine isolato (posizioni gestite dall'orchestratore)")
            logging.info(f"    📊 Posizioni attive nell'orchestratore: {len(open_positions)}")
            
            engine = IntegratedRevolutionaryTradingEngine(capital=capital, open_positions=[], performance_db_path=str(AI_DB_FILE))
            engine.trade_history = closed_trades.copy()
            engine._register_historical_trades_in_ai()
            
            try:
                engine.run_integrated_trading_session_for_backtest(analysis_data_path=str(ANALYSIS_FILE_PATH), sp500_data_full=sp500_data, current_backtest_date=current_date)
            except Exception as e:
                logging.warning(f"    ⚠️ Errore nella generazione segnali (continuando backtest): {str(e)[:100]}...")
                
        except Exception as e:
            logging.error(f"    ❌ ERRORE nella generazione segnali: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        logging.info(f"  📊 Fine giornata: Capitale=${capital:,.2f}, Posizioni Aperte={len(open_positions)}, Trade Chiusi Totali={len(closed_trades)}")
        
        if day_num % 50 == 0:
            logging.info(f"\n🚀 PROGRESSO BACKTEST: {(day_num / len(date_range)) * 100:.1f}% completato ({day_num}/{len(date_range)} giorni)")
    
    logging.info(f"\n{'='*80}")
    logging.info("SIMULAZIONE COMPLETATA")
    logging.info("="*80)
    
    final_portfolio_value = capital
    logging.info(f"💰 Capitale finale: ${capital:,.2f}")
    
    if open_positions:
        logging.info(f"📊 Posizioni aperte da liquidare: {len(open_positions)}")
        for pos in open_positions:
            try:
                last_price = float(all_historical_data[pos['ticker']]['Close'].iloc[-1])
                position_value = pos['quantity'] * last_price
                final_portfolio_value += position_value
                logging.info(f"  - {pos['ticker']}: {pos['quantity']} azioni @ ${last_price:.2f} = ${position_value:,.2f}")
            except Exception:
                final_portfolio_value += pos['trade_value']
                logging.warning(f"  - {pos['ticker']}: Valore di acquisto ${pos['trade_value']:,.2f} (prezzo finale non disponibile)")
    
    logging.info(f"💎 Valore finale totale portafoglio: ${final_portfolio_value:,.2f}")
    logging.info(f"📈 Trade completati durante il backtest: {len(closed_trades)}")
    
    return closed_trades, final_portfolio_value

def save_backtest_results(closed_trades, final_value):
    """FASE 4: Salvataggio dei risultati del backtest"""
    logging.info("\n" + "="*80)
    logging.info("FASE 4: SALVATAGGIO DEI RISULTATI DEL BACKTEST")
    logging.info("="*80)
    
    if not closed_trades:
        logging.warning("⚠️ ATTENZIONE: Nessun trade è stato chiuso durante il backtest.")
        return
    
    df_results = pd.DataFrame(closed_trades)
    output_filename = "backtest_results.csv"
    df_results.to_csv(output_filename, index=False)
    
    logging.info(f"✅ Risultati salvati in '{output_filename}'")
    logging.info(f"\n📊 STATISTICHE BACKTEST:")
    logging.info(f"  📅 Periodo: {START_DATE} a {END_DATE}")
    logging.info(f"  💰 Capitale Iniziale: ${INITIAL_CAPITAL:,.2f}")
    logging.info(f"  💎 Valore Finale: ${final_value:,.2f}")
    
    profit = final_value - INITIAL_CAPITAL
    profit_pct = (profit / INITIAL_CAPITAL) * 100
    logging.info(f"  📈 Profitto/Perdita: ${profit:,.2f} ({profit_pct:+.2f}%)")
    
    logging.info(f"  🔄 Trade Chiusi: {len(closed_trades)}")
    
    if closed_trades:
        profitable_trades = [t for t in closed_trades if t['profit'] > 0]
        win_rate = (len(profitable_trades) / len(closed_trades)) * 100 if closed_trades else 0
        avg_profit = sum(t['profit'] for t in closed_trades) / len(closed_trades) if closed_trades else 0
        avg_hold_days = sum(t['hold_days'] for t in closed_trades) / len(closed_trades) if closed_trades else 0
        
        logging.info(f"  📊 Win Rate: {win_rate:.1f}% ({len(profitable_trades)}/{len(closed_trades)})")
        logging.info(f"  💵 Profitto Medio per Trade: ${avg_profit:,.2f}")
        logging.info(f"  ⏱️ Giorni di Holding Medi: {avg_hold_days:.1f}")
    
    logging.info("="*80)

if __name__ == "__main__":
    logging.info("🚀 AVVIO BACKTEST ORCHESTRATOR")
    logging.info("="*80)
    
    setup_backtest_environment()
    
    logging.info("ESECUZIONE SCREENING INIZIALE TITOLI...")
    tickers_for_backtest = run_one_time_screening_for_backtest()
    
    if not tickers_for_backtest:
        logging.error("❌ ERRORE CRITICO: Nessun ticker ottenuto dallo screening iniziale.")
        exit(1)
    
    logging.info(f"✅ Screening completato: {len(tickers_for_backtest)} titoli selezionati")
    
    historical_data_store = pre_fetch_all_historical_data(tickers_for_backtest)
    
    if not historical_data_store or '^GSPC' not in historical_data_store:
        logging.error("❌ ERRORE CRITICO: Dati storici o S&P 500 non disponibili.")
        exit(1)
    
    logging.info(f"✅ Dati storici pronti per {len(historical_data_store)} simboli")
    
    final_trades, final_portfolio_value = run_backtest_simulation(
        all_historical_data=historical_data_store,
        tickers_to_analyze=tickers_for_backtest
    )
    
    save_backtest_results(final_trades, final_portfolio_value)
    
    logging.info("\n🎉 BACKTEST COMPLETATO CON SUCCESSO!")
    logging.info(f"   Controlla il file 'backtest_results.csv' per i risultati dettagliati.")
    logging.info(f"   Il log completo di questa esecuzione è stato salvato in: {log_filepath}")
