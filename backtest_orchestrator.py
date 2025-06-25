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


import logging
from pathlib import Path
from datetime import datetime

# Definisci le directory qui, in modo che siano accessibili
BASE_DIR = Path.cwd()
REPORTS_DIR = BASE_DIR / "data_backtest" / "reports"
# Assicurati che la cartella dei report esista
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Definisci il nome del file di log (fisso, per sovrascriverlo)
log_filepath = REPORTS_DIR / "backtest_log.txt"

# 2. Configura il sistema di logging.
#    - level=logging.INFO: Cattura tutti i messaggi informativi, di avviso e di errore.
#    - format: Definisce come appare ogni riga del log (data, ora, livello, messaggio).
#    - handlers: Specifica dove inviare i log.
#    - filemode='w': Questa è la chiave! 'w' sta per 'write', e dice a Python
#      di sovrascrivere il file ogni volta che lo script parte.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath, mode='w'), # Sovrascrive il file ad ogni esecuzione
        logging.StreamHandler() # Mantiene l'output anche nel terminale
    ]
)


# --- IMPORTAZIONE DELLE LOGICHE MODIFICATE ---
try:
    from best_buy_backtest import run_one_time_screening_for_backtest
    from stock_analyzer_backtest import run_analysis_for_date
    from trading_engine_backtest import IntegratedRevolutionaryTradingEngine
except ImportError as e:
    print(f"ERRORE CRITICO: Assicurati che i file 'best_buy_backtest.py', 'stock_analyzer_backtest.py', e 'trading_engine_backtest.py' esistano e siano nella stessa cartella.")
    print(f"Dettaglio errore: {e}")
    exit()

# --- CONFIGURAZIONE DEL BACKTEST ---
START_DATE = '2015-01-01'
END_DATE = '2015-03-31'
INITIAL_CAPITAL = 100000.0

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data_backtest"
SIGNALS_DIR = BASE_DIR / "data"  # Directory dove il trading engine salva i segnali

AI_LEARNING_DIR = DATA_DIR / "ai_learning"
REPORTS_DIR = DATA_DIR / "reports"
SIGNALS_HISTORY_DIR = DATA_DIR / "signals_history"
AI_DB_FILE = AI_LEARNING_DIR / "performance.db"
ANALYSIS_FILE_PATH = DATA_DIR / "latest_analysis.json"
EXECUTION_SIGNALS_FILE = SIGNALS_DIR / "execution_signals.json"
HISTORICAL_EXECUTION_SIGNALS_FILE = SIGNALS_DIR / "historical_execution_signals.json"

def setup_backtest_environment():
    """FASE 1: Setup dell'ambiente di backtest"""
    logging.info("FASE 1: Setup dell'ambiente di backtest...")
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        logging.info("  - Rimossa directory backtest esistente")
    
    # Crea tutte le directory necessarie
    for directory in [DATA_DIR, AI_LEARNING_DIR, REPORTS_DIR, SIGNALS_HISTORY_DIR, SIGNALS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logging.info(f"  - Creata directory: {directory}")
    
    # Inizializza il database AI vuoto
    AI_DB_FILE.touch()
    logging.info(f"  - Inizializzato database AI: {AI_DB_FILE}")
    
    # Inizializza il file dei segnali storici
    with open(HISTORICAL_EXECUTION_SIGNALS_FILE, 'w') as f:
        json.dump({"historical_signals": [], "last_updated": "", "total_signals": 0}, f, indent=2)
    logging.info(f"  - Inizializzato file segnali storici: {HISTORICAL_EXECUTION_SIGNALS_FILE}")
    
    logging.info("✅ Ambiente di backtest pronto.\n")

def pre_fetch_all_historical_data(tickers):
    """FASE 2: Download di tutti i dati storici necessari"""
    logging.info("FASE 2: Download di tutti i dati storici...")
    
    # Scarica dati extra per indicatori tecnici
    fetch_start_date = (pd.to_datetime(START_DATE) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end_date = (pd.to_datetime(END_DATE) + timedelta(days=1)).strftime('%Y-%m-%d')
    
    logging.info(f"  - Periodo download: {fetch_start_date} a {fetch_end_date}")
    logging.info(f"  - Titoli da scaricare: {len(tickers)} + S&P 500")
    
    all_data = {}
    tickers_to_fetch = tickers + ['^GSPC']  # Include S&P 500
    
    successful_downloads = 0
    for i, ticker in enumerate(tickers_to_fetch):
        try:
            time.sleep(0.1)  # Rate limiting gentile
            print(f"  Scaricando {ticker} ({i+1}/{len(tickers_to_fetch)})...", end=" ")  # Lascia questo print per vedere il progresso in tempo reale
            
            data = yf.download(
                ticker, 
                start=fetch_start_date, 
                end=fetch_end_date, 
                progress=False, 
                timeout=10,
                auto_adjust=True  # Prezzi aggiustati automaticamente
            )
            
            if not data.empty and len(data) > 100:  # Almeno 100 giorni di dati
                all_data[ticker] = data
                successful_downloads += 1
                print("✅")
            else:
                print("❌ (dati insufficienti)")
                
        except Exception as e:
            print(f"❌ (errore: {str(e)[:50]}...)")
    
    logging.info(f"\n✅ Download completato: {successful_downloads}/{len(tickers_to_fetch)} titoli scaricati con successo.")
    return all_data

def convert_positions_for_trading_engine(orchestrator_positions, current_date):
    """Converte le posizioni dell'orchestratore nel formato atteso dal trading engine"""
    engine_positions = []
    
    for pos in orchestrator_positions:
        # Converte nel formato che il trading engine si aspetta per generate_sell_signals
        engine_position = {
            'ticker': pos['ticker'],
            'entry': pos['entry_price'],
            'date': pos['entry_date'].isoformat(),  # Trading engine si aspetta formato ISO string
            'quantity': pos['quantity'],
            'amount_invested': pos['trade_value'],
            'take_profit': pos.get('take_profit'),
            'stop_loss': pos.get('stop_loss'),
            # Aggiungi tutti i campi che il trading engine potrebbe cercare
            'entry_price': pos['entry_price'],
            'entry_date': pos['entry_date'],
            'position_id': pos.get('position_id', f"{pos['ticker']}_{current_date.strftime('%Y%m%d')}")
        }
        engine_positions.append(engine_position)
    
    return engine_positions

def execute_signals_for_day(signals, all_historical_data, current_date, capital, open_positions, closed_trades):
    """Esegue i segnali di trading per il giorno corrente"""
    if not signals:
        return capital, open_positions, closed_trades
    
    current_date_str = current_date.strftime('%Y-%m-%d')
    print(f"  📈 Esecuzione segnali per {current_date_str}...")
    
    # PASSO 1: ESECUZIONE VENDITE (priorità alta)
    sell_signals = signals.get('sells', [])
    if sell_signals:
        print(f"    🔄 Processando {len(sell_signals)} segnali di vendita...")
        positions_to_remove = []
        
        for sell_signal in sell_signals:
            ticker = sell_signal.get('ticker')
            if not ticker:
                continue
                
            # Trova la posizione corrispondente
            matching_position = next((pos for pos in open_positions if pos['ticker'] == ticker), None)
            if not matching_position:
                print(f"      ⚠️ VENDITA SALTATA: {ticker} non trovato in posizioni aperte")
                continue
            
            try:
                # Usa il prezzo di apertura del giorno corrente
                if ticker not in all_historical_data:
                    print(f"      ⚠️ VENDITA SALTATA: Nessun dato per {ticker}")
                    continue
                    
                ticker_data = all_historical_data[ticker]
                if current_date_str not in ticker_data.index:
                    print(f"      ⚠️ VENDITA SALTATA: Nessun dato per {ticker} il {current_date_str}")
                    continue
                
                exit_price = float(ticker_data.loc[current_date_str]['Open'])
                quantity = matching_position['quantity']
                exit_value = exit_price * quantity
                
                # Aggiorna capitale
                capital += exit_value
                
                # Registra trade chiuso con TUTTI i campi necessari per l'AI
                trade_record = {
                    'ticker': ticker,
                    'date': matching_position['entry_date'].strftime('%Y-%m-%d'),  # Data di entrata
                    'entry_date': matching_position['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': current_date_str,
                    'entry': matching_position['entry_price'],
                    'entry_price': matching_position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'amount_invested': matching_position['trade_value'],
                    'profit': exit_value - matching_position['trade_value'],
                    'profit_percentage': ((exit_value - matching_position['trade_value']) / matching_position['trade_value']) * 100,
                    'hold_days': (current_date - matching_position['entry_date']).days,
                    'sell_reason': sell_signal.get('reason', 'Strategy Exit'),
                    'regime_at_buy': 'unknown',  # Potremmo salvarlo quando creiamo la posizione
                    'method': 'Backtest_System',
                    'ref_score_or_roi': 12.0,  # Valore di default
                    'advanced_indicators_at_buy': {},  # Vuoto per ora
                    'ai_evaluation_details': {}  # Vuoto per ora
                }
                
                closed_trades.append(trade_record)
                positions_to_remove.append(matching_position)
                
                print(f"      ✅ VENDUTO: {quantity} {ticker} @ ${exit_price:.2f} = ${exit_value:,.2f} (P/L: {trade_record['profit_percentage']:+.1f}%)")
                
            except Exception as e:
                print(f"      ❌ ERRORE VENDITA {ticker}: {e}")
        
        # Rimuovi posizioni vendute
        for pos in positions_to_remove:
            open_positions.remove(pos)
        
        print(f"    ✅ Vendite completate. Capitale aggiornato: ${capital:,.2f}")
    else:
        print("    ℹ️ Nessun segnale di vendita da processare")
    
    # PASSO 2: ESECUZIONE ACQUISTI
    buy_signals = signals.get('buys', [])
    if buy_signals:
        print(f"    🔄 Processando {len(buy_signals)} segnali di acquisto...")
        
        for buy_signal in buy_signals:
            ticker = buy_signal.get('ticker')
            quantity_estimated = buy_signal.get('quantity_estimated', 0)
            
            if not ticker or quantity_estimated <= 0:
                print(f"      ⚠️ ACQUISTO SALTATO: Dati segnale invalidi per {ticker}")
                continue
            
            try:
                # Usa il prezzo di apertura del giorno corrente
                if ticker not in all_historical_data:
                    print(f"      ⚠️ ACQUISTO SALTATO: Nessun dato per {ticker}")
                    continue
                    
                ticker_data = all_historical_data[ticker]
                if current_date_str not in ticker_data.index:
                    print(f"      ⚠️ ACQUISTO SALTATO: Nessun dato per {ticker} il {current_date_str}")
                    continue
                
                entry_price = float(ticker_data.loc[current_date_str]['Open'])
                trade_value = entry_price * quantity_estimated
                
                # Verifica disponibilità capitale
                if capital >= trade_value:
                    # Esegui acquisto
                    capital -= trade_value
                    
                    # Crea nuova posizione con tutti i campi necessari
                    new_position = {
                        'ticker': ticker,
                        'entry_price': entry_price,
                        'quantity': quantity_estimated,
                        'trade_value': trade_value,
                        'entry_date': current_date,
                        'stop_loss': buy_signal.get('stop_loss'),
                        'take_profit': buy_signal.get('take_profit'),
                        'position_id': f"{ticker}_{current_date.strftime('%Y%m%d')}_{int(entry_price*100)}"
                    }
                    
                    open_positions.append(new_position)
                    
                    print(f"      ✅ ACQUISTATO: {quantity_estimated} {ticker} @ ${entry_price:.2f} = ${trade_value:,.2f}")
                else:
                    print(f"      ⚠️ ACQUISTO SALTATO: {ticker} - Capitale insufficiente (serve ${trade_value:,.2f}, disponibile ${capital:,.2f})")
                    
            except Exception as e:
                print(f"      ❌ ERRORE ACQUISTO {ticker}: {e}")
        
        print(f"    ✅ Acquisti completati. Capitale rimanente: ${capital:,.2f}")
    else:
        print("    ℹ️ Nessun segnale di acquisto da processare")
    
    return capital, open_positions, closed_trades

# In backtest_orchestrator.py, sostituisci l'INTERA funzione run_backtest_simulation

def run_backtest_simulation(all_historical_data, tickers_to_analyze):
    """FASE 3: Simulazione di trading giornaliera con sincronizzazione del DB AI."""
    logging.info("\n" + "="*80)
    logging.info("FASE 3: INIZIO SIMULAZIONE DI TRADING GIORNALIERA")
    logging.info("="*80)
    
    # INIZIALIZZAZIONE STATO PORTAFOGLIO
    capital = INITIAL_CAPITAL
    open_positions = []
    closed_trades = []
    
    # Definiamo i percorsi del DB qui per chiarezza
    PERSISTENT_AI_DB_FILE = AI_DB_FILE # Il DB gestito dall'orchestratore
    ENGINE_AI_DIR = BASE_DIR / "data" / "ai_learning" # La cartella che l'engine si aspetta
    ENGINE_AI_DB_FILE = ENGINE_AI_DIR / "performance.db" # Il DB che l'engine usa

    # Genera range di date di trading
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    sp500_data = all_historical_data.get('^GSPC')
    
    if sp500_data is None or sp500_data.empty:
        print("❌ ERRORE: Dati S&P 500 non disponibili per la simulazione")
        return closed_trades, capital
    
    print(f"📅 Periodo simulazione: {len(date_range)} giorni lavorativi")
    print(f"💰 Capitale iniziale: ${capital:,.2f}")
    print(f"📊 Titoli nel portafoglio: {len(tickers_to_analyze)}")
    print()
    
    # LOOP PRINCIPALE DI SIMULAZIONE
    for day_num, current_date in enumerate(date_range, 1):
        current_date_str = current_date.strftime('%Y-%m-%d')
        
        print(f"\n{'='*20} GIORNO {day_num}/{len(date_range)}: {current_date_str} {'='*20}")
        
        portfolio_value = capital
        for pos in open_positions:
            try:
                current_price_data = all_historical_data[pos['ticker']].loc[current_date_str]['Close']
                portfolio_value += pos['quantity'] * float(current_price_data)
            except:
                portfolio_value += pos['trade_value']
        
        print(f"💰 Stato Portfolio: Capitale=${capital:,.2f}, Posizioni={len(open_positions)}, Valore Totale≈${portfolio_value:,.2f}")
        
        if EXECUTION_SIGNALS_FILE.exists():
            print("  📋 Esecuzione ordini del giorno precedente...")
            try:
                with open(EXECUTION_SIGNALS_FILE, 'r') as f:
                    signals = json.load(f).get('signals', {})
                if signals:
                    capital, open_positions, closed_trades = execute_signals_for_day(
                        signals, all_historical_data, current_date, capital, open_positions, closed_trades
                    )
                else:
                    print("    ℹ️ Nessun segnale da eseguire")
                os.remove(EXECUTION_SIGNALS_FILE)
            except Exception as e:
                print(f"    ❌ ERRORE processando segnali: {e}")
        else:
            print("  ℹ️ Nessun file di segnali da processare")
        
        print("  🔍 Generazione analisi e segnali per domani...")
        try:
            run_analysis_for_date(tickers_to_analyze, all_historical_data, current_date, ANALYSIS_FILE_PATH)
            
            engine_positions = convert_positions_for_trading_engine(open_positions, current_date)
            
            # --- INIZIO BLOCCO DI SINCRONIZZAZIONE DB (Pre-Esecuzione) ---
            print("    🔄 Sincronizzazione Database AI (Pre-Esecuzione)...")
            try:
                ENGINE_AI_DIR.mkdir(parents=True, exist_ok=True)
                if PERSISTENT_AI_DB_FILE.exists():
                    shutil.copy2(PERSISTENT_AI_DB_FILE, ENGINE_AI_DB_FILE)
                    print(f"      ✅ Copiato DB persistente in {ENGINE_AI_DB_FILE} per l'engine.")
                else:
                    print(f"      ℹ️ Nessun DB persistente trovato. L'engine ne creerà uno nuovo.")
            except Exception as sync_err:
                print(f"      ❌ Errore durante la sincronizzazione Pre-Esecuzione: {sync_err}")
            # --- FINE BLOCCO DI SINCRONIZZAZIONE DB ---

            print(f"    🔧 Inizializzazione trading engine con {len(engine_positions)} posizioni aperte")
            
            # L'engine ora troverà il DB nel suo percorso atteso "data/ai_learning/performance.db"
            engine = IntegratedRevolutionaryTradingEngine(
                capital=capital,
                open_positions=engine_positions,
                performance_db_path=str(ENGINE_AI_DB_FILE) # Passiamo il percorso che userà
            )
            
            engine.trade_history = closed_trades.copy()
            engine._register_historical_trades_in_ai()
            
            print(f"    📊 Trading engine configurato: Capital=${capital:,.2f}, Posizioni={len(engine_positions)}")
            if engine_positions:
                print(f"    📋 Posizioni da valutare per vendita: {[pos['ticker'] for pos in engine_positions]}")

            try:
                engine.run_integrated_trading_session_for_backtest(
                    analysis_data_path=str(ANALYSIS_FILE_PATH),
                    sp500_data_full=sp500_data,
                    current_backtest_date=current_date
                )
                print("    ✅ Sessione di trading dell'engine completata.")
            except Exception as e:
                print(f"    ⚠️ Errore nell'esecuzione dell'engine: {str(e)[:100]}...")
                import traceback
                traceback.print_exc()

            # --- INIZIO BLOCCO DI SINCRONIZZAZIONE DB (Post-Esecuzione) ---
            print("    🔄 Sincronizzazione Database AI (Post-Esecuzione)...")
            try:
                if ENGINE_AI_DB_FILE.exists():
                    shutil.copy2(ENGINE_AI_DB_FILE, PERSISTENT_AI_DB_FILE)
                    print(f"      ✅ Copiato DB aggiornato dall'engine in {PERSISTENT_AI_DB_FILE} per la persistenza.")
                else:
                    print(f"      ⚠️ L'engine non ha creato/modificato il suo file DB. Nessuna sincronizzazione Post-Esecuzione.")
            except Exception as sync_err:
                print(f"      ❌ Errore durante la sincronizzazione Post-Esecuzione: {sync_err}")
            # --- FINE BLOCCO DI SINCRONIZZAZIONE DB ---

        except Exception as e:
            print(f"    ❌ ERRORE nella generazione segnali: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"  📊 Fine giornata: Capitale=${capital:,.2f}, Posizioni Aperte={len(open_positions)}, Trade Chiusi Totali={len(closed_trades)}")
        if len(closed_trades) > 0:
            last_trade = closed_trades[-1]
            print(f"  🔍 Ultimo trade chiuso: {last_trade['ticker']} P/L={last_trade['profit_percentage']:+.1f}%")
        
        if day_num % 50 == 0:
            progress_pct = (day_num / len(date_range)) * 100
            print(f"\n🚀 PROGRESSO BACKTEST: {progress_pct:.1f}% completato ({day_num}/{len(date_range)} giorni)")
            print(f"   📊 Trade chiusi finora: {len(closed_trades)}")
    
    print(f"\n{'='*80}")
    print("SIMULAZIONE COMPLETATA")
    print("="*80)
    
    final_portfolio_value = capital
    print(f"💰 Capitale finale: ${capital:,.2f}")
    
    if open_positions:
        print(f"📊 Posizioni aperte da liquidare: {len(open_positions)}")
        for pos in open_positions:
            try:
                last_price = float(all_historical_data[pos['ticker']]['Close'].iloc[-1])
                position_value = pos['quantity'] * last_price
                final_portfolio_value += position_value
                print(f"  - {pos['ticker']}: {pos['quantity']} azioni @ ${last_price:.2f} = ${position_value:,.2f}")
            except:
                final_portfolio_value += pos['trade_value']
                print(f"  - {pos['ticker']}: Valore di acquisto ${pos['trade_value']:,.2f} (prezzo finale non disponibile)")
    
    print(f"💎 Valore finale totale portafoglio: ${final_portfolio_value:,.2f}")
    print(f"📈 Trade completati durante il backtest: {len(closed_trades)}")
    
    return closed_trades, final_portfolio_value


def save_backtest_results(closed_trades, final_value):
    """FASE 4: Salvataggio dei risultati del backtest"""
    logging.info("\n" + "="*80)
    logging.info("FASE 4: SALVATAGGIO DEI RISULTATI DEL BACKTEST")
    logging.info("="*80)
    
    if not closed_trades:
        logging.warning("⚠️ ATTENZIONE: Nessun trade è stato chiuso durante il backtest.")
        logging.info("   Questo può indicare:")
        logging.info("   - Sistema troppo conservativo")
        logging.info("   - Mancanza di segnali di vendita")
        logging.info("   - Errori nella generazione dei segnali")
        logging.info("   Il file CSV non verrà creato.")
        return
    
    # Prepara DataFrame per CSV
    trades_for_csv = []
    for trade in closed_trades:
        trades_for_csv.append({
            'entry_price': trade['entry_price'],
            'exit_price': trade['exit_price'],
            'quantity': trade['quantity']
        })
    
    df_results = pd.DataFrame(trades_for_csv)
    output_filename = "backtest_results.csv"
    df_results.to_csv(output_filename, index=False)
    
    # Statistiche dettagliate
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
        win_rate = (len(profitable_trades) / len(closed_trades)) * 100
        avg_profit = sum(t['profit'] for t in closed_trades) / len(closed_trades)
        avg_hold_days = sum(t['hold_days'] for t in closed_trades) / len(closed_trades)
        
        logging.info(f"  📊 Win Rate: {win_rate:.1f}% ({len(profitable_trades)}/{len(closed_trades)})")
        logging.info(f"  💵 Profitto Medio per Trade: ${avg_profit:,.2f}")
        logging.info(f"  ⏱️ Giorni di Holding Medi: {avg_hold_days:.1f}")
        
        # Verifica se il database AI è stato popolato
        try:
            with sqlite3.connect(AI_DB_FILE) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM trades WHERE exit_date IS NOT NULL')
                ai_trades_count = cursor.fetchone()[0]
                logging.info(f"  🧠 Trade nel database AI: {ai_trades_count}")
                
                if ai_trades_count >= 80:
                    logging.info("  🎉 DATABASE AI PRONTO! L'AI può ora essere attivata per trading reale.")
                else:
                    logging.info(f"  ⏳ Database AI in crescita: {80 - ai_trades_count} trade ancora necessari per attivazione completa")
        except Exception as e:
            logging.warning(f"  ⚠️ Errore verifica database AI: {e}")
    
    logging.info("="*80)


if __name__ == "__main__":
    # --- INIZIO BLOCCO MODIFICATO ---

    # Import necessari per il logging e la gestione dei file

    logging.info("🚀 AVVIO BACKTEST ORCHESTRATOR")
    logging.info("="*80)

    # FASE 1: Setup ambiente
    # Anche i print() dentro questa funzione verranno catturati se lo script
    # è eseguito da un terminale, ma per coerenza convertiamo i messaggi principali.
    setup_backtest_environment()

    # FASE 2: Screening iniziale titoli
    logging.info("ESECUZIONE SCREENING INIZIALE TITOLI...")
    tickers_for_backtest = run_one_time_screening_for_backtest()

    if not tickers_for_backtest:
        logging.error("❌ ERRORE CRITICO: Nessun ticker ottenuto dallo screening iniziale.")
        logging.error("   Controlla la connessione internet e riprova.")
        exit(1)

    logging.info(f"✅ Screening completato: {len(tickers_for_backtest)} titoli selezionati")

    # FASE 3: Download dati storici
    historical_data_store = pre_fetch_all_historical_data(tickers_for_backtest)

    if not historical_data_store:
        logging.error("❌ ERRORE CRITICO: Nessun dato storico scaricato.")
        logging.error("   Controlla la connessione internet e riprova.")
        exit(1)

    if '^GSPC' not in historical_data_store:
        logging.error("❌ ERRORE CRITICO: Dati S&P 500 non disponibili.")
        exit(1)

    logging.info(f"✅ Dati storici pronti per {len(historical_data_store)} simboli")

    # FASE 4: Esecuzione simulazione
    final_trades, final_portfolio_value = run_backtest_simulation(
        all_historical_data=historical_data_store,
        tickers_to_analyze=tickers_for_backtest
    )

    # FASE 5: Salvataggio risultati
    save_backtest_results(final_trades, final_portfolio_value)

    logging.info("\n🎉 BACKTEST COMPLETATO CON SUCCESSO!")
    logging.info("   Controlla il file 'backtest_results.csv' per i risultati dettagliati.")
    logging.info(f"   Il log completo di questa esecuzione è stato salvato in: {log_filepath}")
