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
    from trading_engine_backtest import IntegratedRevolutionaryTradingEngine  # CORREZIONE: Nome corretto
except ImportError as e:
    print(f"ERRORE CRITICO: Assicurati che i file 'best_buy_backtest.py', 'stock_analyzer_backtest.py', e 'trading_engine_backtest.py' esistano e siano nella stessa cartella.")
    print(f"Dettaglio errore: {e}")
    exit()

# --- CONFIGURAZIONE DEL BACKTEST ---
START_DATE = '2015-01-01'
END_DATE = '2015-12-31'
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
    print("FASE 1: Setup dell'ambiente di backtest...")
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        print("  - Rimossa directory backtest esistente")
    
    # Crea tutte le directory necessarie
    for directory in [DATA_DIR, AI_LEARNING_DIR, REPORTS_DIR, SIGNALS_HISTORY_DIR, SIGNALS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  - Creata directory: {directory}")
    
    # Inizializza il database AI vuoto
    AI_DB_FILE.touch()
    print(f"  - Inizializzato database AI: {AI_DB_FILE}")
    
    # Inizializza il file dei segnali storici
    with open(HISTORICAL_EXECUTION_SIGNALS_FILE, 'w') as f:
        json.dump({"historical_signals": [], "last_updated": "", "total_signals": 0}, f, indent=2)
    print(f"  - Inizializzato file segnali storici: {HISTORICAL_EXECUTION_SIGNALS_FILE}")
    
    print("✅ Ambiente di backtest pronto.\n")

def pre_fetch_all_historical_data(tickers):
    """FASE 2: Download di tutti i dati storici necessari"""
    print("FASE 2: Download di tutti i dati storici...")
    
    # Scarica dati extra per indicatori tecnici
    fetch_start_date = (pd.to_datetime(START_DATE) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end_date = (pd.to_datetime(END_DATE) + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"  - Periodo download: {fetch_start_date} a {fetch_end_date}")
    print(f"  - Titoli da scaricare: {len(tickers)} + S&P 500")
    
    all_data = {}
    tickers_to_fetch = tickers + ['^GSPC']  # Include S&P 500
    
    successful_downloads = 0
    for i, ticker in enumerate(tickers_to_fetch):
        try:
            time.sleep(0.1)  # Rate limiting gentile
            print(f"  Scaricando {ticker} ({i+1}/{len(tickers_to_fetch)})...", end=" ")
            
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
    
    print(f"\n✅ Download completato: {successful_downloads}/{len(tickers_to_fetch)} titoli scaricati con successo.")
    return all_data

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
                
                # Registra trade chiuso
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
                
                print(f"      ✅ VENDUTO: {quantity} {ticker} @ ${exit_price:.2f} = ${exit_value:,.2f} (P/L: {trade_record['profit_pct']:+.1f}%)")
                
            except Exception as e:
                print(f"      ❌ ERRORE VENDITA {ticker}: {e}")
        
        # Rimuovi posizioni vendute
        for pos in positions_to_remove:
            open_positions.remove(pos)
        
        print(f"    ✅ Vendite completate. Capitale aggiornato: ${capital:,.2f}")
    
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
                    
                    # CORREZIONE: Aggiungi tutte le informazioni necessarie per il trading engine
                    new_position = {
                        'ticker': ticker,
                        'entry_price': entry_price,
                        'quantity': quantity_estimated,
                        'trade_value': trade_value,
                        'entry_date': current_date,
                        'stop_loss': buy_signal.get('stop_loss'),
                        'take_profit': buy_signal.get('take_profit'),
                        # Aggiungi informazioni aggiuntive che il trading engine potrebbe cercare
                        'entry_p': entry_price,  # Alias per compatibilità
                        'entry_d': current_date,  # Alias per compatibilità
                        'entry_date_str': current_date.strftime('%Y-%m-%d'),
                        'position_id': f"{ticker}_{current_date.strftime('%Y%m%d')}_{int(entry_price*100)}"
                    }
                    
                    open_positions.append(new_position)
                    
                    print(f"      ✅ ACQUISTATO: {quantity_estimated} {ticker} @ ${entry_price:.2f} = ${trade_value:,.2f}")
                else:
                    print(f"      ⚠️ ACQUISTO SALTATO: {ticker} - Capitale insufficiente (serve ${trade_value:,.2f}, disponibile ${capital:,.2f})")
                    
            except Exception as e:
                print(f"      ❌ ERRORE ACQUISTO {ticker}: {e}")
        
        print(f"    ✅ Acquisti completati. Capitale rimanente: ${capital:,.2f}")
    
    return capital, open_positions, closed_trades

def run_backtest_simulation(all_historical_data, tickers_to_analyze):
    """FASE 3: Simulazione di trading giornaliera"""
    print("\n" + "="*80)
    print("FASE 3: INIZIO SIMULAZIONE DI TRADING GIORNALIERA")
    print("="*80)
    
    # INIZIALIZZAZIONE STATO PORTAFOGLIO
    capital = INITIAL_CAPITAL
    open_positions = []
    closed_trades = []
    
    # Genera range di date di trading (solo giorni lavorativi)
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
        
        # STATO INIZIALE DEL GIORNO
        portfolio_value = capital
        for pos in open_positions:
            try:
                current_price_data = all_historical_data[pos['ticker']].loc[current_date_str]['Close']
                # CORREZIONE: Assicurati che sia un numero singolo
                if hasattr(current_price_data, 'iloc'):
                    current_price = float(current_price_data.iloc[0])
                else:
                    current_price = float(current_price_data)
                portfolio_value += pos['quantity'] * current_price
            except:
                portfolio_value += pos['trade_value']  # Fallback al valore di acquisto
        
        print(f"💰 Stato Portfolio: Capitale=${capital:,.2f}, Posizioni={len(open_positions)}, Valore Totale≈${portfolio_value:,.2f}")
        
        # STEP 1: ESECUZIONE ORDINI DEL GIORNO PRECEDENTE
        if EXECUTION_SIGNALS_FILE.exists():
            print("  📋 Esecuzione ordini del giorno precedente...")
            try:
                with open(EXECUTION_SIGNALS_FILE, 'r') as f:
                    signals_data = json.load(f)
                
                signals = signals_data.get('signals', {})
                if signals:
                    capital, open_positions, closed_trades = execute_signals_for_day(
                        signals, all_historical_data, current_date, capital, open_positions, closed_trades
                    )
                else:
                    print("    ℹ️ Nessun segnale da eseguire")
                
                # Rimuovi il file dei segnali dopo l'esecuzione
                os.remove(EXECUTION_SIGNALS_FILE)
                
            except Exception as e:
                print(f"    ❌ ERRORE processando segnali: {e}")
        else:
            print("  ℹ️ Nessun file di segnali da processare")
        
        # STEP 2: GENERAZIONE ANALISI E NUOVI SEGNALI
        print("  🔍 Generazione analisi e segnali per domani...")
        try:
            # Genera analisi per la data corrente
            run_analysis_for_date(tickers_to_analyze, all_historical_data, current_date, ANALYSIS_FILE_PATH)
            
            # Istanza del trading engine con stato corrente
            # CORREZIONE: Converti le posizioni nel formato atteso dal trading engine
            trading_engine_positions = []
            for pos in open_positions:
                trading_engine_pos = {
                    'ticker': pos['ticker'],
                    'entry_p': pos['entry_price'],  # Il trading engine cerca 'entry_p'
                    'entry_d': pos['entry_date'].strftime('%Y-%m-%d'),  # Il trading engine cerca 'entry_d' come string
                    'quantity': pos['quantity'],
                    'trade_value': pos['trade_value'],
                    'entry_price': pos['entry_price'],  # Mantieni anche l'originale
                    'entry_date': pos['entry_date'],
                    'stop_loss': pos.get('stop_loss'),
                    'take_profit': pos.get('take_profit'),
                    'position_id': f"{pos['ticker']}_{pos['entry_date'].strftime('%Y%m%d')}_{int(pos['entry_price']*100)}"
                }
                trading_engine_positions.append(trading_engine_pos)
            
            # Istanza del trading engine con stato corrente
            engine = IntegratedRevolutionaryTradingEngine(
                capital=capital,
                open_positions=trading_engine_positions,  # Usa le posizioni convertite
                performance_db_path=str(AI_DB_FILE)
            )
            
            # Passa la cronologia dei trade all'engine per l'apprendimento AI
            engine.trade_history = closed_trades.copy()
            
            # Esegue la sessione di trading per generare i segnali
            try:
                success = engine.run_integrated_trading_session_for_backtest(
                    analysis_data_path=str(ANALYSIS_FILE_PATH),
                    sp500_data_full=sp500_data,
                    current_backtest_date=current_date
                )
            
                if success:
                    print("    ✅ Segnali generati con successo")
                else:
                    print("    ⚠️ Generazione segnali completata con avvisi")
                    
            except Exception as e:
                print(f"    ⚠️ Errore nella generazione segnali (continuando backtest): {str(e)[:100]}...")
                # Continua il backtest anche se ci sono errori nella generazione segnali
                # Questo evita che il backtest si fermi per errori non critici
                
        except Exception as e:
            print(f"    ❌ ERRORE nella generazione segnali: {e}")
            import traceback
            traceback.print_exc()
        
        # LOG FINALE DEL GIORNO
        print(f"  📊 Fine giornata: Capitale=${capital:,.2f}, Posizioni Aperte={len(open_positions)}, Trade Chiusi Totali={len(closed_trades)}")
        
        # Progress update ogni 50 giorni
        if day_num % 50 == 0:
            progress_pct = (day_num / len(date_range)) * 100
            print(f"\n🚀 PROGRESSO BACKTEST: {progress_pct:.1f}% completato ({day_num}/{len(date_range)} giorni)")
    
    # CALCOLO VALORE FINALE
    print(f"\n{'='*80}")
    print("SIMULAZIONE COMPLETATA")
    print("="*80)
    
    final_portfolio_value = capital
    print(f"💰 Capitale finale: ${capital:,.2f}")
    
    if open_positions:
        print(f"📊 Posizioni aperte da liquidare: {len(open_positions)}")
        for pos in open_positions:
            try:
                last_price_data = all_historical_data[pos['ticker']]['Close'].iloc[-1]
                # CORREZIONE: Assicurati che sia un numero singolo
                last_price = float(last_price_data)
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
    print("\n" + "="*80)
    print("FASE 4: SALVATAGGIO DEI RISULTATI DEL BACKTEST")
    print("="*80)
    
    if not closed_trades:
        print("⚠️ ATTENZIONE: Nessun trade è stato chiuso durante il backtest.")
        print("   Questo può indicare:")
        print("   - Sistema troppo conservativo")
        print("   - Mancanza di segnali di vendita")
        print("   - Errori nella generazione dei segnali")
        print("   Il file CSV non verrà creato.")
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
    print(f"✅ Risultati salvati in '{output_filename}'")
    print(f"\n📊 STATISTICHE BACKTEST:")
    print(f"  📅 Periodo: {START_DATE} a {END_DATE}")
    print(f"  💰 Capitale Iniziale: ${INITIAL_CAPITAL:,.2f}")
    print(f"  💎 Valore Finale: ${final_value:,.2f}")
    
    profit = final_value - INITIAL_CAPITAL
    profit_pct = (profit / INITIAL_CAPITAL) * 100
    print(f"  📈 Profitto/Perdita: ${profit:,.2f} ({profit_pct:+.2f}%)")
    
    print(f"  🔄 Trade Chiusi: {len(closed_trades)}")
    
    if closed_trades:
        profitable_trades = [t for t in closed_trades if t['profit'] > 0]
        win_rate = (len(profitable_trades) / len(closed_trades)) * 100
        avg_profit = sum(t['profit'] for t in closed_trades) / len(closed_trades)
        avg_hold_days = sum(t['hold_days'] for t in closed_trades) / len(closed_trades)
        
        print(f"  📊 Win Rate: {win_rate:.1f}% ({len(profitable_trades)}/{len(closed_trades)})")
        print(f"  💵 Profitto Medio per Trade: ${avg_profit:,.2f}")
        print(f"  ⏱️ Giorni di Holding Medi: {avg_hold_days:.1f}")
    
    print("="*80)

if __name__ == "__main__":
    print("🚀 AVVIO BACKTEST ORCHESTRATOR")
    print("="*80)
    
    # FASE 1: Setup ambiente
    setup_backtest_environment()
    
    # FASE 2: Screening iniziale titoli
    print("ESECUZIONE SCREENING INIZIALE TITOLI...")
    tickers_for_backtest = run_one_time_screening_for_backtest()
    
    if not tickers_for_backtest:
        print("❌ ERRORE CRITICO: Nessun ticker ottenuto dallo screening iniziale.")
        print("   Controlla la connessione internet e riprova.")
        exit(1)
    
    print(f"✅ Screening completato: {len(tickers_for_backtest)} titoli selezionati")
    
    # FASE 3: Download dati storici
    historical_data_store = pre_fetch_all_historical_data(tickers_for_backtest)
    
    if not historical_data_store:
        print("❌ ERRORE CRITICO: Nessun dato storico scaricato.")
        print("   Controlla la connessione internet e riprova.")
        exit(1)
    
    if '^GSPC' not in historical_data_store:
        print("❌ ERRORE CRITICO: Dati S&P 500 non disponibili.")
        exit(1)
    
    print(f"✅ Dati storici pronti per {len(historical_data_store)} simboli")
    
    # FASE 4: Esecuzione simulazione
    final_trades, final_portfolio_value = run_backtest_simulation(
        all_historical_data=historical_data_store,
        tickers_to_analyze=tickers_for_backtest
    )
    
    # FASE 5: Salvataggio risultati
    save_backtest_results(final_trades, final_portfolio_value)
    
    print("\n🎉 BACKTEST COMPLETATO CON SUCCESSO!")
    print("   Controlla il file 'backtest_results.csv' per i risultati dettagliati.")
