# backtest_orchestrator.py

# ==============================================================================
# --- PATCH DI COMPATIBILITÀ PER NUMPY 2.0 ---
# Questo blocco risolve l'errore 'cannot import name 'NaN' from 'numpy''.
# Deve essere eseguito PRIMA di qualsiasi altra importazione che usa numpy (come pandas).
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
# ==============================================================================

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
import shutil
import warnings

# Sopprimi warning comuni per un output più pulito
warnings.filterwarnings('ignore')

# --- IMPORTAZIONE DELLE LOGICHE MODIFICATE DAGLI ALTRI SCRIPT ---
# Assicurati che questi file modificati siano nella stessa cartella dell'orchestratore.
# Verranno fornite le istruzioni per modificare questi file nella PARTE 2.
try:
    from best_buy_backtest import run_one_time_screening_for_backtest
    from stock_analyzer_backtest import run_analysis_for_date
    from trading_engine_backtest import IntegratedRevolutionaryTradingEngine
except ImportError as e:
    print(f"ERRORE CRITICO: Assicurati di aver rinominato i file modificati in 'best_buy_backtest.py', 'stock_analyzer_backtest.py', e 'trading_engine_backtest.py'")
    print(f"Dettaglio errore: {e}")
    exit()

# ==============================================================================
# --- FASE 1: CONFIGURAZIONE E SETUP DELL'AMBIENTE DI BACKTEST ---
# ==============================================================================

print("--- AVVIO ORCHESTRATORE DI BACKTEST ---")

# --- Parametri del Backtest ---
START_DATE = '2015-01-01'
END_DATE = '2015-12-31'
INITIAL_CAPITAL = 100000.0

# --- Struttura delle Directory ---
# Definiamo i percorsi delle cartelle che il nostro sistema di trading si aspetta di trovare.
# Usiamo Path per una gestione robusta dei percorsi su qualsiasi sistema operativo.
BASE_DIR = Path.cwd()  # Directory corrente dove viene eseguito lo script
DATA_DIR = BASE_DIR / "data_backtest"
AI_LEARNING_DIR = DATA_DIR / "ai_learning"
REPORTS_DIR = DATA_DIR / "reports"
SIGNALS_HISTORY_DIR = DATA_DIR / "signals_history"

# --- Percorsi dei File Chiave ---
AI_DB_FILE = AI_LEARNING_DIR / "performance.db"
ANALYSIS_FILE_PATH = DATA_DIR / "latest_analysis.json"
EXECUTION_SIGNALS_FILE = DATA_DIR / "execution_signals.json"
HISTORICAL_EXECUTION_SIGNALS_FILE = DATA_DIR / "historical_execution_signals.json" # Anche questo è usato dall'AI

def setup_backtest_environment():
    """
    Crea la struttura di directory necessaria per il backtest.
    Pulisce l'ambiente da esecuzioni precedenti per garantire un inizio pulito.
    """
    print("FASE 1: Setup dell'ambiente di backtest...")

    # Se la directory di backtest esiste già, la rimuoviamo per partire da zero.
    # Questo è FONDAMENTALE per garantire che ogni backtest sia indipendente.
    if DATA_DIR.exists():
        print(f"Trovata directory di backtest precedente '{DATA_DIR}'. La rimuovo per un inizio pulito...")
        shutil.rmtree(DATA_DIR)

    # Creiamo tutte le directory necessarie. `parents=True` crea anche le cartelle madri se non esistono.
    print("Creazione della struttura delle directory...")
    for directory in [DATA_DIR, AI_LEARNING_DIR, REPORTS_DIR, SIGNALS_HISTORY_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  - Creata: {directory}")

    # Creiamo un file vuoto per il database dell'AI. Il Trading Engine lo popolerà.
    # Questo è il file che persiste durante tutto il backtest per l'apprendimento.
    AI_DB_FILE.touch()
    print(f"  - Creato file database AI vuoto: {AI_DB_FILE}")

    # Crea un file storico vuoto per i segnali, se richiesto dal Trading Engine
    with open(HISTORICAL_EXECUTION_SIGNALS_FILE, 'w') as f:
        json.dump({"historical_signals": []}, f)
    print(f"  - Creato file storico segnali vuoto: {HISTORICAL_EXECUTION_SIGNALS_FILE}")

    print("✅ Ambiente di backtest pronto.\n")

# ==============================================================================
# --- FASE 2: PRE-CARICAMENTO DI TUTTI I DATI STORICI ---
# ==============================================================================

def pre_fetch_all_historical_data(tickers):
    """
    Scarica tutti i dati storici necessari per il backtest in una sola volta.
    Questo evita chiamate API ripetute durante la simulazione.
    """
    print("FASE 2: Download di tutti i dati storici necessari (potrebbe richiedere tempo)...")
    
    # Abbiamo bisogno di dati precedenti al 2015 per calcolare gli indicatori (es. medie a 200 giorni).
    # Scarichiamo dati a partire dal 1 Gennaio 2014.
    fetch_start_date = (pd.to_datetime(START_DATE) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end_date = (pd.to_datetime(END_DATE) + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Periodo di download: da {fetch_start_date} a {fetch_end_date}")

    all_data = {}
    failed_tickers = []
    
    # Aggiungiamo l'indice S&P 500 (^GSPC) per l'analisi del trend di mercato.
    tickers_to_fetch = tickers + ['^GSPC']

    for i, ticker in enumerate(tickers_to_fetch):
        try:
            print(f"  Scaricando {ticker} ({i+1}/{len(tickers_to_fetch)})...")
            data = yf.download(ticker, start=fetch_start_date, end=fetch_end_date, progress=False)
            if not data.empty:
                all_data[ticker] = data
            else:
                failed_tickers.append(ticker)
        except Exception as e:
            print(f"    ERRORE scaricando {ticker}: {e}")
            failed_tickers.append(ticker)

    print(f"\n✅ Dati storici scaricati per {len(all_data)} simboli.")
    if failed_tickers:
        print(f"⚠️ Non è stato possibile scaricare i dati per: {failed_tickers}")
    
    return all_data

# ==============================================================================
# --- FASE 3: ESECUZIONE DEL LOOP DI SIMULAZIONE GIORNALIERA ---
# ==============================================================================

# In backtest_orchestrator.py, sostituisci l'intera funzione run_backtest_simulation

# In backtest_orchestrator.py, sostituisci l'INTERA funzione run_backtest_simulation

def run_backtest_simulation(all_historical_data, tickers_to_analyze):
    """
    Il cuore del backtest. Simula il passare dei giorni e orchestra gli script.
    """
    print("\n--- FASE 3: INIZIO SIMULAZIONE DI TRADING GIORNALIERA ---")
    
    # --- Stato del Portafoglio ---
    capital = INITIAL_CAPITAL
    open_positions = []
    closed_trades_for_csv = []

    # --- Loop Giornaliero ---
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    sp500_data = all_historical_data['^GSPC']

    for current_date in date_range:
        current_date_str = current_date.strftime('%Y-%m-%d')
        print(f"\n{'='*20} SIMULAZIONE GIORNO: {current_date_str} {'='*20}")
        print(f"Stato: Capitale ${capital:,.2f}, Posizioni Aperte: {len(open_positions)}")
        
        # 3.1: Esegui Stock Analyzer
        print("  - 3.1: Esecuzione Stock Analyzer (offline)...")
        run_analysis_for_date(
            tickers_to_analyze=tickers_to_analyze,
            all_historical_data=all_historical_data,
            current_date=current_date,
            output_path=ANALYSIS_FILE_PATH
        )

        # 3.2: Esegui Trading Engine
        print("  - 3.2: Esecuzione Trading Engine (con stato attuale)...")
        engine = IntegratedRevolutionaryTradingEngine(
            capital=capital,
            open_positions=open_positions,
            performance_db_path=AI_DB_FILE
        )
        # Passa anche i trade chiusi all'engine per l'apprendimento
        engine.trade_history = closed_trades_for_csv
        
        engine.run_integrated_trading_session_for_backtest(
            analysis_data_path=ANALYSIS_FILE_PATH,
            sp500_data_full=sp500_data,
            current_backtest_date=current_date
        )

        # 3.3: Processa i segnali di Esecuzione (Acquisti e Vendite)
        print("  - 3.3: Processamento segnali di esecuzione generati...")
        if not EXECUTION_SIGNALS_FILE.exists():
            print("    [LOG CHIARIFICATORE] Nessun file execution_signals.json generato oggi. Nessuna azione intrapresa.")
            continue
        
        with open(EXECUTION_SIGNALS_FILE, 'r') as f:
            content = f.read()
            if not content:
                print("    [LOG CHIARIFICATORE] Il file execution_signals.json è VUOTO. Nessuna azione intrapresa.")
                continue
            signals = json.loads(content).get('signals', {})
        
        buy_signals = signals.get('buys', [])
        sell_signals = signals.get('sells', [])

        print(f"    [LOG CHIARIFICATORE] Trovati {len(buy_signals)} segnali di ACQUISTO e {len(sell_signals)} di VENDITA da processare.")
        
        # Trova il giorno di trading successivo per l'esecuzione
        next_trading_day = current_date + pd.tseries.offsets.BDay(1)
        # Controlla se il giorno successivo è ancora nel range del backtest
        if next_trading_day > date_range[-1]:
            print("    Fine del periodo di backtest, non si eseguono nuovi trade.")
            continue
        next_trading_day_str = next_trading_day.strftime('%Y-%m-%d')

        # ==============================================================================
        # --- BLOCCO DI ESECUZIONE (LA PARTE MANCANTE) ---
        # ==============================================================================
        
        # --- Processa le VENDITE PRIMA, per liberare capitale ---
        if sell_signals:
            positions_to_remove = []
            for sell_signal in sell_signals:
                ticker = sell_signal['ticker']
                position_to_close = next((p for p in open_positions if p['ticker'] == ticker), None)
                
                if position_to_close:
                    try:
                        # Prendi il prezzo di apertura del giorno successivo per la vendita
                        exit_price = all_historical_data[ticker].loc[next_trading_day_str]['Open']
                        quantity = position_to_close['quantity']
                        
                        exit_value = exit_price * quantity
                        capital += exit_value
                        
                        print(f"    -> VENDITA ESEGUITA: {quantity} di {ticker} @ ${exit_price:.2f}. Capitale aggiornato: ${capital:,.2f}")

                        # Registra il trade chiuso per il CSV finale
                        closed_trades_for_csv.append({
                            'entry_price': position_to_close['entry_price'],
                            'exit_price': exit_price,
                            'quantity': quantity
                        })
                        
                        positions_to_remove.append(position_to_close)
                    except KeyError:
                        print(f"    ATTENZIONE: Nessun dato per {ticker} il {next_trading_day_str}. Vendita non eseguita.")
                    except Exception as e:
                        print(f"    ERRORE ESECUZIONE VENDITA {ticker}: {e}")
            
            open_positions = [p for p in open_positions if p not in positions_to_remove]

        # --- Processa gli ACQUISTI DOPO, usando il capitale aggiornato ---
        if buy_signals:
            for buy_signal in buy_signals:
                ticker = buy_signal['ticker']
                quantity = buy_signal['quantity_estimated']
                
                try:
                    # Prendi il prezzo di apertura del giorno successivo per l'acquisto
                    entry_price = all_historical_data[ticker].loc[next_trading_day_str]['Open']
                    trade_value = entry_price * quantity
                    
                    if capital >= trade_value:
                        capital -= trade_value
                        
                        # Aggiungi alla lista delle posizioni aperte
                        open_positions.append({
                            'ticker': ticker,
                            'entry_price': entry_price,
                            'quantity': quantity,
                            'entry_date': next_trading_day,
                            'trade_value': trade_value
                        })
                        print(f"    -> ACQUISTO ESEGUITO: {quantity} di {ticker} @ ${entry_price:.2f}. Capitale aggiornato: ${capital:,.2f}")
                    else:
                        print(f"    ACQUISTO SALTATO: {ticker} - Capitale insufficiente (${trade_value:,.2f} > ${capital:,.2f}).")
                except KeyError:
                    print(f"    ATTENZIONE: Nessun dato per {ticker} il {next_trading_day_str}. Acquisto non eseguito.")
                except Exception as e:
                    print(f"    ERRORE ESECUZIONE ACQUISTO {ticker}: {e}")
        # ==============================================================================
        # --- FINE BLOCCO DI ESECUZIONE ---
        # ==============================================================================

    print("\n--- SIMULAZIONE COMPLETATA ---")
    
    final_portfolio_value = capital
    for pos in open_positions:
        try:
            # Usa l'ultimo prezzo disponibile alla fine del backtest per valutare le posizioni aperte
            last_price = all_historical_data[pos['ticker']]['Close'].iloc[-1]
            final_portfolio_value += pos['quantity'] * last_price
        except:
            final_portfolio_value += pos['trade_value']
    return closed_trades_for_csv, final_portfolio_value


# ==============================================================================
# --- FASE 4: FINALIZZAZIONE E SALVATAGGIO RISULTATI ---
# ==============================================================================

def save_backtest_results(closed_trades, final_value):
    """
    Salva i risultati del backtest nel file CSV richiesto.
    """
    print("\n--- FASE 4: SALVATAGGIO DEI RISULTATI DEL BACKTEST ---")
    
    if not closed_trades:
        print("⚠️ Nessun trade è stato chiuso durante il backtest. Il file CSV non verrà creato.")
        return

    df_results = pd.DataFrame(closed_trades)
    
    # Assicurati che le colonne richieste siano presenti e nell'ordine giusto
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


# ==============================================================================
# --- FUNZIONE PRINCIPALE (MAIN) ---
# ==============================================================================

if __name__ == "__main__":
    
    # 1. Prepara l'ambiente
    setup_backtest_environment()
    
    # 2. Esegui lo screening una sola volta per ottenere la lista di titoli
    # Questa funzione simula l'output di 1_best_buy.py
    tickers_for_backtest = run_one_time_screening_for_backtest()
    if not tickers_for_backtest:
        print("ERRORE: Nessun ticker ottenuto dallo screening iniziale. Impossibile procedere.")
        exit()
    print(f"Ottenuta lista statica di {len(tickers_for_backtest)} titoli per il backtest.")
    
    # 3. Scarica tutti i dati storici in anticipo
    historical_data_store = pre_fetch_all_historical_data(tickers_for_backtest)
    if not historical_data_store:
        print("ERRORE: Nessun dato storico scaricato. Impossibile procedere.")
        exit()
        
    # 4. Esegui la simulazione di backtest
    final_closed_trades, final_portfolio_value = run_backtest_simulation(
        all_historical_data=historical_data_store,
        tickers_to_analyze=tickers_for_backtest
    )
    
    # 5. Salva i risultati finali
    save_backtest_results(final_closed_trades, final_portfolio_value)
    
    print("\n--- BACKTEST COMPLETATO CON SUCCESSO ---")
