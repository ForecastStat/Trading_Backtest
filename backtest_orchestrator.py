import os
import sys
import json
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import time
import sqlite3
from ai_trade_recorder import AITradeRecorder

class BacktestOrchestrator:
    def __init__(self):
        self.start_date = datetime(2015, 1, 1)
        self.end_date = datetime(2015, 3, 31)
        self.initial_capital = 100000.0
        self.current_capital = self.initial_capital
        self.open_positions = {}
        self.closed_trades = []
        self.trading_days = []
        self.stock_data = {}
        self.results_file = "backtest_results.csv"
        
        # Inizializza il registratore AI
        self.ai_recorder = AITradeRecorder()
        
        # Setup logging
        self.setup_logging()
        
        # Setup directories
        self.setup_directories()
        
        # Lista dei ticker per il backtest
        self.tickers = [
            'AAPL', 'ABBV', 'ADBE', 'AMGN', 'AMZN', 'AXP', 'BA', 'BAC', 'BIIB', 'BLK',
            'C', 'CAT', 'COP', 'CRM', 'CSCO', 'CVX', 'DIS', 'EOG', 'GE', 'GILD',
            'GOOGL', 'GS', 'HD', 'HON', 'IWM', 'JNJ', 'JPM', 'LMT', 'LOW', 'MCD',
            'MMM', 'MRK', 'MS', 'MSFT', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PFE', 'QQQ',
            'RTX', 'SBUX', 'SLB', 'SPY', 'TJX', 'TMO', 'UNH', 'UPS', 'WFC', 'XLE',
            'XLF', 'XLI', 'XLK', 'XLV', 'XOM'
        ]

    def setup_logging(self):
        """Configura il sistema di logging"""
        log_dir = Path("data_backtest")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "trading_integrated.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Crea le directory necessarie"""
        directories = [
            "data_backtest",
            "data_backtest/ai_learning",
            "data_backtest/reports", 
            "data_backtest/signals_history",
            "data"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            self.logger.info(f"  - Creata directory: {os.path.abspath(directory)}")

    def run_stock_screening(self):
        """Esegue lo screening dei titoli una sola volta"""
        self.logger.info("ESECUZIONE SCREENING INIZIALE TITOLI...")
        self.logger.info("Esecuzione screening una tantum per il backtest...")
        # Nel backtest, usiamo una lista statica di titoli
        self.logger.info("Screening completato. 55 titoli statici selezionati per il backtest.")
        self.logger.info(f"✅ Screening completato: {len(self.tickers)} titoli selezionati")

    def download_historical_data(self):
        """Scarica tutti i dati storici necessari"""
        self.logger.info("FASE 2: Download di tutti i dati storici...")
        
        # Periodo esteso per avere dati sufficienti per gli indicatori
        download_start = self.start_date - timedelta(days=365)
        download_end = self.end_date + timedelta(days=30)
        
        self.logger.info(f"  - Periodo download: {download_start.strftime('%Y-%m-%d')} a {download_end.strftime('%Y-%m-%d')}")
        
        # Aggiungi S&P 500 per il benchmark
        all_symbols = self.tickers + ['^GSPC']
        self.logger.info(f"  - Titoli da scaricare: {len(self.tickers)} + S&P 500")
        
        success_count = 0
        for i, symbol in enumerate(all_symbols, 1):
            try:
                self.logger.info(f"  Scaricando {symbol} ({i}/{len(all_symbols)})...")
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=download_start.strftime('%Y-%m-%d'),
                    end=download_end.strftime('%Y-%m-%d'),
                    interval='1d'
                )
                
                if not data.empty:
                    # Salva i dati
                    data_file = f"data_backtest/{symbol}_data.csv"
                    data.to_csv(data_file)
                    self.stock_data[symbol] = data
                    success_count += 1
                    self.logger.info(f"  ✅ {symbol} scaricato con successo")
                else:
                    self.logger.warning(f"  ❌ {symbol} - nessun dato disponibile")
                    
            except Exception as e:
                self.logger.error(f"  ❌ {symbol} - errore: {e}")
                continue
        
        self.logger.info(f"✅ Download completato: {success_count}/{len(all_symbols)} titoli scaricati con successo.")
        self.logger.info(f"✅ Dati storici pronti per {len(self.stock_data)} simboli")

    def get_trading_days(self):
        """Genera la lista dei giorni di trading (esclude weekend)"""
        current_date = self.start_date
        trading_days = []
        
        while current_date <= self.end_date:
            # Solo giorni feriali (0=Lunedì, 6=Domenica)
            if current_date.weekday() < 5:
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        self.trading_days = trading_days
        return trading_days

    def get_stock_price(self, ticker, date):
        """Ottiene il prezzo di un titolo per una data specifica"""
        try:
            if ticker not in self.stock_data:
                return None
            
            data = self.stock_data[ticker]
            date_str = date.strftime('%Y-%m-%d')
            
            # Cerca la data esatta o la più vicina precedente
            available_dates = data.index.strftime('%Y-%m-%d')
            if date_str in available_dates:
                return data.loc[date_str, 'Close']
            else:
                # Trova la data più vicina precedente
                before_dates = [d for d in available_dates if d <= date_str]
                if before_dates:
                    closest_date = max(before_dates)
                    return data.loc[closest_date, 'Close']
            
            return None
        except Exception as e:
            self.logger.error(f"Errore nel recupero prezzo per {ticker}: {e}")
            return None

    def process_signals_file(self, date):
        """Processa il file dei segnali generato dal trading engine"""
        signals_file = "data/execution_signals.json"
        
        if not os.path.exists(signals_file):
            self.logger.info("  ℹ️ Nessun file di segnali da processare")
            return
        
        try:
            with open(signals_file, 'r') as f:
                signals_data = json.load(f)
            
            buy_signals = signals_data.get('buy_signals', [])
            sell_signals = signals_data.get('sell_signals', [])
            
            self.logger.info(f"  📈 Esecuzione segnali per {date.strftime('%Y-%m-%d')}...")
            
            # Processa prima le vendite
            if sell_signals:
                self.process_sell_signals(sell_signals, date)
            else:
                self.logger.info("    ℹ️ Nessun segnale di vendita da processare")
            
            # Poi gli acquisti
            if buy_signals:
                self.process_buy_signals(buy_signals, date)
            else:
                self.logger.info("    ℹ️ Nessun segnale di acquisto da processare")
                
        except Exception as e:
            self.logger.error(f"Errore nel processare i segnali: {e}")

    def process_sell_signals(self, sell_signals, date):
        """Processa i segnali di vendita"""
        self.logger.info(f"    🔄 Processando {len(sell_signals)} segnali di vendita...")
        
        for signal in sell_signals:
            ticker = signal['ticker']
            quantity = signal['quantity']
            reason = signal.get('reason', 'Manual')
            
            if ticker in self.open_positions:
                position = self.open_positions[ticker]
                entry_price = position['entry_price']
                entry_date = position['entry_date']
                
                # Ottieni il prezzo di vendita
                exit_price = self.get_stock_price(ticker, date)
                if exit_price is None:
                    self.logger.warning(f"    ⚠️ Impossibile ottenere prezzo per {ticker}")
                    continue
                
                # Calcola P/L
                total_proceeds = exit_price * quantity
                total_cost = entry_price * quantity
                profit = total_proceeds - total_cost
                profit_pct = (profit / total_cost) * 100
                
                # Aggiorna il capitale
                self.current_capital += total_proceeds
                
                # Registra il trade chiuso
                trade_record = {
                    'ticker': ticker,
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'exit_reason': reason
                }
                self.closed_trades.append(trade_record)
                
                # *** NUOVO: Registra nel database AI ***
                self.ai_recorder.record_trade_from_backtest(
                    ticker=ticker,
                    entry_date=entry_date.strftime('%Y-%m-%d'),
                    exit_date=date.strftime('%Y-%m-%d'),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    profit_pct=profit_pct,
                    exit_reason=reason
                )
                
                # Rimuovi dalla posizioni aperte
                del self.open_positions[ticker]
                
                self.logger.info(f"      ✅ VENDUTO: {quantity} {ticker} @ ${exit_price:.2f} = ${total_proceeds:,.2f} (P/L: {profit_pct:+.1f}%)")
        
        # Mostra il nuovo capitale dopo le vendite
        self.logger.info(f"    ✅ Vendite completate. Capitale aggiornato: ${self.current_capital:,.2f}")
        
        # Mostra quanti trade sono nel database AI
        trade_count = self.ai_recorder.get_trade_count()
        if trade_count > 0:
            self.logger.info(f"    📊 Database AI ora contiene {trade_count} trade")

    def process_buy_signals(self, buy_signals, date):
        """Processa i segnali di acquisto"""
        self.logger.info(f"    🔄 Processando {len(buy_signals)} segnali di acquisto...")
        
        for signal in buy_signals:
            ticker = signal['ticker']
            quantity = signal['quantity']
            
            # Ottieni il prezzo di acquisto
            entry_price = self.get_stock_price(ticker, date)
            if entry_price is None:
                self.logger.warning(f"    ⚠️ Impossibile ottenere prezzo per {ticker}")
                continue
            
            # Calcola il costo totale
            total_cost = entry_price * quantity
            
            # Verifica se abbiamo abbastanza capitale
            if total_cost > self.current_capital:
                self.logger.warning(f"    ⚠️ Capitale insufficiente per {ticker}: serve ${total_cost:,.2f}, disponibile ${self.current_capital:,.2f}")
                continue
            
            # Esegui l'acquisto
            self.current_capital -= total_cost
            
            # Registra la posizione aperta
            self.open_positions[ticker] = {
                'ticker': ticker,
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_date': date,
                'total_cost': total_cost
            }
            
            self.logger.info(f"      ✅ ACQUISTATO: {quantity} {ticker} @ ${entry_price:.2f} = ${total_cost:,.2f}")
        
        self.logger.info(f"    ✅ Acquisti completati. Capitale rimanente: ${self.current_capital:,.2f}")

    def calculate_portfolio_value(self, date):
        """Calcola il valore totale del portafoglio"""
        portfolio_value = self.current_capital
        
        for ticker, position in self.open_positions.items():
            current_price = self.get_stock_price(ticker, date)
            if current_price:
                position_value = current_price * position['quantity']
                portfolio_value += position_value
        
        return portfolio_value

    def run_trading_engine(self, date):
        """Esegue il trading engine per una specifica data"""
        try:
            # Prepara i parametri per il trading engine
            date_str = date.strftime('%Y-%m-%d')
            
            self.logger.info(f"  🔍 Generazione analisi e segnali per domani...")
            
            # Esegui analisi per tutti i ticker
            self.logger.info(f"  Esecuzione analisi per {len(self.tickers)} tickers in data {date_str}")
            
            # Simula l'analisi per tutti i ticker
            for ticker in self.tickers:
                self.logger.info(f"  Analisi (offline) per {ticker}...")
            
            # Salva i risultati dell'analisi
            analysis_file = "data_backtest/latest_analysis.json"
            analysis_data = {
                'date': date_str,
                'tickers_analyzed': len(self.tickers),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            self.logger.info(f"✅ Risultati analisi salvati in '{os.path.abspath(analysis_file)}'.")
            self.logger.info(f"  -> Analisi del giorno salvata in '{os.path.abspath(analysis_file)}'")
            
            # Esegui il trading engine
            self.logger.info(f"    🔧 Inizializzazione trading engine con {len(self.open_positions)} posizioni aperte")
            self.logger.info(f"    📊 Trading engine configurato: Capital=${self.current_capital:,.2f}, Posizioni={len(self.open_positions)}")
            
            if self.open_positions:
                open_tickers = list(self.open_positions.keys())
                self.logger.info(f"    📋 Posizioni da valutare per vendita: {open_tickers}")
            
            # Esegui il trading engine
            cmd = ["python", "trading_engine_backtest.py", date_str]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.logger.info(f"    ✅ Segnali generati con successo")
                
                # Conta i segnali generati (se il file esiste)
                signals_file = "data/execution_signals.json"
                if os.path.exists(signals_file):
                    try:
                        with open(signals_file, 'r') as f:
                            signals_data = json.load(f)
                        buy_count = len(signals_data.get('buy_signals', []))
                        sell_count = len(signals_data.get('sell_signals', []))
                        self.logger.info(f"    📊 Segnali generati: {buy_count} acquisti, {sell_count} vendite")
                    except:
                        self.logger.info(f"    📊 Segnali generati: file creato")
                else:
                    self.logger.info(f"    📊 Segnali generati: 0 acquisti, 0 vendite")
            else:
                self.logger.error(f"Errore nell'esecuzione del trading engine: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout nell'esecuzione del trading engine")
        except Exception as e:
            self.logger.error(f"Errore nell'esecuzione del trading engine: {e}")

    def generate_day_summary(self, date, day_number, total_days):
        """Genera il riassunto della giornata di trading"""
        portfolio_value = self.calculate_portfolio_value(date)
        total_trades = len(self.closed_trades)
        last_trade_info = ""
        
        if self.closed_trades:
            last_trade = self.closed_trades[-1]
            last_trade_info = f"\n  🔍 Ultimo trade chiuso: {last_trade['ticker']} P/L={last_trade['profit_pct']:+.1f}%"
        
        self.logger.info(f"  📊 Fine giornata: Capitale=${self.current_capital:,.2f}, Posizioni Aperte={len(self.open_positions)}, Trade Chiusi Totali={total_trades}{last_trade_info}")

    def save_results(self):
        """Salva i risultati del backtest"""
        try:
            # Prepara i dati per il CSV
            results_data = []
            
            for trade in self.closed_trades:
                results_data.append({
                    'Ticker': trade['ticker'],
                    'Entry_Date': trade['entry_date'].strftime('%Y-%m-%d'),
                    'Exit_Date': trade['exit_date'].strftime('%Y-%m-%d'),
                    'Entry_Price': trade['entry_price'],
                    'Exit_Price': trade['exit_price'],
                    'Quantity': trade['quantity'],
                    'Profit': trade['profit'],
                    'Profit_Pct': trade['profit_pct'],
                    'Exit_Reason': trade['exit_reason']
                })
            
            # Salva in CSV
            if results_data:
                df = pd.DataFrame(results_data)
                df.to_csv(self.results_file, index=False)
                self.logger.info(f"✅ Risultati salvati in {self.results_file}")
            
            # Calcola statistiche finali
            final_portfolio_value = self.calculate_portfolio_value(self.end_date)
            total_return = ((final_portfolio_value - self.initial_capital) / self.initial_capital) * 100
            
            # Genera report finale
            self.generate_final_report(final_portfolio_value, total_return)
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio risultati: {e}")

    def generate_final_report(self, final_value, total_return):
        """Genera il report finale HTML"""
        try:
            winning_trades = [t for t in self.closed_trades if t['profit'] > 0]
            losing_trades = [t for t in self.closed_trades if t['profit'] < 0]
            
            win_rate = (len(winning_trades) / len(self.closed_trades) * 100) if self.closed_trades else 0
            avg_profit = sum(t['profit'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            # Database AI stats
            ai_trade_count = self.ai_recorder.get_trade_count()
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Report Backtest Trading 2015</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Report Backtest Trading System</h1>
        <p>Periodo: {self.start_date.strftime('%Y-%m-%d')} - {self.end_date.strftime('%Y-%m-%d')}</p>
    </div>
    
    <div class="section">
        <h2>📈 Performance Generale</h2>
        <div class="metric">
            <strong>Capitale Iniziale:</strong> ${self.initial_capital:,.2f}
        </div>
        <div class="metric">
            <strong>Capitale Finale:</strong> ${final_value:,.2f}
        </div>
        <div class="metric">
            <strong>Rendimento Totale:</strong> 
            <span class="{'positive' if total_return > 0 else 'negative'}">{total_return:+.2f}%</span>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Statistiche Trading</h2>
        <div class="metric">
            <strong>Trade Totali:</strong> {len(self.closed_trades)}
        </div>
        <div class="metric">
            <strong>Trade Vincenti:</strong> {len(winning_trades)}
        </div>
        <div class="metric">
            <strong>Trade Perdenti:</strong> {len(losing_trades)}
        </div>
        <div class="metric">
            <strong>Win Rate:</strong> {win_rate:.1f}%
        </div>
        <div class="metric">
            <strong>Profitto Medio:</strong> ${avg_profit:.2f}
        </div>
        <div class="metric">
            <strong>Perdita Media:</strong> ${avg_loss:.2f}
        </div>
    </div>
    
    <div class="section">
        <h2>🤖 Database AI</h2>
        <div class="metric">
            <strong>Trade nel Database AI:</strong> {ai_trade_count}
        </div>
        <div class="metric">
            <strong>Stato Bootstrap:</strong> {'Attivo' if ai_trade_count < 80 else 'Completato'}
        </div>
    </div>
    
    <div class="section">
        <h2>💼 Posizioni Aperte</h2>
        <p>Posizioni ancora aperte: {len(self.open_positions)}</p>
    </div>
    
    <div class="section">
        <p><strong>Report generato il:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
            
            report_file = "data_backtest/reports/Report_Olga_Trade.html"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"✅ Report HTML generato: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione del report: {e}")

    def run_backtest(self):
        """Esegue l'intero backtest"""
        self.logger.info("🚀 AVVIO BACKTEST ORCHESTRATOR")
        self.logger.info("=" * 80)
        
        # Fase 1: Setup
        self.logger.info("FASE 1: Setup dell'ambiente di backtest...")
        self.logger.info("✅ Ambiente di backtest pronto.\n")
        
        # Screening
        self.run_stock_screening()
        
        # Fase 2: Download dati
        self.download_historical_data()
        
        # Verifica che abbiamo almeno alcuni dati
        if len(self.stock_data) == 0:
            self.logger.error("❌ ERRORE: Nessun dato scaricato. Impossibile continuare il backtest.")
            return
        
        # Fase 3: Simulazione
        self.logger.info("=" * 80)
        self.logger.info("FASE 3: INIZIO SIMULAZIONE DI TRADING GIORNALIERA")
        self.logger.info("=" * 80)
        
        trading_days = self.get_trading_days()
        
        self.logger.info(f"📅 Periodo simulazione: {len(trading_days)} giorni lavorativi")
        self.logger.info(f"💰 Capitale iniziale: ${self.initial_capital:,.2f}")
        self.logger.info(f"📊 Titoli nel portafoglio: {len(self.tickers)}\n")
        
        # BACKTEST COMPLETO - TUTTI I GIORNI
        for day_num, trading_date in enumerate(trading_days, 1):
            self.logger.info("=" * 20 + f" GIORNO {day_num}/{len(trading_days)}: {trading_date.strftime('%Y-%m-%d')} " + "=" * 20)
            
            # Mostra stato portfolio
            portfolio_value = self.calculate_portfolio_value(trading_date)
            self.logger.info(f"💰 Stato Portfolio: Capitale=${self.current_capital:,.2f}, Posizioni={len(self.open_positions)}, Valore Totale≈${portfolio_value:,.2f}")
            
            # Processa segnali del giorno precedente (se esistono)
            self.logger.info("  📋 Esecuzione ordini del giorno precedente...")
            self.process_signals_file(trading_date)
            
            # Genera nuovi segnali per il giorno successivo
            self.run_trading_engine(trading_date)
            
            # Riassunto della giornata
            self.generate_day_summary(trading_date, day_num, len(trading_days))
            
            self.logger.info("")  # Riga vuota per separare i giorni
        
        # Salva risultati finali
        self.logger.info("=" * 80)
        self.logger.info("COMPLETAMENTO BACKTEST E GENERAZIONE REPORT")
        self.logger.info("=" * 80)
        
        self.save_results()
        
        self.logger.info("🎉 BACKTEST COMPLETATO CON SUCCESSO!")

def main():
    """Funzione principale"""
    try:
        orchestrator = BacktestOrchestrator()
        orchestrator.run_backtest()
        
    except KeyboardInterrupt:
        print("\n❌ Backtest interrotto dall'utente")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Errore critico nel backtest: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
