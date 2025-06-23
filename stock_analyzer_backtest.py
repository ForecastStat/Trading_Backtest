import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup
import urllib3
from urllib3.util.ssl_ import create_urllib3_context
import certifi
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import os
from pathlib import Path
from datetime import datetime
import traceback
from finvizfinance.screener.overview import Overview as FinvizOverview

# Configurazione SSL avanzata con certificati aggiornati
ctx = create_urllib3_context()
ctx.load_verify_locations(certifi.where())

# Configurazione globale per sopprimere gli avvisi HTTPS
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Inizializza l'analizzatore di sentiment VADER (fuori dalla classe)
analyzer = SentimentIntensityAnalyzer()

# Definisci la directory base per i file di dati
BASE_DIR = Path(os.path.dirname(__file__))
BEST_BUY_FILE = BASE_DIR / 'data' / 'Best_buy.json'

# --- FUNZIONI PER INDICATORI AVANZATI (Placeholder/Euristiche) ---
# Queste funzioni servono SOLO per lo screening interno dello StockAnalyzer,
# non verranno usate per popolare il JSON finale, che deve mantenere la sua struttura originale.
# Le implementazioni "corrette" richiederebbero librerie esterne che l'Analyzer non deve per forza avere.

def calculate_permutation_entropy_heuristic(series, order=3, delay=1):
    """
    Euristica per Permutation Entropy.
    Un basso valore suggerisce maggiore prevedibilità per lo screening.
    """
    if len(series) < order + (order - 1) * delay:
        return np.nan
    
    # Heuristica semplificata basata sulla volatilità relativa
    std_dev = np.std(series)
    mean_abs_change = np.mean(np.abs(np.diff(series)))
    
    if mean_abs_change == 0: return 0.0 # Completamente prevedibile
    
    # Maggiore la volatilità relativa, maggiore l'entropia euristica
    heuristic_entropy = min(1.0, std_dev / (mean_abs_change * 5 + 1e-9)) # Scalatura euristica
    return heuristic_entropy

def calculate_rqa_metrics_heuristic(series, embedding_dim=2, delay=1, radius_pct=0.1):
    """
    Euristica per Recurrence Quantification Analysis (RQA).
    Recurrence Rate: maggiore se la serie è "piatta" o ripete valori.
    Determinism: maggiore se c'è una chiara direzione/trend.
    Laminarity: maggiore se ci sono periodi di bassa attività.
    """
    if len(series) < embedding_dim + (embedding_dim - 1) * delay:
        return {"recurrence_rate": np.nan, "determinism": np.nan, "laminarity": np.nan}
    
    # Euristiche basate su statistiche semplici
    price_changes = np.diff(series)
    
    # Recurrence Rate (higher for less change, more flat/repeating)
    # Consideriamo alta la ricorrenza se la deviazione standard è bassa rispetto al range.
    q_75, q_25 = np.percentile(series, [75, 25])
    iqr = q_75 - q_25
    heuristic_rr = max(0.0, 1.0 - (np.std(series) / (iqr + 1e-9) * 0.5))
    
    # Determinism (higher for stronger directional moves)
    # Basato sulla persistenza della direzione
    directional_persistence = np.abs(np.mean(np.sign(price_changes)))
    heuristic_det = directional_persistence
    
    # Laminarity (higher for periods of small changes)
    # Basato sulla frequenza di piccoli cambiamenti
    small_changes_ratio = np.sum(np.abs(price_changes) < np.std(price_changes) * 0.1) / len(price_changes) if len(price_changes) > 0 else 0
    heuristic_lam = small_changes_ratio
    
    return {
        "recurrence_rate": heuristic_rr,
        "determinism": heuristic_det,
        "laminarity": heuristic_lam
    }

def calculate_adaptive_market_state_metrics_heuristic(series, lookback_period=20):
    """
    Euristica per l'analisi adattiva dello stato del mercato (signal, trend, noise).
    Non è un'implementazione DSP di Ehlers, ma una euristica utile per lo screening.
    """
    if len(series) < lookback_period:
        return None
    
    recent_series = series[-lookback_period:]
    
    # Volatility (as a proxy for noise)
    volatility = np.std(recent_series)
    
    # Trend strength (slope of linear regression over the period)
    try:
        x = np.arange(len(recent_series))
        slope, intercept, r_value, p_value, std_err = np.polyfit(x, recent_series, 1, full=False)
        # Normalize slope by the average price to make it scale-independent
        mean_price = np.mean(recent_series)
        trend_strength = slope / (mean_price + 1e-9) if mean_price != 0 else 0.0
    except Exception:
        trend_strength = 0.0
    
    signal_quality = 0.0
    noise_ratio = 0.0
    
    if volatility > 0:
        # Signal Quality: Higher if trend strength is high relative to volatility
        signal_quality = np.abs(trend_strength) / (volatility / (mean_price + 1e-9) + 1e-9) # Normalize volatility by price too
        
        # Noise Ratio: Higher if volatility is high relative to mean price (relative noise)
        noise_ratio = volatility / (mean_price + 1e-9)
    
    return {
        "signal_quality": signal_quality,
        "trend_strength": trend_strength,
        "noise_ratio": noise_ratio
    }

# --- FINE FUNZIONI PER INDICATORI AVANZATI (Placeholder/Euristiche) ---


def parse_market_cap(market_cap_value):
    try:
        return float(market_cap_value)
    except (ValueError, TypeError):
        return 0

def flatten_columns(df):
    if isinstance(df, pd.DataFrame):
        df = df.copy() 
        new_columns = []
        for col in df.columns:
            if isinstance(col, tuple):
                new_columns.append("_".join([str(c) for c in col if c]))
            else:
                new_columns.append(str(col)) 
        df.columns = new_columns
    return df

def remove_ticker_suffix_from_columns(df, ticker):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    
    ticker_suffix = f"_{ticker}"
    new_cols = {}
    renamed_count = 0
    for col in df.columns:
        if isinstance(col, str) and col.endswith(ticker_suffix):
            new_cols[col] = col[:-len(ticker_suffix)]
            renamed_count += 1
        else:
            new_cols[col] = col
    if renamed_count > 0:
        df.rename(columns=new_cols, inplace=True)
    return df

def load_best_buy_tickers(filepath):
    try:
        if not filepath.exists():
            print(f"⚠️ File Best Buy non trovato: {filepath}")
            return []
        with open(filepath, 'r') as f:
            tickers = json.load(f)
            if not isinstance(tickers, list):
                print(f"⚠️ Contenuto del file Best Buy non è una lista: {filepath}")
                return []
            print(f"✅ Caricati {len(tickers)} ticker dal file Best Buy.")
            return [t for t in tickers if isinstance(t, str) and t.strip()] 
    except json.JSONDecodeError:
        print(f"❌ Errore decodifica JSON nel file Best Buy: {filepath}")
        return []
    except Exception as e:
        print(f"❌ Errore generico caricando file Best Buy {filepath}: {e}")
        return []


class StockAnalyzer:
    def __init__(self):
        self.json_url = ("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
                         "?scrIds=day_gainers&count=100")
        self.fallback_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'JPM', 'V'] 
        self.session = requests.Session()
        self.session.verify = certifi.where()  
        
        # Lista delle colonne che DEVONO essere nell'output JSON finale.
        # Ho rimosso tutti gli indicatori per mantenere la compatibilità.
        self.final_output_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 'Ticker']

    def get_top_gainers(self, n=10, cp_threshold=5, vol_threshold=1e6,
                    price_lower=5, price_upper=200, mcap_lower=1e9, mcap_upper=5e10):
        """Versione Backtest: disattivata, non usa dati live."""
        return [] 

    def get_news_sentiment(self, ticker):
        """Versione Backtest: disattivata, restituisce sempre 'Neutral'."""
        return "Neutral" 

    def analyze_stock(self, ticker, historical_data_for_ticker):
        """
        Versione Backtest: Non scarica dati, ma li riceve come argomento.
        L'orchestratore fornirà i dati storici già filtrati fino alla data corrente.
        """
        print(f"  Analisi (offline) per {ticker}...")
        try:
            # Usa i dati passati come argomento invece di scaricarli
            data = historical_data_for_ticker
            
            if data is None or not isinstance(data, pd.DataFrame) or data.empty:
                print(f"  ⚠️ Dati storici non validi o vuoti per {ticker}.")
                return None
            
            data = flatten_columns(data) 
            data = remove_ticker_suffix_from_columns(data, ticker) 
            
            if 'Adj Close' in data.columns and 'Close' not in data.columns:
                 data['Close'] = data['Adj Close']
            elif 'close' in data.columns and 'Close' not in data.columns: 
                 data.rename(columns={'close': 'Close'}, inplace=True)
            if 'open' in data.columns and 'Open' not in data.columns: data.rename(columns={'open': 'Open'}, inplace=True)
            if 'high' in data.columns and 'High' not in data.columns: data.rename(columns={'high': 'High'}, inplace=True)
            if 'low' in data.columns and 'Low' not in data.columns: data.rename(columns={'low': 'Low'}, inplace=True)
            if 'volume' in data.columns and 'Volume' not in data.columns: data.rename(columns={'volume': 'Volume'}, inplace=True)
    
            required_ohlcv_original_case = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_ohlcv_original_case):
                 missing_cols = [col for col in required_ohlcv_original_case if col not in data.columns]
                 print(f"  ⚠️ Colonne OHLCV standard mancanti per {ticker}: {missing_cols}. Colonne presenti: {data.columns.tolist()}")
                 return None
            
            for col in required_ohlcv_original_case:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data.dropna(subset=required_ohlcv_original_case, inplace=True)
            if data.empty:
                print(f"  ⚠️ Dati vuoti per {ticker} dopo pulizia OHLCV numerici.")
                return None
            
            if not isinstance(data.index, pd.DatetimeIndex):
                 data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            data = data.replace([float('inf'), float('-inf')], pd.NA).dropna(subset=required_ohlcv_original_case)
            if data.empty:
                print(f"  ⚠️ Dati vuoti per {ticker} dopo rimozione inf/NA finali.")
                return None
            
            data['Sentiment'] = self.get_news_sentiment(ticker) 
            data['Ticker'] = ticker
            
            if isinstance(data.index, pd.DatetimeIndex):
                 data.index.name = 'Date'
                 if 'Date' not in data.columns:
                     data = data.reset_index()
                     if data.columns[0] == 'index':
                         data.rename(columns={'index': 'Date'}, inplace=True)
            
            data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
            final_df = data[self.final_output_columns].copy()
            final_df = final_df.replace({pd.NA: None, np.nan: None})
    
            if final_df.empty: 
                print(f"  ⚠️ Dati {ticker} vuoti dopo selezione colonne finali.")
                return None
            
            return final_df
        except Exception as e_main_analyze:
            print(f"  ❌ Errore critico durante l'analisi di {ticker}: {e_main_analyze}")
            traceback.print_exc()
            return None

    def get_finviz_screened_tickers(self, n_fallback=50, min_market_cap_str='+Mid (over $2bln)', min_avg_volume_str='Over 500K', min_price_str='Over $5'):
        """Versione Backtest: disattivata, non usa dati live."""
        print("  (Backtest) Chiamata a get_finviz_screened_tickers disattivata.")
        return []

    def get_volume_breakouts(self, n=10, stocks_to_scan=None, min_avg_vol=500000, vol_multiplier=2.0):
        """Versione Backtest: disattivata."""
        return []

    def get_trending_stocks(self, n=10, stocks_to_scan=None, ma_short=20, ma_long=50):
        """Versione Backtest: disattivata."""
        return []

    def get_relative_strength_stocks(self, n=10, stocks_to_scan=None, benchmark_ticker='SPY', lookback_days=20, rs_threshold_pct=3.0):
        """Versione Backtest: disattivata."""
        return []
        
    def get_valuation_anomalies(self, n=10):
        """Versione Backtest: disattivata."""
        return []
    
    def get_technical_breakouts(self, n=10, stocks_to_scan=None):
        """Versione Backtest: disattivata."""
        return []
    
    def get_institutional_flow_candidates(self, n=10):
        """Versione Backtest: disattivata."""
        return []

    def get_insider_buying_candidates(self, n=10):
        """Versione Backtest: disattivata."""
        return []

    def get_analyst_upgrades_candidates(self, n=10):
        """Versione Backtest: disattivata."""
        return []
            
    def get_catalyst_candidates(self, n=10):
        """Versione Backtest: disattivata."""
        return []

    def get_advanced_signal_candidates(self, n=10, stocks_to_scan=None):
        """Versione Backtest: disattivata."""
        return []

            
    def get_diversified_candidates(self, max_stocks=10, min_history_days=150):
        """
        Versione Backtest: disattivata. Restituisce una tupla vuota per compatibilità.
        L'orchestratore userà una lista statica di titoli, quindi questa funzione non serve.
        """
        return [], 0

    def save_analysis_results(self, results, filename_str='data/latest_analysis.json'):
        filepath = BASE_DIR / filename_str 
        def custom_serializer(obj):
            if pd.isna(obj) or obj is None:  
                return None
            if isinstance(obj, (pd.Timestamp, np.datetime64)):
                try: return pd.Timestamp(obj).isoformat()
                except: return str(obj) 
            if isinstance(obj, np.generic):
                try: return obj.item()
                except ValueError: return str(obj) 
            if isinstance(obj, (np.ndarray, pd.core.series.Series)):
                 return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                 return None
            if isinstance(obj, (int, float, str, bool, type(None))):
                 return obj
            return str(obj)
        
        serializable = {}
        for ticker, df in results.items():
            if df is None or df.empty: continue 
            try:
                # La logica di pulizia e formattazione è stata spostata direttamente in analyze_stock
                # per garantire che il DataFrame 'df' qui sia già nel formato finale desiderato.
                
                df_processed = df.copy() # Lavora su una copia
                
                # Assicurati che 'Date' sia sempre presente come colonna per la serializzazione JSON
                # Se analyze_stock ha già messo Date come colonna, non fare nulla.
                # Se è rimasto come indice, resetta l'indice.
                if isinstance(df_processed.index, pd.DatetimeIndex) and df_processed.index.name == 'Date':
                    if 'Date' not in df_processed.columns: # Se Date è indice ma non colonna
                        df_processed = df_processed.reset_index()
                
                data = []
                for _, row in df_processed.iterrows():
                    clean_row = {}
                    for col, val in row.items():
                        try:
                            clean_row[col] = custom_serializer(val)
                        except Exception as e:
                            clean_row[col] = str(val)  
                    data.append(clean_row)
                serializable[ticker] = {
                    'metadata': {
                        'last_updated': pd.Timestamp.now().isoformat(),
                        'columns': df_processed.columns.tolist(),
                        'dtypes': {col: str(df_processed[col].dtype) for col in df_processed.columns}
                    },
                    'data': data 
                }
            except Exception as e:
                print(f"❌ Errore processando {ticker} per serializzazione: {str(e)}")
                traceback.print_exc()
                continue 
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False, default=custom_serializer) 
            print(f"✅ Risultati analisi salvati in '{filepath}'.")
        except Exception as e:
            print(f"❌ Errore nel salvataggio del file JSON '{filepath}': {e}")


# In stock_analyzer_backtest.py, sostituisci la parte main:

def run_analysis_for_date(tickers_to_analyze, all_historical_data, current_date, output_path):
    """
    Funzione orchestrata per il backtest.
    Esegue l'analisi per una specifica data usando dati pre-caricati e salva l'output.
    """
    stock_analyzer = StockAnalyzer()
    analysis_results = {}
    
    print(f"  Esecuzione analisi per {len(tickers_to_analyze)} tickers in data {current_date.strftime('%Y-%m-%d')}")

    for ticker in tickers_to_analyze:
        if ticker in all_historical_data:
            # Filtra i dati storici fino alla data corrente del backtest
            historical_slice = all_historical_data[ticker].loc[:current_date]
            
            if not historical_slice.empty:
                # Chiama la versione modificata di analyze_stock
                stock_data = stock_analyzer.analyze_stock(ticker, historical_slice)
                if stock_data is not None and not stock_data.empty:
                    analysis_results[ticker] = stock_data
    
    # Salva i risultati nel formato richiesto dal Trading Engine
    stock_analyzer.save_analysis_results(analysis_results, filename_str=output_path)
    print(f"  -> Analisi del giorno salvata in '{output_path}'")
