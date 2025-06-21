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

# --- INIZIO BLOCCO MODIFICA PER BACKTEST ---
# --- INIZIO BLOCCO MODIFICA PER BACKTEST ---
import os
from datetime import datetime
def get_current_date():
    simulated_date_str = os.environ.get('SIMULATED_DATE')
    if simulated_date_str:
        return datetime.strptime(simulated_date_str, '%Y-%m-%d')
    return datetime.now()
    
def is_backtest_mode():
    return 'SIMULATED_DATE' in os.environ

BACKTEST_TICKER_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JNJ', 'WMT', 'JPM', 'PG', 'XOM', 'CVX', 
    'UNH', 'HD', 'BAC', 'KO', 'PFE', 'VZ', 'DIS', 'CSCO', 'PEP', 'INTC',
    'MCD', 'T', 'BA', 'IBM', 'CAT', 'GE', 'MMM', 'HON', 'AXP', 'NKE',
    'GS', 'MRK', 'ORCL', 'UPS', 'LMT', 'COST', 'SBUX', 'SPY', 'QQQ', 'IWM'
]
# --- FINE BLOCCO MODIFICA PER BACKTEST ---


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
        """
        MODIFICATA PER BACKTEST: In modalità backtest, simula i gainers usando dati storici.
        In modalità live, usa l'API di Yahoo.
        """
        if is_backtest_mode():
            print("  [BACKTEST MODE] get_top_gainers: Simulo screener su dati storici.")
            simulated_date = get_current_date()
            gainers = []
            for ticker in BACKTEST_TICKER_UNIVERSE:
                try:
                    # Scarica gli ultimi 2 giorni di dati per calcolare la variazione
                    data = yf.download(ticker, end=simulated_date, period='2d', interval='1d', progress=False)
                    if data is None or len(data) < 2:
                        continue
                    
                    last_day = data.iloc[-1]
                    prev_day = data.iloc[-2]
                    
                    cp = ((last_day['Close'] / prev_day['Close']) - 1) * 100
                    vol = last_day['Volume']
                    price = last_day['Close']
    
                    if cp > cp_threshold and vol > vol_threshold and price_lower < price < price_upper:
                        gainers.append({'ticker': ticker, 'cp': cp})
                except Exception:
                    continue
            
            # Ordina per performance e restituisci i migliori
            sorted_gainers = sorted(gainers, key=lambda x: x['cp'], reverse=True)
            return [g['ticker'] for g in sorted_gainers[:n]]
    
        # --- Il tuo codice originale rimane qui ---
        filtered_symbols = []
        try:
            response = self.session.get(
                self.json_url,
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=10
            )
            response.raise_for_status() 
            data = response.json()
            # ... (il resto del tuo codice originale) ...
            # ... (tutta la logica di parsing del JSON) ...
            # ...
            for quote in quotes:
                try:
                    # ... (il tuo codice di parsing) ...
                    if (symbol and isinstance(symbol, str) and
                        cp > cp_threshold and
                        vol > vol_threshold and
                        price_lower < price < price_upper and
                        mcap_lower < mcap < mcap_upper):
                        filtered_symbols.append(symbol)
                except Exception:
                    continue
            return filtered_symbols[:n]
        except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:
            print(f"❌ Errore recupero Day Gainers: {e}")
            return filtered_symbols

    def get_news_sentiment(self, ticker):
        try:
            news_url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}®ion=US&lang=en-US"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            time.sleep(1) 
            response = self.session.get(news_url, headers=headers, timeout=15) 
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            news_titles = []
            for a_tag in soup.find_all('a', {'class': 'js-content-viewer', 'data-uuid': True}):
                 title = a_tag.text.strip()
                 if title: news_titles.append(title)
            if not news_titles:
                for a_tag in soup.find_all('a', class_='Fw(b) Fz(18px) Lh(23px) LineClamp(2,46px) PV(0px)'): 
                    title = a_tag.text.strip()
                    if title: news_titles.append(title)
            if not news_titles:
                for h3_tag in soup.find_all('h3'):
                     title = h3_tag.text.strip()
                     if title: news_titles.append(title)
            if news_titles:
                news_titles_sample = news_titles[:10] 
                compound_scores = [analyzer.polarity_scores(title)['compound'] for title in news_titles_sample]
                avg_compound_score = sum(compound_scores) / len(news_titles_sample)
                if avg_compound_score >= 0.1: 
                    return "Positive"
                elif avg_compound_score <= -0.1:
                    return "Negative"
                else:
                    return "Neutral"
            else:
                return "Neutral" 
        except (requests.exceptions.RequestException, Exception) as e:
            return "Neutral" 

    def analyze_stock(self, ticker):
        print(f"  Inizio analisi per {ticker}...")
        try:
            # Scarica 3 anni di dati
            simulated_end_date = get_current_date()
            data = yf.download(ticker, end=simulated_end_date, period='3y', interval='1d', progress=False, auto_adjust=False)
            if data is None:
                print(f"  ⚠️ yf.download per {ticker} ha restituito None.")
                return None
            if not isinstance(data, pd.DataFrame):
                print(f"  ⚠️ yf.download per {ticker} non ha restituito un DataFrame (tipo: {type(data)}).")
                return None
            if data.empty:
                print(f"  ⚠️ yf.download per {ticker} ha restituito un DataFrame vuoto.")
                return None
            
            data = flatten_columns(data) 
            data = remove_ticker_suffix_from_columns(data, ticker) 
            
            # Standardize column names (case-insensitive)
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
            
            # Calcola il Sentiment (QUESTO VIENE AGGIUNTO AL JSON)
            data['Sentiment'] = self.get_news_sentiment(ticker) 

            # Aggiungi Ticker (QUESTO VIENE AGGIUNTO AL JSON)
            data['Ticker'] = ticker
            
            # === PUNTO CRUCIALE: Selezione e ordinamento delle colonne per l'output JSON ===
            # Qui garantiamo che solo le colonne desiderate (OHLCV + Sentiment + Ticker) siano nell'output.
            # Tutti gli altri indicatori che potrebbero essere calcolati sopra o sotto questo punto
            # NON saranno inclusi nel DataFrame finale restituito per il JSON.
            
            # Assicurati che l'indice 'Date' diventi una colonna per l'output JSON.
            if isinstance(data.index, pd.DatetimeIndex):
                 data.index.name = 'Date' # Set index name first
                 # Reset index to make 'Date' a regular column, then rename it if it became 'index'
                 if 'Date' not in data.columns: # If 'Date' is not already a column
                     data = data.reset_index()
                     if data.columns[0] == 'index': # If the reset index column is named 'index'
                         data.rename(columns={'index': 'Date'}, inplace=True)
            
            # Rimpiazza inf/-inf con NaN, poi con 0, per sicurezza prima della selezione finale
            data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Crea il DataFrame finale con SOLO le colonne specificate
            final_df = data[self.final_output_columns].copy()

            # Sostituisci eventuali NaN residui o pd.NA con None per la compatibilità JSON
            final_df = final_df.replace({pd.NA: None, np.nan: None})

            if final_df.empty: 
                print(f"  ⚠️ Dati {ticker} vuoti dopo selezione colonne finali.")
                return None
            
            return final_df
        except Exception as e_main_analyze:
            print(f"  ❌ Errore critico durante l'analisi di {ticker}: {e_main_analyze}")
            traceback.print_exc() # Print full traceback for debugging
            return None

    def get_finviz_screened_tickers(self, n_fallback=50, min_market_cap_str='+Mid (over $2bln)', min_avg_volume_str='Over 500K', min_price_str='Over $5'):
        """
        MODIFICATA PER BACKTEST: In modalità backtest, restituisce semplicemente l'universo statico.
        In modalità live, usa Finviz.
        """
        if is_backtest_mode():
            print("  [BACKTEST MODE] get_finviz_screened_tickers: Restituisco l'universo di ticker statico.")
            return BACKTEST_TICKER_UNIVERSE
        
        # --- Il tuo codice originale rimane qui ---
        print(f"  Recupero titoli da Finviz per scansione base...")
        try:
            foverview = FinvizOverview()
            
            filters_dict = {
                'Country': 'USA', 
                'Market Cap.': f'{min_market_cap_str}', 
                'Average Volume': min_avg_volume_str,
                'Price': min_price_str                  
            }
            print(f"  Applicazione filtri Finviz: {filters_dict}")
            foverview.set_filter(filters_dict=filters_dict)
            
            df = foverview.screener_view(order='Market Cap.', ascend=False, verbose=0) 
            time.sleep(3)
    
            if df is not None and not df.empty and 'Ticker' in df.columns:
                symbols = df['Ticker'].tolist()
                symbols = list(set([s for s in symbols if isinstance(s, str) and s.strip() and '.' not in s and len(s) <= 5]))
                if not symbols:
                    print(f"  ⚠️ Nessun ticker valido trovato da Finviz con i filtri applicati. Uso fallback.")
                    return self.fallback_tickers[:n_fallback]
                print(f"  ✅ Trovati {len(symbols)} tickers da Finviz (prima di eventuale limite).")
                return symbols
            else:
                print(f"  ⚠️ Nessun dato o colonna 'Ticker' da Finviz con i filtri applicati. Uso fallback.")
                return self.fallback_tickers[:n_fallback]
    
        except Exception as e:
            print(f"❌ Errore nel recupero titoli da Finviz: {str(e)}. Uso fallback.")
            return self.fallback_tickers[:n_fallback]

    def get_volume_breakouts(self, n=10, stocks_to_scan=None, min_avg_vol=500000, vol_multiplier=2.0):
        # Questo metodo calcola internamente indicatori, ma non vanno nel JSON finale
        if stocks_to_scan is None:
            stocks_to_scan = self.get_finviz_screened_tickers()[:50] 
        
        result = []
        for ticker in stocks_to_scan:
            if not isinstance(ticker, str) or not ticker.strip(): continue
            try:
                simulated_end_date = get_current_date()
                data = yf.download(ticker, end=simulated_end_date, period='1mo', interval='1d', progress=False, auto_adjust=False)
                #time.sleep(0.5) # Ritardo per YFinance, se necessario
                if data is None or data.empty or len(data) < 10: 
                    continue
                data = flatten_columns(data) 
                data = remove_ticker_suffix_from_columns(data, ticker) 
                if 'Volume' not in data.columns and 'volume' in data.columns:
                    data.rename(columns={'volume':'Volume'}, inplace=True)
                if 'Volume' not in data.columns: continue
                data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
                data.dropna(subset=['Volume'], inplace=True)
                if data.empty or len(data) < 10: continue 
                avg_vol = data['Volume'].iloc[:-1].rolling(window=10, min_periods=5).mean().iloc[-1]
                last_vol = data['Volume'].iloc[-1]
                if pd.isna(avg_vol) or avg_vol == 0: continue
                if last_vol > avg_vol * vol_multiplier and last_vol > min_avg_vol:
                    result.append(ticker)
                if len(result) >= n:
                    break
            except Exception: 
                continue
        if not result and n > 0: 
            return self.fallback_tickers[:n] 
        return result

    def get_trending_stocks(self, n=10, stocks_to_scan=None, ma_short=20, ma_long=50):
        # Questo metodo calcola internamente indicatori, ma non vanno nel JSON finale
        if stocks_to_scan is None:
            stocks_to_scan = self.get_finviz_screened_tickers()[:50] 
        
        result = []
        for ticker in stocks_to_scan:
            if not isinstance(ticker, str) or not ticker.strip(): continue
            try:
                simulated_end_date = get_current_date()
                data = yf.download(ticker, end=simulated_end_date, period='3mo', interval='1d', progress=False, auto_adjust=False)
                #time.sleep(0.5) # Ritardo per YFinance, se necessario
                if data is None or data.empty or len(data) < ma_long: 
                    continue
                data = flatten_columns(data)
                data = remove_ticker_suffix_from_columns(data, ticker) 
                if 'Close' not in data.columns and 'close' in data.columns:
                    data.rename(columns={'close':'Close'}, inplace=True)
                if 'Close' not in data.columns: continue
                data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
                data.dropna(subset=['Close'], inplace=True)
                if data.empty or len(data) < ma_long: continue
                data[f'MA{ma_short}'] = data['Close'].rolling(window=ma_short, min_periods=ma_short//2).mean()
                data[f'MA{ma_long}'] = data['Close'].rolling(window=ma_long, min_periods=ma_long//2).mean()
                last_row = data.iloc[-1]
                if (pd.notna(last_row['Close']) and pd.notna(last_row[f'MA{ma_short}']) and pd.notna(last_row[f'MA{ma_long}']) and
                    last_row['Close'] > last_row[f'MA{ma_short}'] > last_row[f'MA{ma_long}']):
                    result.append(ticker)
                if len(result) >= n:
                    break
            except Exception:
                continue
        if not result and n > 0:
            return self.fallback_tickers[:n]
        return result

    def get_relative_strength_stocks(self, n=10, stocks_to_scan=None, benchmark_ticker='SPY', lookback_days=20, rs_threshold_pct=3.0):
        if stocks_to_scan is None:
            stocks_to_scan = self.get_finviz_screened_tickers()[:50] 
        result = []
        try:
            simulated_end_date = get_current_date()
            benchmark_data = yf.download(benchmark_ticker, end=simulated_end_date, period=f'{lookback_days+5}d', interval='1d', progress=False, auto_adjust=False)
            #time.sleep(0.5) # Ritardo per YFinance, se necessario
            if benchmark_data is None or benchmark_data.empty or len(benchmark_data) < lookback_days:
                return self.fallback_tickers[:n]
            benchmark_data = flatten_columns(benchmark_data)
            benchmark_data = remove_ticker_suffix_from_columns(benchmark_data, benchmark_ticker) 
            if 'Close' not in benchmark_data.columns and 'close' in benchmark_data.columns: 
                benchmark_data.rename(columns={'close': 'Close'}, inplace=True)
            if 'Close' not in benchmark_data.columns:
                print(f"⚠️ Colonna 'Close' mancante per benchmark '{benchmark_ticker}' dopo rimozione suffisso. Colonne: {benchmark_data.columns.tolist()}")
                return self.fallback_tickers[:n]
            benchmark_data['Close'] = pd.to_numeric(benchmark_data['Close'], errors='coerce')
            benchmark_data.dropna(subset=['Close'], inplace=True)
            if len(benchmark_data) < lookback_days:
                return self.fallback_tickers[:n]
            benchmark_returns = benchmark_data['Close'].pct_change(periods=lookback_days).iloc[-1] * 100
            if pd.isna(benchmark_returns):
                 return self.fallback_tickers[:n]
            for ticker in stocks_to_scan:
                if not isinstance(ticker, str) or not ticker.strip(): continue
                try:
                    simulated_end_date = get_current_date()
                    stock_data = yf.download(ticker, end=simulated_end_date, period=f'{lookback_days+5}d', interval='1d', progress=False, auto_adjust=False)
                    #time.sleep(0.5) # Ritardo per YFinance, se necessario
                    if stock_data is None or stock_data.empty or len(stock_data) < lookback_days:
                        continue
                    stock_data = flatten_columns(stock_data)
                    stock_data = remove_ticker_suffix_from_columns(stock_data, ticker) 
                    if 'Close' not in stock_data.columns and 'close' in stock_data.columns:
                        stock_data.rename(columns={'close':'Close'}, inplace=True)
                    if 'Close' not in stock_data.columns: continue
                    stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
                    stock_data.dropna(subset=['Close'], inplace=True)
                    if len(stock_data) < lookback_days: continue
                    stock_return = stock_data['Close'].pct_change(periods=lookback_days).iloc[-1] * 100
                    if pd.isna(stock_return): continue
                    if stock_return > benchmark_returns + rs_threshold_pct:
                        result.append(ticker)
                    if len(result) >= n:
                        break
                except Exception:
                    continue
            if not result and n > 0:
                return self.fallback_tickers[:n]
            return result
        except Exception as e:
            print(f"❌ Errore generale in get_relative_strength_stocks: {e}")
            return self.fallback_tickers[:n]
        
    def get_valuation_anomalies(self, n=10):
        
        if is_backtest_mode():
            print(f"  [BACKTEST MODE] {self.get_valuation_anomalies.__name__}: Funzione disabilitata, restituisco lista vuota.")
            # Durante il backtest, non possiamo ottenere dati fondamentali storici da Finviz.
            # Restituire una lista vuota è il comportamento più sicuro.
            return []
        
        """
        Trova titoli con disconnessione prezzo/valore (PERLE RARE - Potenziato)
        Rimosso Return on Equity e Operating Margin a causa di problemi con i filtri
        """
        print(f"  💎 Ricerca anomalie di valutazione (Potenziato)...")
        try:
            foverview = FinvizOverview()
            
            filters_dict = {
                'P/E': 'Under 15',                   
                'PEG': 'Under 1',                     
                'P/B': 'Under 2',                     
                'P/S': 'Under 2',                     
                # 'Return on Equity': 'Over +15%',      # Rimosso: Problemi con i filtri Finviz
                'Sales growthpast 5 years': 'Over 10%', 
                'Market Cap.': '+Mid (over $2bln)',
                'Debt/Equity': 'Under 0.5',               
                'Quick Ratio': 'Over 1.5',            
                # 'Operating Margin': 'Over +15%'           # Rimosso: Problemi con i filtri Finviz
            }
            
            foverview.set_filter(filters_dict=filters_dict)
            df = foverview.screener_view(order='Price/Earnings', ascend=True, verbose=0) 
            time.sleep(3) # Aumento delay dopo ogni screener_view

            if df is not None and not df.empty and 'Ticker' in df.columns:
                candidates = df['Ticker'].head(n).tolist()
                candidates = [t for t in candidates if isinstance(t, str) and t.strip() and '.' not in t]
                print(f"    ✅ Trovate {len(candidates)} anomalie di valutazione (Potenziato)")
                return candidates
            return []
        except Exception as e:
            print(f"    ❌ Errore valuation anomalies: {e}")
            return []
    
    def get_technical_breakouts(self, n=10, stocks_to_scan=None):
        # Questo metodo calcola internamente indicatori, ma non vanno nel JSON finale
        print(f"  💎 Ricerca breakout tecnici (Squeeze Play / VCP)...")
        if stocks_to_scan is None:
            stocks_to_scan = self.get_finviz_screened_tickers()[:100]  
        
        breakout_candidates = []
        
        for ticker in stocks_to_scan:
            try:
                simulated_end_date = get_current_date()
                data = yf.download(ticker, end=simulated_end_date, period='6mo', interval='1d', progress=False, auto_adjust=False)
                #time.sleep(0.5) # Ritardo per YFinance, se necessario
                if data is None or len(data) < 100:
                    continue
                    
                data = flatten_columns(data)
                data = remove_ticker_suffix_from_columns(data, ticker)
                
                if not all(col in data.columns for col in ['Close', 'Volume', 'High', 'Low']):
                    continue
                    
                close = pd.to_numeric(data['Close'], errors='coerce').dropna()
                volume = pd.to_numeric(data['Volume'], errors='coerce').dropna()
                high = pd.to_numeric(data['High'], errors='coerce').dropna()
                low = pd.to_numeric(data['Low'], errors='coerce').dropna()
                
                if len(close) < 50 or len(volume) < 50: 
                    continue
                
                bbands_dfs = ta.bbands(close=close, length=20, std=2)
                if not isinstance(bbands_dfs, pd.DataFrame) or 'BBL_20_2.0' not in bbands_dfs.columns:
                    continue
                
                bb_lower = bbands_dfs['BBL_20_2.0']
                bb_upper = bbands_dfs['BBU_20_2.0']
                bb_mid = bbands_dfs['BBM_20_2.0']
                bb_width = (bb_upper - bb_lower) / bb_mid
                
                if len(bb_width.dropna()) < 30: continue
                
                is_squeezing = bb_width.iloc[-1] < bb_width.iloc[-30:-1].quantile(0.2) 
                
                avg_volume = volume.rolling(20).mean()
                if pd.isna(avg_volume.iloc[-1]) or avg_volume.iloc[-1] == 0: continue
                volume_spike = volume.iloc[-1] > avg_volume.iloc[-1] * 1.5 
                
                recent_high = high.iloc[-50:-1].max() 
                price_breakout = close.iloc[-1] > recent_high * 1.01 
                
                close_near_high = (close.iloc[-1] - low.iloc[-1]) / (high.iloc[-1] - low.iloc[-1] + 1e-9) > 0.7 
                
                if is_squeezing and volume_spike and price_breakout and close_near_high:
                    breakout_candidates.append(ticker)
                    
                if len(breakout_candidates) >= n:
                    break
                    
            except Exception:
                continue
        
        print(f"    ✅ Trovati {len(breakout_candidates)} breakout tecnici (Squeeze Play / VCP)")
        return breakout_candidates
    
    def get_institutional_flow_candidates(self, n=10):
        
        
        if is_backtest_mode():
            print(f"  [BACKTEST MODE] {self.get_valuation_anomalies.__name__}: Funzione disabilitata, restituisco lista vuota.")
            # Durante il backtest, non possiamo ottenere dati fondamentali storici da Finviz.
            # Restituire una lista vuota è il comportamento più sicuro.
            return []
        
        
        """
        Identifica titoli con flussi istituzionali positivi (PERLE RARE)
        CORRETTO: 'Institutional Ownership' a 'InstitutionalOwnership'
        CORRETTO: 'Insider Ownership' a 'InsiderOwnership'
        """
        print(f"  💎 Ricerca flussi istituzionali...")
        try:
            foverview = FinvizOverview()
            filters_dict = {
                'InstitutionalOwnership': 'Over 70%',  # Corretto nome filtro
                'InsiderOwnership': 'Over 10%',        # Corretto nome filtro
                'Performance': 'Quarter Up',           
                'Float Short': 'Under 10%'            
            }
            
            foverview.set_filter(filters_dict=filters_dict)
            df = foverview.screener_view(order='Institutional Ownership', ascend=False, verbose=0)  
            time.sleep(3) # Aumento delay dopo ogni screener_view

            if df is not None and not df.empty and 'Ticker' in df.columns:
                candidates = df['Ticker'].head(n).tolist()
                candidates = [t for t in candidates if isinstance(t, str) and t.strip() and '.' not in t]
                print(f"    ✅ Trovati {len(candidates)} con flussi istituzionali")
                return candidates
            return []
        except Exception as e:
            print(f"    ❌ Errore institutional flow: {e}")
            return []

    def get_insider_buying_candidates(self, n=10):
        
        if is_backtest_mode():
            print(f"  [BACKTEST MODE] {self.get_valuation_anomalies.__name__}: Funzione disabilitata, restituisco lista vuota.")
            # Durante il backtest, non possiamo ottenere dati fondamentali storici da Finviz.
            # Restituire una lista vuota è il comportamento più sicuro.
            return []
        
        
        """
        NUOVO: Identifica titoli con acquisti recenti significativi da insider (PERLE RARE - Segnale Forte)
        CORRETTO: 'Insider Trading' a 'InsiderTransactions' e opzione 'Recent Buying' a 'Positive (>0%)'
        """
        print(f"  💎 Ricerca titoli con acquisti insider...")
        try:
            foverview = FinvizOverview()
            filters_dict = {
                'InsiderTransactions': 'Positive (>0%)', # Corretto nome filtro e opzione
                'Market Cap.': '+Mid (over $2bln)',
                'Average Volume': 'Over 500K' # Corretto qui anche
            }
            
            foverview.set_filter(filters_dict=filters_dict)
            df = foverview.screener_view(order='Insider Transactions', ascend=False, verbose=0) 
            time.sleep(3) # Aumento delay dopo ogni screener_view

            if df is not None and not df.empty and 'Ticker' in df.columns:
                candidates = df['Ticker'].head(n).tolist()
                candidates = [t for t in candidates if isinstance(t, str) and t.strip() and '.' not in t]
                print(f"    ✅ Trovati {len(candidates)} con acquisti insider recenti")
                return candidates
            return []
        except Exception as e:
            print(f"    ❌ Errore insider buying: {e}")
            return []

    def get_analyst_upgrades_candidates(self, n=10):
        
        
        if is_backtest_mode():
            print(f"  [BACKTEST MODE] {self.get_valuation_anomalies.__name__}: Funzione disabilitata, restituisco lista vuota.")
            # Durante il backtest, non possiamo ottenere dati fondamentali storici da Finviz.
            # Restituire una lista vuota è il comportamento più sicuro.
            return []
        
        """
        NUOVO: Identifica titoli con upgrade recenti di rating da analisti (PERLE RARE - Catalizzatore)
        CORRETTO: Rimossa 'Avg Volume' non pertinente.
        """
        print(f"  💎 Ricerca titoli con upgrade analisti...")
        try:
            foverview = FinvizOverview()
            filters_dict = {
                'Analyst Recom.': 'Buy or better', # Corretta l'opzione
                # 'Average Volume': 'Over 500K' # Rimosso questo filtro che causava errore
            }
            
            foverview.set_filter(filters_dict=filters_dict)
            df = foverview.screener_view(order='Analyst Recommendation', ascend=False, verbose=0) 
            time.sleep(3) # Aumento delay dopo ogni screener_view

            if df is not None and not df.empty and 'Ticker' in df.columns:
                candidates = df['Ticker'].head(n).tolist()
                candidates = [t for t in candidates if isinstance(t, str) and t.strip() and '.' not in t]
                print(f"    ✅ Trovati {len(candidates)} con upgrade analisti recenti")
                return candidates
            return []
        except Exception as e:
            print(f"    ❌ Errore analyst upgrades: {e}")
            return []
            
    def get_catalyst_candidates(self, n=10):
        
        if is_backtest_mode():
            print(f"  [BACKTEST MODE] {self.get_valuation_anomalies.__name__}: Funzione disabilitata, restituisco lista vuota.")
            # Durante il backtest, non possiamo ottenere dati fondamentali storici da Finviz.
            # Restituire una lista vuota è il comportamento più sicuro.
            return []
        
        
        """
        Cerca titoli con potenziali catalizzatori (PERLE RARE - Potenziato)
        Rimosso Return on Investment a causa di problemi con i filtri
        """
        print(f"  💎 Ricerca catalizzatori fondamentali (Potenziato)...")
        try:
            foverview = FinvizOverview()
            
            filters_dict = {
                'EPS growthnext year': 'Over 20%',    
                'Sales growthpast 5 years': 'Over 15%', 
                # 'Return on Investment': 'Over +15%',   # Rimosso: Problemi con i filtri Finviz
                'Current Ratio': 'Over 2',            
                'LT Debt/Equity': 'Under 0.3',        
                'Analyst Recom.': 'Buy or better' 
            }
            
            foverview.set_filter(filters_dict=filters_dict)
            df = foverview.screener_view(order='EPS growth next year', ascend=False, verbose=0)  
            time.sleep(3) # Aumento delay dopo ogni screener_view

            if df is not None and not df.empty and 'Ticker' in df.columns:
                candidates = df['Ticker'].head(n).tolist()
                candidates = [t for t in candidates if isinstance(t, str) and t.strip() and '.' not in t]
                print(f"    ✅ Trovati {len(candidates)} con catalizzatori")
                return candidates
            return []
        except Exception as e:
            print(f"    ❌ Errore catalyst candidates: {e}")
            return []

    def get_advanced_signal_candidates(self, n=10, stocks_to_scan=None):
        """
        Scansione per segnali basati su Permutation Entropy, RQA e Adaptive Timeframe (euristici).
        Questi sono indicatori più "speciali" che possono rivelare la natura del trend.
        Sono usati SOLO per la selezione dei candidati, non per l'output JSON.
        """
        print(f"  💎 Ricerca segnali avanzati (Permutation Entropy, RQA, Adaptive Timeframe - euristiche)...")
        
        # Le librerie nolds/pyRQA non sono più un requisito per lo stock_analyzer per il suo output,
        # ma i placeholder euristiche funzionano anche senza di esse, quindi rimuovo il warning qui.
        
        if stocks_to_scan is None:
            # Aggiungo un sleep qui perché get_finviz_screened_tickers potrebbe essere la prima chiamata Finviz del blocco
            stocks_to_scan = self.get_finviz_screened_tickers()[:100] 
            time.sleep(3) # Sleep after getting base scan list

        signal_candidates = []
        for ticker in stocks_to_scan:
            try:
                simulated_end_date = get_current_date()
                # Scarica più dati per un'analisi affidabile degli indicatori euristici
                data = yf.download(ticker, end=simulated_end_date, period='1y', interval='1d', progress=False, auto_adjust=False)
                #time.sleep(0.5) # Ritardo per YFinance, se necessario
                if data is None or data.empty or len(data) < 150: 
                    continue
                
                data = flatten_columns(data)
                data = remove_ticker_suffix_from_columns(data, ticker)
                if 'Close' not in data.columns: continue
                close_series_np = pd.to_numeric(data['Close'], errors='coerce').dropna().to_numpy()

                if len(close_series_np) < 150: continue 

                # Calculate advanced heuristic indicators
                pe_score = calculate_permutation_entropy_heuristic(close_series_np[-100:])
                rqa_scores = calculate_rqa_metrics_heuristic(close_series_np[-100:])
                adapt_scores = calculate_adaptive_market_state_metrics_heuristic(close_series_np[-120:])

                # Define conditions for "good" advanced signals (heuristic thresholds)
                is_low_entropy = pd.notna(pe_score) and pe_score < 0.35 
                is_deterministic = pd.notna(rqa_scores.get('determinism')) and rqa_scores.get('determinism') > 0.7 
                is_high_signal_quality = pd.notna(adapt_scores.get('signal_quality')) and adapt_scores.get('signal_quality') > 2.0 
                is_low_noise_ratio = pd.notna(adapt_scores.get('noise_ratio')) and adapt_scores.get('noise_ratio') < 0.015 
                
                # Combine conditions
                if (is_low_entropy or is_deterministic) and (is_high_signal_quality or is_low_noise_ratio):
                    signal_candidates.append(ticker)
                
                if len(signal_candidates) >= n:
                    break
            except Exception as e:
                continue
        print(f"    ✅ Trovati {len(signal_candidates)} candidati con segnali avanzati (euristici)")
        return signal_candidates

            
    def get_diversified_candidates(self, max_stocks=10, min_history_days=150):
        # Questo contatore terrà traccia dei fallback
        num_fallbacks_in_final_selection = 0
        
        print("\n🔍 Selezione candidati diversificati con ricerca PERLE RARE (v2.0)...")
        # Inizializza un set per tenere traccia dei candidati trovati dalle strategie (non fallback)
        initial_candidates_from_strategies_set = set()

        base_scan_list_raw = self.get_finviz_screened_tickers(
            min_market_cap_str='+Mid (over $2bln)', 
            min_avg_volume_str='Over 500K',      
            min_price_str='Over $5'              
        )
        # Determina se base_scan_list_raw è effettivamente la lista di fallback o una lista da Finviz
        # Confronta con la lista di fallback *predefinita*, non solo il subset.
        if base_scan_list_raw and (set(base_scan_list_raw) == set(self.fallback_tickers[:50]) or not base_scan_list_raw):
            print("  ℹ️ Finviz returned fallback list. Will count these towards fallbacks if selected.")
            base_scan_list = self.fallback_tickers[:50] # Forzare l'uso della lista fallback se Finviz fallisce
            # Non aggiungere questi alla initial_candidates_from_strategies_set se sono già i fallback
        else:
            base_scan_list = base_scan_list_raw
            initial_candidates_from_strategies_set.update(base_scan_list) # Tutti i tickers da Finviz sono da strategie

        if len(base_scan_list) > 200:
            print(f"  INFO: La lista base da Finviz ha {len(base_scan_list)} tickers, la limito a 200 per le strategie.")
            base_scan_list = base_scan_list[:200]

        candidates_scores = {} 
        
        # === STRATEGIE TRADIZIONALI (peso 1-2) ===
        try:
            day_gainers = self.get_top_gainers(n=30, cp_threshold=3, vol_threshold=5e5, price_lower=1, mcap_lower=3e8)
            for ticker in day_gainers:
                candidates_scores[ticker] = candidates_scores.get(ticker, 0) + 2 
                initial_candidates_from_strategies_set.add(ticker)
            time.sleep(3) # Aumento delay dopo ogni screener_view (o equivalente)
        except Exception as e:
            print(f"⚠️ Errore nel recupero Day Gainers: {str(e)}")
        
        try:
            volume_breakouts = self.get_volume_breakouts(n=30, stocks_to_scan=base_scan_list, min_avg_vol=3e5, vol_multiplier=1.8)
            for ticker in volume_breakouts:
                candidates_scores[ticker] = candidates_scores.get(ticker, 0) + 1
                initial_candidates_from_strategies_set.add(ticker)
        except Exception as e:
            print(f"⚠️ Errore nel calcolo dei volume breakouts: {str(e)}")
        
        try:
            trending_stocks = self.get_trending_stocks(n=30, stocks_to_scan=base_scan_list, ma_short=20, ma_long=50)
            for ticker in trending_stocks:
                candidates_scores[ticker] = candidates_scores.get(ticker, 0) + 1
                initial_candidates_from_strategies_set.add(ticker)
        except Exception as e:
            print(f"⚠️ Errore nel calcolo dei trending stocks: {str(e)}")
        
        try:
            relative_strength = self.get_relative_strength_stocks(n=30, stocks_to_scan=base_scan_list, rs_threshold_pct=2.0)
            for ticker in relative_strength:
                candidates_scores[ticker] = candidates_scores.get(ticker, 0) + 1
                initial_candidates_from_strategies_set.add(ticker)
        except Exception as e:
            print(f"⚠️ Errore nel calcolo della relative strength: {str(e)}")

        # === NUOVE STRATEGIE PERLE RARE (peso 3-5) ===
        try:
            valuation_anomalies = self.get_valuation_anomalies(n=15)
            for ticker in valuation_anomalies:
                candidates_scores[ticker] = candidates_scores.get(ticker, 0) + 4  
                initial_candidates_from_strategies_set.add(ticker)
        except Exception as e:
            print(f"⚠️ Errore valuation anomalies: {e}")
        
        try:
            technical_breakouts = self.get_technical_breakouts(n=15, stocks_to_scan=base_scan_list[:100]) 
            for ticker in technical_breakouts:
                candidates_scores[ticker] = candidates_scores.get(ticker, 0) + 3 
                initial_candidates_from_strategies_set.add(ticker)
        except Exception as e:
            print(f"⚠️ Errore technical breakouts: {e}")
        
        try:
            institutional_flow = self.get_institutional_flow_candidates(n=15)
            for ticker in institutional_flow:
                candidates_scores[ticker] = candidates_scores.get(ticker, 0) + 3
                initial_candidates_from_strategies_set.add(ticker)
        except Exception as e:
            print(f"⚠️ Errore institutional flow: {e}")

        try:
            insider_buying = self.get_insider_buying_candidates(n=15) 
            for ticker in insider_buying:
                candidates_scores[ticker] = candidates_scores.get(ticker, 0) + 5 
                initial_candidates_from_strategies_set.add(ticker)
        except Exception as e:
            print(f"⚠️ Errore insider buying: {e}")

        try:
            analyst_upgrades = self.get_analyst_upgrades_candidates(n=15) 
            for ticker in analyst_upgrades:
                candidates_scores[ticker] = candidates_scores.get(ticker, 0) + 4 
                initial_candidates_from_strategies_set.add(ticker)
        except Exception as e:
            print(f"⚠️ Errore analyst upgrades: {e}")

        try:
            catalyst_candidates = self.get_catalyst_candidates(n=15)
            for ticker in catalyst_candidates:
                candidates_scores[ticker] = candidates_scores.get(ticker, 0) + 4  
                initial_candidates_from_strategies_set.add(ticker)
        except Exception as e:
            print(f"⚠️ Errore catalyst candidates: {e}")

        try:
            advanced_signals = self.get_advanced_signal_candidates(n=15, stocks_to_scan=base_scan_list[:150]) 
            for ticker in advanced_signals:
                candidates_scores[ticker] = candidates_scores.get(ticker, 0) + 5 
                initial_candidates_from_strategies_set.add(ticker)
        except Exception as e:
            print(f"⚠️ Errore advanced signal candidates: {str(e)}") # Modificato per stampare l'errore specifico se ce n'è uno.


        if not candidates_scores:
            print("⚠️ Nessun candidato trovato da nessuna strategia. Uso fallback tickers.")
            # Se nessun candidato è stato trovato dalle strategie, popola con i fallback
            final_selected_candidates = self.fallback_tickers[:max_stocks]
            num_fallbacks_in_final_selection = len(final_selected_candidates) # Tutti sono fallback
            return final_selected_candidates, num_fallbacks_in_final_selection
        
        sorted_candidates_by_score = sorted(candidates_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Mostra top candidates con score
        print(f"\n🏆 TOP CANDIDATES BY SCORE:")
        for i, (ticker, score) in enumerate(sorted_candidates_by_score[:20]):
            emoji = "💎" if score >= 4 else "🌟" if score >= 3 else "📈"
            print(f"  {i+1:2d}. {emoji} {ticker}: {score} points")
        
        print(f"\n🔍 Verifica storico (min {min_history_days}gg) per {len(sorted_candidates_by_score)} candidati...")
        final_verified_candidates = []
        
        checked_count = 0
        for ticker, score in sorted_candidates_by_score:
            if len(final_verified_candidates) >= max_stocks:
                break 
            checked_count += 1
            try:
                simulated_end_date = get_current_date()
                data_check = yf.download(ticker, end=simulated_end_date, period='18mo', interval='1d', progress=False, auto_adjust=False)
                if data_check is not None and not data_check.empty:
                    data_check = flatten_columns(data_check) 
                    data_check = remove_ticker_suffix_from_columns(data_check, ticker) 
                    if 'Close' not in data_check.columns and 'close' in data_check.columns:
                        data_check.rename(columns={'close':'Close'}, inplace=True)
                    if 'Close' in data_check.columns and len(data_check['Close'].dropna()) >= min_history_days:
                        final_verified_candidates.append(ticker)
                        # Se il ticker non era tra quelli inizialmente trovati dalle strategie (ovvero, non era nel set)
                        # ma era tra i fallback, allora lo contiamo come un fallback aggiunto alla selezione finale.
                        if ticker in self.fallback_tickers and ticker not in initial_candidates_from_strategies_set:
                            num_fallbacks_in_final_selection += 1
            except Exception as e:
                pass # Errore nel download o verifica, ignora questo ticker

        # Aggiungi fallback tickers se non abbiamo raggiunto il numero desiderato di candidati
        if len(final_verified_candidates) < max_stocks:
            for fb_ticker in self.fallback_tickers:
                if len(final_verified_candidates) >= max_stocks:
                    break
                # Aggiungi solo se non è già stato verificato o selezionato
                if fb_ticker not in final_verified_candidates:
                    final_verified_candidates.append(fb_ticker)
                    # Conta questo come fallback aggiunto
                    num_fallbacks_in_final_selection += 1
        
        final_selected_candidates = final_verified_candidates[:max_stocks]
        
        if not final_selected_candidates: 
            print("‼️ Nessun candidato finale selezionato. Controllare strategie.")
            return [], 0 # Nessun candidato, 0 fallback
        
        print(f"\n🏆 SELEZIONE FINALE ({len(final_selected_candidates)} perle): {', '.join(final_selected_candidates)}")
        # Ritorna la lista dei candidati e il conteggio dei fallback
        return final_selected_candidates, num_fallbacks_in_final_selection

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


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.2f}'.format) 
    print("--- START stock_analyzer_2_0.py ---")
    print(f"Script avviato alle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stock_analyzer = StockAnalyzer()
    position_tickers = []
    try:
        trading_state_file = BASE_DIR / 'data' / 'trading_state.json'
        if trading_state_file.exists():
            with open(trading_state_file, 'r') as f:
                trading_state = json.load(f)
                open_positions = trading_state.get('open_positions', [])
                position_tickers = [pos.get('ticker') for pos in open_positions if isinstance(pos.get('ticker'), str) and pos.get('ticker').strip()]
                position_tickers = list(set(position_tickers)) 
                print(f"\n📋 Posizioni attualmente aperte ({len(position_tickers)}): {', '.join(position_tickers)}")
        else:
             print(f"\n⚠️ File di stato trading '{trading_state_file}' non trovato. Nessuna posizione aperta caricata.")
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        position_tickers = []
        print(f"\n⚠️ Errore nel caricamento delle posizioni aperte da '{trading_state_file}': {str(e)}")
    
    num_candidates_from_sources = 20 
    # Ora get_diversified_candidates restituisce una tupla (lista_tickers, num_fallback_tickers)
    new_candidates_from_sources, num_fallbacks_in_new_candidates = stock_analyzer.get_diversified_candidates(max_stocks=num_candidates_from_sources)
    
    best_buy_tickers = load_best_buy_tickers(BEST_BUY_FILE)
    
    # Raccogli tutti i ticker unici da analizzare, inclusi i fallback dalla selezione diversificata
    all_potential_tickers_raw = position_tickers + new_candidates_from_sources + best_buy_tickers
    all_potential_tickers = list(set(all_potential_tickers_raw)) 
    print(f"\n🎯 Totale ticker unici potenziali da analizzare ({len(all_potential_tickers)}): {', '.join(sorted(all_potential_tickers))}")
    
    tickers_to_analyze = all_potential_tickers 
    print(f"\n🔬 Avvio analisi dettagliata per {len(tickers_to_analyze)} ticker...")
    analysis_results = {}
    for ticker_item in tickers_to_analyze: 
        if not isinstance(ticker_item, str) or not ticker_item.strip():
             print(f"⚠️ Saltato ticker non valido: '{ticker_item}'")
             continue
        stock_data = stock_analyzer.analyze_stock(ticker_item) 
        if stock_data is not None and not stock_data.empty:
            analysis_results[ticker_item] = stock_data
            try:
                last_row = stock_data.tail(1)
                close_col = 'Close' 
                if close_col in last_row.columns and 'Sentiment' in last_row.columns:
                    print(f"✅ {ticker_item} - Analisi OK (fine). Chiusura: {last_row[close_col].values[0]:.2f}. Sentiment: {last_row['Sentiment'].values[0]}")
                elif close_col in last_row.columns: 
                    print(f"✅ {ticker_item} - Analisi OK (fine). Chiusura: {last_row[close_col].values[0]:.2f}. Sentiment: N/D")
                else:
                    print(f"✅ {ticker_item} - Analisi OK (fine). Prezzo di chiusura non trovato.")
            except Exception as e:
                print(f"✅ {ticker_item} - Analisi OK (fine). Errore visualizzazione sommario: {str(e)}")
    
    stock_analyzer.save_analysis_results(analysis_results)
    
    # Aggiungi questa riga per il conteggio finale e il fallback
    print(f"\n📊 Totale titoli analizzati e passati al Trading Engine: {len(analysis_results)} (di cui {num_fallbacks_in_new_candidates} fallback).")
    print("\n✅ Processo di analisi completato.")
    print("--- END stock_analyzer_2_0.py ---")
