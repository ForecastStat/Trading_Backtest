import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore') # Mantieni per sopprimere warning comuni come quello di yfinance

# Import della libreria finvizfinance
from finvizfinance.screener.overview import Overview
import traceback

# --- INIZIO BLOCCO MODIFICA PER BACKTEST ---
# --- INIZIO BLOCCO MODIFICA PER BACKTEST ---
import os
from datetime import datetime
def get_current_date():
    simulated_date_str = os.environ.get('SIMULATED_DATE')
    if simulated_date_str:
        return datetime.strptime(simulated_date_str, '%Y-%m-%d')
    return datetime.now()
# --- FINE BLOCCO MODIFICA PER BACKTEST ---

# --- Funzione helper per Finviz: convert_market_cap_original_robust ---
def convert_market_cap_original_robust(val):
    """Versione robusta della tua funzione originale convert_market_cap."""
    if isinstance(val, str):
        val = val.strip()
        if val == '-' or val == '':
            return None
        if val.endswith('B'):
            try:
                return float(val[:-1]) * 1e9
            except ValueError:
                return None
        elif val.endswith('M'):
            try:
                return float(val[:-1]) * 1e6
            except ValueError:
                return None
        else:
            try:
                return float(val)
            except ValueError:
                return None
    elif isinstance(val, (int, float)):
        return float(val)
    return None

# --- Classe principale EnhancedBestBuySelector ---
class EnhancedBestBuySelector:
    def __init__(self):
        self.alpha_vantage_key = "QCA8K5F0O19WGSEN"
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        self.sector_universe = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'CRM', 'ADBE', 'NFLX', 'AMD', 'AVGO', 'ORCL', 'CSCO', 'ACN'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'ABT', 'MRK', 'BMY', 'AMGN', 'GILD', 'VRTX', 'BIIB', 'MRNA', 'BNTX'],
            'financials': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SCHW', 'USB', 'PNC', 'TFC', 'COF', 'CME', 'ICE'],
            'consumer_discretionary': ['AMZN', 'TSLA', 'HD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'MCD', 'DIS', 'ABNB', 'UBER', 'LYFT'],
            'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'DVN', 'FANG', 'APA'],
            'industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD', 'FDX', 'UNP', 'CSX'],
            'etfs': ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLV', 'XLK', 'XLI', 'ARKK', 'ICLN', 'VWO', 'EWJ']
        }
        
        self.macro_indicators = {
            'VIX': '^VIX',
            'DXY': 'DX-Y.NYB',
            'TNX': '^TNX',
            'YIELD_CURVE': None
        }

    def _get_watchlist_from_finviz(self, num_top_tickers_finviz=100):
        """
        Estrae una watchlist di ticker da Finviz basandosi su filtri.
        Questa è la Fase 1 dello screening.
        Restituisce una lista di ticker.
        Copiata esattamente la logica di best_buy 10 06 2025.py.txt
        """
        print(f"ℹ️ Fase 1: Estrazione watchlist iniziale da Finviz...")
        try:
            overview = Overview()
            filters_dict = {'Analyst Recom.': 'Strong Buy (1)'}
            overview.set_filter(filters_dict=filters_dict)

            df_raw = overview.screener_view(order='Price/Earnings', ascend=True)

            if df_raw is None or df_raw.empty:
                print("⚠️ Finviz: Nessun dato restituito dallo screener di Finviz.")
                return [] 

            print(f"✅ Finviz: Scaricati {df_raw.shape[0]} titoli da Finviz con 'Strong Buy'.")

            df = df_raw.copy()

            if "Price/Earnings" in df.columns:
                df = df.rename(columns={"Price/Earnings": "P/E"})
            else:
                if "P/E" not in df.columns: df["P/E"] = pd.NA

            expected_columns_to_select = ["Ticker", "Company", "Country", "Market Cap", "P/E", "Price", "Change", "Volume"]
            actual_columns_present = [col for col in expected_columns_to_select if col in df.columns]
            
            if not actual_columns_present or "Ticker" not in actual_columns_present:
                 print("❌ Finviz: ERRORE CRITICO: La colonna 'Ticker' non è disponibile dopo il download da Finviz.")
                 return []
            
            df = df[actual_columns_present]

            if "P/E" in df.columns:
                df["P/E"] = pd.to_numeric(df["P/E"], errors='coerce')
            if "Market Cap" in df.columns:
                df["Market Cap"] = df["Market Cap"].apply(convert_market_cap_original_robust)
            if "Price" in df.columns:
                df["Price"] = pd.to_numeric(df["Price"], errors='coerce')
            if "Change" in df.columns:
                change_series = df["Change"]
                if pd.api.types.is_string_dtype(change_series):
                    df["Change"] = pd.to_numeric(change_series.str.rstrip('%'), errors='coerce') / 100.0
                elif pd.api.types.is_numeric_dtype(change_series):
                    df["Change"] = change_series / 100.0
                else:
                    df["Change"] = pd.to_numeric(change_series, errors='coerce') / 100.0
            if "Volume" in df.columns:
                volume_series = df["Volume"]
                if pd.api.types.is_string_dtype(volume_series):
                     df["Volume"] = pd.to_numeric(volume_series.str.replace(',', '', regex=False), errors='coerce')
                else:
                     df["Volume"] = pd.to_numeric(volume_series, errors='coerce')

            cols_for_dropna = []
            if "P/E" in df.columns: cols_for_dropna.append("P/E")
            if "Change" in df.columns: cols_for_dropna.append("Change")
            if "Market Cap" in df.columns: cols_for_dropna.append("Market Cap")
            if cols_for_dropna:
                df = df.dropna(subset=cols_for_dropna)
            
            if df.empty:
                print("⚠️ Finviz: Nessun titolo rimasto dopo la rimozione dei NaN.")
                return []

            conditions = []
            if "P/E" in df.columns: conditions.append((df["P/E"] > 0) & (df["P/E"] < 20))
            if "Change" in df.columns: conditions.append(df["Change"] > 0)
            
            if not conditions:
                filtered_df = df.copy()
            else:
                final_condition = pd.Series(True, index=df.index)
                for cond in conditions:
                    final_condition &= cond
                filtered_df = df[final_condition]

            if filtered_df.empty:
                print("⚠️ Finviz: Nessun titolo soddisfa i filtri (P/E > 0 e < 20, Change > 0).")
                return []
                
            print(f"✅ Finviz: Trovati {filtered_df.shape[0]} titoli dopo i filtri P/E e Change.")

            sort_by_cols = []
            ascending_order = []
            if "P/E" in filtered_df.columns:
                sort_by_cols.append("P/E")
                ascending_order.append(True)
            if "Change" in filtered_df.columns:
                sort_by_cols.append("Change")
                ascending_order.append(False)

            if not sort_by_cols:
                top_picks_df = filtered_df
            else:
                top_picks_df = filtered_df.sort_values(by=sort_by_cols, ascending=ascending_order)
            
            final_top_tickers_list = []
            if "Ticker" in top_picks_df.columns:
                final_top_tickers_list = top_picks_df["Ticker"].head(num_top_tickers_finviz).tolist()
            
            if not final_top_tickers_list:
                print("⚠️ Finviz: Nessun ticker finale selezionato dopo l'ordinamento e head().")
                return []
            
            print(f"✅ Finviz: Selezionati {len(final_top_tickers_list)} ticker dalla Fase 1 (Finviz).")
            return final_top_tickers_list

        except Exception as e:
            print(f"❌ Finviz: Si è verificato un errore durante l'estrazione da Finviz: {e}")
            traceback.print_exc()
            return []

    # --- NUOVA FUNZIONE HELPER per estrarre valori scalari in modo sicuro ---
    def _get_last_scalar_value(self, series):
        """
        Extracts the last scalar value from a pandas Series.
        Returns np.nan if the series is empty or the last value is not a scalar.
        """
        if series is None or series.empty:
            return np.nan
        try:
            val = series.iloc[-1]
            # Ensure it's not a Series of 1, convert to scalar if it is
            if isinstance(val, pd.Series):
                if not val.empty and val.size == 1:
                    return val.item()
                else: # empty or multi-element Series after iloc[-1] (should not happen with a 1D Series)
                    return np.nan
            return float(val) # Ensure the final type is float, as np.nan is also a float
        except (IndexError, ValueError, TypeError): # Catch errors from iloc, item, or float conversion
            return np.nan
        except Exception: # Catch any other unexpected errors
            return np.nan

    def get_market_regime(self):
        """Determina il regime di mercato attuale"""
        try:
            simulated_end_date = get_current_date()
            # Increased period for more robust MA calculation
            spy_data = yf.download('SPY', end=simulated_end_date, period='2y', interval='1d', progress=False)
            vix_data = yf.download('^VIX', end=simulated_end_date, period='1y', interval='1d', progress=False)
            
            if spy_data.empty or vix_data.empty:
                print("DEBUG: SPY or VIX data is empty for market regime analysis. Returning neutral.")
                return 'neutral'
            
            spy_close = spy_data['Close']
            vix_close = vix_data['Close']

            # Check if there's enough *raw* data to compute the longest MAs
            # This is the primary check for data sufficiency.
            if len(spy_close) < 200: # Need at least 200 days for SPY MA200
                print(f"DEBUG: SPY Close data length ({len(spy_close)}) is less than 200. Cannot calculate MA200. Returning neutral.")
                return 'neutral'
            if len(vix_close) < 20: # Need at least 20 days for VIX MA20
                print(f"DEBUG: VIX Close data length ({len(vix_close)}) is less than 20. Cannot calculate MA20. Returning neutral.")
                return 'neutral'

            # Calculate Moving Averages. These will have leading NaNs, but if raw data is sufficient,
            # their *last* value should be calculable (unless underlying data has NaNs).
            spy_ma_50_series = spy_close.rolling(50).mean()
            spy_ma_200_series = spy_close.rolling(200).mean()
            vix_ma_20_series = vix_close.rolling(20).mean()

            # Extract scalar values from the end of the series using the robust helper.
            # This helper ensures that the values are floats or np.nan, never a Series.
            spy_price = self._get_last_scalar_value(spy_close)
            spy_ma_50 = self._get_last_scalar_value(spy_ma_50_series)
            spy_ma_200 = self._get_last_scalar_value(spy_ma_200_series)
            vix_current = self._get_last_scalar_value(vix_close)
            vix_ma_20 = self._get_last_scalar_value(vix_ma_20_series)
            
            # --- DEBUGGING PRINTS --- (Keep them for now, they are crucial for verification)
            print(f"DEBUG: spy_price: {spy_price} (type: {type(spy_price)})")
            print(f"DEBUG: spy_ma_50: {spy_ma_50} (type: {type(spy_ma_50)})")
            print(f"DEBUG: spy_ma_200: {spy_ma_200} (type: {type(spy_ma_200)})")
            print(f"DEBUG: vix_current: {vix_current} (type: {type(vix_current)})")
            print(f"DEBUG: vix_ma_20: {vix_ma_20} (type: {type(vix_ma_20)})")
            # --- END DEBUGGING PRINTS ---

            # Final check that all extracted values are valid numbers.
            # `any(pd.isna([...]))` works correctly because `pd.isna` will receive individual scalar floats/np.nan.
            if any(pd.isna([spy_price, spy_ma_50, spy_ma_200, vix_current, vix_ma_20])):
                print("DEBUG: One or more critical market regime values are NaN after extraction. Returning neutral.")
                return 'neutral'

            # Determine trend based on valid scalar values
            is_spy_bullish = (spy_price > spy_ma_50) and (spy_ma_50 > spy_ma_200)
            is_spy_bearish = (spy_price < spy_ma_50) and (spy_ma_50 < spy_ma_200)
            is_vix_calm = (vix_current < vix_ma_20)
            is_vix_volatile = (vix_current > vix_ma_20)

            if is_spy_bullish and is_vix_calm:
                return 'bull'
            elif is_spy_bearish and is_vix_volatile:
                return 'bear'
            else:
                return 'neutral'
        except Exception as e:
            # This catch-all exception is for unexpected errors outside of NaN handling.
            print(f"❌ Errore critico nel determinare il regime di mercato: {e}. Restituzione 'neutral'.")
            traceback.print_exc()
            return 'neutral'

    def calculate_quality_score(self, ticker):
        """Calcola quality score basato su metriche fondamentali"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            score = 0
            
            roe = info.get('returnOnEquity')
            if roe is not None and roe > 0.15: score += 20
            
            debt_to_equity = info.get('debtToEquity')
            if debt_to_equity is not None and debt_to_equity < 50: score += 20
            
            current_ratio = info.get('currentRatio')
            if current_ratio is not None and current_ratio > 1.5: score += 15
            
            revenue_growth = info.get('revenueGrowth')
            if revenue_growth is not None and revenue_growth > 0.10: score += 20
            
            operating_margin = info.get('operatingMargins')
            if operating_margin is not None and operating_margin > 0.10: score += 15
            
            free_cash_flow = info.get('freeCashflow')
            if free_cash_flow is not None and free_cash_flow > 0: score += 10
            
            return score
        except Exception: # No need to print for every ticker failing
            return 0

    def calculate_growth_score(self, ticker):
        """Calcola growth score"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            score = 0
            
            eps_growth = info.get('earningsGrowth')
            if eps_growth is not None:
                if eps_growth > 0.20: score += 30
                elif eps_growth > 0.10: score += 20
            
            revenue_growth = info.get('revenueGrowth')
            if revenue_growth is not None:
                if revenue_growth > 0.15: score += 25
                elif revenue_growth > 0.10: score += 15
            
            peg_ratio = info.get('pegRatio')
            if peg_ratio is not None and peg_ratio > 0:
                if peg_ratio < 1: score += 25
                elif peg_ratio < 1.5: score += 15
            
            forward_pe = info.get('forwardPE')
            if forward_pe is not None and forward_pe > 0:
                if 5 < forward_pe < 20: score += 20
            
            return score
        except Exception:
            return 0

    def calculate_technical_score(self, ticker):
        """Calcola technical score"""
        try:
            simulated_end_date = get_current_date()
            data = yf.download(ticker, end=simulated_end_date, period='6mo', interval='1d', progress=False)
            
            if data.empty: return 0
            
            if not data.index.is_unique: data = data[~data.index.duplicated(keep='first')]
            data.sort_index(inplace=True)

            required_cols = ['Close', 'Volume', 'High']
            for col in required_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                else:
                    return 0

            data.dropna(subset=required_cols, inplace=True)
            if data.empty or len(data) < 60: return 0
            
            score = 0
            close = data['Close']
            volume = data['Volume']
            
            ma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
            ma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
            ma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
            current_price = close.iloc[-1]
            
            if not pd.isna(current_price) and not pd.isna(ma_20) and not pd.isna(ma_50):
                if current_price > ma_20:
                    score += 15
                    if not pd.isna(ma_200) and ma_20 > ma_50 and ma_50 > ma_200: score += 10
            
            if len(volume) >= 20:
                avg_volume_20 = volume.rolling(20).mean().iloc[-1]
                recent_volume = volume.iloc[-5:].mean()
                if not pd.isna(recent_volume) and not pd.isna(avg_volume_20):
                    if recent_volume > avg_volume_20 * 1.5: score += 20
                    elif recent_volume > avg_volume_20 * 1.2: score += 10
            
            if len(close) >= 14:
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().fillna(0)
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().fillna(0)
                
                latest_gain = gain.iloc[-1]
                latest_loss = loss.iloc[-1]

                if latest_loss == 0: rs = np.inf
                else: rs = latest_gain / latest_loss
                
                if np.isinf(rs) or pd.isna(rs): current_rsi = np.nan
                else: current_rsi = 100 - (100 / (1 + rs))
                
                if not pd.isna(current_rsi) and 30 < current_rsi < 70: score += 15
            
            if len(close) >= 60:
                three_month_return = (close.iloc[-1] / close.iloc[-60] - 1) * 100
                if not pd.isna(three_month_return):
                    if three_month_return > 10: score += 25
                    elif three_month_return > 5: score += 15
                    elif three_month_return > 0: score += 10
            
            if len(close.pct_change().dropna()) > 1:
                volatility = close.pct_change().std() * np.sqrt(252)
                if not pd.isna(volatility):
                    if volatility < 0.20: score += 15
                    elif volatility < 0.30: score += 10
            
            return score
        except Exception:
            return 0

    def calculate_composite_score(self, ticker, market_regime):
        """Calcola score composito finale basato su regime di mercato."""
        quality_score = self.calculate_quality_score(ticker)
        growth_score = self.calculate_growth_score(ticker)
        technical_score = self.calculate_technical_score(ticker)
        
        if market_regime == 'bull':
            composite = (quality_score * 0.20 + growth_score * 0.40 + technical_score * 0.40)
        elif market_regime == 'bear':
            composite = (quality_score * 0.50 + growth_score * 0.25 + technical_score * 0.25)
        else:
            composite = (quality_score * 0.35 + growth_score * 0.35 + technical_score * 0.30)
        
        return composite

    def get_best_buy_candidates(self, num_candidates=15):
        """
        Funzione principale per identificare i migliori candidati.
        Integra Fase 1 (Finviz) e Fase 2 (Analisi Approfondita) + Fase 3 (Selezione Finale).
        """
        print("🚀 Avvio Enhanced Best Buy Selector (v4.0)...")
        
        market_regime = self.get_market_regime()
        print(f"📊 Regime di mercato rilevato: {market_regime.upper()}")
        
        all_candidates_set = set()

        finviz_watchlist = self._get_watchlist_from_finviz(num_top_tickers_finviz=100)
        all_candidates_set.update(finviz_watchlist)

        for sector_tickers in self.sector_universe.values():
            all_candidates_set.update(sector_tickers)
        
        all_candidates_list = list(all_candidates_set)
        
        print(f"🎯 Trovati {len(all_candidates_list)} candidati unici per screening finale (Fase 2)...")
        
        candidate_scores = []
        for i, ticker in enumerate(all_candidates_list):
            try:
                if (i + 1) % 10 == 0 or i == len(all_candidates_list) - 1:
                    print(f"   Processing ticker {i+1}/{len(all_candidates_list)}: {ticker}...")
                
                score = self.calculate_composite_score(ticker, market_regime)
                if score > 40:
                    candidate_scores.append((ticker, score))
            except Exception as e:
                print(f"⚠️ Errore calcolo score per {ticker}: {str(e)}")
                continue
        
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        final_candidates_with_scores = candidate_scores[:num_candidates]
        final_candidates = [ticker for ticker, score in final_candidates_with_scores]

        print(f"\n🏆 TOP {len(final_candidates)} BEST BUY CANDIDATES:")
        if not final_candidates:
            print("  Nessun candidato ha soddisfatto i criteri finali di punteggio.")
        for i, (ticker, score) in enumerate(final_candidates_with_scores, 1):
            print(f"  {i:2d}. {ticker}: {score:.1f} points")
        
        return final_candidates

def main():
    """Funzione principale per GitHub Actions (o esecuzione locale)"""
    selector = EnhancedBestBuySelector()
    
    try:
        best_candidates = selector.get_best_buy_candidates(num_candidates=15)
        
        if not best_candidates:
            print("⚠️ Nessun candidato trovato o score troppo basso, uso fallback tickers...")
            best_candidates = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'JNJ', 'SPY', 'QQQ']
        
        output_dir = "data"
        output_file = "Best_buy.json"
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        
        with open(output_path, 'w') as f:
            json.dump(best_candidates, f, indent=4)
        
        print(f"\n✅ Salvati {len(best_candidates)} best buy candidates in {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Errore critico in main(): {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
