import sqlite3
import os
from datetime import datetime
import logging

class AITradeRecorder:
    def __init__(self, db_path="data_backtest/ai_learning/performance.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Assicura che il database e la directory esistano"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Inizializza database SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT,
                        entry_date TEXT,
                        exit_date TEXT,
                        entry_price REAL,
                        exit_price REAL,
                        quantity INTEGER,
                        profit REAL,
                        profit_pct REAL,
                        hold_days INTEGER,
                        market_regime TEXT,
                        volatility REAL,
                        rsi_at_entry REAL,
                        macd_at_entry REAL,
                        adx_at_entry REAL,
                        volume_ratio REAL,
                        entropy REAL,
                        determinism REAL,
                        signal_quality REAL,
                        trend_strength REAL,
                        noise_ratio REAL,
                        prediction_confidence REAL,
                        actual_outcome REAL,
                        ai_decision_score REAL,
                        exit_reason TEXT,
                        signal_method TEXT,
                        ref_score_or_roi REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        unique_trade_id TEXT UNIQUE
                    )
                ''')
                self.logger.info("✅ Database AI inizializzato correttamente")
                    
        except sqlite3.Error as e:
            self.logger.error(f"Errore inizializzazione database: {e}")
    
    def record_trade_from_backtest(self, ticker, entry_date, exit_date, entry_price, exit_price, quantity, profit_pct, exit_reason):
        """Registra un trade chiuso dal backtest"""
        try:
            # Genera ID unico
            unique_id = f"{ticker}_{entry_date}_{exit_date}"
            
            # Calcola i valori
            profit = (exit_price - entry_price) * quantity
            hold_days = (datetime.strptime(exit_date, '%Y-%m-%d') - datetime.strptime(entry_date, '%Y-%m-%d')).days
            
            # Determina l'outcome per l'AI
            if profit_pct > 5:
                actual_outcome = 1.0  # Successo
            elif profit_pct < -3:
                actual_outcome = 0.0  # Fallimento
            else:
                actual_outcome = 0.5  # Neutro
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO trades (
                        unique_trade_id, ticker, entry_date, exit_date, entry_price, exit_price,
                        quantity, profit, profit_pct, hold_days, market_regime, volatility,
                        rsi_at_entry, macd_at_entry, adx_at_entry, volume_ratio,
                        entropy, determinism, signal_quality, trend_strength, noise_ratio,
                        prediction_confidence, actual_outcome, ai_decision_score, exit_reason,
                        signal_method, ref_score_or_roi, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    unique_id, ticker, entry_date, exit_date, entry_price, exit_price,
                    quantity, profit, profit_pct, hold_days, 'unknown', 0.2,
                    50, 0, 30, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                    actual_outcome, 0.5, exit_reason, 'ensemble', 0, datetime.now().isoformat()
                ))
                
                self.logger.info(f"✅ Trade registrato nel DB AI: {ticker} - P/L: {profit_pct:.2f}%")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Errore registrazione trade: {e}")
            return False
    
    def get_trade_count(self):
        """Conta i trade nel database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM trades")
                return cursor.fetchone()[0]
        except:
            return 0
