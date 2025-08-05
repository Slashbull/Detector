# ============================================
# pattern_detector.py
# ============================================

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

from config import CONFIG
from utils import PerformanceMonitor

logger = logging.getLogger(__name__)

class PatternDetector:
    """Detect all patterns using fully vectorized operations"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.3)
    def detect_all_patterns_optimized(df: pd.DataFrame) -> pd.DataFrame:
        """Detect all 25 patterns with fully vectorized numpy operations - O(n) complexity"""
        
        if df.empty:
            df['patterns'] = ''
            return df
        
        pattern_results = {}
        
        if 'category_percentile' in df.columns:
            pattern_results['🔥 CAT LEADER'] = df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['category_leader']
        
        if 'category_percentile' in df.columns and 'percentile' in df.columns:
            pattern_results['💎 HIDDEN GEM'] = (
                (df['category_percentile'] >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & 
                (df['percentile'] < 70)
            )
        
        if 'acceleration_score' in df.columns:
            pattern_results['🚀 ACCELERATING'] = df['acceleration_score'] >= CONFIG.PATTERN_THRESHOLDS['acceleration']
        
        if 'volume_score' in df.columns and 'vol_ratio_90d_180d' in df.columns:
            pattern_results['🏦 INSTITUTIONAL'] = (
                (df['volume_score'] >= CONFIG.PATTERN_THRESHOLDS['institutional']) &
                (df['vol_ratio_90d_180d'] > 1.1)
            )
        
        if 'rvol' in df.columns:
            pattern_results['⚡ VOL EXPLOSION'] = df['rvol'] > 3
        
        if 'breakout_score' in df.columns:
            pattern_results['🎯 BREAKOUT'] = df['breakout_score'] >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
        
        if 'percentile' in df.columns:
            pattern_results['👑 MARKET LEADER'] = df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['market_leader']
        
        if 'momentum_score' in df.columns and 'acceleration_score' in df.columns:
            pattern_results['🌊 MOMENTUM WAVE'] = (
                (df['momentum_score'] >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) &
                (df['acceleration_score'] >= 70)
            )
        
        if 'liquidity_score' in df.columns and 'percentile' in df.columns:
            pattern_results['💰 LIQUID LEADER'] = (
                (df['liquidity_score'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) &
                (df['percentile'] >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
            )
        
        if 'long_term_strength' in df.columns:
            pattern_results['💪 LONG STRENGTH'] = df['long_term_strength'] >= CONFIG.PATTERN_THRESHOLDS['long_strength']
        
        if 'trend_quality' in df.columns:
            pattern_results['📈 QUALITY TREND'] = df['trend_quality'] >= 80
        
        if 'pe' in df.columns and 'master_score' in df.columns:
            has_valid_pe = (df['pe'].notna() & (df['pe'] > 0) & (df['pe'] < 10000))
            pattern_results['💎 VALUE MOMENTUM'] = has_valid_pe & (df['pe'] < 15) & (df['master_score'] >= 70)
        
        if 'eps_change_pct' in df.columns and 'acceleration_score' in df.columns:
            has_eps_growth = df['eps_change_pct'].notna()
            extreme_growth = has_eps_growth & (df['eps_change_pct'] > 1000)
            normal_growth = has_eps_growth & (df['eps_change_pct'] > 50) & (df['eps_change_pct'] <= 1000)
            
            pattern_results['📊 EARNINGS ROCKET'] = (
                (extreme_growth & (df['acceleration_score'] >= 80)) |
                (normal_growth & (df['acceleration_score'] >= 70))
            )
        
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            has_complete_data = (
                df['pe'].notna() & 
                df['eps_change_pct'].notna() & 
                (df['pe'] > 0) &
                (df['pe'] < 10000)
            )
            pattern_results['🏆 QUALITY LEADER'] = (
                has_complete_data &
                (df['pe'].between(10, 25)) &
                (df['eps_change_pct'] > 20) &
                (df['percentile'] >= 80)
            )
        
        if 'eps_change_pct' in df.columns and 'volume_score' in df.columns:
            has_eps = df['eps_change_pct'].notna()
            mega_turnaround = has_eps & (df['eps_change_pct'] > 500) & (df['volume_score'] >= 60)
            strong_turnaround = has_eps & (df['eps_change_pct'] > 100) & (df['eps_change_pct'] <= 500) & (df['volume_score'] >= 70)
            
            pattern_results['⚡ TURNAROUND'] = mega_turnaround | strong_turnaround
        
        if 'pe' in df.columns:
            has_valid_pe = df['pe'].notna() & (df['pe'] > 0)
            pattern_results['⚠️ HIGH PE'] = has_valid_pe & (df['pe'] > 100)
        
        if 'from_high_pct' in df.columns and 'volume_score' in df.columns and 'momentum_score' in df.columns:
            pattern_results['🎯 52W HIGH APPROACH'] = (
                (df['from_high_pct'] > -5) & 
                (df['volume_score'] >= 70) & 
                (df['momentum_score'] >= 60)
            )
        
        if all(col in df.columns for col in ['from_low_pct', 'acceleration_score', 'ret_30d']):
            pattern_results['🔄 52W LOW BOUNCE'] = (
                (df['from_low_pct'] < 20) & 
                (df['acceleration_score'] >= 80) & 
                (df['ret_30d'] > 10)
            )
        
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct', 'trend_quality']):
            pattern_results['👑 GOLDEN ZONE'] = (
                (df['from_low_pct'] > 60) & 
                (df['from_high_pct'] > -40) & 
                (df['trend_quality'] >= 70)
            )
        
        if all(col in df.columns for col in ['vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d']):
            pattern_results['📊 VOL ACCUMULATION'] = (
                (df['vol_ratio_30d_90d'] > 1.2) & 
                (df['vol_ratio_90d_180d'] > 1.1) & 
                (df['ret_30d'] > 5)
            )
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            ret_7d_arr = df['ret_7d'].fillna(0).values
            ret_30d_arr = df['ret_30d'].fillna(0).values
            
            daily_7d_pace = np.where(ret_7d_arr != 0, ret_7d_arr / 7, 0)
            daily_30d_pace = np.where(ret_30d_arr != 0, ret_30d_arr / 30, 0)
            
            pattern_results['🔀 MOMENTUM DIVERGE'] = (
                (daily_7d_pace > daily_30d_pace * 1.5) & 
                (df['acceleration_score'] >= 85) & 
                (df['rvol'] > 2)
            )
        
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'from_low_pct']):
            high_arr = df['high_52w'].fillna(0).values
            low_arr = df['low_52w'].fillna(0).values
            
            range_pct = np.where(
                low_arr > 0,
                ((high_arr - low_arr) / low_arr) * 100,
                100
            )
            
            pattern_results['🎯 RANGE COMPRESS'] = (range_pct < 50) & (df['from_low_pct'] > 30)
        
        if all(col in df.columns for col in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']):
            ret_7d_arr = df['ret_7d'].fillna(0).values
            ret_30d_arr = df['ret_30d'].fillna(0).values
            
            ret_ratio = np.where(ret_30d_arr != 0, ret_7d_arr / (ret_30d_arr / 4), 0)
            
            pattern_results['🤫 STEALTH'] = (
                (df['vol_ratio_90d_180d'] > 1.1) &
                (df['vol_ratio_30d_90d'].between(0.9, 1.1)) &
                (df['from_low_pct'] > 40) &
                (ret_ratio > 1)
            )
        
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']):
            ret_1d_arr = df['ret_1d'].fillna(0).values
            ret_7d_arr = df['ret_7d'].fillna(0).values
            
            daily_pace_ratio = np.where(ret_7d_arr != 0, ret_1d_arr / (ret_7d_arr / 7), 0)
            
            pattern_results['🧛 VAMPIRE'] = (
                (daily_pace_ratio > 2) &
                (df['rvol'] > 3) &
                (df['from_high_pct'] > -15) &
                (df['category'].isin(['Small Cap', 'Micro Cap']))
            )
        
        if 'momentum_harmony' in df.columns and 'master_score' in df.columns:
            pattern_results['⛈️ PERFECT STORM'] = (
                (df['momentum_harmony'] == 4) &
                (df['master_score'] > 80)
            )
        
        pattern_names = list(pattern_results.keys())
        if pattern_names:
            pattern_matrix = np.column_stack([pattern_results[name].values for name in pattern_names])
            
            df['patterns'] = [
                ' | '.join([pattern_names[i] for i, val in enumerate(row) if val])
                for row in pattern_matrix
            ]
        else:
            df['patterns'] = ''
        
        df['patterns'] = df['patterns'].fillna('')
        
        return df
