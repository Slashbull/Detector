# ============================================
# market_intelligence.py
# ============================================

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

from config import CONFIG

class MarketIntelligence:
    """Advanced market analysis and regime detection"""
    
    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Detect current market regime with supporting data"""
        
        if df.empty:
            return "😴 NO DATA", {}
        
        metrics = {}
        
        if 'category' in df.columns and 'master_score' in df.columns:
            category_scores = df.groupby('category')['master_score'].mean()
            
            micro_small_avg = category_scores[category_scores.index.isin(['Micro Cap', 'Small Cap'])].mean()
            large_mega_avg = category_scores[category_scores.index.isin(['Large Cap', 'Mega Cap'])].mean()
            
            metrics['micro_small_avg'] = micro_small_avg
            metrics['large_mega_avg'] = large_mega_avg
            metrics['category_spread'] = micro_small_avg - large_mega_avg
        else:
            micro_small_avg = 50
            large_mega_avg = 50
        
        if 'ret_30d' in df.columns:
            breadth = len(df[df['ret_30d'] > 0]) / len(df)
            metrics['breadth'] = breadth
        else:
            breadth = 0.5
        
        if 'rvol' in df.columns:
            avg_rvol = df['rvol'].median()
            metrics['avg_rvol'] = avg_rvol
        else:
            avg_rvol = 1.0
        
        if micro_small_avg > large_mega_avg + 10 and breadth > 0.6:
            regime = "🔥 RISK-ON BULL"
        elif large_mega_avg > micro_small_avg + 10 and breadth < 0.4:
            regime = "🛡️ RISK-OFF DEFENSIVE"
        elif avg_rvol > 1.5 and breadth > 0.5:
            regime = "⚡ VOLATILE OPPORTUNITY"
        else:
            regime = "😴 RANGE-BOUND"
        
        metrics['regime'] = regime
        
        return regime, metrics
    
    @staticmethod
    def calculate_advance_decline_ratio(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advance/decline ratio and related metrics"""
        
        ad_metrics = {}
        
        if 'ret_1d' in df.columns:
            advancing = len(df[df['ret_1d'] > 0])
            declining = len(df[df['ret_1d'] < 0])
            unchanged = len(df[df['ret_1d'] == 0])
            
            ad_metrics['advancing'] = advancing
            ad_metrics['declining'] = declining
            ad_metrics['unchanged'] = unchanged
            
            if declining > 0:
                ad_metrics['ad_ratio'] = advancing / declining
            else:
                ad_metrics['ad_ratio'] = float('inf') if advancing > 0 else 1.0
            
            ad_metrics['ad_line'] = advancing - declining
            ad_metrics['breadth_pct'] = (advancing / len(df)) * 100 if len(df) > 0 else 0
        
        return ad_metrics
    
    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        """Detect sector rotation patterns with smart normalized analysis"""
        
        if 'sector' not in df.columns or df.empty:
            return pd.DataFrame()
        
        sector_dfs = []
        
        for sector in df['sector'].unique():
            if sector != 'Unknown' and pd.notna(sector):
                sector_df = df[df['sector'] == sector].copy()
                sector_size = len(sector_df)
                
                if sector_size == 1:
                    sample_count = 1
                elif 2 <= sector_size <= 5:
                    sample_count = sector_size
                elif 6 <= sector_size <= 10:
                    sample_count = max(3, int(sector_size * 0.80))
                elif 11 <= sector_size <= 25:
                    sample_count = max(5, int(sector_size * 0.60))
                elif 26 <= sector_size <= 50:
                    sample_count = max(10, int(sector_size * 0.50))
                elif 51 <= sector_size <= 100:
                    sample_count = max(20, int(sector_size * 0.40))
                elif 101 <= sector_size <= 200:
                    sample_count = max(30, int(sector_size * 0.30))
                else:
                    sample_count = min(60, int(sector_size * 0.25))
                
                if sample_count > 0:
                    sector_df = sector_df.nlargest(sample_count, 'master_score')
                else:
                    sector_df = pd.DataFrame()
                
                if not sector_df.empty:
                    sector_dfs.append(sector_df)
        
        if sector_dfs:
            normalized_df = pd.concat(sector_dfs, ignore_index=True)
        else:
            return pd.DataFrame()
        
        sector_metrics = normalized_df.groupby('sector').agg({
            'master_score': ['mean', 'median', 'std', 'count'],
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'rvol': 'mean',
            'ret_30d': 'mean',
            'money_flow_mm': 'sum' if 'money_flow_mm' in normalized_df.columns else lambda x: 0
        }).round(2)
        
        if 'money_flow_mm' in normalized_df.columns:
            sector_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 
                                     'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'total_money_flow']
        else:
            sector_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 
                                     'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'dummy_money_flow']
            sector_metrics = sector_metrics.drop('dummy_money_flow', axis=1)
        
        original_counts = df.groupby('sector').size().rename('total_stocks')
        sector_metrics = sector_metrics.join(original_counts, how='left')
        sector_metrics['analyzed_stocks'] = sector_metrics['count']
        
        sector_metrics['flow_score'] = (
            sector_metrics['avg_score'] * 0.3 +
            sector_metrics['median_score'] * 0.2 +
            sector_metrics['avg_momentum'] * 0.25 +
            sector_metrics['avg_volume'] * 0.25
        )
        
        sector_metrics['rank'] = sector_metrics['flow_score'].rank(ascending=False)
        
        sector_metrics['sampling_pct'] = (
            (sector_metrics['analyzed_stocks'] / sector_metrics['total_stocks'] * 100)
            .round(1)
        )
        
        return sector_metrics.sort_values('flow_score', ascending=False)
    
    @staticmethod
    def detect_industry_rotation(df: pd.DataFrame) -> pd.DataFrame:
        """Detect industry rotation patterns with smart normalized analysis"""
        
        if 'industry' not in df.columns or df.empty:
            return pd.DataFrame()
        
        industry_dfs = []
        
        for industry in df['industry'].unique():
            if industry != 'Unknown' and pd.notna(industry):
                industry_df = df[df['industry'] == industry].copy()
                industry_size = len(industry_df)
                
                if industry_size == 1:
                    sample_count = 1
                elif 2 <= industry_size <= 5:
                    sample_count = industry_size
                elif 6 <= industry_size <= 10:
                    sample_count = max(3, int(industry_size * 0.80))
                elif 11 <= industry_size <= 25:
                    sample_count = max(5, int(industry_size * 0.60))
                elif 26 <= industry_size <= 50:
                    sample_count = max(10, int(industry_size * 0.40))
                elif 51 <= industry_size <= 100:
                    sample_count = max(15, int(industry_size * 0.30))
                elif 101 <= industry_size <= 250:
                    sample_count = max(25, int(industry_size * 0.20))
                elif 251 <= industry_size <= 550:
                    sample_count = max(40, int(industry_size * 0.15))
                else:
                    sample_count = min(75, int(industry_size * 0.10))
                
                if sample_count > 0:
                    industry_df = industry_df.nlargest(sample_count, 'master_score')
                else:
                    industry_df = pd.DataFrame()
                
                if not industry_df.empty:
                    industry_dfs.append(industry_df)
        
        if industry_dfs:
            normalized_df = pd.concat(industry_dfs, ignore_index=True)
        else:
            return pd.DataFrame()
        
        industry_metrics = normalized_df.groupby('industry').agg({
            'master_score': ['mean', 'median', 'std', 'count'],
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'rvol': 'mean',
            'ret_30d': 'mean',
            'money_flow_mm': 'sum' if 'money_flow_mm' in normalized_df.columns else lambda x: 0
        }).round(2)
        
        if 'money_flow_mm' in normalized_df.columns:
            industry_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 
                                       'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'total_money_flow']
        else:
            industry_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 
                                       'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'dummy_money_flow']
            industry_metrics = industry_metrics.drop('dummy_money_flow', axis=1)
        
        original_counts = df.groupby('industry').size().rename('total_stocks')
        industry_metrics = industry_metrics.join(original_counts, how='left')
        industry_metrics['analyzed_stocks'] = industry_metrics['count']
        
        industry_metrics['flow_score'] = (
            industry_metrics['avg_score'] * 0.3 +
            industry_metrics['median_score'] * 0.2 +
            industry_metrics['avg_momentum'] * 0.25 +
            industry_metrics['avg_volume'] * 0.25
        )
        
        industry_metrics['rank'] = industry_metrics['flow_score'].rank(ascending=False)
        
        industry_metrics['sampling_pct'] = (
            (industry_metrics['analyzed_stocks'] / industry_metrics['total_stocks'] * 100)
            .round(1)
        )
        
        return industry_metrics.sort_values('flow_score', ascending=False)
    
    @staticmethod
    def detect_category_performance(df: pd.DataFrame) -> pd.DataFrame:
        """Detect category performance patterns with smart normalized analysis"""
        
        if 'category' not in df.columns or df.empty:
            return pd.DataFrame()
        
        category_dfs = []
        
        for category in df['category'].unique():
            if category != 'Unknown' and pd.notna(category):
                category_df = df[df['category'] == category].copy()
                category_size = len(category_df)
                
                if category_size == 1:
                    sample_count = 1
                elif 2 <= category_size <= 10:
                    sample_count = category_size
                elif 11 <= category_size <= 50:
                    sample_count = max(5, int(category_size * 0.60))
                elif 51 <= category_size <= 100:
                    sample_count = max(20, int(category_size * 0.40))
                elif 101 <= category_size <= 200:
                    sample_count = max(30, int(category_size * 0.30))
                else:
                    sample_count = min(50, int(category_size * 0.25))
                
                if sample_count > 0:
                    category_df = category_df.nlargest(sample_count, 'master_score')
                else:
                    category_df = pd.DataFrame()
                
                if not category_df.empty:
                    category_dfs.append(category_df)
        
        if category_dfs:
            normalized_df = pd.concat(category_dfs, ignore_index=True)
        else:
            return pd.DataFrame()
        
        category_metrics = normalized_df.groupby('category').agg({
            'master_score': ['mean', 'median', 'std', 'count'],
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'rvol': 'mean',
            'ret_30d': 'mean',
            'acceleration_score': 'mean',
            'breakout_score': 'mean',
            'money_flow_mm': 'sum' if 'money_flow_mm' in normalized_df.columns else lambda x: 0
        }).round(2)
        
        if 'money_flow_mm' in normalized_df.columns:
            category_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 
                                       'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d',
                                       'avg_acceleration', 'avg_breakout', 'total_money_flow']
        else:
            category_metrics.columns = ['avg_score', 'median_score', 'std_score', 'count', 
                                       'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d',
                                       'avg_acceleration', 'avg_breakout', 'dummy_money_flow']
            category_metrics = category_metrics.drop('dummy_money_flow', axis=1)
        
        original_counts = df.groupby('category').size().rename('total_stocks')
        category_metrics = category_metrics.join(original_counts, how='left')
        category_metrics['analyzed_stocks'] = category_metrics['count']
        
        category_metrics['flow_score'] = (
            category_metrics['avg_score'] * 0.35 +
            category_metrics['median_score'] * 0.20 +
            category_metrics['avg_momentum'] * 0.20 +
            category_metrics['avg_acceleration'] * 0.15 +
            category_metrics['avg_volume'] * 0.10
        )
        
        category_metrics['rank'] = category_metrics['flow_score'].rank(ascending=False)
        
        category_metrics['sampling_pct'] = (
            (category_metrics['analyzed_stocks'] / category_metrics['total_stocks'] * 100)
            .round(1)
        )
        
        category_order = ['Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap']
        category_metrics = category_metrics.reindex(
            [cat for cat in category_order if cat in category_metrics.index]
        )
        
        return category_metrics
