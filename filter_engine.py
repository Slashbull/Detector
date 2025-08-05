# ============================================
# filter_engine.py
# ============================================

import pandas as pd
import logging
from typing import Dict, List, Any
import re

from utils import PerformanceMonitor

logger = logging.getLogger(__name__)

class FilterEngine:
    """Handle all filtering operations efficiently with a robust approach"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.2)
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply all filters with optimized performance"""
        
        if df.empty:
            return df
        
        mask = pd.Series(True, index=df.index)
        
        categories = filters.get('categories', [])
        if categories and 'All' not in categories:
            mask &= df['category'].isin(categories)
        
        sectors = filters.get('sectors', [])
        if sectors and 'All' not in sectors:
            mask &= df['sector'].isin(sectors)
        
        industries = filters.get('industries', [])
        if industries and 'All' not in industries and 'industry' in df.columns:
            mask &= df['industry'].isin(industries)
        
        min_score = filters.get('min_score', 0)
        if min_score > 0:
            mask &= df['master_score'] >= min_score
        
        min_eps_change = filters.get('min_eps_change')
        if min_eps_change is not None and 'eps_change_pct' in df.columns:
            mask &= (df['eps_change_pct'] >= min_eps_change) | df['eps_change_pct'].isna()
        
        patterns = filters.get('patterns', [])
        if patterns and 'patterns' in df.columns:
            pattern_regex = '|'.join([re.escape(p) for p in patterns])
            mask &= df['patterns'].str.contains(pattern_regex, case=False, na=False, regex=True)
        
        if filters.get('trend_range') and filters.get('trend_filter') != 'All Trends':
            min_trend, max_trend = filters['trend_range']
            if 'trend_quality' in df.columns:
                mask &= (df['trend_quality'] >= min_trend) & (df['trend_quality'] <= max_trend)
        
        min_pe = filters.get('min_pe')
        if min_pe is not None and 'pe' in df.columns:
            mask &= df['pe'].isna() | ((df['pe'] > 0) & (df['pe'] >= min_pe))
        
        max_pe = filters.get('max_pe')
        if max_pe is not None and 'pe' in df.columns:
            mask &= df['pe'].isna() | ((df['pe'] > 0) & (df['pe'] <= max_pe))
        
        for tier_type in ['eps_tiers', 'pe_tiers', 'price_tiers']:
            tier_values = filters.get(tier_type, [])
            if tier_values and 'All' not in tier_values:
                col_name = tier_type.replace('_tiers', '_tier')
                if col_name in df.columns:
                    mask &= df[col_name].isin(tier_values)
        
        if filters.get('require_fundamental_data', False):
            if 'pe' in df.columns and 'eps_change_pct' in df.columns:
                mask &= df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna()
        
        wave_states = filters.get('wave_states', [])
        if wave_states and 'All' not in wave_states and 'wave_state' in df.columns:
            mask &= df['wave_state'].isin(wave_states)

        wave_strength_range = filters.get('wave_strength_range')
        if wave_strength_range and 'overall_wave_strength' in df.columns:
            min_ws, max_ws = wave_strength_range
            mask &= (df['overall_wave_strength'] >= min_ws) & (df['overall_wave_strength'] <= max_ws)

        filtered_df = df[mask].copy()
        
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks")
        
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Dict[str, Any]) -> List[str]:
        """Get available filter options with a robust approach"""
        
        if df.empty or column not in df.columns:
            return []
        
        temp_df = df.copy()
        
        if 'categories' in current_filters and 'category' in temp_df.columns:
            if current_filters['categories']:
                temp_df = temp_df[temp_df['category'].isin(current_filters['categories'])]
        
        if 'sectors' in current_filters and 'sector' in temp_df.columns:
            if current_filters['sectors'] and column != 'sectors':
                temp_df = temp_df[temp_df['sector'].isin(current_filters['sectors'])]
        
        if 'industries' in current_filters and 'industry' in temp_df.columns:
            if current_filters['industries'] and column != 'industries':
                temp_df = temp_df[temp_df['industry'].isin(current_filters['industries'])]
        
        values = temp_df[column].dropna().unique()
        
        values = [v for v in values if str(v).strip() not in ['Unknown', 'unknown', '', 'nan', 'NaN', 'None']]
        
        try:
            values = sorted(values, key=lambda x: float(str(x).replace(',', '').replace('.', '').replace('-', '').replace('x', '').replace('+', '').replace('%', '')) if str(x).replace(',', '').replace('.', '').replace('-', '').replace('x', '').replace('+', '').replace('%', '').isdigit() else x)
        except:
            values = sorted(values, key=str)
        
        return values
