# ============================================
# utils.py
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Tuple, Optional
from functools import wraps
import time
import gc

from config import CONFIG

logger = logging.getLogger(__name__)

class RobustSessionState:
    """Bulletproof session state management - prevents all KeyErrors."""
    
    # Complete list of ALL session state keys with their default values
    # This ensures no KeyErrors and provides predictable behavior.
    STATE_DEFAULTS = {
        # Core states
        'search_query': "",
        'last_refresh': None,
        'data_source': "sheet",
        'sheet_id': "1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM",
        'gid': "1823439984",
        'user_preferences': {
            'default_top_n': 50,
            'display_mode': 'Technical',
            'last_filters': {}
        },
        'filters': {},
        'active_filter_count': 0,
        'quick_filter': None,
        'quick_filter_applied': False,
        'show_debug': False,
        'performance_metrics': {},
        'data_quality': {},
        'trigger_clear': False,
        
        # All filter states with proper defaults
        'category_filter': [],
        'sector_filter': [],
        'industry_filter': [],
        'min_score': 0,
        'patterns': [],
        'trend_filter': "All Trends",
        'eps_tier_filter': [],
        'pe_tier_filter': [],
        'price_tier_filter': [],
        'min_eps_change': "",
        'min_pe': "",
        'max_pe': "",
        'require_fundamental_data': False,
        'wave_states_filter': [],
        'wave_strength_range_slider': (0, 100),
        'show_sensitivity_details': False,
        'show_market_regime': True,
        'wave_timeframe_select': "All Waves",
        'wave_sensitivity': "Balanced",
        'export_template_radio': "Full Analysis (All Data)",
        'display_mode_toggle': 0,
        
        # Data states
        'ranked_df': None,
        'data_timestamp': None,
        'last_good_data': None,
        
        # UI states
        'search_input': ""
    }
    
    @staticmethod
    def safe_get(key: str, default: Any = None) -> Any:
        """Safely get a session state value with a fallback to a default value."""
        if key not in st.session_state:
            st.session_state[key] = RobustSessionState.STATE_DEFAULTS.get(key, default)
        return st.session_state[key]
    
    @staticmethod
    def safe_set(key: str, value: Any) -> None:
        """Safely set a session state value."""
        st.session_state[key] = value
    
    @staticmethod
    def initialize():
        """Initialize all session state variables with their defaults if they don't exist."""
        for key, default_value in RobustSessionState.STATE_DEFAULTS.items():
            if key not in st.session_state:
                if key == 'last_refresh' and default_value is None:
                    st.session_state[key] = datetime.now(timezone.utc)
                else:
                    st.session_state[key] = default_value
    
    @staticmethod
    def clear_filters():
        """Clear all filter states back to their default values safely."""
        filter_keys = [
            'category_filter', 'sector_filter', 'industry_filter', 'eps_tier_filter',
            'pe_tier_filter', 'price_tier_filter', 'patterns',
            'min_score', 'trend_filter', 'min_eps_change',
            'min_pe', 'max_pe', 'require_fundamental_data',
            'quick_filter', 'quick_filter_applied',
            'wave_states_filter', 'wave_strength_range_slider',
            'show_sensitivity_details', 'show_market_regime',
            'wave_timeframe_select', 'wave_sensitivity'
        ]
        
        for key in filter_keys:
            if key in RobustSessionState.STATE_DEFAULTS:
                RobustSessionState.safe_set(key, RobustSessionState.STATE_DEFAULTS[key])
        
        RobustSessionState.safe_set('filters', {})
        RobustSessionState.safe_set('active_filter_count', 0)
        RobustSessionState.safe_set('trigger_clear', False)

class PerformanceMonitor:
    """Track and report performance metrics using a decorator."""
    
    @staticmethod
    def timer(target_time: Optional[float] = None):
        """A decorator to measure function execution time."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    
                    if target_time and elapsed > target_time:
                        logger.warning(f"{func.__name__} took {elapsed:.2f}s (target: {target_time}s)")
                    elif elapsed > 1.0:
                        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
                    
                    perf_metrics = RobustSessionState.safe_get('performance_metrics', {})
                    perf_metrics[func.__name__] = elapsed
                    RobustSessionState.safe_set('performance_metrics', perf_metrics)
                    
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
                    raise
            return wrapper
        return decorator

class DataValidator:
    """Comprehensive data validation and sanitization, essential for real-world data."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str], context: str) -> Tuple[bool, str]:
        """Validate dataframe structure and data quality upon load."""
        if df is None:
            return False, f"{context}: DataFrame is None"
        
        if df.empty:
            return False, f"{context}: DataFrame is empty"
        
        missing_critical = [col for col in CONFIG.CRITICAL_COLUMNS if col not in df.columns]
        if missing_critical:
            return False, f"{context}: Missing critical columns: {missing_critical}"
        
        duplicates = df['ticker'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"{context}: Found {duplicates} duplicate tickers")
        
        total_cells = len(df) * len(df.columns)
        filled_cells = df.notna().sum().sum()
        completeness = (filled_cells / total_cells * 100) if total_cells > 0 else 0
        
        if completeness < 50:
            logger.warning(f"{context}: Low data completeness ({completeness:.1f}%)")
        
        data_quality = RobustSessionState.safe_get('data_quality', {})
        data_quality.update({
            'completeness': completeness,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_tickers': duplicates,
            'context': context,
            'timestamp': datetime.now(timezone.utc)
        })
        RobustSessionState.safe_set('data_quality', data_quality)
        
        logger.info(f"{context}: Validated {len(df)} rows, {len(df.columns)} columns, {completeness:.1f}% complete")
        return True, "Valid"
    
    @staticmethod
    def clean_numeric_value(value: Any, is_percentage: bool = False, bounds: Optional[Tuple[float, float]] = None) -> Optional[float]:
        """Clean and convert numeric values from strings, with robust error handling and bounds checking."""
        if pd.isna(value) or value == '' or value is None:
            return np.nan
        
        try:
            cleaned = str(value).strip()
            
            if cleaned.upper() in ['', '-', 'N/A', 'NA', 'NAN', 'NONE', '#VALUE!', '#ERROR!', '#DIV/0!', 'INF', '-INF']:
                return np.nan
            
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            
            result = float(cleaned)
            
            if bounds:
                min_val, max_val = bounds
                if result < min_val or result > max_val:
                    logger.debug(f"Value {result} outside bounds [{min_val}, {max_val}]")
                    result = np.clip(result, min_val, max_val)
            
            if np.isnan(result) or np.isinf(result):
                return np.nan
            
            return result
            
        except (ValueError, TypeError, AttributeError):
            return np.nan
    
    @staticmethod
    def sanitize_string(value: Any, default: str = "Unknown") -> str:
        """Sanitize string values, replacing common empty/error representations with a default."""
        if pd.isna(value) or value is None:
            return default
        
        cleaned = str(value).strip()
        if cleaned.upper() in ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-']:
            return default
        
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
