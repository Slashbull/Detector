# ============================================
# data_loader.py
# ============================================

import streamlit as st
import pandas as pd
from datetime import datetime, timezone
import logging
from typing import Dict, Any, Tuple
import time
import re
import gc

from config import CONFIG
from utils import RobustSessionState, PerformanceMonitor, DataValidator
from data_processor import DataProcessor
from ranking_engine import RankingEngine
from pattern_detector import PatternDetector
from advanced_metrics import AdvancedMetrics

logger = logging.getLogger(__name__)

@st.cache_data(persist="disk", show_spinner=False)
def load_and_process_data(source_type: str = "sheet", file_data=None, 
                         sheet_id: str = None, gid: str = None,
                         data_version: str = "1.0") -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    """Load and process data with smart caching and versioning"""
    
    start_time = time.perf_counter()
    metadata = {
        'source_type': source_type,
        'data_version': data_version,
        'processing_start': datetime.now(timezone.utc),
        'errors': [],
        'warnings': []
    }
    
    try:
        if source_type == "upload" and file_data is not None:
            logger.info("Loading data from uploaded CSV")
            df = pd.read_csv(file_data, low_memory=False)
            metadata['source'] = "User Upload"
        else:
            if not sheet_id:
                raise ValueError("Please enter a Google Sheets ID")
            
            sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_id)
            if sheet_id_match:
                sheet_id = sheet_id_match.group(1)
            
            if not gid:
                gid = CONFIG.DEFAULT_GID
            
            csv_url = CONFIG.CSV_URL_TEMPLATE.format(sheet_id=sheet_id, gid=gid)
            
            logger.info(f"Loading data from Google Sheets ID: {sheet_id}")
            
            try:
                df = pd.read_csv(csv_url, low_memory=False)
                metadata['source'] = "Google Sheets"
                metadata['sheet_id'] = sheet_id
            except Exception as e:
                logger.error(f"Failed to load from Google Sheets: {str(e)}")
                metadata['errors'].append(f"Sheet load error: {str(e)}")
                
                last_good_data = RobustSessionState.safe_get('last_good_data', None)
                if last_good_data:
                    logger.info("Using cached data as fallback")
                    df, timestamp, old_metadata = last_good_data
                    metadata['warnings'].append("Using cached data due to load failure")
                    metadata['cache_used'] = True
                    return df, timestamp, metadata
                raise
        
        is_valid, validation_msg = DataValidator.validate_dataframe(df, CONFIG.CRITICAL_COLUMNS, "Initial load")
        if not is_valid:
            raise ValueError(validation_msg)
        
        df = DataProcessor.process_dataframe(df, metadata)
        
        df = RankingEngine.calculate_all_scores(df)
        
        df = PatternDetector.detect_all_patterns_optimized(df)
        
        df = AdvancedMetrics.calculate_all_metrics(df)
        
        is_valid, validation_msg = DataValidator.validate_dataframe(df, ['master_score', 'rank'], "Final processed")
        if not is_valid:
            raise ValueError(validation_msg)
        
        timestamp = datetime.now(timezone.utc)
        RobustSessionState.safe_set('last_good_data', (df.copy(), timestamp, metadata))
        
        processing_time = time.perf_counter() - start_time
        metadata['processing_time'] = processing_time
        metadata['processing_end'] = datetime.now(timezone.utc)
        
        logger.info(f"Data processing complete: {len(df)} stocks in {processing_time:.2f}s")
        
        gc.collect()
        
        return df, timestamp, metadata
        
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}")
        metadata['errors'].append(str(e))
        raise
