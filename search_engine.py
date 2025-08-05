# ============================================
# search_engine.py
# ============================================

import pandas as pd
import logging
from typing import List, Tuple, Any

from utils import PerformanceMonitor

logger = logging.getLogger(__name__)

class SearchEngine:
    """Optimized search functionality with exact match prioritization"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks with exact match prioritization"""
        
        if not query or df.empty:
            return pd.DataFrame()
        
        try:
            query = query.upper().strip()
            
            results = df.copy()
            results['relevance'] = 0
            
            exact_ticker_mask = results['ticker'].str.upper() == query
            results.loc[exact_ticker_mask, 'relevance'] += 1000
            
            ticker_starts_mask = results['ticker'].str.upper().str.startswith(query)
            results.loc[ticker_starts_mask & ~exact_ticker_mask, 'relevance'] += 500
            
            ticker_contains_mask = results['ticker'].str.upper().str.contains(query, na=False, regex=False)
            results.loc[ticker_contains_mask & ~ticker_starts_mask, 'relevance'] += 200
            
            if 'company_name' in results.columns:
                company_exact_mask = results['company_name'].str.upper() == query
                results.loc[company_exact_mask, 'relevance'] += 800
                
                company_starts_mask = results['company_name'].str.upper().str.startswith(query)
                results.loc[company_starts_mask & ~company_exact_mask, 'relevance'] += 300
                
                company_contains_mask = results['company_name'].str.upper().str.contains(query, na=False, regex=False)
                results.loc[company_contains_mask & ~company_starts_mask, 'relevance'] += 100
                
                def word_match_score(company_name):
                    if pd.isna(company_name):
                        return 0
                    words = str(company_name).upper().split()
                    for word in words:
                        if word.startswith(query):
                            return 50
                    return 0
                
                word_scores = results['company_name'].apply(word_match_score)
                results['relevance'] += word_scores
            
            matches = results[results['relevance'] > 0].copy()
            
            if matches.empty:
                return pd.DataFrame()
            
            matches = matches.sort_values(['relevance', 'master_score'], ascending=[False, False])
            
            return matches.drop('relevance', axis=1)
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return pd.DataFrame()
