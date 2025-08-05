# ============================================
# config.py
# ============================================

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class Config:
    """
    Centralized configuration for the Slashbull application.
    This dataclass holds all constants, weights, and thresholds.
    """

    # --- Data Source Configuration ---
    DEFAULT_SHEET_URL_TEMPLATE: str = "https://docs.google.com/spreadsheets/d/{sheet_id}/edit?usp=sharing"
    CSV_URL_TEMPLATE: str = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    DEFAULT_GID: str = "1823439984"

    # --- Performance and Caching ---
    CACHE_TTL: int = 3600  # 1 hour in seconds
    STALE_DATA_HOURS: int = 24
    
    # --- Ranking Weights (Master Score Calculation) ---
    POSITION_WEIGHT: float = 0.30
    VOLUME_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.15
    ACCELERATION_WEIGHT: float = 0.10
    BREAKOUT_WEIGHT: float = 0.10
    RVOL_WEIGHT: float = 0.10

    # --- UI & Display Settings ---
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])

    # --- Data Schema and Validation ---
    CRITICAL_COLUMNS: List[str] = field(default_factory=lambda: ['ticker', 'price', 'volume_1d'])
    
    PERCENTAGE_COLUMNS: List[str] = field(default_factory=lambda: [
        'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_3d', 'ret_7d', 
        'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'eps_change_pct'
    ])
    
    VOLUME_RATIO_COLUMNS: List[str] = field(default_factory=lambda: [
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d'
    ])
    
    # --- Pattern Detection Thresholds ---
    PATTERN_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "category_leader": 90, "hidden_gem": 80, "acceleration": 85,
        "institutional": 75, "vol_explosion": 95, "breakout_ready": 80,
        "market_leader": 95, "momentum_wave": 75, "liquid_leader": 80,
        "long_strength": 80, "52w_high_approach": 90, "52w_low_bounce": 85,
        "golden_zone": 85, "vol_accumulation": 80, "momentum_diverge": 90,
        "range_compress": 75, "stealth": 70, "vampire": 85,
        "perfect_storm": 80
    })
    
    # --- Data Tiering & Boundaries ---
    VALUE_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1_000_000), 'rvol': (0.01, 1_000_000.0),
        'pe': (-10000, 10000), 'returns': (-99.99, 9999.99),
        'volume': (0, 1e12)
    })
    
    TIERS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "eps": {"Loss": (-float('inf'), 0), "0-5": (0, 5), "5-10": (5, 10),
                "10-20": (10, 20), "20-50": (20, 50), "50-100": (50, 100),
                "100+": (100, float('inf'))},
        "pe": {"Negative/NA": (-float('inf'), 0), "0-10": (0, 10),
               "10-15": (10, 15), "15-20": (15, 20), "20-30": (20, 30),
               "30-50": (30, 50), "50+": (50, float('inf'))},
        "price": {"0-100": (0, 100), "100-250": (100, 250),
                  "250-500": (250, 500), "500-1000": (500, 1000),
                  "1000-2500": (1000, 2500), "2500-5000": (2500, 5000),
                  "5000+": (5000, float('inf'))}
    })

CONFIG = Config()
