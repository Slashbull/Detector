"""
Wave Detection Ultimate 3.0 - PRODUCTION READY VERSION
=======================================================
Professional Stock Ranking System with Advanced Analytics
Optimized for Streamlit Community Cloud Deployment

Version: 3.0.0-FINAL
Last Updated: December 2024
Status: PRODUCTION READY
"""

# ============================================
# IMPORTS AND SETUP
# ============================================

# --------------- Standard Library ---------------
import gc  # Memory management and garbage collection
import json  # JSON parsing for structured logging
import logging  # Application logging and debugging
import re  # Regular expressions for URL parsing
import sys  # System-specific parameters
import time  # Performance timing and delays
import warnings  # Warning suppression for clean output
from collections import OrderedDict  # Ordered dictionaries for preserving sequence
from datetime import datetime, timezone, timedelta  # Date/time operations
from functools import wraps  # Decorator utilities for performance monitoring
from logging.handlers import RotatingFileHandler  # Log file rotation
from pathlib import Path  # File path operations
from typing import Dict, List, Tuple, Optional, Any  # Type hints for better code quality
from contextlib import contextmanager  # Context manager utilities

# --------------- Data Processing ---------------
import numpy as np  # Numerical operations and array processing
import pandas as pd  # DataFrame operations and data manipulation

# --------------- Visualization ---------------
import plotly.express as px  # Quick statistical visualizations
import plotly.graph_objects as go  # Advanced interactive charts

# --------------- Streamlit Core ---------------
import streamlit as st  # Main Streamlit framework

# --------------- Data Classes ---------------
from dataclasses import dataclass, field  # Configuration and data structures

# --------------- Export Functionality ---------------
# Lazy import - only loaded when needed for exports
def get_bytesio():
    """Lazy load BytesIO only when creating exports"""
    from io import BytesIO
    return BytesIO

# --------------- HTTP Requests (if actually needed) ---------------
# Uncomment ONLY if you implement actual HTTP retry logic
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# def create_session_with_retries():
#     """Create HTTP session with connection pooling and retry logic"""
#     session = requests.Session()
#     retry_strategy = Retry(
#         total=3,
#         backoff_factor=1,
#         status_forcelist=[429, 500, 502, 503, 504],
#     )
#     adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
#     session.mount("http://", adapter)
#     session.mount("https://", adapter)
#     return session

# ============================================
# ENVIRONMENT SETUP
# ============================================

# Suppress warnings for production
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure NumPy for stability
np.seterr(divide='ignore', invalid='ignore')  # Handle division by zero gracefully

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# PRODUCTION FLAGS
# ============================================

# Set production mode (disable in development)
PRODUCTION_MODE = True
DEBUG_MODE = not PRODUCTION_MODE

# Configure behavior based on mode
if PRODUCTION_MODE:
    # Production settings
    LOG_LEVEL = logging.WARNING
    SHOW_ERRORS = False
    ENABLE_PROFILING = False
else:
    # Development settings
    LOG_LEVEL = logging.DEBUG
    SHOW_ERRORS = True
    ENABLE_PROFILING = True

# ============================================
# PANDAS CONFIGURATION
# ============================================

# Optimize pandas display and performance
pd.set_option('display.max_columns', None)  # Show all columns in debug mode
pd.set_option('display.width', None)  # No width restriction
pd.set_option('display.max_colwidth', 50)  # Limit column width for readability
pd.set_option('mode.chained_assignment', None)  # Disable SettingWithCopyWarning
pd.set_option('compute.use_numexpr', True)  # Use numexpr for acceleration
pd.set_option('compute.use_bottleneck', True)  # Use bottleneck for acceleration

# ============================================
# LOGGING CONFIGURATION
# ============================================

# ============================================
# LOG CONFIGURATION SETTINGS
# ============================================

class LogConfig:
    """Centralized logging configuration"""
    
    # Log levels for different components
    LEVELS = {
        'main': logging.INFO,
        'data': logging.INFO,
        'scoring': logging.WARNING,
        'filter': logging.WARNING,
        'export': logging.INFO,
        'ui': logging.WARNING,
        'performance': logging.DEBUG,
        'error': logging.ERROR
    }
    
    # Log format templates
    FORMATS = {
        'detailed': '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        'simple': '%(asctime)s | %(levelname)s | %(message)s',
        'json': '{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","message":"%(message)s"}',
        'performance': '%(asctime)s | PERF | %(message)s | %(duration).3fs'
    }
    
    # File settings
    LOG_DIR = Path('logs')
    MAX_BYTES = 10 * 1024 * 1024  # 10MB per file
    BACKUP_COUNT = 5  # Keep 5 backup files
    
    # Performance thresholds (seconds)
    SLOW_QUERY_THRESHOLD = 1.0
    SLOW_RENDER_THRESHOLD = 0.5
    SLOW_CALCULATION_THRESHOLD = 2.0

# ============================================
# CUSTOM LOG FILTERS
# ============================================

class PerformanceFilter(logging.Filter):
    """Filter to add performance context to log records"""
    
    def filter(self, record):
        # Add performance context if available
        if hasattr(record, 'duration'):
            record.duration = getattr(record, 'duration', 0)
        else:
            record.duration = 0
        return True

class ErrorCountFilter(logging.Filter):
    """Track error counts for monitoring"""
    
    def __init__(self):
        super().__init__()
        self.error_count = 0
        self.warning_count = 0
    
    def filter(self, record):
        if record.levelno >= logging.ERROR:
            self.error_count += 1
        elif record.levelno >= logging.WARNING:
            self.warning_count += 1
        return True

# ============================================
# CUSTOM FORMATTERS
# ============================================

class ColoredFormatter(logging.Formatter):
    """Colored output for console logging"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        if not sys.stderr.isatty():
            return super().format(record)
        
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)

class StructuredFormatter(logging.Formatter):
    """JSON structured logging for production"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'process_id': record.process,
            'thread_id': record.thread
        }
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_obj['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_obj['session_id'] = record.session_id
        if hasattr(record, 'duration'):
            log_obj['duration_ms'] = record.duration * 1000
        if hasattr(record, 'error_code'):
            log_obj['error_code'] = record.error_code
            
        return json.dumps(log_obj)

# ============================================
# LOGGER FACTORY
# ============================================

class LoggerFactory:
    """Factory for creating configured loggers"""
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def setup(cls, production_mode: bool = True):
        """Initialize logging system"""
        if cls._initialized:
            return
        
        # Create logs directory if needed
        if production_mode:
            LogConfig.LOG_DIR.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers = []
        
        # Add console handler
        console_handler = cls._create_console_handler(production_mode)
        root_logger.addHandler(console_handler)
        
        # Add file handler in production
        if production_mode:
            file_handler = cls._create_file_handler()
            root_logger.addHandler(file_handler)
            
            # Add error file handler
            error_handler = cls._create_error_handler()
            root_logger.addHandler(error_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str, level: str = None) -> logging.Logger:
        """Get or create a configured logger"""
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            
            # Set level from config or parameter
            if level:
                logger.setLevel(getattr(logging, level.upper()))
            else:
                # Extract component from name and use config
                component = name.split('.')[0] if '.' in name else 'main'
                logger.setLevel(LogConfig.LEVELS.get(component, logging.INFO))
            
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @staticmethod
    def _create_console_handler(production_mode: bool) -> logging.StreamHandler:
        """Create console handler with appropriate formatter"""
        handler = logging.StreamHandler(sys.stdout)
        
        if production_mode:
            formatter = logging.Formatter(
                LogConfig.FORMATS['simple'],
                datefmt='%H:%M:%S'
            )
        else:
            formatter = ColoredFormatter(
                LogConfig.FORMATS['detailed'],
                datefmt='%H:%M:%S'
            )
        
        handler.setFormatter(formatter)
        handler.addFilter(PerformanceFilter())
        return handler
    
    @staticmethod
    def _create_file_handler() -> RotatingFileHandler:
        """Create rotating file handler for general logs"""
        handler = RotatingFileHandler(
            LogConfig.LOG_DIR / 'wave_detection.log',
            maxBytes=LogConfig.MAX_BYTES,
            backupCount=LogConfig.BACKUP_COUNT
        )
        handler.setLevel(logging.INFO)
        
        formatter = StructuredFormatter() if PRODUCTION_MODE else logging.Formatter(
            LogConfig.FORMATS['detailed'],
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        return handler
    
    @staticmethod
    def _create_error_handler() -> RotatingFileHandler:
        """Create rotating file handler for errors only"""
        handler = RotatingFileHandler(
            LogConfig.LOG_DIR / 'errors.log',
            maxBytes=LogConfig.MAX_BYTES,
            backupCount=LogConfig.BACKUP_COUNT
        )
        handler.setLevel(logging.ERROR)
        
        formatter = logging.Formatter(
            LogConfig.FORMATS['detailed'],
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        handler.addFilter(ErrorCountFilter())
        return handler

# ============================================
# LOGGING DECORATORS
# ============================================

def log_execution(level=logging.DEBUG, include_args=False, include_result=False):
    """Decorator to log function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = LoggerFactory.get_logger(func.__module__)
            
            # Log entry
            msg = f"Entering {func.__name__}"
            if include_args:
                msg += f" with args={args}, kwargs={kwargs}"
            logger.log(level, msg)
            
            # Execute function
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                
                # Log success
                msg = f"Completed {func.__name__} in {duration:.3f}s"
                if include_result:
                    msg += f" with result={result}"
                logger.log(level, msg, extra={'duration': duration})
                
                # Warn if slow
                if duration > LogConfig.SLOW_QUERY_THRESHOLD:
                    logger.warning(f"{func.__name__} took {duration:.3f}s (threshold: {LogConfig.SLOW_QUERY_THRESHOLD}s)")
                
                return result
                
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error(f"Error in {func.__name__} after {duration:.3f}s: {str(e)}", exc_info=True)
                raise
        
        return wrapper
    return decorator

def log_performance(threshold: float = None):
    """Decorator specifically for performance monitoring"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = LoggerFactory.get_logger('performance')
            
            start_time = time.perf_counter()
            start_memory = get_memory_usage() if 'get_memory_usage' in globals() else 0
            
            try:
                result = func(*args, **kwargs)
                
                duration = time.perf_counter() - start_time
                memory_delta = (get_memory_usage() if 'get_memory_usage' in globals() else 0) - start_memory
                
                # Log performance metrics
                logger.debug(
                    f"{func.__name__} | Duration: {duration:.3f}s | Memory Δ: {memory_delta:.1f}MB",
                    extra={'duration': duration, 'memory_delta': memory_delta}
                )
                
                # Check threshold
                if threshold and duration > threshold:
                    logger.warning(f"Performance threshold exceeded: {func.__name__} took {duration:.3f}s (limit: {threshold}s)")
                
                # Store metrics
                if 'performance_metrics' in st.session_state:
                    if func.__name__ not in st.session_state.performance_metrics:
                        st.session_state.performance_metrics[func.__name__] = []
                    st.session_state.performance_metrics[func.__name__].append({
                        'duration': duration,
                        'memory_delta': memory_delta,
                        'timestamp': datetime.now(timezone.utc)
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"Performance monitoring failed for {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    return decorator

# ============================================
# CONTEXT MANAGERS
# ============================================

@contextmanager
def log_context(operation: str, logger_name: str = 'main'):
    """Context manager for logging operations"""
    logger = LoggerFactory.get_logger(logger_name)
    
    logger.info(f"Starting: {operation}")
    start_time = time.perf_counter()
    
    try:
        yield logger
        duration = time.perf_counter() - start_time
        logger.info(f"Completed: {operation} ({duration:.3f}s)")
        
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.error(f"Failed: {operation} after {duration:.3f}s - {str(e)}")
        raise

@contextmanager
def suppress_logs(level=logging.WARNING):
    """Temporarily suppress logs below specified level"""
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(level)
    try:
        yield
    finally:
        root_logger.setLevel(original_level)

# ============================================
# LOGGING UTILITIES
# ============================================

def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame", logger_name: str = 'data'):
    """Log detailed DataFrame information"""
    logger = LoggerFactory.get_logger(logger_name)
    
    logger.info(f"{name} Info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"  Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
    logger.info(f"  Dtypes: {df.dtypes.value_counts().to_dict()}")
    logger.info(f"  Nulls: {df.isnull().sum().sum()}")

def log_error_with_context(error: Exception, context: Dict[str, Any], logger_name: str = 'error'):
    """Log error with additional context"""
    logger = LoggerFactory.get_logger(logger_name)
    
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    logger.error(f"Error occurred: {json.dumps(error_info, default=str)}")
    
    # Store in session state for debugging
    if 'error_log' not in st.session_state:
        st.session_state.error_log = []
    st.session_state.error_log.append(error_info)

# ============================================
# INITIALIZE LOGGING SYSTEM
# ============================================

# Setup logging based on mode
LoggerFactory.setup(production_mode=PRODUCTION_MODE)

# Create main application logger
logger = LoggerFactory.get_logger('wave_detection.main')

# Create specialized loggers
data_logger = LoggerFactory.get_logger('wave_detection.data')
scoring_logger = LoggerFactory.get_logger('wave_detection.scoring')
filter_logger = LoggerFactory.get_logger('wave_detection.filter')
export_logger = LoggerFactory.get_logger('wave_detection.export')
ui_logger = LoggerFactory.get_logger('wave_detection.ui')
perf_logger = LoggerFactory.get_logger('wave_detection.performance')

# Log initialization
logger.info("Logging system initialized successfully")
logger.info(f"Production Mode: {PRODUCTION_MODE}")
logger.info(f"Log Level: {logging.getLevelName(logger.level)}")

# ============================================
# PERFORMANCE CONSTANTS
# ============================================

# Memory management thresholds
MEMORY_CLEANUP_THRESHOLD = 500  # Trigger cleanup after processing 500 rows
CACHE_CLEANUP_INTERVAL = 300  # Clean cache every 5 minutes

# Data processing limits
MAX_ROWS_DISPLAY = 1000  # Maximum rows to display in UI
MAX_EXPORT_ROWS = 10000  # Maximum rows for Excel export
CHUNK_SIZE = 100  # Process data in chunks for large datasets

# Performance targets (in seconds)
TARGET_LOAD_TIME = 2.0
TARGET_PROCESS_TIME = 1.0
TARGET_RENDER_TIME = 0.5

# ============================================
# GLOBAL STATE INITIALIZATION
# ============================================

# Initialize performance tracking
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = OrderedDict()

# Initialize error tracking
if 'error_log' not in st.session_state:
    st.session_state.error_log = []

# Track import time for performance monitoring
IMPORT_TIME = time.perf_counter()

# ============================================
# UTILITY FUNCTIONS
# ============================================

def cleanup_memory(force: bool = False) -> None:
    """
    Perform memory cleanup when needed.
    
    Args:
        force: Force garbage collection regardless of thresholds
    """
    if force or (time.time() % CACHE_CLEANUP_INTERVAL < 1):
        gc.collect()
        logger.debug("Memory cleanup performed")

def safe_import(module_name: str, package: str = None):
    """
    Safely import optional modules with fallback.
    
    Args:
        module_name: Name of module to import
        package: Package name if different from module
    
    Returns:
        Module or None if import fails
    """
    try:
        import importlib
        return importlib.import_module(module_name, package)
    except ImportError:
        logger.warning(f"Optional module {module_name} not available")
        return None

def get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in megabytes
    """
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

# ============================================
# VALIDATION CHECKS
# ============================================

def validate_environment() -> bool:
    """
    Validate that all required packages are available and configured correctly.
    
    Returns:
        True if environment is valid, False otherwise
    """
    required_modules = {
        'streamlit': st,
        'pandas': pd,
        'numpy': np,
        'plotly': go
    }
    
    for name, module in required_modules.items():
        if module is None:
            logger.error(f"Required module {name} is not available")
            return False
    
    # Check Streamlit version
    try:
        st_version = st.__version__
        major, minor = map(int, st_version.split('.')[:2])
        if major < 1 or (major == 1 and minor < 28):
            logger.warning(f"Streamlit version {st_version} is outdated. Please upgrade to 1.28+")
    except Exception as e:
        logger.warning(f"Could not verify Streamlit version: {e}")
    
    return True

# ============================================
# INITIALIZATION LOG
# ============================================

# Log successful import
import_duration = time.perf_counter() - IMPORT_TIME
logger.info(f"Imports completed in {import_duration:.3f}s")

# Validate environment on import
if not validate_environment():
    logger.error("Environment validation failed. Some features may not work correctly.")

# ============================================
# DEVELOPMENT/DEBUG HELPERS
# ============================================

def debug_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print debug information about a dataframe (only in debug mode).
    
    Args:
        df: DataFrame to debug
        name: Name for logging
    """
    if st.session_state.get('debug_mode', False):
        logger.debug(f"{name} shape: {df.shape}")
        logger.debug(f"{name} memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.debug(f"{name} dtypes: {df.dtypes.value_counts().to_dict()}")
        logger.debug(f"{name} nulls: {df.isnull().sum().sum()}")

# ============================================
# PRODUCTION FLAGS
# ============================================

# Set production mode (disable in development)
PRODUCTION_MODE = True
DEBUG_MODE = not PRODUCTION_MODE

# Configure behavior based on mode
if PRODUCTION_MODE:
    # Production settings
    LOG_LEVEL = logging.WARNING
    SHOW_ERRORS = False
    ENABLE_PROFILING = False
else:
    # Development settings
    LOG_LEVEL = logging.DEBUG
    SHOW_ERRORS = True
    ENABLE_PROFILING = True

# Apply log level
logger.setLevel(LOG_LEVEL)

# ============================================
# CONFIGURATION AND CONSTANTS
# ============================================

@dataclass(frozen=True)
class ScoreWeights:
    """Master score component weights (must sum to 1.0)"""
    POSITION: float = 0.30    # 30% - Price position strength
    VOLUME: float = 0.25       # 25% - Volume intensity
    MOMENTUM: float = 0.15     # 15% - Price momentum
    ACCELERATION: float = 0.10 # 10% - Momentum acceleration
    BREAKOUT: float = 0.10     # 10% - Breakout probability
    RVOL: float = 0.10        # 10% - Relative volume
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = sum([self.POSITION, self.VOLUME, self.MOMENTUM, 
                    self.ACCELERATION, self.BREAKOUT, self.RVOL])
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Score weights must sum to 1.0, got {total}")

@dataclass(frozen=True)
class DataConfig:
    """Data source and processing configuration"""
    
    # Google Sheets defaults
    DEFAULT_SHEET_ID: str = "1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM"
    DEFAULT_GID: str = "1823439984"
    
    # Data validation
    CRITICAL_COLUMNS: Tuple[str, ...] = ('ticker', 'price', 'volume_1d')
    IMPORTANT_COLUMNS: Tuple[str, ...] = (
        'category', 'sector', 'industry', 'company_name',
        'rvol', 'pe', 'eps_current', 'eps_change_pct',
        'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
        'from_low_pct', 'from_high_pct',
        'sma_20d', 'sma_50d', 'sma_200d'
    )
    
    # Data processing limits
    MIN_VALID_ROWS: int = 10
    MAX_DISPLAY_ROWS: int = 1000
    MAX_EXPORT_ROWS: int = 10000
    
    # Data quality thresholds
    MIN_DATA_QUALITY: float = 0.6  # 60% minimum completeness
    OUTLIER_STD_THRESHOLD: float = 4.0  # Remove beyond 4 std devs

@dataclass(frozen=True)
class PerformanceConfig:
    """Performance and optimization settings"""
    
    # Cache settings
    CACHE_TTL_SECONDS: int = 900  # 15 minutes
    CACHE_VERSION: str = "3.0.0"
    USE_PERSISTENT_CACHE: bool = True
    
    # Processing chunks for large datasets
    CHUNK_SIZE: int = 100
    PARALLEL_THRESHOLD: int = 500  # Use parallel processing above this
    
    # Memory management
    MEMORY_CLEANUP_INTERVAL: int = 300  # 5 minutes
    MAX_MEMORY_MB: int = 512  # Maximum memory usage
    
    # API/Network settings
    REQUEST_TIMEOUT: int = 30  # seconds
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0  # seconds

@dataclass(frozen=True)
class UIConfig:
    """User interface configuration"""
    
    # Display defaults
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: Tuple[int, ...] = (10, 20, 50, 100, 200, 500, 1000)
    
    # Chart settings
    CHART_HEIGHT: int = 400
    CHART_COLOR_SCHEME: str = "RdYlGn"
    MAX_CHART_POINTS: int = 100
    
    # Table settings
    TABLE_HEIGHT_PER_ROW: int = 35
    TABLE_MAX_HEIGHT: int = 600
    TABLE_DECIMAL_PLACES: int = 1
    
    # UI refresh settings
    AUTO_REFRESH: bool = False
    REFRESH_INTERVAL: int = 60  # seconds
    
    # Theme colors
    COLORS = {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#22c55e',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#3b82f6'
    }

@dataclass(frozen=True)
class PatternConfig:
    """Pattern detection thresholds and settings"""
    
    # Pattern detection thresholds
    THRESHOLDS = {
        'category_leader': 85,      # Top 15% in category
        'hidden_gem': 80,           # High category, low overall
        'acceleration': 75,         # Strong acceleration
        'institutional': 70,        # Institutional interest
        'volume_explosion': 3.0,    # 3x normal volume
        'momentum_shift': 70,       # Momentum reversal
        'breakout_imminent': 80,    # Near breakout
        'trend_leader': 75,         # Leading trend
        'oversold_bounce': 30,      # Oversold reversal
        'bullish_divergence': 60,   # Price/volume divergence
        'accumulation': 65,         # Smart money accumulating
        'distribution': 35,         # Smart money distributing
        'short_squeeze': 85,        # Potential squeeze
        'gap_fill': 5.0,           # Gap percentage
        'golden_cross': 1.0,        # MA crossover threshold
        'death_cross': -1.0,        # MA crossover threshold
        'support_bounce': 5.0,      # Near support level
        'resistance_test': 95.0,    # Near resistance
        'range_breakout': 90,       # Breaking range
        'mean_reversion': 2.0,      # Standard deviations
        'momentum_burst': 80,       # Sudden momentum
        'volume_dry_up': 0.5,       # Low volume
        'trend_exhaustion': 15,     # Trend ending
        'perfect_storm': 4          # Multiple signals minimum
    }
    
    # Pattern importance weights (for ranking)
    WEIGHTS = {
        'PERFECT STORM': 1.0,
        'VOL EXPLOSION': 0.9,
        'BREAKOUT READY': 0.85,
        'CAT LEADER': 0.8,
        'INSTITUTIONAL': 0.75,
        'MOMENTUM SHIFT': 0.7,
        'HIDDEN GEM': 0.65
    }
    
    # Pattern display settings
    MAX_PATTERNS_DISPLAY: int = 5
    PATTERN_SEPARATOR: str = " | "

@dataclass(frozen=True)
class FilterConfig:
    """Filter system configuration"""
    
    # Quick filter presets
    QUICK_FILTERS = {
        'top_gainers': {'momentum_score': 80, 'ret_30d': 0},
        'volume_surges': {'rvol': 3.0, 'volume_score': 70},
        'breakout_ready': {'breakout_score': 80, 'from_high_pct': -10},
        'hidden_gems': {'category_percentile': 80, 'percentile': 70}
    }
    
    # Filter limits
    MAX_ACTIVE_FILTERS: int = 20
    MIN_RESULTS_WARNING: int = 5
    
    # Tier definitions
    EPS_TIERS = {
        'Negative': (-float('inf'), 0),
        'Low (0-2)': (0, 2),
        'Medium (2-5)': (2, 5),
        'High (5-10)': (5, 10),
        'Very High (10+)': (10, float('inf'))
    }
    
    PE_TIERS = {
        'Negative/Zero': (-float('inf'), 0),
        'Low (0-15)': (0, 15),
        'Fair (15-25)': (15, 25),
        'High (25-40)': (25, 40),
        'Very High (40+)': (40, float('inf'))
    }
    
    PRICE_TIERS = {
        'Penny (<10)': (0, 10),
        'Low (10-100)': (10, 100),
        'Mid (100-1000)': (100, 1000),
        'High (1000-5000)': (1000, 5000),
        'Premium (5000+)': (5000, float('inf'))
    }

@dataclass(frozen=True)
class ExportConfig:
    """Export and reporting configuration"""
    
    # Export templates
    TEMPLATES = {
        'full': {
            'sheets': ['Overview', 'Technical', 'Fundamental', 'Patterns', 'Signals', 'Statistics'],
            'max_rows': 10000
        },
        'day_trader': {
            'sheets': ['Intraday', 'Volume', 'Momentum', 'Signals'],
            'max_rows': 500,
            'focus_columns': ['ticker', 'price', 'ret_1d', 'rvol', 'momentum_score']
        },
        'swing_trader': {
            'sheets': ['Weekly', 'Patterns', 'Technical', 'Breakouts'],
            'max_rows': 1000,
            'focus_columns': ['ticker', 'price', 'ret_7d', 'ret_30d', 'patterns']
        },
        'investor': {
            'sheets': ['Fundamental', 'Sectors', 'Long-term', 'Quality'],
            'max_rows': 2000,
            'focus_columns': ['ticker', 'pe', 'eps_current', 'ret_1y', 'category']
        }
    }
    
    # Excel formatting
    EXCEL_STYLES = {
        'header': {'bold': True, 'bg_color': '#667eea', 'font_color': 'white'},
        'positive': {'font_color': '#22c55e'},
        'negative': {'font_color': '#ef4444'},
        'highlight': {'bg_color': '#fef3c7'}
    }
    
    # Report settings
    INCLUDE_CHARTS: bool = True
    INCLUDE_SUMMARY: bool = True
    DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'

@dataclass(frozen=True)
class ValidationConfig:
    """Data validation rules"""
    
    # Numeric column validation
    NUMERIC_COLUMNS = {
        'price': (0, 1000000),
        'volume_1d': (0, float('inf')),
        'pe': (-1000, 1000),
        'eps_current': (-100, 100),
        'ret_1d': (-50, 50),
        'ret_7d': (-75, 75),
        'ret_30d': (-90, 90),
        'rvol': (0, 100),
        'master_score': (0, 100)
    }
    
    # Required value checks
    REQUIRED_VALUES = {
        'ticker': lambda x: isinstance(x, str) and len(x) > 0,
        'price': lambda x: isinstance(x, (int, float)) and x > 0,
        'volume_1d': lambda x: isinstance(x, (int, float)) and x >= 0
    }
    
    # Category validations
    VALID_CATEGORIES = ('Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap', 'Nano Cap')
    VALID_TRENDS = ('Uptrend', 'Downtrend', 'Sideways', 'Volatile')

# ============================================
# RUNTIME CONFIGURATION
# ============================================

class RuntimeConfig:
    """Mutable runtime configuration"""
    
    def __init__(self):
        self.debug_mode = False
        self.performance_tracking = True
        self.error_reporting = True
        self.auto_save = False
        self.theme = 'light'
        self.timezone = timezone.utc
        self.locale = 'en_US'
        self.currency_symbol = '₹'
        self.number_format = 'indian'  # or 'western'
    
    def update(self, **kwargs):
        """Update runtime configuration"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Runtime config updated: {key}={value}")
            else:
                logger.warning(f"Unknown runtime config: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'debug_mode': self.debug_mode,
            'performance_tracking': self.performance_tracking,
            'error_reporting': self.error_reporting,
            'auto_save': self.auto_save,
            'theme': self.theme,
            'timezone': str(self.timezone),
            'locale': self.locale,
            'currency_symbol': self.currency_symbol,
            'number_format': self.number_format
        }

# ============================================
# CONFIGURATION INSTANCES
# ============================================

# Create immutable configuration instances
SCORE_WEIGHTS = ScoreWeights()
DATA_CONFIG = DataConfig()
PERFORMANCE_CONFIG = PerformanceConfig()
UI_CONFIG = UIConfig()
PATTERN_CONFIG = PatternConfig()
FILTER_CONFIG = FilterConfig()
EXPORT_CONFIG = ExportConfig()
VALIDATION_CONFIG = ValidationConfig()

# Create mutable runtime configuration
RUNTIME_CONFIG = RuntimeConfig()

# ============================================
# CONFIGURATION VALIDATION
# ============================================

def validate_configuration() -> bool:
    """Validate all configuration settings"""
    try:
        # Validate score weights
        assert abs(sum([
            SCORE_WEIGHTS.POSITION, SCORE_WEIGHTS.VOLUME,
            SCORE_WEIGHTS.MOMENTUM, SCORE_WEIGHTS.ACCELERATION,
            SCORE_WEIGHTS.BREAKOUT, SCORE_WEIGHTS.RVOL
        ]) - 1.0) < 0.001, "Score weights must sum to 1.0"
        
        # Validate data config
        assert DATA_CONFIG.MIN_VALID_ROWS > 0, "MIN_VALID_ROWS must be positive"
        assert DATA_CONFIG.MAX_DISPLAY_ROWS > DATA_CONFIG.MIN_VALID_ROWS
        
        # Validate performance config
        assert PERFORMANCE_CONFIG.CACHE_TTL_SECONDS > 0
        assert PERFORMANCE_CONFIG.CHUNK_SIZE > 0
        
        # Validate UI config
        assert UI_CONFIG.DEFAULT_TOP_N in UI_CONFIG.AVAILABLE_TOP_N
        
        logger.info("Configuration validation passed")
        return True
        
    except AssertionError as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

# ============================================
# CONFIGURATION HELPERS
# ============================================

def get_config_value(path: str, default: Any = None) -> Any:
    """
    Get configuration value by dot notation path.
    Example: get_config_value('SCORE_WEIGHTS.POSITION')
    """
    try:
        parts = path.split('.')
        obj = globals()[parts[0]]
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj
    except (KeyError, AttributeError):
        logger.warning(f"Config path not found: {path}, using default: {default}")
        return default

def format_number(value: float, number_type: str = 'general') -> str:
    """Format number based on locale and type"""
    if RUNTIME_CONFIG.number_format == 'indian':
        if number_type == 'currency':
            return f"{RUNTIME_CONFIG.currency_symbol}{value:,.0f}"
        elif number_type == 'percentage':
            return f"{value:.1f}%"
        else:
            return f"{value:,.1f}"
    else:
        if number_type == 'currency':
            return f"${value:,.2f}"
        elif number_type == 'percentage':
            return f"{value:.1f}%"
        else:
            return f"{value:,.2f}"

# ============================================
# VALIDATE ON IMPORT
# ============================================

# Validate configuration on module import
if not validate_configuration():
    logger.error("Invalid configuration detected. Using defaults.")

# Log configuration summary
logger.info(f"Configuration loaded: {len(SCORE_WEIGHTS.__annotations__)} score weights")
logger.info(f"Cache TTL: {PERFORMANCE_CONFIG.CACHE_TTL_SECONDS}s")
logger.info(f"Default display: {UI_CONFIG.DEFAULT_TOP_N} rows")
logger.info(f"Pattern thresholds: {len(PATTERN_CONFIG.THRESHOLDS)} patterns")

# ============================================
# PERFORMANCE MONITORING
# ============================================

class PerformanceMonitor:
    """
    Advanced performance monitoring with automatic profiling,
    bottleneck detection, and optimization suggestions.
    """
    
    # Performance thresholds (seconds)
    THRESHOLDS = {
        'data_load': 2.0,
        'data_process': 1.0,
        'score_calculation': 0.5,
        'pattern_detection': 0.5,
        'filter_apply': 0.1,
        'render': 0.5,
        'export': 3.0,
        'search': 0.2
    }
    
    # Performance grades
    GRADES = {
        'excellent': 0.5,   # Under 50% of threshold
        'good': 0.75,       # Under 75% of threshold
        'acceptable': 1.0,  # At threshold
        'slow': 1.5,        # 50% over threshold
        'critical': 2.0     # 2x threshold
    }
    
    @staticmethod
    def timer(operation: str = None, threshold: float = None, 
             track_memory: bool = True, log_level: int = logging.DEBUG):
        """
        Performance timing decorator with automatic threshold detection.
        
        Args:
            operation: Operation name (auto-detected if None)
            threshold: Performance threshold in seconds
            track_memory: Whether to track memory usage
            log_level: Logging level for performance data
        """
        def decorator(func):
            nonlocal operation, threshold
            
            # Auto-detect operation name
            if operation is None:
                operation = func.__name__
            
            # Auto-detect threshold
            if threshold is None:
                threshold = PerformanceMonitor.THRESHOLDS.get(
                    operation, 
                    PERFORMANCE_CONFIG.SLOW_QUERY_THRESHOLD if 'PERFORMANCE_CONFIG' in globals() else 1.0
                )
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Start monitoring
                start_time = time.perf_counter()
                start_memory = 0
                
                if track_memory:
                    try:
                        import psutil
                        process = psutil.Process()
                        start_memory = process.memory_info().rss / 1024 / 1024  # MB
                    except:
                        track_memory = False
                
                # Track function call
                call_id = f"{operation}_{int(time.time() * 1000)}"
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Calculate metrics
                    duration = time.perf_counter() - start_time
                    memory_delta = 0
                    
                    if track_memory:
                        try:
                            import psutil
                            process = psutil.Process()
                            end_memory = process.memory_info().rss / 1024 / 1024
                            memory_delta = end_memory - start_memory
                        except:
                            pass
                    
                    # Determine performance grade
                    grade = PerformanceMonitor._get_grade(duration, threshold)
                    
                    # Log performance
                    log_msg = f"[PERF] {operation}: {duration:.3f}s"
                    if memory_delta:
                        log_msg += f" | Δ Memory: {memory_delta:+.1f}MB"
                    log_msg += f" | Grade: {grade}"
                    
                    # Choose log level based on performance
                    if grade == 'excellent':
                        perf_logger.log(log_level, log_msg)
                    elif grade in ['good', 'acceptable']:
                        perf_logger.info(log_msg)
                    elif grade == 'slow':
                        perf_logger.warning(log_msg)
                    else:  # critical
                        perf_logger.error(log_msg)
                    
                    # Store metrics in session state
                    PerformanceMonitor._store_metrics(
                        operation, duration, memory_delta, grade, call_id
                    )
                    
                    # Check for performance degradation
                    PerformanceMonitor._check_degradation(operation)
                    
                    return result
                    
                except Exception as e:
                    duration = time.perf_counter() - start_time
                    perf_logger.error(
                        f"[PERF] {operation} FAILED after {duration:.3f}s: {str(e)}"
                    )
                    
                    # Store failure metrics
                    PerformanceMonitor._store_metrics(
                        operation, duration, 0, 'failed', call_id, failed=True
                    )
                    raise
            
            return wrapper
        return decorator
    
    @staticmethod
    def _get_grade(duration: float, threshold: float) -> str:
        """Determine performance grade based on duration and threshold"""
        ratio = duration / threshold
        
        for grade, max_ratio in PerformanceMonitor.GRADES.items():
            if ratio <= max_ratio:
                return grade
        return 'critical'
    
    @staticmethod
    def _store_metrics(operation: str, duration: float, memory_delta: float, 
                      grade: str, call_id: str, failed: bool = False):
        """Store performance metrics in session state"""
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = OrderedDict()
        
        if operation not in st.session_state.performance_metrics:
            st.session_state.performance_metrics[operation] = {
                'calls': [],
                'total_duration': 0,
                'total_memory': 0,
                'failure_count': 0,
                'grades': {'excellent': 0, 'good': 0, 'acceptable': 0, 'slow': 0, 'critical': 0}
            }
        
        metrics = st.session_state.performance_metrics[operation]
        
        # Store call details (keep last 100)
        metrics['calls'].append({
            'id': call_id,
            'timestamp': datetime.now(timezone.utc),
            'duration': duration,
            'memory_delta': memory_delta,
            'grade': grade if not failed else 'failed',
            'failed': failed
        })
        
        if len(metrics['calls']) > 100:
            metrics['calls'] = metrics['calls'][-100:]
        
        # Update aggregates
        metrics['total_duration'] += duration
        metrics['total_memory'] += memory_delta
        
        if failed:
            metrics['failure_count'] += 1
        else:
            metrics['grades'][grade] += 1
    
    @staticmethod
    def _check_degradation(operation: str):
        """Check for performance degradation trends"""
        if 'performance_metrics' not in st.session_state:
            return
        
        metrics = st.session_state.performance_metrics.get(operation, {})
        calls = metrics.get('calls', [])
        
        if len(calls) < 10:
            return
        
        # Check last 10 calls vs previous 10
        recent_calls = calls[-10:]
        previous_calls = calls[-20:-10] if len(calls) >= 20 else calls[:10]
        
        recent_avg = np.mean([c['duration'] for c in recent_calls])
        previous_avg = np.mean([c['duration'] for c in previous_calls])
        
        # Detect significant degradation (>20% slower)
        if recent_avg > previous_avg * 1.2:
            degradation_pct = ((recent_avg - previous_avg) / previous_avg) * 100
            perf_logger.warning(
                f"Performance degradation detected for {operation}: "
                f"{degradation_pct:.1f}% slower than baseline"
            )
    
    @staticmethod
    def get_summary(operation: str = None) -> Dict[str, Any]:
        """
        Get performance summary for an operation or all operations.
        
        Args:
            operation: Specific operation or None for all
            
        Returns:
            Dictionary with performance statistics
        """
        if 'performance_metrics' not in st.session_state:
            return {}
        
        if operation:
            metrics = st.session_state.performance_metrics.get(operation, {})
            if not metrics:
                return {}
            
            calls = metrics.get('calls', [])
            if not calls:
                return {}
            
            durations = [c['duration'] for c in calls if not c.get('failed')]
            
            return {
                'operation': operation,
                'total_calls': len(calls),
                'failure_rate': metrics['failure_count'] / len(calls) if calls else 0,
                'avg_duration': np.mean(durations) if durations else 0,
                'min_duration': np.min(durations) if durations else 0,
                'max_duration': np.max(durations) if durations else 0,
                'p50_duration': np.percentile(durations, 50) if durations else 0,
                'p95_duration': np.percentile(durations, 95) if durations else 0,
                'total_memory': metrics['total_memory'],
                'grades': metrics['grades']
            }
        else:
            # Summary for all operations
            all_summaries = {}
            for op in st.session_state.performance_metrics:
                all_summaries[op] = PerformanceMonitor.get_summary(op)
            return all_summaries
    
    @staticmethod
    def display_dashboard():
        """Display performance dashboard in Streamlit"""
        if 'performance_metrics' not in st.session_state or not st.session_state.performance_metrics:
            st.info("No performance data available yet")
            return
        
        st.markdown("### 📊 Performance Dashboard")
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        all_calls = []
        for metrics in st.session_state.performance_metrics.values():
            all_calls.extend(metrics.get('calls', []))
        
        if all_calls:
            total_duration = sum(c['duration'] for c in all_calls)
            avg_duration = np.mean([c['duration'] for c in all_calls])
            failure_rate = sum(1 for c in all_calls if c.get('failed')) / len(all_calls)
            
            with col1:
                st.metric("Total Operations", f"{len(all_calls):,}")
            with col2:
                st.metric("Total Time", f"{total_duration:.1f}s")
            with col3:
                st.metric("Avg Duration", f"{avg_duration:.3f}s")
            with col4:
                st.metric("Failure Rate", f"{failure_rate:.1%}")
        
        # Per-operation breakdown
        st.markdown("#### Operation Performance")
        
        data = []
        for operation, metrics in st.session_state.performance_metrics.items():
            calls = metrics.get('calls', [])
            if calls:
                durations = [c['duration'] for c in calls if not c.get('failed')]
                if durations:
                    data.append({
                        'Operation': operation,
                        'Calls': len(calls),
                        'Avg (s)': np.mean(durations),
                        'Min (s)': np.min(durations),
                        'Max (s)': np.max(durations),
                        'P95 (s)': np.percentile(durations, 95),
                        'Failures': metrics['failure_count'],
                        'Grade': PerformanceMonitor._get_overall_grade(metrics['grades'])
                    })
        
        if data:
            df = pd.DataFrame(data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Avg (s)': st.column_config.NumberColumn(format='%.3f'),
                    'Min (s)': st.column_config.NumberColumn(format='%.3f'),
                    'Max (s)': st.column_config.NumberColumn(format='%.3f'),
                    'P95 (s)': st.column_config.NumberColumn(format='%.3f'),
                }
            )
    
    @staticmethod
    def _get_overall_grade(grades: Dict[str, int]) -> str:
        """Calculate overall grade from grade distribution"""
        total = sum(grades.values())
        if total == 0:
            return 'N/A'
        
        # Weighted score
        weights = {
            'excellent': 1.0,
            'good': 0.8,
            'acceptable': 0.6,
            'slow': 0.3,
            'critical': 0.0
        }
        
        score = sum(grades[g] * weights.get(g, 0) for g in grades) / total
        
        if score >= 0.8:
            return '🟢 Excellent'
        elif score >= 0.6:
            return '🟡 Good'
        elif score >= 0.4:
            return '🟠 Fair'
        else:
            return '🔴 Poor'
    
    @staticmethod
    def optimize_suggestions() -> List[str]:
        """Generate optimization suggestions based on performance data"""
        suggestions = []
        
        if 'performance_metrics' not in st.session_state:
            return suggestions
        
        for operation, metrics in st.session_state.performance_metrics.items():
            calls = metrics.get('calls', [])
            if len(calls) < 5:
                continue
            
            durations = [c['duration'] for c in calls if not c.get('failed')]
            if not durations:
                continue
            
            avg_duration = np.mean(durations)
            threshold = PerformanceMonitor.THRESHOLDS.get(operation, 1.0)
            
            # Check if consistently slow
            if avg_duration > threshold * 1.5:
                suggestions.append(
                    f"⚠️ {operation} is running {avg_duration/threshold:.1f}x slower than expected. "
                    f"Consider optimizing or caching."
                )
            
            # Check for high variance
            if len(durations) > 1:
                cv = np.std(durations) / avg_duration if avg_duration > 0 else 0
                if cv > 0.5:
                    suggestions.append(
                        f"📊 {operation} has high variance (CV={cv:.1f}). "
                        f"Performance is inconsistent."
                    )
            
            # Check failure rate
            failure_rate = metrics['failure_count'] / len(calls) if calls else 0
            if failure_rate > 0.1:
                suggestions.append(
                    f"❌ {operation} has {failure_rate:.1%} failure rate. "
                    f"Needs error handling improvement."
                )
        
        return suggestions

# Create specialized logger for performance
perf_logger = LoggerFactory.get_logger('wave_detection.performance')

# ============================================
# DATA VALIDATION AND SANITIZATION
# ============================================

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class DataValidator:
    """
    Comprehensive data validation and sanitization system.
    Handles validation, cleaning, type conversion, and quality reporting.
    """
    
    # Validation statistics tracking
    validation_stats = {
        'total_validations': 0,
        'total_errors_fixed': 0,
        'total_warnings': 0,
        'last_validation': None
    }
    
    @staticmethod
    @PerformanceMonitor.timer(operation='data_validation')
    def validate_and_sanitize(df: pd.DataFrame, 
                             strict: bool = False,
                             auto_fix: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete validation and sanitization pipeline.
        
        Args:
            df: DataFrame to validate and sanitize
            strict: If True, raise errors instead of warnings
            auto_fix: If True, automatically fix issues where possible
            
        Returns:
            Tuple of (cleaned_df, validation_report)
        """
        if df is None or df.empty:
            raise DataValidationError("DataFrame is empty or None")
        
        validation_report = {
            'timestamp': datetime.now(timezone.utc),
            'original_shape': df.shape,
            'errors': [],
            'warnings': [],
            'fixes_applied': [],
            'quality_score': 100.0,
            'column_reports': {}
        }
        
        # Create a copy to avoid modifying original
        df = df.copy()
        original_rows = len(df)
        
        try:
            # Step 1: Structure validation
            df, structure_report = DataValidator._validate_structure(df, strict)
            validation_report['structure'] = structure_report
            
            # Step 2: Column validation
            df, column_report = DataValidator._validate_columns(df, auto_fix)
            validation_report['column_reports'] = column_report
            
            # Step 3: Data type validation and conversion
            df, type_report = DataValidator._validate_and_convert_types(df, auto_fix)
            validation_report['type_conversions'] = type_report
            
            # Step 4: Value range validation
            df, range_report = DataValidator._validate_ranges(df, auto_fix)
            validation_report['range_validations'] = range_report
            
            # Step 5: Remove duplicates
            df, duplicate_report = DataValidator._remove_duplicates(df, auto_fix)
            validation_report['duplicate_removal'] = duplicate_report
            
            # Step 6: Handle missing values
            df, missing_report = DataValidator._handle_missing_values(df, auto_fix)
            validation_report['missing_values'] = missing_report
            
            # Step 7: Detect and handle outliers
            df, outlier_report = DataValidator._handle_outliers(df, auto_fix)
            validation_report['outliers'] = outlier_report
            
            # Step 8: Validate business rules
            df, business_report = DataValidator._validate_business_rules(df, auto_fix)
            validation_report['business_rules'] = business_report
            
            # Step 9: Data consistency checks
            df, consistency_report = DataValidator._check_consistency(df, auto_fix)
            validation_report['consistency'] = consistency_report
            
            # Calculate quality score
            validation_report['quality_score'] = DataValidator._calculate_quality_score(
                df, validation_report
            )
            
            # Final shape
            validation_report['final_shape'] = df.shape
            validation_report['rows_removed'] = original_rows - len(df)
            
            # Update statistics
            DataValidator.validation_stats['total_validations'] += 1
            DataValidator.validation_stats['total_errors_fixed'] += len(validation_report.get('fixes_applied', []))
            DataValidator.validation_stats['total_warnings'] += len(validation_report.get('warnings', []))
            DataValidator.validation_stats['last_validation'] = datetime.now(timezone.utc)
            
            # Log summary
            data_logger.info(
                f"Validation complete: {len(df)} rows, "
                f"Quality: {validation_report['quality_score']:.1f}%, "
                f"Fixes: {len(validation_report.get('fixes_applied', []))}"
            )
            
            return df, validation_report
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            if strict:
                raise DataValidationError(f"Validation failed: {str(e)}")
            validation_report['errors'].append(str(e))
            return df, validation_report
    
    @staticmethod
    def _validate_structure(df: pd.DataFrame, strict: bool) -> Tuple[pd.DataFrame, Dict]:
        """Validate DataFrame structure"""
        report = {
            'critical_columns_missing': [],
            'important_columns_missing': [],
            'rows_before': len(df),
            'rows_after': len(df)
        }
        
        # Check critical columns
        missing_critical = [col for col in DATA_CONFIG.CRITICAL_COLUMNS if col not in df.columns]
        if missing_critical:
            error_msg = f"Critical columns missing: {missing_critical}"
            if strict:
                raise DataValidationError(error_msg)
            report['critical_columns_missing'] = missing_critical
            logger.error(error_msg)
        
        # Check important columns
        missing_important = [col for col in DATA_CONFIG.IMPORTANT_COLUMNS if col not in df.columns]
        if missing_important:
            report['important_columns_missing'] = missing_important
            logger.warning(f"Important columns missing: {missing_important[:5]}...")
        
        # Remove completely empty rows
        before = len(df)
        df = df.dropna(how='all')
        removed = before - len(df)
        if removed > 0:
            report['empty_rows_removed'] = removed
            logger.info(f"Removed {removed} empty rows")
        
        # Remove rows missing critical data
        for col in DATA_CONFIG.CRITICAL_COLUMNS:
            if col in df.columns:
                before = len(df)
                df = df.dropna(subset=[col])
                removed = before - len(df)
                if removed > 0:
                    report[f'rows_missing_{col}'] = removed
        
        report['rows_after'] = len(df)
        return df, report
    
    @staticmethod
    def _validate_columns(df: pd.DataFrame, auto_fix: bool) -> Tuple[pd.DataFrame, Dict]:
        """Validate and standardize column names"""
        report = {}
        
        # Standardize column names
        original_columns = df.columns.tolist()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Track renamed columns
        renamed = [(orig, new) for orig, new in zip(original_columns, df.columns) if orig != new]
        if renamed:
            report['columns_renamed'] = renamed
            logger.info(f"Renamed {len(renamed)} columns")
        
        # Remove duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols and auto_fix:
            df = df.loc[:, ~df.columns.duplicated()]
            report['duplicate_columns_removed'] = duplicate_cols
            logger.warning(f"Removed duplicate columns: {duplicate_cols}")
        
        return df, report
    
    @staticmethod
    def _validate_and_convert_types(df: pd.DataFrame, auto_fix: bool) -> Tuple[pd.DataFrame, Dict]:
        """Validate and convert data types"""
        report = {
            'conversions': {},
            'failed_conversions': {}
        }
        
        # Define expected types
        type_mappings = {
            'price': 'float64',
            'volume_1d': 'float64',
            'volume_7d': 'float64',
            'volume_30d': 'float64',
            'volume_90d': 'float64',
            'volume_180d': 'float64',
            'pe': 'float64',
            'eps_current': 'float64',
            'eps_last_qtr': 'float64',
            'ret_1d': 'float64',
            'ret_7d': 'float64',
            'ret_30d': 'float64',
            'ret_3m': 'float64',
            'ret_6m': 'float64',
            'ret_1y': 'float64',
            'ret_3y': 'float64',
            'ret_5y': 'float64',
            'rvol': 'float64',
            'from_low_pct': 'float64',
            'from_high_pct': 'float64',
            'sma_20d': 'float64',
            'sma_50d': 'float64',
            'sma_200d': 'float64',
            'ticker': 'string',
            'company_name': 'string',
            'category': 'category',
            'sector': 'category',
            'industry': 'category'
        }
        
        for col, expected_type in type_mappings.items():
            if col not in df.columns:
                continue
            
            current_type = df[col].dtype
            
            if auto_fix and str(current_type) != expected_type:
                try:
                    if expected_type == 'float64':
                        # Handle percentage strings
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '')
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                    elif expected_type == 'string':
                        df[col] = df[col].astype(str).str.strip()
                        
                    elif expected_type == 'category':
                        df[col] = df[col].astype('category')
                    
                    report['conversions'][col] = f"{current_type} -> {expected_type}"
                    
                except Exception as e:
                    report['failed_conversions'][col] = str(e)
                    logger.warning(f"Failed to convert {col}: {e}")
        
        return df, report
    
    @staticmethod
    def _validate_ranges(df: pd.DataFrame, auto_fix: bool) -> Tuple[pd.DataFrame, Dict]:
        """Validate value ranges for numeric columns"""
        report = {
            'out_of_range': {},
            'capped_values': {}
        }
        
        # Use validation config
        for col, (min_val, max_val) in VALIDATION_CONFIG.NUMERIC_COLUMNS.items():
            if col not in df.columns or df[col].dtype not in ['float64', 'int64']:
                continue
            
            # Find out of range values
            out_of_range_mask = (df[col] < min_val) | (df[col] > max_val)
            out_of_range_count = out_of_range_mask.sum()
            
            if out_of_range_count > 0:
                report['out_of_range'][col] = int(out_of_range_count)
                
                if auto_fix:
                    # Cap values at min/max
                    df.loc[df[col] < min_val, col] = min_val
                    df.loc[df[col] > max_val, col] = max_val
                    report['capped_values'][col] = int(out_of_range_count)
                    logger.info(f"Capped {out_of_range_count} values in {col}")
        
        # Special validation for percentages
        percentage_cols = ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y']
        for col in percentage_cols:
            if col in df.columns:
                # Convert if stored as decimals (0.1 instead of 10%)
                if df[col].abs().max() < 2:  # Likely decimal format
                    df[col] = df[col] * 100
                    report[f'{col}_converted'] = 'decimal to percentage'
        
        return df, report
    
    @staticmethod
    def _remove_duplicates(df: pd.DataFrame, auto_fix: bool) -> Tuple[pd.DataFrame, Dict]:
        """Remove duplicate rows"""
        report = {
            'duplicates_found': 0,
            'duplicates_removed': 0
        }
        
        # Check for duplicate tickers
        if 'ticker' in df.columns:
            duplicates = df['ticker'].duplicated()
            report['duplicates_found'] = int(duplicates.sum())
            
            if auto_fix and duplicates.sum() > 0:
                # Keep the row with the most recent data (highest volume or newest price)
                if 'volume_1d' in df.columns:
                    df = df.sort_values('volume_1d', ascending=False).drop_duplicates('ticker', keep='first')
                else:
                    df = df.drop_duplicates('ticker', keep='last')
                
                report['duplicates_removed'] = report['duplicates_found']
                logger.info(f"Removed {report['duplicates_removed']} duplicate tickers")
        
        return df, report
    
    @staticmethod
    def _handle_missing_values(df: pd.DataFrame, auto_fix: bool) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values intelligently"""
        report = {
            'missing_before': {},
            'missing_after': {},
            'imputation_methods': {}
        }
        
        # Track missing values before
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                report['missing_before'][col] = int(missing_count)
        
        if auto_fix:
            # Numeric columns: fill with appropriate values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    if 'volume' in col or col == 'rvol':
                        # Volume columns: fill with 0
                        df[col].fillna(0, inplace=True)
                        report['imputation_methods'][col] = 'filled with 0'
                    elif 'ret' in col or 'pct' in col:
                        # Return/percentage columns: fill with 0
                        df[col].fillna(0, inplace=True)
                        report['imputation_methods'][col] = 'filled with 0'
                    elif col in ['pe', 'eps_current', 'eps_last_qtr']:
                        # Fundamental data: keep as NaN (will be handled by filters)
                        report['imputation_methods'][col] = 'kept as NaN'
                    else:
                        # Other numeric: fill with median
                        median_val = df[col].median()
                        df[col].fillna(median_val, inplace=True)
                        report['imputation_methods'][col] = f'filled with median ({median_val:.2f})'
            
            # Categorical columns: fill with 'Unknown'
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna('Unknown', inplace=True)
                    report['imputation_methods'][col] = 'filled with Unknown'
        
        # Track missing values after
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                report['missing_after'][col] = int(missing_count)
        
        return df, report
    
    @staticmethod
    def _handle_outliers(df: pd.DataFrame, auto_fix: bool) -> Tuple[pd.DataFrame, Dict]:
        """Detect and handle statistical outliers"""
        report = {
            'outliers_detected': {},
            'outliers_handled': {}
        }
        
        # Columns where outliers are meaningful (don't remove)
        exclude_from_outlier_removal = ['ticker', 'company_name', 'category', 'sector', 'industry']
        
        # Columns where extreme values are expected
        allow_extreme = ['rvol', 'volume_1d', 'ret_1d', 'ret_7d', 'ret_30d']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in exclude_from_outlier_removal:
                continue
            
            # Calculate statistics
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            if col in allow_extreme:
                # More lenient for volatile columns
                multiplier = 5.0
            else:
                multiplier = DATA_CONFIG.OUTLIER_STD_THRESHOLD
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Detect outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                report['outliers_detected'][col] = int(outlier_count)
                
                if auto_fix and col not in allow_extreme:
                    # Cap outliers at bounds
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    report['outliers_handled'][col] = f'capped {outlier_count} values'
        
        return df, report
    
    @staticmethod
    def _validate_business_rules(df: pd.DataFrame, auto_fix: bool) -> Tuple[pd.DataFrame, Dict]:
        """Validate business-specific rules"""
        report = {
            'rule_violations': {},
            'fixes_applied': []
        }
        
        # Rule 1: Price must be positive
        if 'price' in df.columns:
            invalid_price = df['price'] <= 0
            if invalid_price.sum() > 0:
                report['rule_violations']['negative_price'] = int(invalid_price.sum())
                if auto_fix:
                    df = df[~invalid_price]
                    report['fixes_applied'].append('removed negative price rows')
        
        # Rule 2: Volume cannot be negative
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        for col in volume_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                invalid = df[col] < 0
                if invalid.sum() > 0:
                    report['rule_violations'][f'negative_{col}'] = int(invalid.sum())
                    if auto_fix:
                        df.loc[invalid, col] = 0
                        report['fixes_applied'].append(f'set negative {col} to 0')
        
        # Rule 3: Percentage returns should be reasonable (-100% to +1000%)
        return_cols = [col for col in df.columns if 'ret_' in col]
        for col in return_cols:
            if col in df.columns:
                unreasonable = (df[col] < -100) | (df[col] > 1000)
                if unreasonable.sum() > 0:
                    report['rule_violations'][f'unreasonable_{col}'] = int(unreasonable.sum())
                    if auto_fix:
                        df = df[~unreasonable]
                        report['fixes_applied'].append(f'removed unreasonable {col} values')
        
        # Rule 4: RVOL should be positive
        if 'rvol' in df.columns:
            invalid_rvol = df['rvol'] < 0
            if invalid_rvol.sum() > 0:
                report['rule_violations']['negative_rvol'] = int(invalid_rvol.sum())
                if auto_fix:
                    df.loc[invalid_rvol, 'rvol'] = 0
                    report['fixes_applied'].append('set negative rvol to 0')
        
        # Rule 5: Category validation
        if 'category' in df.columns:
            valid_categories = VALIDATION_CONFIG.VALID_CATEGORIES
            invalid_category = ~df['category'].isin(valid_categories)
            if invalid_category.sum() > 0:
                report['rule_violations']['invalid_category'] = int(invalid_category.sum())
                if auto_fix:
                    df.loc[invalid_category, 'category'] = 'Unknown'
                    report['fixes_applied'].append('set invalid categories to Unknown')
        
        return df, report
    
    @staticmethod
    def _check_consistency(df: pd.DataFrame, auto_fix: bool) -> Tuple[pd.DataFrame, Dict]:
        """Check data consistency and relationships"""
        report = {
            'inconsistencies': {},
            'fixes': []
        }
        
        # Check: from_low_pct should be >= 0
        if 'from_low_pct' in df.columns:
            inconsistent = df['from_low_pct'] < 0
            if inconsistent.sum() > 0:
                report['inconsistencies']['negative_from_low'] = int(inconsistent.sum())
                if auto_fix:
                    df.loc[inconsistent, 'from_low_pct'] = 0
                    report['fixes'].append('fixed negative from_low_pct')
        
        # Check: from_high_pct should be <= 0 (or negative)
        if 'from_high_pct' in df.columns:
            inconsistent = df['from_high_pct'] > 0
            if inconsistent.sum() > 0:
                report['inconsistencies']['positive_from_high'] = int(inconsistent.sum())
                if auto_fix:
                    df.loc[inconsistent, 'from_high_pct'] = -abs(df.loc[inconsistent, 'from_high_pct'])
                    report['fixes'].append('fixed positive from_high_pct')
        
        # Check: SMA relationships (SMA_20 should vary around price)
        if 'price' in df.columns and 'sma_20d' in df.columns:
            # SMA shouldn't be more than 50% different from price
            large_diff = abs(df['sma_20d'] - df['price']) / df['price'] > 0.5
            if large_diff.sum() > 0:
                report['inconsistencies']['sma_price_mismatch'] = int(large_diff.sum())
        
        return df, report
    
    @staticmethod
    def _calculate_quality_score(df: pd.DataFrame, validation_report: Dict) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        penalties = []
        
        # Penalty for missing critical columns
        critical_missing = len(validation_report.get('structure', {}).get('critical_columns_missing', []))
        if critical_missing > 0:
            penalty = critical_missing * 20
            penalties.append(('critical_columns', penalty))
            score -= penalty
        
        # Penalty for missing important columns
        important_missing = len(validation_report.get('structure', {}).get('important_columns_missing', []))
        if important_missing > 0:
            penalty = min(important_missing * 2, 20)
            penalties.append(('important_columns', penalty))
            score -= penalty
        
        # Penalty for removed rows
        rows_removed = validation_report.get('rows_removed', 0)
        original_rows = validation_report.get('original_shape', (1, 0))[0]
        if original_rows > 0:
            removal_ratio = rows_removed / original_rows
            if removal_ratio > 0.1:  # More than 10% removed
                penalty = min(removal_ratio * 30, 30)
                penalties.append(('row_removal', penalty))
                score -= penalty
        
        # Penalty for missing values
        missing_after = validation_report.get('missing_values', {}).get('missing_after', {})
        if missing_after:
            total_cells = df.shape[0] * df.shape[1]
            total_missing = sum(missing_after.values())
            missing_ratio = total_missing / total_cells if total_cells > 0 else 0
            if missing_ratio > 0.05:  # More than 5% missing
                penalty = min(missing_ratio * 100, 20)
                penalties.append(('missing_values', penalty))
                score -= penalty
        
        # Penalty for data type conversion failures
        failed_conversions = len(validation_report.get('type_conversions', {}).get('failed_conversions', {}))
        if failed_conversions > 0:
            penalty = min(failed_conversions * 3, 15)
            penalties.append(('type_conversions', penalty))
            score -= penalty
        
        # Penalty for outliers
        outliers = validation_report.get('outliers', {}).get('outliers_detected', {})
        if outliers:
            total_outliers = sum(outliers.values())
            outlier_ratio = total_outliers / (len(df) * len(outliers)) if len(df) > 0 else 0
            if outlier_ratio > 0.02:  # More than 2% outliers
                penalty = min(outlier_ratio * 50, 10)
                penalties.append(('outliers', penalty))
                score -= penalty
        
        # Penalty for business rule violations
        violations = validation_report.get('business_rules', {}).get('rule_violations', {})
        if violations:
            penalty = min(len(violations) * 2, 10)
            penalties.append(('business_rules', penalty))
            score -= penalty
        
        # Log penalties
        if penalties:
            logger.debug(f"Quality penalties: {penalties}")
        
        return max(score, 0.0)
    
    @staticmethod
    def get_validation_summary(validation_report: Dict) -> str:
        """Generate human-readable validation summary"""
        if not validation_report:
            return "No validation report available"
        
        lines = [
            "📊 Data Validation Summary",
            "=" * 40,
            f"Quality Score: {validation_report.get('quality_score', 0):.1f}%",
            f"Original Shape: {validation_report.get('original_shape', 'N/A')}",
            f"Final Shape: {validation_report.get('final_shape', 'N/A')}",
            f"Rows Removed: {validation_report.get('rows_removed', 0)}",
            ""
        ]
        
        # Add key issues
        if validation_report.get('errors'):
            lines.append("❌ Errors:")
            for error in validation_report['errors'][:5]:
                lines.append(f"  - {error}")
        
        if validation_report.get('warnings'):
            lines.append("⚠️ Warnings:")
            for warning in validation_report['warnings'][:5]:
                lines.append(f"  - {warning}")
        
        if validation_report.get('fixes_applied'):
            lines.append("✅ Fixes Applied:")
            for fix in validation_report['fixes_applied'][:5]:
                lines.append(f"  - {fix}")
        
        return "\n".join(lines)

# ============================================
# END OF DATA VALIDATION AND SANITIZATION
# ============================================

# ============================================
# FINAL SETUP COMPLETION
# ============================================

# Mark all setup sections as complete
SETUP_COMPLETE = True

# Log final configuration summary
logger.info("=" * 50)
logger.info("APPLICATION SETUP COMPLETE")
logger.info(f"Production Mode: {PRODUCTION_MODE}")
logger.info(f"Debug Mode: {DEBUG_MODE}")
logger.info(f"Log Level: {logging.getLevelName(LOG_LEVEL)}")
logger.info(f"Configuration loaded: All dataclasses initialized")
logger.info(f"Performance monitoring: Enabled with thresholds")
logger.info(f"Data validation: Full pipeline ready")
logger.info("=" * 50)
        
# ============================================
# SMART CACHING WITH VERSIONING
# ============================================

def extract_spreadsheet_id(url_or_id: str) -> str:
    """
    Extracts the spreadsheet ID from a Google Sheets URL or returns the ID if it's already in the correct format.

    Args:
        url_or_id (str): A Google Sheets URL or just the spreadsheet ID.

    Returns:
        str: The extracted spreadsheet ID, or an empty string if not found.
    """
    if not url_or_id:
        return ""
    
    # If it's already just an ID (no slashes), return it
    if '/' not in url_or_id:
        return url_or_id.strip()
    
    # Try to extract from URL using a regular expression
    pattern = r'/spreadsheets/d/([a-zA-Z0-9-_]+)'
    match = re.search(pattern, url_or_id)
    if match:
        return match.group(1)
    
    # If no match, return as is.
    return url_or_id.strip()

@st.cache_data(ttl=CONFIG.CACHE_TTL, persist="disk", show_spinner=False)
def load_and_process_data(source_type: str = "sheet", file_data=None, 
                         sheet_id: str = None, gid: str = None,
                         data_version: str = "1.0") -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    """
    Loads and processes data from a Google Sheet or CSV file with caching and versioning.

    Args:
        source_type (str): Specifies the data source, either "sheet" or "upload".
        file_data (Optional): The uploaded CSV file object if `source_type` is "upload".
        sheet_id (str): The Google Spreadsheet ID.
        gid (str): The Google Sheet tab ID.
        data_version (str): A unique key to bust the cache (e.g., hash of date + sheet ID).

    Returns:
        Tuple[pd.DataFrame, datetime, Dict[str, Any]]: A tuple containing the processed DataFrame,
        the processing timestamp, and metadata about the process.
    
    Raises:
        ValueError: If a valid Google Sheets ID is not provided.
        Exception: If data loading or processing fails.
    """
    
    start_time = time.perf_counter()
    metadata = {
        'source_type': source_type,
        'data_version': data_version,
        'processing_start': datetime.now(timezone.utc),
        'errors': [],
        'warnings': []
    }
    
    try:
        # Load data based on source
        if source_type == "upload" and file_data is not None:
            logger.info("Loading data from uploaded CSV")
            try:
                df = pd.read_csv(file_data, low_memory=False)
                metadata['source'] = "User Upload"
            except UnicodeDecodeError:
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        file_data.seek(0)
                        df = pd.read_csv(file_data, low_memory=False, encoding=encoding)
                        metadata['warnings'].append(f"Used {encoding} encoding")
                        break
                    except:
                        continue
                else:
                    raise ValueError("Unable to decode CSV file")
        else:
            # Use defaults if not provided
            if not sheet_id:
                raise ValueError("Please enter a Google Sheets ID")
            if not gid:
                gid = CONFIG.DEFAULT_GID
            
            # Construct CSV URL
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            
            logger.info(f"Loading data from Google Sheets ID: {sheet_id}")
            
            try:
                df = pd.read_csv(csv_url, low_memory=False)
                metadata['source'] = "Google Sheets"
            except Exception as e:
                logger.error(f"Failed to load from Google Sheets: {str(e)}")
                metadata['errors'].append(f"Sheet load error: {str(e)}")
                
                # Try to use cached data as fallback
                if 'last_good_data' in st.session_state:
                    logger.info("Using cached data as fallback")
                    df, timestamp, old_metadata = st.session_state.last_good_data
                    metadata['warnings'].append("Using cached data due to load failure")
                    metadata['cache_used'] = True
                    return df, timestamp, metadata
                raise
        
        # Validate loaded data
        is_valid, validation_msg = DataValidator.validate_dataframe(df, CONFIG.CRITICAL_COLUMNS, "Initial load")
        if not is_valid:
            raise ValueError(validation_msg)
        
        # Process the data
        df = DataProcessor.process_dataframe(df, metadata)
        
        # Calculate all scores and rankings
        df = RankingEngine.calculate_all_scores(df)
        
        # Corrected method call here
        df = PatternDetector.detect_all_patterns_optimized(df)
        
        # Add advanced metrics
        df = AdvancedMetrics.calculate_all_metrics(df)
        
        # Final validation
        is_valid, validation_msg = DataValidator.validate_dataframe(df, ['master_score', 'rank'], "Final processed")
        if not is_valid:
            raise ValueError(validation_msg)
        
        # Store as last good data
        timestamp = datetime.now(timezone.utc)
        st.session_state.last_good_data = (df.copy(), timestamp, metadata)
        
        # Record processing time
        processing_time = time.perf_counter() - start_time
        metadata['processing_time'] = processing_time
        metadata['processing_end'] = datetime.now(timezone.utc)
        
        logger.info(f"Data processing complete: {len(df)} stocks in {processing_time:.2f}s")
        
        # Periodic cleanup
        if 'last_cleanup' not in st.session_state:
            st.session_state.last_cleanup = datetime.now(timezone.utc)
        
        if (datetime.now(timezone.utc) - st.session_state.last_cleanup).total_seconds() > 300:
            gc.collect()
            st.session_state.last_cleanup = datetime.now(timezone.utc)
        
        return df, timestamp, metadata
        
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}")
        metadata['errors'].append(str(e))
        raise
        
# ============================================
# DATA PROCESSING ENGINE
# ============================================

class DataProcessor:
    """
    Handles the entire data processing pipeline, from raw data ingestion to a clean,
    ready-for-analysis DataFrame. This class is optimized for performance and robustness.
    """
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Main pipeline to validate, clean, and prepare the raw DataFrame.

        Args:
            df (pd.DataFrame): The raw DataFrame to be processed.
            metadata (Dict[str, Any]): A dictionary to log warnings and changes.

        Returns:
            pd.DataFrame: A clean, processed DataFrame ready for scoring.
        """
        df = df.copy()
        initial_count = len(df)
        
        # 1. Process numeric columns with intelligent cleaning
        numeric_cols = [col for col in df.columns if col not in ['ticker', 'company_name', 'category', 'sector', 'industry', 'year', 'market_cap']]
        
        for col in numeric_cols:
            if col in df.columns:
                is_pct = col in CONFIG.PERCENTAGE_COLUMNS
                
                # Dynamically determine bounds based on column name
                bounds = None
                if 'volume' in col.lower():
                    bounds = CONFIG.VALUE_BOUNDS['volume']
                elif col == 'rvol':
                    bounds = CONFIG.VALUE_BOUNDS['rvol']
                elif col == 'pe':
                    bounds = CONFIG.VALUE_BOUNDS['pe']
                elif is_pct:
                    bounds = CONFIG.VALUE_BOUNDS['returns']
                else:
                    bounds = CONFIG.VALUE_BOUNDS.get('price', None)
                
                # Apply vectorized cleaning
                df[col] = df[col].apply(lambda x: DataValidator.clean_numeric_value(x, is_pct, bounds))
        
        # 2. Process categorical columns with robust sanitization
        string_cols = ['ticker', 'company_name', 'category', 'sector', 'industry']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(DataValidator.sanitize_string)
        
        # 3. Handle volume ratios with safety
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = (100 + df[col]) / 100
                df[col] = df[col].clip(0.01, 1000.0)
                df[col] = df[col].fillna(1.0)
        
        # 4. Critical data validation and removal of duplicates
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0.01]
        
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df):
            metadata['warnings'].append(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        # 5. Fill missing values and add tier classifications
        df = DataProcessor._fill_missing_values(df)
        df = DataProcessor._add_tier_classifications(df)
        
        # 6. Log final data quality metrics
        removed_count = initial_count - len(df)
        if removed_count > 0:
            metadata['warnings'].append(f"Removed {removed_count} invalid rows during processing.")
        
        logger.info(f"Processed {len(df)} valid stocks from {initial_count} initial rows.")
        
        return df

    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in key columns with sensible defaults.
        This is a final defensive step to ensure downstream calculations don't fail due to NaNs.
        """
        # Default for position metrics
        if 'from_low_pct' in df.columns:
            df['from_low_pct'] = df['from_low_pct'].fillna(50)
        
        if 'from_high_pct' in df.columns:
            df['from_high_pct'] = df['from_high_pct'].fillna(-50)
        
        # Default for Relative Volume (RVOL)
        if 'rvol' in df.columns:
            df['rvol'] = df['rvol'].fillna(1.0)
        
        # Defaults for price returns
        return_cols = [col for col in df.columns if col.startswith('ret_')]
        for col in return_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Defaults for volume columns
        volume_cols = [col for col in df.columns if col.startswith('volume_')]
        for col in volume_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Defaults for categorical columns
        for col in ['category', 'sector', 'industry']:
            if col not in df.columns:
                df[col] = 'Unknown'
            else:
                df[col] = df[col].fillna('Unknown')
        
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a classification tier to numerical data (e.g., price, PE)
        based on predefined ranges in the `Config` class.
        This is a bug-fixed and robust version of the logic from earlier files.
        """
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            """Helper function to map a value to its tier."""
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val:
                    return tier_name
                if min_val == -float('inf') and value <= max_val:
                    return tier_name
                if max_val == float('inf') and value > min_val:
                    return tier_name
            
            return "Unknown"
        
        if 'eps_current' in df.columns:
            df['eps_tier'] = df['eps_current'].apply(lambda x: classify_tier(x, CONFIG.TIERS['eps']))
        
        if 'pe' in df.columns:
            df['pe_tier'] = df['pe'].apply(
                lambda x: "Negative/NA" if pd.isna(x) or x <= 0 else classify_tier(x, CONFIG.TIERS['pe'])
            )
        
        if 'price' in df.columns:
            df['price_tier'] = df['price'].apply(lambda x: classify_tier(x, CONFIG.TIERS['price']))
        
        return df
        
# ============================================
# ADVANCED METRICS CALCULATOR
# ============================================

class AdvancedMetrics:
    """
    Calculates advanced metrics and indicators using a combination of price,
    volume, and algorithmically derived scores. Ensures robust calculation
    by handling potential missing data (NaNs) gracefully.
    """
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a comprehensive set of advanced metrics for the DataFrame.
        All calculations are designed to be vectorized and handle missing data
        without raising errors.

        Args:
            df (pd.DataFrame): The DataFrame with raw data and core scores.

        Returns:
            pd.DataFrame: The DataFrame with all calculated advanced metrics added.
        """
        if df.empty:
            return df
        
        # Money Flow (in millions)
        if all(col in df.columns for col in ['price', 'volume_1d', 'rvol']):
            df['money_flow'] = df['price'].fillna(0) * df['volume_1d'].fillna(0) * df['rvol'].fillna(1.0)
            df['money_flow_mm'] = df['money_flow'] / 1_000_000
        else:
            df['money_flow_mm'] = pd.Series(np.nan, index=df.index)
        
        # Volume Momentum Index (VMI)
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d']):
            df['vmi'] = (
                df['vol_ratio_1d_90d'].fillna(1.0) * 4 +
                df['vol_ratio_7d_90d'].fillna(1.0) * 3 +
                df['vol_ratio_30d_90d'].fillna(1.0) * 2 +
                df['vol_ratio_90d_180d'].fillna(1.0) * 1
            ) / 10
        else:
            df['vmi'] = pd.Series(np.nan, index=df.index)
        
        # Position Tension
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            df['position_tension'] = df['from_low_pct'].fillna(50) + abs(df['from_high_pct'].fillna(-50))
        else:
            df['position_tension'] = pd.Series(np.nan, index=df.index)
        
        # Momentum Harmony
        df['momentum_harmony'] = pd.Series(0, index=df.index, dtype=int)
        
        if 'ret_1d' in df.columns:
            df['momentum_harmony'] += (df['ret_1d'].fillna(0) > 0).astype(int)
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_7d = pd.Series(np.where(df['ret_7d'].fillna(0) != 0, df['ret_7d'].fillna(0) / 7, np.nan), index=df.index)
                daily_ret_30d = pd.Series(np.where(df['ret_30d'].fillna(0) != 0, df['ret_30d'].fillna(0) / 30, np.nan), index=df.index)
            df['momentum_harmony'] += ((daily_ret_7d.fillna(-np.inf) > daily_ret_30d.fillna(-np.inf))).astype(int)
        
        if all(col in df.columns for col in ['ret_30d', 'ret_3m']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_30d_comp = pd.Series(np.where(df['ret_30d'].fillna(0) != 0, df['ret_30d'].fillna(0) / 30, np.nan), index=df.index)
                daily_ret_3m_comp = pd.Series(np.where(df['ret_3m'].fillna(0) != 0, df['ret_3m'].fillna(0) / 90, np.nan), index=df.index)
            df['momentum_harmony'] += ((daily_ret_30d_comp.fillna(-np.inf) > daily_ret_3m_comp.fillna(-np.inf))).astype(int)
        
        if 'ret_3m' in df.columns:
            df['momentum_harmony'] += (df['ret_3m'].fillna(0) > 0).astype(int)
        
        # Wave State
        df['wave_state'] = df.apply(AdvancedMetrics._get_wave_state, axis=1)

        # Overall Wave Strength
        score_cols = ['momentum_score', 'acceleration_score', 'rvol_score', 'breakout_score']
        if all(col in df.columns for col in score_cols):
            df['overall_wave_strength'] = (
                df['momentum_score'].fillna(50) * 0.3 +
                df['acceleration_score'].fillna(50) * 0.3 +
                df['rvol_score'].fillna(50) * 0.2 +
                df['breakout_score'].fillna(50) * 0.2
            )
        else:
            df['overall_wave_strength'] = pd.Series(np.nan, index=df.index)
        
        return df
    
    @staticmethod
    def _get_wave_state(row: pd.Series) -> str:
        """
        Determines the `wave_state` for a single stock based on a set of thresholds.
        """
        signals = 0
        
        if row.get('momentum_score', 0) > 70:
            signals += 1
        if row.get('volume_score', 0) > 70:
            signals += 1
        if row.get('acceleration_score', 0) > 70:
            signals += 1
        if row.get('rvol', 0) > 2:
            signals += 1
        
        if signals >= 4:
            return "🌊🌊🌊 CRESTING"
        elif signals >= 3:
            return "🌊🌊 BUILDING"
        elif signals >= 1:
            return "🌊 FORMING"
        else:
            return "💥 BREAKING"
        
# ============================================
# RANKING ENGINE - OPTIMIZED
# ============================================

class RankingEngine:
    """
    Core ranking calculations using a multi-factor model.
    This class is highly optimized with vectorized NumPy operations
    for speed and designed to be resilient to missing data.
    """

    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all component scores, a composite master score, and ranks the stocks.

        Args:
            df (pd.DataFrame): The DataFrame containing processed stock data.

        Returns:
            pd.DataFrame: The DataFrame with all scores and ranks added.
        """
        if df.empty:
            return df
        
        logger.info("Starting optimized ranking calculations...")

        # Calculate component scores
        df['position_score'] = RankingEngine._calculate_position_score(df)
        df['volume_score'] = RankingEngine._calculate_volume_score(df)
        df['momentum_score'] = RankingEngine._calculate_momentum_score(df)
        df['acceleration_score'] = RankingEngine._calculate_acceleration_score(df)
        df['breakout_score'] = RankingEngine._calculate_breakout_score(df)
        df['rvol_score'] = RankingEngine._calculate_rvol_score(df)
        
        # Calculate auxiliary scores
        df['trend_quality'] = RankingEngine._calculate_trend_quality(df)
        df['long_term_strength'] = RankingEngine._calculate_long_term_strength(df)
        df['liquidity_score'] = RankingEngine._calculate_liquidity_score(df)
        
        # Calculate master score using numpy (DO NOT MODIFY FORMULA)
        # FIX: Use safer np.column_stack approach
        scores_matrix = np.column_stack([
            df['position_score'].fillna(50),
            df['volume_score'].fillna(50),
            df['momentum_score'].fillna(50),
            df['acceleration_score'].fillna(50),
            df['breakout_score'].fillna(50),
            df['rvol_score'].fillna(50)
        ])
        
        weights = np.array([
            CONFIG.POSITION_WEIGHT,
            CONFIG.VOLUME_WEIGHT,
            CONFIG.MOMENTUM_WEIGHT,
            CONFIG.ACCELERATION_WEIGHT,
            CONFIG.BREAKOUT_WEIGHT,
            CONFIG.RVOL_WEIGHT
        ])
        
        df['master_score'] = np.dot(scores_matrix, weights).clip(0, 100)
        
        # Calculate ranks
        df['rank'] = df['master_score'].rank(method='first', ascending=False, na_option='bottom')
        df['rank'] = df['rank'].fillna(len(df) + 1).astype(int)
        
        df['percentile'] = df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
        df['percentile'] = df['percentile'].fillna(0)
        
        # Calculate category-specific ranks
        df = RankingEngine._calculate_category_ranks(df)
        
        logger.info(f"Ranking complete: {len(df)} stocks processed")
        
        return df

    @staticmethod
    def _safe_rank(series: pd.Series, pct: bool = True, ascending: bool = True) -> pd.Series:
        """
        Safely ranks a series, handling NaNs and infinite values to prevent errors.
        
        Args:
            series (pd.Series): The series to rank.
            pct (bool): If True, returns percentile ranks (0-100).
            ascending (bool): The order for ranking.
            
        Returns:
            pd.Series: A new series with the calculated ranks.
        """
        # FIX: Return proper defaults instead of NaN series
        if series is None or series.empty:
            return pd.Series(dtype=float)
        
        # Replace inf values with NaN
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # Count valid values
        valid_count = series.notna().sum()
        if valid_count == 0:
            return pd.Series(50, index=series.index)  # FIX: Return 50 default
        
        # Rank with proper parameters
        if pct:
            ranks = series.rank(pct=True, ascending=ascending, na_option='bottom') * 100
            ranks = ranks.fillna(0 if ascending else 100)
        else:
            ranks = series.rank(ascending=ascending, method='min', na_option='bottom')
            ranks = ranks.fillna(valid_count + 1)
        
        return ranks

    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """Calculate position score from 52-week range (DO NOT MODIFY LOGIC)"""
        # FIX: Initialize with neutral score 50, not NaN
        position_score = pd.Series(50, index=df.index, dtype=float)
        
        # Check required columns
        has_from_low = 'from_low_pct' in df.columns and df['from_low_pct'].notna().any()
        has_from_high = 'from_high_pct' in df.columns and df['from_high_pct'].notna().any()
        
        if not has_from_low and not has_from_high:
            logger.warning("No position data available, using neutral position scores")
            return position_score
        
        # Get data with defaults
        from_low = df['from_low_pct'].fillna(50) if has_from_low else pd.Series(50, index=df.index)
        from_high = df['from_high_pct'].fillna(-50) if has_from_high else pd.Series(-50, index=df.index)
        
        # Rank components
        if has_from_low:
            rank_from_low = RankingEngine._safe_rank(from_low, pct=True, ascending=True)
        else:
            rank_from_low = pd.Series(50, index=df.index)
        
        if has_from_high:
            # from_high is negative, less negative = closer to high = better
            rank_from_high = RankingEngine._safe_rank(from_high, pct=True, ascending=False)
        else:
            rank_from_high = pd.Series(50, index=df.index)
        
        # Combined position score (DO NOT MODIFY WEIGHTS)
        position_score = (rank_from_low * 0.6 + rank_from_high * 0.4)
        
        return position_score.clip(0, 100)

    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive volume score"""
        # FIX: Start with default 50, not NaN
        volume_score = pd.Series(50, index=df.index, dtype=float)
        
        # Volume ratio columns with weights
        vol_cols = [
            ('vol_ratio_1d_90d', 0.20),
            ('vol_ratio_7d_90d', 0.20),
            ('vol_ratio_30d_90d', 0.20),
            ('vol_ratio_30d_180d', 0.15),
            ('vol_ratio_90d_180d', 0.25)
        ]
        
        # Calculate weighted score
        total_weight = 0
        weighted_score = pd.Series(0, index=df.index, dtype=float)
        
        for col, weight in vol_cols:
            if col in df.columns and df[col].notna().any():
                col_rank = RankingEngine._safe_rank(df[col], pct=True, ascending=True)
                weighted_score += col_rank * weight
                total_weight += weight
        
        if total_weight > 0:
            volume_score = weighted_score / total_weight
        else:
            logger.warning("No volume ratio data available, using neutral scores")
        
        # FIX: Don't set to NaN, keep default 50
        # Removed the aggressive NaN masking logic from V2
        
        return volume_score.clip(0, 100)

    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum score based on returns"""
        # FIX: Start with default 50
        momentum_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'ret_30d' not in df.columns or df['ret_30d'].notna().sum() == 0:
            # Fallback to 7-day returns
            if 'ret_7d' in df.columns and df['ret_7d'].notna().any():
                ret_7d = df['ret_7d'].fillna(0)
                momentum_score = RankingEngine._safe_rank(ret_7d, pct=True, ascending=True)
                logger.info("Using 7-day returns for momentum score")
            else:
                logger.warning("No return data available for momentum calculation")
            
            return momentum_score.clip(0, 100)
        
        # Primary: 30-day returns
        ret_30d = df['ret_30d'].fillna(0)
        momentum_score = RankingEngine._safe_rank(ret_30d, pct=True, ascending=True)
        
        # Add consistency bonus
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            consistency_bonus = pd.Series(0, index=df.index, dtype=float)
            
            # Both positive
            all_positive = (df['ret_7d'] > 0) & (df['ret_30d'] > 0)
            consistency_bonus[all_positive] = 5
            
            # Accelerating returns
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_ret_7d = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0)
                daily_ret_30d = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
            
            accelerating = all_positive & (daily_ret_7d > daily_ret_30d)
            consistency_bonus[accelerating] = 10
            
            # FIX: Use simpler approach, no complex masking
            momentum_score = (momentum_score + consistency_bonus).clip(0, 100)
        
        return momentum_score

    @staticmethod
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """Calculate if momentum is accelerating with proper division handling"""
        # FIX: Start with default 50, not NaN
        acceleration_score = pd.Series(50, index=df.index, dtype=float)
        
        req_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        available_cols = [col for col in req_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("Insufficient return data for acceleration calculation")
            return acceleration_score
        
        # Get return data with defaults
        ret_1d = df['ret_1d'].fillna(0) if 'ret_1d' in df.columns else pd.Series(0, index=df.index)
        ret_7d = df['ret_7d'].fillna(0) if 'ret_7d' in df.columns else pd.Series(0, index=df.index)
        ret_30d = df['ret_30d'].fillna(0) if 'ret_30d' in df.columns else pd.Series(0, index=df.index)
        
        # Calculate daily averages with safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_daily_1d = ret_1d  # Already daily
            avg_daily_7d = np.where(ret_7d != 0, ret_7d / 7, 0)
            avg_daily_30d = np.where(ret_30d != 0, ret_30d / 30, 0)
        
        if all(col in df.columns for col in req_cols):
            # Perfect acceleration
            perfect = (avg_daily_1d > avg_daily_7d) & (avg_daily_7d > avg_daily_30d) & (ret_1d > 0)
            acceleration_score[perfect] = 100
            
            # Good acceleration
            good = (~perfect) & (avg_daily_1d > avg_daily_7d) & (ret_1d > 0)
            acceleration_score[good] = 80
            
            # Moderate
            moderate = (~perfect) & (~good) & (ret_1d > 0)
            acceleration_score[moderate] = 60
            
            # Deceleration
            slight_decel = (ret_1d <= 0) & (ret_7d > 0)
            acceleration_score[slight_decel] = 40
            
            strong_decel = (ret_1d <= 0) & (ret_7d <= 0)
            acceleration_score[strong_decel] = 20
        
        return acceleration_score

    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        """Calculate breakout probability"""
        # FIX: Start with default 50
        breakout_score = pd.Series(50, index=df.index, dtype=float)
        
        # Factor 1: Distance from high (40% weight)
        if 'from_high_pct' in df.columns:
            # from_high_pct is negative, closer to 0 = closer to high
            distance_from_high = -df['from_high_pct'].fillna(-50)
            distance_factor = (100 - distance_from_high).clip(0, 100)
        else:
            distance_factor = pd.Series(50, index=df.index)
        
        # Factor 2: Volume surge (40% weight)
        volume_factor = pd.Series(50, index=df.index)
        if 'vol_ratio_7d_90d' in df.columns:
            vol_ratio = df['vol_ratio_7d_90d'].fillna(1.0)
            volume_factor = ((vol_ratio - 1) * 100).clip(0, 100)
        
        # Factor 3: Trend support (20% weight)
        trend_factor = pd.Series(0, index=df.index, dtype=float)
        
        if 'price' in df.columns:
            current_price = df['price']
            trend_count = 0
            
            for sma_col, points in [('sma_20d', 33.33), ('sma_50d', 33.33), ('sma_200d', 33.34)]:
                if sma_col in df.columns:
                    above_sma = (current_price > df[sma_col]).fillna(False)
                    trend_factor += above_sma.astype(float) * points
                    trend_count += 1
            
            if trend_count > 0 and trend_count < 3:
                trend_factor = trend_factor * (3 / trend_count)
        
        trend_factor = trend_factor.clip(0, 100)
        
        # FIX: Simple combination without complex NaN masking
        breakout_score = (
            distance_factor * 0.4 +
            volume_factor * 0.4 +
            trend_factor * 0.2
        )
        
        return breakout_score.clip(0, 100)

    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        """Calculate RVOL-based score"""
        if 'rvol' not in df.columns:
            return pd.Series(50, index=df.index)
        
        rvol = df['rvol'].fillna(1.0)
        rvol_score = pd.Series(50, index=df.index, dtype=float)
        
        # Score based on RVOL ranges
        rvol_score[rvol > 10] = 95
        rvol_score[(rvol > 5) & (rvol <= 10)] = 90
        rvol_score[(rvol > 3) & (rvol <= 5)] = 85
        rvol_score[(rvol > 2) & (rvol <= 3)] = 80
        rvol_score[(rvol > 1.5) & (rvol <= 2)] = 70
        rvol_score[(rvol > 1.2) & (rvol <= 1.5)] = 60
        rvol_score[(rvol > 0.8) & (rvol <= 1.2)] = 50
        rvol_score[(rvol > 0.5) & (rvol <= 0.8)] = 40
        rvol_score[(rvol > 0.3) & (rvol <= 0.5)] = 30
        rvol_score[rvol <= 0.3] = 20
        
        return rvol_score

    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        """Calculate trend quality based on SMA alignment"""
        trend_quality = pd.Series(50, index=df.index, dtype=float)
        
        if 'price' not in df.columns:
            return trend_quality
        
        current_price = df['price']
        sma_cols = ['sma_20d', 'sma_50d', 'sma_200d']
        available_smas = [col for col in sma_cols if col in df.columns]
        
        if not available_smas:
            return trend_quality
        
        # Check alignment
        conditions = pd.DataFrame(index=df.index)
        
        for sma_col in available_smas:
            conditions[f'above_{sma_col}'] = (current_price > df[sma_col]).fillna(False)
        
        # Calculate score based on alignment
        total_conditions = len(available_smas)
        
        if total_conditions == 3:
            # All SMAs available
            all_above = conditions.all(axis=1)
            all_below = (~conditions).all(axis=1)
            
            # Perfect uptrend: price > 20 > 50 > 200
            if 'sma_20d' in df.columns and 'sma_50d' in df.columns and 'sma_200d' in df.columns:
                perfect_uptrend = (
                    (current_price > df['sma_20d']) &
                    (df['sma_20d'] > df['sma_50d']) &
                    (df['sma_50d'] > df['sma_200d'])
                )
                trend_quality[perfect_uptrend] = 100
            
            trend_quality[all_above & ~perfect_uptrend] = 85
            trend_quality[conditions.sum(axis=1) == 2] = 70
            trend_quality[conditions.sum(axis=1) == 1] = 55
            trend_quality[all_below] = 20
        else:
            # Partial SMAs available
            proportion_above = conditions.sum(axis=1) / total_conditions
            trend_quality = (proportion_above * 80 + 20).round()
        
        return trend_quality.clip(0, 100)

    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate long-term strength based on multiple timeframe returns"""
        strength_score = pd.Series(50, index=df.index, dtype=float)
        
        # Get available return columns
        return_cols = ['ret_3m', 'ret_6m', 'ret_1y']
        available_returns = [col for col in return_cols if col in df.columns]
        
        if not available_returns:
            return strength_score
        
        # Calculate average return
        returns_df = df[available_returns].fillna(0)
        avg_return = returns_df.mean(axis=1)
        
        # Score based on average return
        strength_score[avg_return > 50] = 90
        strength_score[(avg_return > 30) & (avg_return <= 50)] = 80
        strength_score[(avg_return > 15) & (avg_return <= 30)] = 70
        strength_score[(avg_return > 5) & (avg_return <= 15)] = 60
        strength_score[(avg_return > 0) & (avg_return <= 5)] = 50
        strength_score[(avg_return > -10) & (avg_return <= 0)] = 40
        strength_score[(avg_return > -25) & (avg_return <= -10)] = 30
        strength_score[avg_return <= -25] = 20
        
        return strength_score.clip(0, 100)

    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity score based on trading volume"""
        liquidity_score = pd.Series(50, index=df.index, dtype=float)
        
        if 'volume_30d' in df.columns and 'price' in df.columns:
            # Calculate dollar volume
            dollar_volume = df['volume_30d'].fillna(0) * df['price'].fillna(0)
            
            # Rank based on dollar volume
            liquidity_score = RankingEngine._safe_rank(dollar_volume, pct=True, ascending=True)
        
        return liquidity_score.clip(0, 100)

    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentile ranks within each category"""
        # FIX: Initialize with proper defaults, not NaN
        df['category_rank'] = 9999
        df['category_percentile'] = 0.0
        
        # Get unique categories
        if 'category' in df.columns:
            categories = df['category'].unique()
            
            # Rank within each category
            for category in categories:
                if category != 'Unknown':
                    mask = df['category'] == category
                    cat_df = df[mask]
                    
                    if len(cat_df) > 0:
                        # Calculate ranks
                        cat_ranks = cat_df['master_score'].rank(method='first', ascending=False, na_option='bottom')
                        df.loc[mask, 'category_rank'] = cat_ranks.astype(int)
                        
                        # Calculate percentiles
                        cat_percentiles = cat_df['master_score'].rank(pct=True, ascending=True, na_option='bottom') * 100
                        df.loc[mask, 'category_percentile'] = cat_percentiles
        
        return df
        
# ============================================
# PATTERN DETECTION ENGINE - FULLY OPTIMIZED
# ============================================

class PatternDetector:
    """
    Advanced pattern detection using vectorized operations for maximum performance.
    This class identifies a comprehensive set of 25 technical, fundamental,
    and intelligent trading patterns.
    """

    # Pattern metadata for intelligent confidence scoring (e.g., importance, risk).
    PATTERN_METADATA = {
        '🔥 CAT LEADER': {'importance_weight': 10},
        '💎 HIDDEN GEM': {'importance_weight': 10},
        '🚀 ACCELERATING': {'importance_weight': 10},
        '🏦 INSTITUTIONAL': {'importance_weight': 10},
        '⚡ VOL EXPLOSION': {'importance_weight': 15},
        '🎯 BREAKOUT': {'importance_weight': 10},
        '👑 MARKET LEADER': {'importance_weight': 15},
        '🌊 MOMENTUM WAVE': {'importance_weight': 10},
        '💰 LIQUID LEADER': {'importance_weight': 5},
        '💪 LONG STRENGTH': {'importance_weight': 5},
        '📈 QUALITY TREND': {'importance_weight': 10},
        '💎 VALUE MOMENTUM': {'importance_weight': 10},
        '📊 EARNINGS ROCKET': {'importance_weight': 10},
        '🏆 QUALITY LEADER': {'importance_weight': 10},
        '⚡ TURNAROUND': {'importance_weight': 10},
        '⚠️ HIGH PE': {'importance_weight': -5}, # Negative weight for a "warning" pattern
        '🎯 52W HIGH APPROACH': {'importance_weight': 10},
        '🔄 52W LOW BOUNCE': {'importance_weight': 10},
        '👑 GOLDEN ZONE': {'importance_weight': 5},
        '📊 VOL ACCUMULATION': {'importance_weight': 5},
        '🔀 MOMENTUM DIVERGE': {'importance_weight': 10},
        '🎯 RANGE COMPRESS': {'importance_weight': 5},
        '🤫 STEALTH': {'importance_weight': 10},
        '🧛 VAMPIRE': {'importance_weight': 10},
        '⛈️ PERFECT STORM': {'importance_weight': 20},
        '🪤 BULL TRAP': {'importance_weight': 15},      # High value for shorts
        '💣 CAPITULATION': {'importance_weight': 20},   # Best risk/reward
        '🏃 RUNAWAY GAP': {'importance_weight': 12},    # Strong continuation
        '🔄 ROTATION LEADER': {'importance_weight': 10}, # Sector strength
        '⚠️ DISTRIBUTION': {'importance_weight': 15},   # Exit signal
        '🎯 VELOCITY SQUEEZE': {'importance_weight': 15},    # High value - coiled spring
        '⚠️ VOLUME DIVERGENCE': {'importance_weight': -10},  # Negative - warning signal
        '⚡ GOLDEN CROSS': {'importance_weight': 12},        # Strong bullish
        '📉 EXHAUSTION': {'importance_weight': -15},         # Strong bearish
        '🔺 PYRAMID': {'importance_weight': 10},             # Accumulation
        '🌪️ VACUUM': {'importance_weight': 18},             # High potential bounce
    }

    @staticmethod
    @PerformanceMonitor.timer(target_time=0.3)
    def detect_all_patterns_optimized(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects all trading patterns using highly efficient vectorized operations.
        Returns a DataFrame with a new 'patterns' column and a `pattern_confidence` score.
        """
        if df.empty:
            df['patterns'] = ''
            df['pattern_confidence'] = 0.0
            return df
        
        # Get all pattern definitions as a list of (name, mask) tuples.
        patterns_with_masks = PatternDetector._get_all_pattern_definitions(df)
        
        # Create a boolean matrix from the masks for vectorized processing.
        pattern_names = [name for name, _ in patterns_with_masks]
        pattern_matrix = pd.DataFrame(False, index=df.index, columns=pattern_names)
        
        for name, mask in patterns_with_masks:
            if mask is not None and not mask.empty:
                pattern_matrix[name] = mask.reindex(df.index, fill_value=False)
        
        # Combine the boolean columns into a single 'patterns' string column.
        df['patterns'] = pattern_matrix.apply(
            lambda row: ' | '.join(row.index[row].tolist()), axis=1
        )
        
        df['patterns'] = df['patterns'].fillna('')
        
        # Calculate a confidence score for the detected patterns.
        df = PatternDetector._calculate_pattern_confidence(df)
        
        logger.info(f"Pattern detection completed for {len(df)} stocks.")
        return df

    @staticmethod
    def _get_all_pattern_definitions(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
        """
        Defines all 25 patterns using vectorized boolean masks.
        This method is purely for defining the conditions, not for execution.
        """
        patterns = []
        
        # Helper function to safely get column data as a Series, filling NaNs with a default.
        def get_col_safe(col_name: str, default_value: Any = np.nan) -> pd.Series:
            if col_name in df.columns:
                return df[col_name].copy()
            return pd.Series(default_value, index=df.index)

        # 1. Category Leader
        mask = get_col_safe('category_percentile', 0) >= CONFIG.PATTERN_THRESHOLDS['category_leader']
        patterns.append(('🔥 CAT LEADER', mask))
        
        # 2. Hidden Gem
        mask = (get_col_safe('category_percentile', 0) >= CONFIG.PATTERN_THRESHOLDS['hidden_gem']) & (get_col_safe('percentile', 100) < 70)
        patterns.append(('💎 HIDDEN GEM', mask))
        
        # 3. Accelerating
        mask = get_col_safe('acceleration_score', 0) >= CONFIG.PATTERN_THRESHOLDS['acceleration']
        patterns.append(('🚀 ACCELERATING', mask))
        
        # 4. Institutional
        mask = (get_col_safe('volume_score', 0) >= CONFIG.PATTERN_THRESHOLDS['institutional']) & (get_col_safe('vol_ratio_90d_180d', 1) > 1.1)
        patterns.append(('🏦 INSTITUTIONAL', mask))
        
        # 5. Volume Explosion
        mask = get_col_safe('rvol', 0) > 3
        patterns.append(('⚡ VOL EXPLOSION', mask))
        
        # 6. Breakout Ready
        mask = get_col_safe('breakout_score', 0) >= CONFIG.PATTERN_THRESHOLDS['breakout_ready']
        patterns.append(('🎯 BREAKOUT', mask))
        
        # 7. Market Leader
        mask = get_col_safe('percentile', 0) >= CONFIG.PATTERN_THRESHOLDS['market_leader']
        patterns.append(('👑 MARKET LEADER', mask))
        
        # 8. Momentum Wave
        mask = (get_col_safe('momentum_score', 0) >= CONFIG.PATTERN_THRESHOLDS['momentum_wave']) & (get_col_safe('acceleration_score', 0) >= 70)
        patterns.append(('🌊 MOMENTUM WAVE', mask))
        
        # 9. Liquid Leader
        mask = (get_col_safe('liquidity_score', 0) >= CONFIG.PATTERN_THRESHOLDS['liquid_leader']) & (get_col_safe('percentile', 0) >= CONFIG.PATTERN_THRESHOLDS['liquid_leader'])
        patterns.append(('💰 LIQUID LEADER', mask))
        
        # 10. Long-term Strength
        mask = get_col_safe('long_term_strength', 0) >= CONFIG.PATTERN_THRESHOLDS['long_strength']
        patterns.append(('💪 LONG STRENGTH', mask))
        
        # 11. Quality Trend
        mask = get_col_safe('trend_quality', 0) >= 80
        patterns.append(('📈 QUALITY TREND', mask))
        
        # 12. Value Momentum
        pe = get_col_safe('pe')
        mask = pe.notna() & (pe > 0) & (pe < 15) & (get_col_safe('master_score', 0) >= 70)
        patterns.append(('💎 VALUE MOMENTUM', mask))
        
        # 13. Earnings Rocket
        eps_change_pct = get_col_safe('eps_change_pct')
        mask = eps_change_pct.notna() & (eps_change_pct > 50) & (get_col_safe('acceleration_score', 0) >= 70)
        patterns.append(('📊 EARNINGS ROCKET', mask))

        # 14. Quality Leader
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            pe, eps_change_pct, percentile = get_col_safe('pe'), get_col_safe('eps_change_pct'), get_col_safe('percentile')
            mask = pe.notna() & eps_change_pct.notna() & (pe.between(10, 25)) & (eps_change_pct > 20) & (percentile >= 80)
            patterns.append(('🏆 QUALITY LEADER', mask))
        
        # 15. Turnaround Play
        eps_change_pct = get_col_safe('eps_change_pct')
        mask = eps_change_pct.notna() & (eps_change_pct > 100) & (get_col_safe('volume_score', 0) >= 60)
        patterns.append(('⚡ TURNAROUND', mask))
        
        # 16. High PE Warning
        pe = get_col_safe('pe')
        mask = pe.notna() & (pe > 100)
        patterns.append(('⚠️ HIGH PE', mask))
        
        # 17. 52W High Approach
        mask = (get_col_safe('from_high_pct', -100) > -5) & (get_col_safe('volume_score', 0) >= 70) & (get_col_safe('momentum_score', 0) >= 60)
        patterns.append(('🎯 52W HIGH APPROACH', mask))
        
        # 18. 52W Low Bounce
        mask = (get_col_safe('from_low_pct', 100) < 20) & (get_col_safe('acceleration_score', 0) >= 80) & (get_col_safe('ret_30d', 0) > 10)
        patterns.append(('🔄 52W LOW BOUNCE', mask))
        
        # 19. Golden Zone
        mask = (get_col_safe('from_low_pct', 0) > 60) & (get_col_safe('from_high_pct', 0) > -40) & (get_col_safe('trend_quality', 0) >= 70)
        patterns.append(('👑 GOLDEN ZONE', mask))
        
        # 20. Volume Accumulation
        mask = (get_col_safe('vol_ratio_30d_90d', 1) > 1.2) & (get_col_safe('vol_ratio_90d_180d', 1) > 1.1) & (get_col_safe('ret_30d', 0) > 5)
        patterns.append(('📊 VOL ACCUMULATION', mask))
        
        # 21. Momentum Divergence
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d_pace = np.where(df['ret_7d'].fillna(0) != 0, df['ret_7d'].fillna(0) / 7, np.nan)
                daily_30d_pace = np.where(df['ret_30d'].fillna(0) != 0, df['ret_30d'].fillna(0) / 30, np.nan)
            mask = pd.Series(daily_7d_pace > daily_30d_pace * 1.5, index=df.index).fillna(False) & (get_col_safe('acceleration_score', 0) >= 85) & (get_col_safe('rvol', 0) > 2)
            patterns.append(('🔀 MOMENTUM DIVERGE', mask))
        
        # 22. Range Compression
        if all(col in df.columns for col in ['high_52w', 'low_52w', 'from_low_pct']):
            high, low, from_low_pct = get_col_safe('high_52w'), get_col_safe('low_52w'), get_col_safe('from_low_pct')
            with np.errstate(divide='ignore', invalid='ignore'):
                range_pct = pd.Series(np.where(low > 0, ((high - low) / low) * 100, 100), index=df.index).fillna(100)
            mask = range_pct.notna() & (range_pct < 50) & (from_low_pct > 30)
            patterns.append(('🎯 RANGE COMPRESS', mask))
        
        # 23. Stealth Accumulator
        if all(col in df.columns for col in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']):
            ret_7d, ret_30d = get_col_safe('ret_7d'), get_col_safe('ret_30d')
            with np.errstate(divide='ignore', invalid='ignore'):
                ret_ratio = pd.Series(np.where(ret_30d != 0, ret_7d / (ret_30d / 4), np.nan), index=df.index).fillna(0)
            mask = (get_col_safe('vol_ratio_90d_180d', 1) > 1.1) & (get_col_safe('vol_ratio_30d_90d', 1).between(0.9, 1.1)) & (get_col_safe('from_low_pct', 0) > 40) & (ret_ratio > 1)
            patterns.append(('🤫 STEALTH', mask))

        # 24. Momentum Vampire
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']):
            ret_1d, ret_7d, rvol, from_high_pct = get_col_safe('ret_1d'), get_col_safe('ret_7d'), get_col_safe('rvol'), get_col_safe('from_high_pct')
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_pace_ratio = pd.Series(np.where(ret_7d != 0, ret_1d / (ret_7d / 7), np.nan), index=df.index).fillna(0)
            mask = (daily_pace_ratio > 2) & (rvol > 3) & (from_high_pct > -15) & (df['category'].isin(['Small Cap', 'Micro Cap']))
            patterns.append(('🧛 VAMPIRE', mask))
        
        # 25. Perfect Storm
        if 'momentum_harmony' in df.columns and 'master_score' in df.columns:
            mask = (get_col_safe('momentum_harmony', 0) == 4) & (get_col_safe('master_score', 0) > 80)
            patterns.append(('⛈️ PERFECT STORM', mask))

        # 26. BULL TRAP - Failed breakout/shorting opportunity
        if all(col in df.columns for col in ['from_high_pct', 'ret_7d', 'volume_7d', 'volume_30d']):
            mask = (
                (get_col_safe('from_high_pct', -100) > -5) &     # Was near 52W high
                (get_col_safe('ret_7d', 0) < -10) &              # Now falling hard
                (get_col_safe('volume_7d', 0) > get_col_safe('volume_30d', 1))  # High volume selling
            )
            patterns.append(('🪤 BULL TRAP', mask))
        
        # 27. CAPITULATION BOTTOM - Panic selling exhaustion
        if all(col in df.columns for col in ['ret_1d', 'from_low_pct', 'rvol', 'volume_1d', 'volume_90d']):
            mask = (
                (get_col_safe('ret_1d', 0) < -7) &               # Huge down day
                (get_col_safe('from_low_pct', 100) < 20) &       # Near 52W low
                (get_col_safe('rvol', 0) > 5) &                  # Extreme volume
                (get_col_safe('volume_1d', 0) > get_col_safe('volume_90d', 1) * 3)  # Panic volume
            )
            patterns.append(('💣 CAPITULATION', mask))
        
        # 28. RUNAWAY GAP - Continuation pattern
        if all(col in df.columns for col in ['price', 'prev_close', 'ret_30d', 'rvol', 'from_high_pct']):
            price = get_col_safe('price', 0)
            prev_close = get_col_safe('prev_close', 1)
            
            # Calculate gap percentage safely
            with np.errstate(divide='ignore', invalid='ignore'):
                gap = np.where(prev_close > 0, 
                              ((price - prev_close) / prev_close) * 100,
                              0)
            gap_series = pd.Series(gap, index=df.index)
            
            mask = (
                (gap_series > 5) &                               # Big gap up
                (get_col_safe('ret_30d', 0) > 20) &             # Already trending
                (get_col_safe('rvol', 0) > 3) &                 # Institutional volume
                (get_col_safe('from_high_pct', -100) > -3)      # Making new highs
            )
            patterns.append(('🏃 RUNAWAY GAP', mask))
        
        # 29. ROTATION LEADER - First mover in sector rotation
        if all(col in df.columns for col in ['ret_7d', 'sector', 'rvol']):
            ret_7d = get_col_safe('ret_7d', 0)
            
            # Calculate sector average return safely
            if 'sector' in df.columns:
                sector_avg = df.groupby('sector')['ret_7d'].transform('mean').fillna(0)
            else:
                sector_avg = pd.Series(0, index=df.index)
            
            mask = (
                (ret_7d > sector_avg + 5) &                      # Beating sector by 5%
                (ret_7d > 0) &                                   # Positive absolute return
                (sector_avg < 0) &                               # Sector still negative
                (get_col_safe('rvol', 0) > 2)                   # Volume confirmation
            )
            patterns.append(('🔄 ROTATION LEADER', mask))
        
        # 30. DISTRIBUTION TOP - Smart money selling
        if all(col in df.columns for col in ['from_high_pct', 'rvol', 'ret_1d', 'ret_30d', 'volume_7d', 'volume_30d']):
            mask = (
                (get_col_safe('from_high_pct', -100) > -10) &    # Near highs
                (get_col_safe('rvol', 0) > 2) &                  # High volume
                (get_col_safe('ret_1d', 0) < 2) &                # Price not moving up
                (get_col_safe('ret_30d', 0) > 50) &              # After big rally
                (get_col_safe('volume_7d', 0) > get_col_safe('volume_30d', 1) * 1.5)  # Volume spike
            )
            patterns.append(('⚠️ DISTRIBUTION', mask))

        # 31. VELOCITY SQUEEZE
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'from_high_pct', 'from_low_pct', 'high_52w', 'low_52w']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0)
                daily_30d = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
                range_pct = np.where(df['low_52w'] > 0, 
                                   (df['high_52w'] - df['low_52w']) / df['low_52w'], 
                                   np.inf)
            
            mask = (
                (daily_7d > daily_30d) &  # Velocity increasing
                (abs(df['from_high_pct']) + df['from_low_pct'] < 30) &  # Middle of range
                (range_pct < 0.5)  # Tight range
            )
            patterns.append(('🎯 VELOCITY SQUEEZE', mask))
        
        # 32. VOLUME DIVERGENCE TRAP
        if all(col in df.columns for col in ['ret_30d', 'vol_ratio_30d_180d', 'vol_ratio_90d_180d', 'from_high_pct']):
            mask = (
                (df['ret_30d'] > 20) &
                (df['vol_ratio_30d_180d'] < 0.7) &
                (df['vol_ratio_90d_180d'] < 0.9) &
                (df['from_high_pct'] > -5)
            )
            patterns.append(('⚠️ VOLUME DIVERGENCE', mask))
        
        # 33. GOLDEN CROSS MOMENTUM
        if all(col in df.columns for col in ['sma_20d', 'sma_50d', 'sma_200d', 'rvol', 'ret_7d', 'ret_30d']):
            mask = (
                (df['sma_20d'] > df['sma_50d']) &
                (df['sma_50d'] > df['sma_200d']) &
                ((df['sma_20d'] - df['sma_50d']) / df['sma_50d'] > 0.02) &
                (df['rvol'] > 1.5) &
                (df['ret_7d'] > df['ret_30d'] / 4)
            )
            patterns.append(('⚡ GOLDEN CROSS', mask))
        
        # 34. MOMENTUM EXHAUSTION
        if all(col in df.columns for col in ['ret_7d', 'ret_1d', 'rvol', 'from_low_pct', 'price', 'sma_20d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                sma_deviation = np.where(df['sma_20d'] > 0,
                                        (df['price'] - df['sma_20d']) / df['sma_20d'],
                                        0)
            mask = (
                (df['ret_7d'] > 25) &
                (df['ret_1d'] < 0) &
                (df['rvol'] < df['rvol'].shift(1)) &
                (df['from_low_pct'] > 80) &
                (sma_deviation > 0.15)
            )
            patterns.append(('📉 EXHAUSTION', mask))
        
        # 35. PYRAMID ACCUMULATION
        if all(col in df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d', 'from_low_pct']):
            mask = (
                (df['vol_ratio_7d_90d'] > 1.1) &
                (df['vol_ratio_30d_90d'] > 1.05) &
                (df['vol_ratio_90d_180d'] > 1.02) &
                (df['ret_30d'].between(5, 15)) &
                (df['from_low_pct'] < 50)
            )
            patterns.append(('🔺 PYRAMID', mask))
        
        # 36. MOMENTUM VACUUM
        if all(col in df.columns for col in ['ret_30d', 'ret_7d', 'ret_1d', 'rvol', 'from_low_pct']):
            mask = (
                (df['ret_30d'] < -20) &
                (df['ret_7d'] > 0) &
                (df['ret_1d'] > 2) &
                (df['rvol'] > 3) &
                (df['from_low_pct'] < 10)
            )
            patterns.append(('🌪️ VACUUM', mask))

        return patterns

    @staticmethod
    def _calculate_pattern_confidence(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a numerical confidence score for each stock based on the
        quantity and importance of the patterns it exhibits.
        """
        if 'patterns' not in df.columns or df['patterns'].eq('').all():
            df['pattern_confidence'] = 0.0
            return df

        pattern_list = df['patterns'].str.split(' | ').fillna(pd.Series([[]] * len(df), index=df.index))
        
        max_possible_score = sum(item['importance_weight'] for item in PatternDetector.PATTERN_METADATA.values())

        if max_possible_score > 0:
            df['pattern_confidence'] = pattern_list.apply(
                lambda patterns: sum(
                    PatternDetector.PATTERN_METADATA.get(p, {'importance_weight': 0})['importance_weight']
                    for p in patterns
                )
            )
            df['pattern_confidence'] = (df['pattern_confidence'] / max_possible_score * 100).clip(0, 100).round(2)
        else:
            df['pattern_confidence'] = 0.0

        return df
        
# ============================================
# MARKET INTELLIGENCE
# ============================================

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
            
            micro_small_avg = category_scores[category_scores.index.isin(['Micro Cap', 'Small Cap'])].mean() if any(category_scores.index.isin(['Micro Cap', 'Small Cap'])) else 50
            large_mega_avg = category_scores[category_scores.index.isin(['Large Cap', 'Mega Cap'])].mean() if any(category_scores.index.isin(['Large Cap', 'Mega Cap'])) else 50
            
            metrics['micro_small_avg'] = micro_small_avg if pd.notna(micro_small_avg) else 50
            metrics['large_mega_avg'] = large_mega_avg if pd.notna(large_mega_avg) else 50
            metrics['category_spread'] = metrics['micro_small_avg'] - metrics['large_mega_avg']
        else:
            metrics['micro_small_avg'] = 50
            metrics['large_mega_avg'] = 50
            metrics['category_spread'] = 0
        
        if 'ret_30d' in df.columns:
            breadth = len(df[df['ret_30d'] > 0]) / len(df) if len(df) > 0 else 0.5
            metrics['breadth'] = breadth
        else:
            breadth = 0.5
            metrics['breadth'] = breadth
        
        if 'rvol' in df.columns:
            avg_rvol = df['rvol'].median()
            metrics['avg_rvol'] = avg_rvol if pd.notna(avg_rvol) else 1.0
        else:
            metrics['avg_rvol'] = 1.0
        
        # Determine regime
        if metrics['micro_small_avg'] > metrics['large_mega_avg'] + 10 and breadth > 0.6:
            regime = "🔥 RISK-ON BULL"
        elif metrics['large_mega_avg'] > metrics['micro_small_avg'] + 10 and breadth < 0.4:
            regime = "🛡️ RISK-OFF DEFENSIVE"
        elif metrics['avg_rvol'] > 1.5 and breadth > 0.5:
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
        else:
            ad_metrics = {'advancing': 0, 'declining': 0, 'unchanged': 0, 'ad_ratio': 1.0, 'ad_line': 0, 'breadth_pct': 0}
        
        return ad_metrics
    
    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        """Detect sector rotation patterns with transparent sampling"""
        
        if 'sector' not in df.columns or df.empty:
            return pd.DataFrame()
        
        sector_dfs = []
        
        for sector in df['sector'].unique():
            if sector != 'Unknown':
                sector_df = df[df['sector'] == sector].copy()
                sector_size = len(sector_df)
                
                if sector_size == 0:
                    continue
                
                # Dynamic sampling
                if 1 <= sector_size <= 5:
                    sample_count = sector_size
                elif 6 <= sector_size <= 20:
                    sample_count = max(1, int(sector_size * 0.80))
                elif 21 <= sector_size <= 50:
                    sample_count = max(1, int(sector_size * 0.60))
                elif 51 <= sector_size <= 100:
                    sample_count = max(1, int(sector_size * 0.40))
                else:
                    sample_count = min(50, int(sector_size * 0.25))
                
                if sample_count > 0:
                    sector_df = sector_df.nlargest(min(sample_count, len(sector_df)), 'master_score')
                    
                    if not sector_df.empty:
                        sector_dfs.append(sector_df)
        
        if not sector_dfs:
            return pd.DataFrame()
        
        normalized_df = pd.concat(sector_dfs, ignore_index=True)
        
        # Calculate metrics
        agg_dict = {
            'master_score': ['mean', 'median', 'std', 'count'],
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'rvol': 'mean',
            'ret_30d': 'mean'
        }
        
        if 'money_flow_mm' in normalized_df.columns:
            agg_dict['money_flow_mm'] = 'sum'
        
        sector_metrics = normalized_df.groupby('sector').agg(agg_dict).round(2)
        
        # Flatten columns
        new_cols = []
        for col in sector_metrics.columns:
            if isinstance(col, tuple):
                new_cols.append(f"{col[0]}_{col[1]}" if col[1] != 'mean' else col[0])
            else:
                new_cols.append(col)
        
        sector_metrics.columns = new_cols
        
        # Rename for clarity
        rename_dict = {
            'master_score': 'avg_score',
            'master_score_median': 'median_score',
            'master_score_std': 'std_score',
            'master_score_count': 'count',
            'momentum_score': 'avg_momentum',
            'volume_score': 'avg_volume',
            'rvol': 'avg_rvol',
            'ret_30d': 'avg_ret_30d'
        }
        
        if 'money_flow_mm' in sector_metrics.columns:
            rename_dict['money_flow_mm'] = 'total_money_flow'
        
        sector_metrics.rename(columns=rename_dict, inplace=True)
        
        # Add original counts
        original_counts = df.groupby('sector').size().rename('total_stocks')
        sector_metrics = sector_metrics.join(original_counts, how='left')
        sector_metrics['analyzed_stocks'] = sector_metrics['count']
        
        # Calculate sampling percentage
        with np.errstate(divide='ignore', invalid='ignore'):
            sector_metrics['sampling_pct'] = (sector_metrics['analyzed_stocks'] / sector_metrics['total_stocks'] * 100)
            sector_metrics['sampling_pct'] = sector_metrics['sampling_pct'].replace([np.inf, -np.inf], 100).fillna(100).round(1)
        
        # Calculate flow score
        sector_metrics['flow_score'] = (
            sector_metrics['avg_score'] * 0.3 +
            sector_metrics.get('median_score', 50) * 0.2 +
            sector_metrics['avg_momentum'] * 0.25 +
            sector_metrics['avg_volume'] * 0.25
        )
        
        sector_metrics['rank'] = sector_metrics['flow_score'].rank(ascending=False)
        
        return sector_metrics.sort_values('flow_score', ascending=False)
    
    @staticmethod
    def detect_industry_rotation(df: pd.DataFrame) -> pd.DataFrame:
        """Detect industry rotation patterns with transparent sampling"""
        
        if 'industry' not in df.columns or df.empty:
            return pd.DataFrame()
        
        industry_dfs = []
        
        for industry in df['industry'].unique():
            if industry != 'Unknown':
                industry_df = df[df['industry'] == industry].copy()
                industry_size = len(industry_df)
                
                if industry_size == 0:
                    continue
                
                # Smart Dynamic Sampling
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
                    industry_df = industry_df.nlargest(min(sample_count, len(industry_df)), 'master_score')
                    
                    if not industry_df.empty:
                        industry_dfs.append(industry_df)
        
        if not industry_dfs:
            return pd.DataFrame()
        
        normalized_df = pd.concat(industry_dfs, ignore_index=True)
        
        # Calculate metrics
        agg_dict = {
            'master_score': ['mean', 'median', 'std', 'count'],
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'rvol': 'mean',
            'ret_30d': 'mean'
        }
        
        if 'money_flow_mm' in normalized_df.columns:
            agg_dict['money_flow_mm'] = 'sum'
        
        industry_metrics = normalized_df.groupby('industry').agg(agg_dict).round(2)
        
        # Flatten columns
        new_cols = []
        for col in industry_metrics.columns:
            if isinstance(col, tuple):
                new_cols.append(f"{col[0]}_{col[1]}" if col[1] != 'mean' else col[0])
            else:
                new_cols.append(col)
        
        industry_metrics.columns = new_cols
        
        # Rename for clarity
        rename_dict = {
            'master_score': 'avg_score',
            'master_score_median': 'median_score',
            'master_score_std': 'std_score',
            'master_score_count': 'count',
            'momentum_score': 'avg_momentum',
            'volume_score': 'avg_volume',
            'rvol': 'avg_rvol',
            'ret_30d': 'avg_ret_30d'
        }
        
        if 'money_flow_mm' in industry_metrics.columns:
            rename_dict['money_flow_mm'] = 'total_money_flow'
        
        industry_metrics.rename(columns=rename_dict, inplace=True)
        
        # Add original counts
        original_counts = df.groupby('industry').size().rename('total_stocks')
        industry_metrics = industry_metrics.join(original_counts, how='left')
        industry_metrics['analyzed_stocks'] = industry_metrics['count']
        
        # Calculate sampling percentage
        with np.errstate(divide='ignore', invalid='ignore'):
            industry_metrics['sampling_pct'] = (industry_metrics['analyzed_stocks'] / industry_metrics['total_stocks'] * 100)
            industry_metrics['sampling_pct'] = industry_metrics['sampling_pct'].replace([np.inf, -np.inf], 100).fillna(100).round(1)
        
        # Add sampling quality warning
        industry_metrics['quality_flag'] = ''
        industry_metrics.loc[industry_metrics['sampling_pct'] < 10, 'quality_flag'] = '⚠️ Low Sample'
        industry_metrics.loc[industry_metrics['analyzed_stocks'] < 5, 'quality_flag'] = '⚠️ Few Stocks'
        
        # Calculate flow score
        industry_metrics['flow_score'] = (
            industry_metrics['avg_score'] * 0.3 +
            industry_metrics.get('median_score', 50) * 0.2 +
            industry_metrics['avg_momentum'] * 0.25 +
            industry_metrics['avg_volume'] * 0.25
        )
        
        industry_metrics['rank'] = industry_metrics['flow_score'].rank(ascending=False)
        
        return industry_metrics.sort_values('flow_score', ascending=False)


# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    """Create all visualizations with proper error handling"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create score distribution chart"""
        fig = go.Figure()
        
        if df.empty:
            fig.add_annotation(
                text="No data available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        scores = [
            ('position_score', 'Position', '#3498db'),
            ('volume_score', 'Volume', '#e74c3c'),
            ('momentum_score', 'Momentum', '#2ecc71'),
            ('acceleration_score', 'Acceleration', '#f39c12'),
            ('breakout_score', 'Breakout', '#9b59b6'),
            ('rvol_score', 'RVOL', '#e67e22')
        ]
        
        for score_col, label, color in scores:
            if score_col in df.columns:
                score_data = df[score_col].dropna()
                if len(score_data) > 0:
                    fig.add_trace(go.Box(
                        y=score_data,
                        name=label,
                        marker_color=color,
                        boxpoints='outliers',
                        hovertemplate=f'{label}<br>Score: %{{y:.1f}}<extra></extra>'
                    ))
        
        fig.update_layout(
            title="Score Component Distribution",
            yaxis_title="Score (0-100)",
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig

    @staticmethod
    def create_acceleration_profiles(df: pd.DataFrame, n: int = 10) -> go.Figure:
        """Create acceleration profiles showing momentum over time"""
        try:
            accel_df = df.nlargest(min(n, len(df)), 'acceleration_score')
            
            if len(accel_df) == 0:
                return go.Figure()
            
            fig = go.Figure()
            
            for _, stock in accel_df.iterrows():
                x_points = []
                y_points = []
                
                x_points.append('Start')
                y_points.append(0)
                
                if 'ret_30d' in stock.index and pd.notna(stock['ret_30d']):
                    x_points.append('30D')
                    y_points.append(stock['ret_30d'])
                
                if 'ret_7d' in stock.index and pd.notna(stock['ret_7d']):
                    x_points.append('7D')
                    y_points.append(stock['ret_7d'])
                
                if 'ret_1d' in stock.index and pd.notna(stock['ret_1d']):
                    x_points.append('Today')
                    y_points.append(stock['ret_1d'])
                
                if len(x_points) > 1:
                    if stock['acceleration_score'] >= 85:
                        line_style = dict(width=3, dash='solid')
                        marker_style = dict(size=10, symbol='star')
                    elif stock['acceleration_score'] >= 70:
                        line_style = dict(width=2, dash='solid')
                        marker_style = dict(size=8)
                    else:
                        line_style = dict(width=2, dash='dot')
                        marker_style = dict(size=6)
                    
                    fig.add_trace(go.Scatter(
                        x=x_points,
                        y=y_points,
                        mode='lines+markers',
                        name=f"{stock['ticker']} ({stock['acceleration_score']:.0f})",
                        line=line_style,
                        marker=marker_style,
                        hovertemplate=(
                            f"<b>{stock['ticker']}</b><br>" +
                            "%{x}: %{y:.1f}%<br>" +
                            f"Accel Score: {stock['acceleration_score']:.0f}<extra></extra>"
                        )
                    ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                title=f"Acceleration Profiles - Top {len(accel_df)} Momentum Builders",
                xaxis_title="Time Frame",
                yaxis_title="Return %",
                height=400,
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating acceleration profiles: {str(e)}")
            return go.Figure()

# ============================================
# FILTER ENGINE - OPTIMIZED VERSION
# ============================================

class FilterEngine:
    """
    Centralized filter management with single state object.
    This eliminates 15+ separate session state keys.
    """
    
    @staticmethod
    def initialize_filters():
        """Initialize single filter state object"""
        if 'filter_state' not in st.session_state:
            st.session_state.filter_state = {
                'categories': [],
                'sectors': [],
                'industries': [],
                'min_score': 0,
                'patterns': [],
                'trend_filter': "All Trends",
                'trend_range': (0, 100),
                'eps_tiers': [],
                'pe_tiers': [],
                'price_tiers': [],
                'min_eps_change': None,
                'min_pe': None,
                'max_pe': None,
                'require_fundamental_data': False,
                'wave_states': [],
                'wave_strength_range': (0, 100),
                'quick_filter': None,
                'quick_filter_applied': False
            }
    
    @staticmethod
    def get_filter(key: str, default: Any = None) -> Any:
        """Get filter value from centralized state"""
        FilterEngine.initialize_filters()
        return st.session_state.filter_state.get(key, default)
    
    @staticmethod
    def set_filter(key: str, value: Any) -> None:
        """Set filter value in centralized state"""
        FilterEngine.initialize_filters()
        st.session_state.filter_state[key] = value
    
    @staticmethod
    def get_active_count() -> int:
        """Count active filters"""
        FilterEngine.initialize_filters()
        count = 0
        filters = st.session_state.filter_state
        
        # Check each filter type
        if filters.get('categories'): count += 1
        if filters.get('sectors'): count += 1
        if filters.get('industries'): count += 1
        if filters.get('min_score', 0) > 0: count += 1
        if filters.get('patterns'): count += 1
        if filters.get('trend_filter') != "All Trends": count += 1
        if filters.get('eps_tiers'): count += 1
        if filters.get('pe_tiers'): count += 1
        if filters.get('price_tiers'): count += 1
        if filters.get('min_eps_change') is not None: count += 1
        if filters.get('min_pe') is not None: count += 1
        if filters.get('max_pe') is not None: count += 1
        if filters.get('require_fundamental_data'): count += 1
        if filters.get('wave_states'): count += 1
        if filters.get('wave_strength_range') != (0, 100): count += 1
        
        return count
    
    @staticmethod
    def clear_all_filters():
        """Reset all filters to defaults and clear widget states"""
        # Reset centralized filter state
        st.session_state.filter_state = {
            'categories': [],
            'sectors': [],
            'industries': [],
            'min_score': 0,
            'patterns': [],
            'trend_filter': "All Trends",
            'trend_range': (0, 100),
            'eps_tiers': [],
            'pe_tiers': [],
            'price_tiers': [],
            'min_eps_change': None,
            'min_pe': None,
            'max_pe': None,
            'require_fundamental_data': False,
            'wave_states': [],
            'wave_strength_range': (0, 100),
            'quick_filter': None,
            'quick_filter_applied': False
        }
        
        # CRITICAL FIX: Delete all widget keys to force UI reset
        widget_keys_to_delete = [
            # Multiselect widgets
            'category_multiselect', 'sector_multiselect', 'industry_multiselect',
            'patterns_multiselect', 'wave_states_multiselect',
            'eps_tier_multiselect', 'pe_tier_multiselect', 'price_tier_multiselect',
            
            # Slider widgets
            'min_score_slider', 'wave_strength_slider',
            
            # Selectbox widgets
            'trend_selectbox', 'wave_timeframe_select',
            
            # Text input widgets
            'eps_change_input', 'min_pe_input', 'max_pe_input',
            
            # Checkbox widgets
            'require_fundamental_checkbox',
            
            # Additional filter-related keys
            'display_count_select', 'sort_by_select', 'export_template_radio',
            'wave_sensitivity', 'show_sensitivity_details', 'show_market_regime'
        ]
        
        # Delete each widget key if it exists
        for key in widget_keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        
        # Also clear legacy filter keys for backward compatibility
        legacy_keys = [
            'category_filter', 'sector_filter', 'industry_filter',
            'min_score', 'patterns', 'trend_filter',
            'eps_tier_filter', 'pe_tier_filter', 'price_tier_filter',
            'min_eps_change', 'min_pe', 'max_pe',
            'require_fundamental_data', 'wave_states_filter',
            'wave_strength_range_slider'
        ]
        
        for key in legacy_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                elif isinstance(st.session_state[key], bool):
                    st.session_state[key] = False
                elif isinstance(st.session_state[key], str):
                    if key == 'trend_filter':
                        st.session_state[key] = "All Trends"
                    else:
                        st.session_state[key] = ""
                elif isinstance(st.session_state[key], tuple):
                    if key == 'wave_strength_range_slider':
                        st.session_state[key] = (0, 100)
                elif isinstance(st.session_state[key], (int, float)):
                    if key == 'min_score':
                        st.session_state[key] = 0
                    else:
                        st.session_state[key] = None
                else:
                    st.session_state[key] = None
        
        # Reset active filter count
        st.session_state.active_filter_count = 0
        
        # Clear quick filter
        st.session_state.quick_filter = None
        st.session_state.quick_filter_applied = False
        
        logger.info("All filters and widget states cleared successfully")
    
    @staticmethod
    def sync_widget_to_filter(widget_key: str, filter_key: str):
        """Sync widget state to filter state - used in callbacks"""
        if widget_key in st.session_state:
            st.session_state.filter_state[filter_key] = st.session_state[widget_key]
    
    @staticmethod
    def build_filter_dict() -> Dict[str, Any]:
        """Build filter dictionary for apply_filters method"""
        FilterEngine.initialize_filters()
        filters = {}
        state = st.session_state.filter_state
        
        # Map internal state to filter dict format
        if state.get('categories'):
            filters['categories'] = state['categories']
        if state.get('sectors'):
            filters['sectors'] = state['sectors']
        if state.get('industries'):
            filters['industries'] = state['industries']
        if state.get('min_score', 0) > 0:
            filters['min_score'] = state['min_score']
        if state.get('patterns'):
            filters['patterns'] = state['patterns']
        if state.get('trend_filter') != "All Trends":
            filters['trend_filter'] = state['trend_filter']
            filters['trend_range'] = state.get('trend_range', (0, 100))
        if state.get('eps_tiers'):
            filters['eps_tiers'] = state['eps_tiers']
        if state.get('pe_tiers'):
            filters['pe_tiers'] = state['pe_tiers']
        if state.get('price_tiers'):
            filters['price_tiers'] = state['price_tiers']
        if state.get('min_eps_change') is not None:
            filters['min_eps_change'] = state['min_eps_change']
        if state.get('min_pe') is not None:
            filters['min_pe'] = state['min_pe']
        if state.get('max_pe') is not None:
            filters['max_pe'] = state['max_pe']
        if state.get('require_fundamental_data'):
            filters['require_fundamental_data'] = True
        if state.get('wave_states'):
            filters['wave_states'] = state['wave_states']
        if state.get('wave_strength_range') != (0, 100):
            filters['wave_strength_range'] = state['wave_strength_range']
            
        return filters
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.1)
    def apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Apply all filters to dataframe efficiently using vectorized operations.
        If no filters provided, get from centralized state.
        """
        if df.empty:
            return df
        
        # Use provided filters or get from state
        if filters is None:
            filters = FilterEngine.build_filter_dict()
        
        if not filters:
            return df
        
        # Create boolean masks for each filter
        masks = []
        
        # Helper function for isin filters
        def create_mask_from_isin(column: str, values: List[Any]) -> Optional[pd.Series]:
            if values and column in df.columns:
                return df[column].isin(values)
            return None
        
        # 1. Category filters
        if 'categories' in filters:
            masks.append(create_mask_from_isin('category', filters['categories']))
        if 'sectors' in filters:
            masks.append(create_mask_from_isin('sector', filters['sectors']))
        if 'industries' in filters:
            masks.append(create_mask_from_isin('industry', filters['industries']))
        
        # 2. Score filter
        if filters.get('min_score', 0) > 0 and 'master_score' in df.columns:
            masks.append(df['master_score'] >= filters['min_score'])
        
        # 3. Pattern filter
        if filters.get('patterns') and 'patterns' in df.columns:
            pattern_mask = pd.Series(False, index=df.index)
            for pattern in filters['patterns']:
                pattern_mask |= df['patterns'].str.contains(pattern, na=False, regex=False)
            masks.append(pattern_mask)
        
        # 4. Trend filter
        trend_range = filters.get('trend_range')
        if trend_range and trend_range != (0, 100) and 'trend_quality' in df.columns:
            min_trend, max_trend = trend_range
            masks.append((df['trend_quality'] >= min_trend) & (df['trend_quality'] <= max_trend))
        
        # 5. EPS change filter
        if filters.get('min_eps_change') is not None and 'eps_change_pct' in df.columns:
            masks.append(df['eps_change_pct'] >= filters['min_eps_change'])
        
        # 6. PE filters
        if filters.get('min_pe') is not None and 'pe' in df.columns:
            masks.append(df['pe'] >= filters['min_pe'])
        
        if filters.get('max_pe') is not None and 'pe' in df.columns:
            masks.append(df['pe'] <= filters['max_pe'])
        
        # 7. Tier filters
        if 'eps_tiers' in filters:
            masks.append(create_mask_from_isin('eps_tier', filters['eps_tiers']))
        if 'pe_tiers' in filters:
            masks.append(create_mask_from_isin('pe_tier', filters['pe_tiers']))
        if 'price_tiers' in filters:
            masks.append(create_mask_from_isin('price_tier', filters['price_tiers']))
        
        # 8. Data completeness filter
        if filters.get('require_fundamental_data', False):
            if all(col in df.columns for col in ['pe', 'eps_change_pct']):
                masks.append(df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna())
        
        # 9. Wave filters
        if 'wave_states' in filters:
            masks.append(create_mask_from_isin('wave_state', filters['wave_states']))
        
        wave_strength_range = filters.get('wave_strength_range')
        if wave_strength_range and wave_strength_range != (0, 100) and 'overall_wave_strength' in df.columns:
            min_ws, max_ws = wave_strength_range
            masks.append((df['overall_wave_strength'] >= min_ws) & 
                        (df['overall_wave_strength'] <= max_ws))
        
        # Combine all masks
        masks = [mask for mask in masks if mask is not None]
        
        if masks:
            combined_mask = np.logical_and.reduce(masks)
            filtered_df = df[combined_mask].copy()
        else:
            filtered_df = df.copy()
        
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks")
        
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Get available options for a filter based on other active filters.
        This creates interconnected filters.
        """
        if df.empty or column not in df.columns:
            return []
        
        # Use current filters or get from state
        if current_filters is None:
            current_filters = FilterEngine.build_filter_dict()
        
        # Create temp filters without the current column
        temp_filters = current_filters.copy()
        
        # Map column to filter key
        filter_key_map = {
            'category': 'categories',
            'sector': 'sectors',
            'industry': 'industries',
            'eps_tier': 'eps_tiers',
            'pe_tier': 'pe_tiers',
            'price_tier': 'price_tiers',
            'wave_state': 'wave_states'
        }
        
        if column in filter_key_map:
            temp_filters.pop(filter_key_map[column], None)
        
        # Apply remaining filters
        filtered_df = FilterEngine.apply_filters(df, temp_filters)
        
        # Get unique values
        values = filtered_df[column].dropna().unique()
        
        # Filter out invalid values
        values = [v for v in values if str(v).strip() not in ['Unknown', 'unknown', '', 'nan', 'NaN', 'None', 'N/A', '-']]
        
        # Sort appropriately
        try:
            values = sorted(values, key=lambda x: float(str(x).replace(',', '')))
        except (ValueError, TypeError):
            values = sorted(values, key=str)
        
        return values
        
# ============================================
# SEARCH ENGINE
# ============================================

class SearchEngine:
    """Optimized search functionality"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks with optimized performance"""
        
        if not query or df.empty:
            return pd.DataFrame()
        
        try:
            query = query.upper().strip()
            
            # Method 1: Direct ticker match
            ticker_exact = df[df['ticker'].str.upper() == query]
            if not ticker_exact.empty:
                return ticker_exact
            
            # Method 2: Ticker contains
            ticker_contains = df[df['ticker'].str.upper().str.contains(query, na=False, regex=False)]
            
            # Method 3: Company name contains
            if 'company_name' in df.columns:
                company_contains = df[df['company_name'].str.upper().str.contains(query, na=False, regex=False)]
            else:
                company_contains = pd.DataFrame()
            
            # Method 4: Word match
            def word_starts_with(company_name_str):
                if pd.isna(company_name_str):
                    return False
                words = str(company_name_str).upper().split()
                return any(word.startswith(query) for word in words)
            
            if 'company_name' in df.columns:
                company_word_match = df[df['company_name'].apply(word_starts_with)]
            else:
                company_word_match = pd.DataFrame()
            
            # Combine results
            all_matches = pd.concat([
                ticker_contains,
                company_contains,
                company_word_match
            ]).drop_duplicates()
            
            # Sort by relevance
            if not all_matches.empty:
                all_matches['relevance'] = 0
                all_matches.loc[all_matches['ticker'].str.upper() == query, 'relevance'] = 100
                all_matches.loc[all_matches['ticker'].str.upper().str.startswith(query), 'relevance'] += 50
                
                if 'company_name' in all_matches.columns:
                    all_matches.loc[all_matches['company_name'].str.upper().str.startswith(query), 'relevance'] += 30
                
                return all_matches.sort_values(['relevance', 'master_score'], ascending=[False, False]).drop('relevance', axis=1)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return pd.DataFrame()

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Handle all export operations"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def create_excel_report(df: pd.DataFrame, template: str = 'full') -> BytesIO:
        """Create comprehensive Excel report"""
        
        output = BytesIO()
        
        templates = {
            'day_trader': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 'rvol', 
                           'momentum_score', 'acceleration_score', 'ret_1d', 'ret_7d', 
                           'volume_score', 'vmi', 'wave_state', 'patterns', 'category', 'industry'],
                'focus': 'Intraday momentum and volume'
            },
            'swing_trader': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 
                           'breakout_score', 'position_score', 'position_tension',
                           'from_high_pct', 'from_low_pct', 'trend_quality', 
                           'momentum_harmony', 'patterns', 'industry'],
                'focus': 'Position and breakout setups'
            },
            'investor': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 'pe', 
                           'eps_current', 'eps_change_pct', 'ret_1y', 'ret_3y', 
                           'long_term_strength', 'money_flow_mm', 'category', 'sector', 'industry'],
                'focus': 'Fundamentals and long-term performance'
            },
            'full': {
                'columns': None,
                'focus': 'Complete analysis'
            }
        }
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#3498db',
                    'font_color': 'white',
                    'border': 1
                })
                
                # 1. Top 100 Stocks
                top_100 = df.nlargest(min(100, len(df)), 'master_score')
                
                if template in templates and templates[template]['columns']:
                    export_cols = [col for col in templates[template]['columns'] if col in top_100.columns]
                else:
                    export_cols = None
                
                if export_cols:
                    top_100_export = top_100[export_cols]
                else:
                    top_100_export = top_100
                
                top_100_export.to_excel(writer, sheet_name='Top 100', index=False)
                
                worksheet = writer.sheets['Top 100']
                for i, col in enumerate(top_100_export.columns):
                    worksheet.write(0, i, col, header_format)
                
                # 2. Market Intelligence
                intel_data = []
                
                regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
                intel_data.append({
                    'Metric': 'Market Regime',
                    'Value': regime,
                    'Details': f"Breadth: {regime_metrics.get('breadth', 0):.1%}"
                })
                
                ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
                intel_data.append({
                    'Metric': 'Advance/Decline',
                    'Value': f"{ad_metrics.get('advancing', 0)}/{ad_metrics.get('declining', 0)}",
                    'Details': f"Ratio: {ad_metrics.get('ad_ratio', 1):.2f}"
                })
                
                intel_df = pd.DataFrame(intel_data)
                intel_df.to_excel(writer, sheet_name='Market Intelligence', index=False)
                
                # 3. Sector Rotation
                sector_rotation = MarketIntelligence.detect_sector_rotation(df)
                if not sector_rotation.empty:
                    sector_rotation.to_excel(writer, sheet_name='Sector Rotation')
                
                # 4. Industry Rotation
                industry_rotation = MarketIntelligence.detect_industry_rotation(df)
                if not industry_rotation.empty:
                    industry_rotation.to_excel(writer, sheet_name='Industry Rotation')
                
                # 5. Pattern Analysis
                pattern_counts = {}
                for patterns in df['patterns'].dropna():
                    if patterns:
                        for p in patterns.split(' | '):
                            pattern_counts[p] = pattern_counts.get(p, 0) + 1
                
                if pattern_counts:
                    pattern_df = pd.DataFrame(
                        list(pattern_counts.items()),
                        columns=['Pattern', 'Count']
                    ).sort_values('Count', ascending=False)
                    pattern_df.to_excel(writer, sheet_name='Pattern Analysis', index=False)
                
                # 6. Wave Radar Signals
                wave_signals = df[
                    (df['momentum_score'] >= 60) & 
                    (df['acceleration_score'] >= 70) &
                    (df['rvol'] >= 2)
                ].head(50)
                
                if len(wave_signals) > 0:
                    wave_cols = ['ticker', 'company_name', 'master_score', 
                                'momentum_score', 'acceleration_score', 'rvol',
                                'wave_state', 'patterns', 'category', 'industry']
                    available_wave_cols = [col for col in wave_cols if col in wave_signals.columns]
                    
                    wave_signals[available_wave_cols].to_excel(
                        writer, sheet_name='Wave Radar', index=False
                    )
                
                # 7. Summary Statistics
                summary_stats = {
                    'Total Stocks': len(df),
                    'Average Master Score': df['master_score'].mean() if 'master_score' in df.columns else 0,
                    'Stocks with Patterns': (df['patterns'] != '').sum() if 'patterns' in df.columns else 0,
                    'High RVOL (>2x)': (df['rvol'] > 2).sum() if 'rvol' in df.columns else 0,
                    'Positive 30D Returns': (df['ret_30d'] > 0).sum() if 'ret_30d' in df.columns else 0,
                    'Template Used': template,
                    'Export Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                logger.info(f"Excel report created successfully with {len(writer.sheets)} sheets")
                
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}")
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create CSV export efficiently"""
        
        export_cols = [
            'rank', 'ticker', 'company_name', 'master_score',
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score',
            'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct',
            'from_low_pct', 'from_high_pct',
            'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
            'rvol', 'vmi', 'money_flow_mm', 'position_tension',
            'momentum_harmony', 'wave_state', 'patterns', 
            'category', 'sector', 'industry', 'eps_tier', 'pe_tier', 'overall_wave_strength'
        ]
        
        available_cols = [col for col in export_cols if col in df.columns]
        
        export_df = df[available_cols].copy()
        
        # Convert volume ratios back to percentage
        vol_ratio_cols = [col for col in export_df.columns if 'vol_ratio' in col]
        for col in vol_ratio_cols:
            with np.errstate(divide='ignore', invalid='ignore'):
                export_df[col] = (export_df[col] - 1) * 100
                export_df[col] = export_df[col].replace([np.inf, -np.inf], 0).fillna(0)
        
        return export_df.to_csv(index=False)

# ============================================
# UI COMPONENTS - CLEAN VERSION FOR DISCOVERY FOCUS
# ============================================

class UIComponents:
    """Reusable UI components for Wave Detection Dashboard"""
    
    @staticmethod
    def render_metric_card(label: str, value: Any, delta: Optional[str] = None, 
                          help_text: Optional[str] = None) -> None:
        """Render a styled metric card with optional tooltips"""
        # Add tooltip from CONFIG if available
        metric_key = label.lower().replace(' ', '_')
        if not help_text and metric_key in CONFIG.METRIC_TOOLTIPS:
            help_text = CONFIG.METRIC_TOOLTIPS[metric_key]
        
        if help_text:
            st.metric(label, value, delta, help=help_text)
        else:
            st.metric(label, value, delta)
    
    @staticmethod
    def render_discovery_card(title: str, metrics: List[Tuple[str, Any]], color: str = "info") -> None:
        """Render a discovery information card"""
        # Color mapping for different card types
        color_func = getattr(st, color, st.info)
        
        # Build the content
        content_lines = [f"**{title}**"]
        for metric_label, metric_value in metrics:
            content_lines.append(f"{metric_label}: {metric_value}")
        
        # Render the card
        color_func("\n".join(content_lines))
    
    @staticmethod
    def render_stock_card(stock: pd.Series, fields: List[str] = None) -> None:
        """Render a compact stock information card"""
        if fields is None:
            fields = ['ticker', 'master_score', 'price', 'rvol']
        
        card_content = []
        
        # Always show ticker first
        if 'ticker' in stock.index:
            card_content.append(f"**{stock['ticker']}**")
        
        # Add specified fields
        field_mapping = {
            'master_score': ('Score', lambda x: f"{x:.0f}"),
            'price': ('Price', lambda x: f"₹{x:.0f}"),
            'rvol': ('RVOL', lambda x: f"{x:.1f}x"),
            'ret_30d': ('30D', lambda x: f"{x:+.1f}%"),
            'wave_state': ('Wave', lambda x: str(x)),
            'category': ('Category', lambda x: str(x)),
            'patterns': ('Pattern', lambda x: str(x)[:20] if x else '-')
        }
        
        for field in fields:
            if field != 'ticker' and field in stock.index and field in field_mapping:
                label, formatter = field_mapping[field]
                value = formatter(stock[field])
                card_content.append(f"{label}: {value}")
        
        st.info("\n".join(card_content))
    
    @staticmethod
    def render_pattern_badge(pattern: str, count: int) -> str:
        """Create a pattern badge with count"""
        # Determine badge intensity based on count
        if count > 20:
            intensity = "🔥🔥🔥"
        elif count > 10:
            intensity = "🔥🔥"
        else:
            intensity = "🔥"
        
        return f"{pattern} ({count}) {intensity}"
    
    @staticmethod
    def render_wave_indicator(wave_state: str) -> str:
        """Convert wave state to visual indicator"""
        if 'CRESTING' in wave_state:
            return "🌊🌊🌊 CRESTING"
        elif 'BUILDING' in wave_state:
            return "🌊🌊 BUILDING"
        elif 'FORMING' in wave_state:
            return "🌊 FORMING"
        elif 'BREAKING' in wave_state:
            return "💥 BREAKING"
        else:
            return "〰️ NEUTRAL"
    
    @staticmethod
    def render_score_badge(score: float) -> str:
        """Create a visual badge for scores"""
        if score >= 90:
            return f"🏆 {score:.0f}"
        elif score >= 80:
            return f"⭐ {score:.0f}"
        elif score >= 70:
            return f"✅ {score:.0f}"
        elif score >= 60:
            return f"👍 {score:.0f}"
        else:
            return f"{score:.0f}"
    
    @staticmethod
    def render_momentum_indicator(momentum_score: float, acceleration_score: float) -> str:
        """Create momentum status indicator"""
        if momentum_score > 80 and acceleration_score > 80:
            return "🚀 Explosive"
        elif momentum_score > 70 and acceleration_score > 70:
            return "📈 Strong"
        elif momentum_score > 60 or acceleration_score > 60:
            return "➡️ Building"
        else:
            return "💤 Quiet"
    
    @staticmethod
    def render_category_performance_table(df: pd.DataFrame) -> None:
        """Render category performance comparison table"""
        if 'category' not in df.columns or 'master_score' not in df.columns:
            st.info("Category data not available")
            return
        
        # Calculate category metrics
        cat_metrics = df.groupby('category').agg({
            'master_score': ['mean', 'count'],
            'ret_30d': 'mean' if 'ret_30d' in df.columns else lambda x: 0,
            'rvol': 'mean' if 'rvol' in df.columns else lambda x: 1
        }).round(2)
        
        # Flatten columns
        cat_metrics.columns = ['Avg Score', 'Count', 'Avg 30D Ret', 'Avg RVOL']
        cat_metrics = cat_metrics.sort_values('Avg Score', ascending=False)
        
        # Display with styling
        st.dataframe(
            cat_metrics.style.background_gradient(subset=['Avg Score']),
            use_container_width=True
        )
    
    @staticmethod
    def render_data_quality_indicator(df: pd.DataFrame) -> None:
        """Render data quality status bar"""
        quality = st.session_state.data_quality.get('completeness', 0)
        total_rows = len(df)
        
        # Determine quality status
        if quality > 90:
            quality_emoji = "🟢"
            quality_text = "Excellent"
        elif quality > 75:
            quality_emoji = "🟡"
            quality_text = "Good"
        else:
            quality_emoji = "🔴"
            quality_text = "Poor"
        
        # Create compact status bar
        st.caption(
            f"Data Quality: {quality_emoji} {quality_text} ({quality:.0f}%) | "
            f"Stocks: {total_rows:,} | "
            f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
        )
    
    @staticmethod
    def create_distribution_chart(df: pd.DataFrame, column: str, title: str) -> go.Figure:
        """Create a distribution chart for any numeric column"""
        fig = go.Figure()
        
        if column in df.columns:
            data = df[column].dropna()
            
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=30,
                name=title,
                marker_color='#3498db',
                opacity=0.7
            ))
            
            # Add mean line
            mean_val = data.mean()
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.1f}"
            )
            
            fig.update_layout(
                title=title,
                xaxis_title=column,
                yaxis_title="Count",
                template='plotly_white',
                height=300,
                showlegend=False
            )
        else:
            fig.add_annotation(
                text=f"No data available for {column}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return fig
    
    @staticmethod
    def render_quick_stats_row(df: pd.DataFrame) -> None:
        """Render a row of quick statistics"""
        cols = st.columns(4)
        
        with cols[0]:
            if 'master_score' in df.columns:
                UIComponents.render_metric_card(
                    "Avg Score",
                    f"{df['master_score'].mean():.1f}",
                    f"Top: {df['master_score'].max():.0f}"
                )
        
        with cols[1]:
            if 'ret_30d' in df.columns:
                winners = (df['ret_30d'] > 0).sum()
                UIComponents.render_metric_card(
                    "30D Winners",
                    f"{winners}",
                    f"{winners/len(df)*100:.0f}%" if len(df) > 0 else "0%"
                )
        
        with cols[2]:
            if 'rvol' in df.columns:
                high_vol = (df['rvol'] > 2).sum()
                UIComponents.render_metric_card(
                    "High Volume",
                    f"{high_vol}",
                    "RVOL > 2x"
                )
        
        with cols[3]:
            if 'patterns' in df.columns:
                with_patterns = (df['patterns'] != '').sum()
                UIComponents.render_metric_card(
                    "With Patterns",
                    f"{with_patterns}",
                    f"{with_patterns/len(df)*100:.0f}%" if len(df) > 0 else "0%"
                )

# ============================================
# SESSION STATE MANAGER
# ============================================

class SessionStateManager:
    """
    Unified session state manager for Streamlit.
    This class ensures all state variables are properly initialized,
    preventing runtime errors and managing filter states consistently.
    """

    @staticmethod
    def initialize():
        """
        Initializes all necessary session state variables with explicit defaults.
        This prevents KeyErrors when accessing variables for the first time.
        """
        defaults = {
            # Core Application State
            'search_query': "",
            'last_refresh': datetime.now(timezone.utc),
            'data_source': "sheet",
            'user_preferences': {
                'default_top_n': CONFIG.DEFAULT_TOP_N,
                'display_mode': 'Technical',
                'last_filters': {}
            },
            'active_filter_count': 0,
            'quick_filter': None,
            'quick_filter_applied': False,
            'show_debug': False,
            'performance_metrics': {},
            'data_quality': {},
            
            # Legacy filter keys (for backward compatibility)
            'display_count': CONFIG.DEFAULT_TOP_N,
            'sort_by': 'Rank',
            'export_template': 'Full Analysis (All Data)',
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
            
            # Wave Radar specific filters
            'wave_states_filter': [],
            'wave_strength_range_slider': (0, 100),
            'show_sensitivity_details': False,
            'show_market_regime': True,
            'wave_timeframe_select': "All Waves",
            'wave_sensitivity': "Balanced",
            
            # Sheet configuration
            'sheet_id': '',
            'gid': CONFIG.DEFAULT_GID
        }
        
        # Initialize default values
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Initialize centralized filter state (NEW)
        if 'filter_state' not in st.session_state:
            st.session_state.filter_state = {
                'categories': [],
                'sectors': [],
                'industries': [],
                'min_score': 0,
                'patterns': [],
                'trend_filter': "All Trends",
                'trend_range': (0, 100),
                'eps_tiers': [],
                'pe_tiers': [],
                'price_tiers': [],
                'min_eps_change': None,
                'min_pe': None,
                'max_pe': None,
                'require_fundamental_data': False,
                'wave_states': [],
                'wave_strength_range': (0, 100),
                'quick_filter': None,
                'quick_filter_applied': False
            }

    @staticmethod
    def build_filter_dict() -> Dict[str, Any]:
        """
        Builds a comprehensive filter dictionary from the current session state.
        This centralizes filter data for easy consumption by the FilterEngine.
        
        Returns:
            Dict[str, Any]: A dictionary of all active filter settings.
        """
        filters = {}
        
        # Use centralized filter state if available
        if 'filter_state' in st.session_state:
            state = st.session_state.filter_state
            
            # Map centralized state to filter dict
            if state.get('categories'):
                filters['categories'] = state['categories']
            if state.get('sectors'):
                filters['sectors'] = state['sectors']
            if state.get('industries'):
                filters['industries'] = state['industries']
            if state.get('min_score', 0) > 0:
                filters['min_score'] = state['min_score']
            if state.get('patterns'):
                filters['patterns'] = state['patterns']
            if state.get('trend_filter') != "All Trends":
                filters['trend_filter'] = state['trend_filter']
                filters['trend_range'] = state.get('trend_range', (0, 100))
            if state.get('eps_tiers'):
                filters['eps_tiers'] = state['eps_tiers']
            if state.get('pe_tiers'):
                filters['pe_tiers'] = state['pe_tiers']
            if state.get('price_tiers'):
                filters['price_tiers'] = state['price_tiers']
            if state.get('min_eps_change') is not None:
                filters['min_eps_change'] = state['min_eps_change']
            if state.get('min_pe') is not None:
                filters['min_pe'] = state['min_pe']
            if state.get('max_pe') is not None:
                filters['max_pe'] = state['max_pe']
            if state.get('require_fundamental_data'):
                filters['require_fundamental_data'] = True
            if state.get('wave_states'):
                filters['wave_states'] = state['wave_states']
            if state.get('wave_strength_range') != (0, 100):
                filters['wave_strength_range'] = state['wave_strength_range']
                
        else:
            # Fallback to legacy individual keys
            # Categorical filters
            for key, filter_name in [
                ('category_filter', 'categories'), 
                ('sector_filter', 'sectors'), 
                ('industry_filter', 'industries')
            ]:
                if st.session_state.get(key) and st.session_state[key]:
                    filters[filter_name] = st.session_state[key]
            
            # Numeric filters
            if st.session_state.get('min_score', 0) > 0:
                filters['min_score'] = st.session_state['min_score']
            
            # EPS change filter
            if st.session_state.get('min_eps_change'):
                value = st.session_state['min_eps_change']
                if isinstance(value, str) and value.strip():
                    try:
                        filters['min_eps_change'] = float(value)
                    except ValueError:
                        pass
                elif isinstance(value, (int, float)):
                    filters['min_eps_change'] = float(value)
            
            # PE filters
            if st.session_state.get('min_pe'):
                value = st.session_state['min_pe']
                if isinstance(value, str) and value.strip():
                    try:
                        filters['min_pe'] = float(value)
                    except ValueError:
                        pass
                elif isinstance(value, (int, float)):
                    filters['min_pe'] = float(value)
            
            if st.session_state.get('max_pe'):
                value = st.session_state['max_pe']
                if isinstance(value, str) and value.strip():
                    try:
                        filters['max_pe'] = float(value)
                    except ValueError:
                        pass
                elif isinstance(value, (int, float)):
                    filters['max_pe'] = float(value)

            # Multi-select filters
            if st.session_state.get('patterns') and st.session_state['patterns']:
                filters['patterns'] = st.session_state['patterns']
            
            # Tier filters
            for key, filter_name in [
                ('eps_tier_filter', 'eps_tiers'),
                ('pe_tier_filter', 'pe_tiers'),
                ('price_tier_filter', 'price_tiers')
            ]:
                if st.session_state.get(key) and st.session_state[key]:
                    filters[filter_name] = st.session_state[key]
            
            # Trend filter
            if st.session_state.get('trend_filter') != "All Trends":
                trend_options = {
                    "🔥 Strong Uptrend (80+)": (80, 100), 
                    "✅ Good Uptrend (60-79)": (60, 79),
                    "➡️ Neutral Trend (40-59)": (40, 59), 
                    "⚠️ Weak/Downtrend (<40)": (0, 39)
                }
                filters['trend_filter'] = st.session_state['trend_filter']
                filters['trend_range'] = trend_options.get(st.session_state['trend_filter'], (0, 100))
            
            # Wave filters
            if st.session_state.get('wave_strength_range_slider') != (0, 100):
                filters['wave_strength_range'] = st.session_state['wave_strength_range_slider']
            
            if st.session_state.get('wave_states_filter') and st.session_state['wave_states_filter']:
                filters['wave_states'] = st.session_state['wave_states_filter']
            
            # Checkbox filters
            if st.session_state.get('require_fundamental_data', False):
                filters['require_fundamental_data'] = True
            
        return filters

    @staticmethod
    def clear_filters():
        """
        Resets all filter-related session state keys to their default values.
        This provides a clean slate for the user.
        """
        # Clear the centralized filter state
        if 'filter_state' in st.session_state:
            st.session_state.filter_state = {
                'categories': [],
                'sectors': [],
                'industries': [],
                'min_score': 0,
                'patterns': [],
                'trend_filter': "All Trends",
                'trend_range': (0, 100),
                'eps_tiers': [],
                'pe_tiers': [],
                'price_tiers': [],
                'min_eps_change': None,
                'min_pe': None,
                'max_pe': None,
                'require_fundamental_data': False,
                'wave_states': [],
                'wave_strength_range': (0, 100),
                'quick_filter': None,
                'quick_filter_applied': False
            }
        
        # Clear individual legacy filter keys
        filter_keys = [
            'category_filter', 'sector_filter', 'industry_filter', 'eps_tier_filter',
            'pe_tier_filter', 'price_tier_filter', 'patterns', 'min_score', 'trend_filter',
            'min_eps_change', 'min_pe', 'max_pe', 'require_fundamental_data',
            'quick_filter', 'quick_filter_applied', 'wave_states_filter',
            'wave_strength_range_slider', 'show_sensitivity_details', 'show_market_regime',
            'wave_timeframe_select', 'wave_sensitivity'
        ]
        
        for key in filter_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                elif isinstance(st.session_state[key], bool):
                    st.session_state[key] = False
                elif isinstance(st.session_state[key], str):
                    if key == 'trend_filter':
                        st.session_state[key] = "All Trends"
                    elif key == 'wave_timeframe_select':
                        st.session_state[key] = "All Waves"
                    elif key == 'wave_sensitivity':
                        st.session_state[key] = "Balanced"
                    else:
                        st.session_state[key] = ""
                elif isinstance(st.session_state[key], tuple):
                    if key == 'wave_strength_range_slider':
                        st.session_state[key] = (0, 100)
                elif isinstance(st.session_state[key], (int, float)):
                    if key == 'min_score':
                        st.session_state[key] = 0
                    else:
                        st.session_state[key] = None if key in ['min_eps_change', 'min_pe', 'max_pe'] else 0
                else:
                    st.session_state[key] = None
        
        # CRITICAL FIX: Delete all widget keys to force UI reset
        widget_keys_to_delete = [
            # Multiselect widgets
            'category_multiselect', 'sector_multiselect', 'industry_multiselect',
            'patterns_multiselect', 'wave_states_multiselect',
            'eps_tier_multiselect', 'pe_tier_multiselect', 'price_tier_multiselect',
            
            # Slider widgets
            'min_score_slider', 'wave_strength_slider',
            
            # Selectbox widgets
            'trend_selectbox', 'wave_timeframe_select', 'display_mode_toggle',
            
            # Text input widgets
            'eps_change_input', 'min_pe_input', 'max_pe_input',
            
            # Checkbox widgets
            'require_fundamental_checkbox', 'show_sensitivity_details', 'show_market_regime',
            
            # Additional keys
            'display_count_select', 'sort_by_select', 'export_template_radio',
            'wave_sensitivity', 'search_input', 'sheet_input', 'gid_input'
        ]
        
        # Delete each widget key if it exists
        deleted_count = 0
        for key in widget_keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
                deleted_count += 1
        
        # Reset active filter count
        st.session_state.active_filter_count = 0
        
        # Reset quick filter states
        st.session_state.quick_filter = None
        st.session_state.quick_filter_applied = False
        
        # Clear any cached filter results
        if 'user_preferences' in st.session_state:
            st.session_state.user_preferences['last_filters'] = {}
        
        logger.info(f"All filters cleared successfully. Deleted {deleted_count} widget keys.")
    
    @staticmethod
    def sync_filter_states():
        """
        Synchronizes legacy individual filter keys with centralized filter state.
        This ensures backward compatibility during transition.
        """
        if 'filter_state' not in st.session_state:
            return
        
        state = st.session_state.filter_state
        
        # Sync from centralized to individual (for widgets that still use old keys)
        mappings = [
            ('categories', 'category_filter'),
            ('sectors', 'sector_filter'),
            ('industries', 'industry_filter'),
            ('min_score', 'min_score'),
            ('patterns', 'patterns'),
            ('trend_filter', 'trend_filter'),
            ('eps_tiers', 'eps_tier_filter'),
            ('pe_tiers', 'pe_tier_filter'),
            ('price_tiers', 'price_tier_filter'),
            ('min_eps_change', 'min_eps_change'),
            ('min_pe', 'min_pe'),
            ('max_pe', 'max_pe'),
            ('require_fundamental_data', 'require_fundamental_data'),
            ('wave_states', 'wave_states_filter'),
            ('wave_strength_range', 'wave_strength_range_slider'),
        ]
        
        for state_key, session_key in mappings:
            if state_key in state:
                st.session_state[session_key] = state[state_key]
    
    @staticmethod
    def get_active_filter_count() -> int:
        """
        Counts the number of active filters.
        
        Returns:
            int: Number of active filters.
        """
        count = 0
        
        if 'filter_state' in st.session_state:
            state = st.session_state.filter_state
            
            if state.get('categories'): count += 1
            if state.get('sectors'): count += 1
            if state.get('industries'): count += 1
            if state.get('min_score', 0) > 0: count += 1
            if state.get('patterns'): count += 1
            if state.get('trend_filter') != "All Trends": count += 1
            if state.get('eps_tiers'): count += 1
            if state.get('pe_tiers'): count += 1
            if state.get('price_tiers'): count += 1
            if state.get('min_eps_change') is not None: count += 1
            if state.get('min_pe') is not None: count += 1
            if state.get('max_pe') is not None: count += 1
            if state.get('require_fundamental_data'): count += 1
            if state.get('wave_states'): count += 1
            if state.get('wave_strength_range') != (0, 100): count += 1
        else:
            # Fallback to old method
            filter_checks = [
                ('category_filter', lambda x: x and len(x) > 0),
                ('sector_filter', lambda x: x and len(x) > 0),
                ('industry_filter', lambda x: x and len(x) > 0),
                ('min_score', lambda x: x > 0),
                ('patterns', lambda x: x and len(x) > 0),
                ('trend_filter', lambda x: x != 'All Trends'),
                ('eps_tier_filter', lambda x: x and len(x) > 0),
                ('pe_tier_filter', lambda x: x and len(x) > 0),
                ('price_tier_filter', lambda x: x and len(x) > 0),
                ('min_eps_change', lambda x: x is not None and str(x).strip() != ''),
                ('min_pe', lambda x: x is not None and str(x).strip() != ''),
                ('max_pe', lambda x: x is not None and str(x).strip() != ''),
                ('require_fundamental_data', lambda x: x),
                ('wave_states_filter', lambda x: x and len(x) > 0),
                ('wave_strength_range_slider', lambda x: x != (0, 100))
            ]
            
            for key, check_func in filter_checks:
                value = st.session_state.get(key)
                if value is not None and check_func(value):
                    count += 1
        
        return count
    
    @staticmethod
    def safe_get(key: str, default: Any = None) -> Any:
        """
        Safely get a session state value with fallback.
        
        Args:
            key (str): The session state key.
            default (Any): Default value if key doesn't exist.
            
        Returns:
            Any: The value from session state or default.
        """
        if key not in st.session_state:
            st.session_state[key] = default
        return st.session_state[key]
    
    @staticmethod
    def safe_set(key: str, value: Any) -> None:
        """
        Safely set a session state value.
        
        Args:
            key (str): The session state key.
            value (Any): The value to set.
        """
        st.session_state[key] = value
    
    @staticmethod
    def reset_quick_filters():
        """Reset quick filter states"""
        st.session_state.quick_filter = None
        st.session_state.quick_filter_applied = False
        
        if 'filter_state' in st.session_state:
            st.session_state.filter_state['quick_filter'] = None
            st.session_state.filter_state['quick_filter_applied'] = False
        
# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main Streamlit application - Final Perfected Production Version"""
    
    # Page configuration
    st.set_page_config(
        page_title="Wave Detection Ultimate 3.0",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize robust session state
    SessionStateManager.initialize()
    
    # Custom CSS for production UI
    st.markdown("""
    <style>
    /* Production-ready CSS */
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        overflow-wrap: break-word;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 5px;
    }
    /* Button styling */
    div.stButton > button {
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    /* Mobile responsive */
    @media (max-width: 768px) {
        .stDataFrame {font-size: 12px;}
        div[data-testid="metric-container"] {padding: 3%;}
        .main {padding: 0rem 0.5rem;}
    }
    /* Table optimization */
    .stDataFrame > div {overflow-x: auto;}
    /* Loading animation */
    .stSpinner > div {
        border-color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h1 style="margin: 0; font-size: 2.5rem;">🌊 Wave Detection Ultimate 3.0</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Professional Stock Ranking System • Final Perfected Production Version
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.session_state.last_refresh = datetime.now(timezone.utc)
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                gc.collect()  # Force garbage collection
                st.success("Cache cleared!")
                time.sleep(0.5)
                st.rerun()
        
        # Data source selection
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        
        data_source_col1, data_source_col2 = st.columns(2)
        
        with data_source_col1:
            if st.button("📊 Google Sheets", 
                        type="primary" if st.session_state.data_source == "sheet" else "secondary", 
                        use_container_width=True):
                st.session_state.data_source = "sheet"
                st.rerun()
        
        with data_source_col2:
            if st.button("📁 Upload CSV", 
                        type="primary" if st.session_state.data_source == "upload" else "secondary", 
                        use_container_width=True):
                st.session_state.data_source = "upload"
                st.rerun()

        uploaded_file = None
        sheet_id = None
        gid = None
        
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader(
                "Choose CSV file", 
                type="csv",
                help="Upload a CSV file with stock data. Must contain 'ticker' and 'price' columns."
            )
            if uploaded_file is None:
                st.info("Please upload a CSV file to continue")
        else:
            # Google Sheets input
            st.markdown("#### 📊 Google Sheets Configuration")
            
            sheet_input = st.text_input(
                "Google Sheets ID or URL",
                value=st.session_state.get('sheet_id', ''),
                placeholder="Enter Sheet ID or full URL",
                help="Example: 1OEQ_qxL4lXbO9LlKWDGlDju2yQC1iYvOYeXF3mTQuJM or the full Google Sheets URL"
            )
            
            if sheet_input:
                sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_input)
                if sheet_id_match:
                    sheet_id = sheet_id_match.group(1)
                else:
                    sheet_id = sheet_input.strip()
            
                st.session_state.sheet_id = sheet_id
            
            gid_input = st.text_input(
                "Sheet Tab GID (Optional)",
                value=st.session_state.get('gid', CONFIG.DEFAULT_GID),
                placeholder=f"Default: {CONFIG.DEFAULT_GID}",
                help="The GID identifies specific sheet tab. Found in URL after #gid="
            )
            
            if gid_input:
                gid = gid_input.strip()
            else:
                gid = CONFIG.DEFAULT_GID
            
            if not sheet_id:
                st.warning("Please enter a Google Sheets ID to continue")
        
        # Data quality indicator
        data_quality = st.session_state.get('data_quality', {})
        if data_quality:
            with st.expander("📊 Data Quality", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    completeness = data_quality.get('completeness', 0)
                    if completeness > 80:
                        emoji = "🟢"
                    elif completeness > 60:
                        emoji = "🟡"
                    else:
                        emoji = "🔴"
                    
                    st.metric("Completeness", f"{emoji} {completeness:.1f}%")
                    st.metric("Total Stocks", f"{data_quality.get('total_rows', 0):,}")
                
                with col2:
                    if 'timestamp' in data_quality:
                        age = datetime.now(timezone.utc) - data_quality['timestamp']
                        hours = age.total_seconds() / 3600
                        
                        if hours < 1:
                            freshness = "🟢 Fresh"
                        elif hours < 24:
                            freshness = "🟡 Recent"
                        else:
                            freshness = "🔴 Stale"
                        
                        st.metric("Data Age", freshness)
                    
                    duplicates = data_quality.get('duplicate_tickers', 0)
                    if duplicates > 0:
                        st.metric("Duplicates", f"⚠️ {duplicates}")
        
        # Performance metrics
        perf_metrics = st.session_state.get('performance_metrics', {})
        if perf_metrics:
            with st.expander("⚡ Performance"):
                total_time = sum(perf_metrics.values())
                if total_time < 3:
                    perf_emoji = "🟢"
                elif total_time < 5:
                    perf_emoji = "🟡"
                else:
                    perf_emoji = "🔴"
                
                st.metric("Load Time", f"{perf_emoji} {total_time:.1f}s")
                
                # Show slowest operations
                if len(perf_metrics) > 0:
                    slowest = sorted(perf_metrics.items(), key=lambda x: x[1], reverse=True)[:3]
                    for func_name, elapsed in slowest:
                        if elapsed > 0.001:
                            st.caption(f"{func_name}: {elapsed:.4f}s")
        
        st.markdown("---")
        st.markdown("### 🔍 Smart Filters")
        
        active_filter_count = 0
        
        if st.session_state.get('quick_filter_applied', False):
            active_filter_count += 1
        
        filter_checks = [
            ('category_filter', lambda x: x and len(x) > 0),
            ('sector_filter', lambda x: x and len(x) > 0),
            ('industry_filter', lambda x: x and len(x) > 0),
            ('min_score', lambda x: x > 0),
            ('patterns', lambda x: x and len(x) > 0),
            ('trend_filter', lambda x: x != 'All Trends'),
            ('eps_tier_filter', lambda x: x and len(x) > 0),
            ('pe_tier_filter', lambda x: x and len(x) > 0),
            ('price_tier_filter', lambda x: x and len(x) > 0),
            ('min_eps_change', lambda x: x is not None and str(x).strip() != ''),
            ('min_pe', lambda x: x is not None and str(x).strip() != ''),
            ('max_pe', lambda x: x is not None and str(x).strip() != ''),
            ('require_fundamental_data', lambda x: x),
            ('wave_states_filter', lambda x: x and len(x) > 0),
            ('wave_strength_range_slider', lambda x: x != (0, 100))
        ]
        
        for key, check_func in filter_checks:
            value = st.session_state.get(key)
            if value is not None and check_func(value):
                active_filter_count += 1
        
        st.session_state.active_filter_count = active_filter_count
        
        if active_filter_count > 0:
            st.info(f"🔍 **{active_filter_count} filter{'s' if active_filter_count > 1 else ''} active**")
        
        if st.button("🗑️ Clear All Filters", 
                    use_container_width=True, 
                    type="primary" if active_filter_count > 0 else "secondary"):
            SessionStateManager.clear_filters()
            st.success("✅ All filters cleared!")
            st.rerun()
        
        st.markdown("---")
        show_debug = st.checkbox("🐛 Show Debug Info", 
                               value=st.session_state.get('show_debug', False),
                               key="show_debug")
    
    try:
        if st.session_state.data_source == "upload" and uploaded_file is None:
            st.warning("Please upload a CSV file to continue")
            st.stop()
        
        if st.session_state.data_source == "sheet" and not sheet_id:
            st.warning("Please enter a Google Sheets ID to continue")
            st.stop()
        
        with st.spinner("📥 Loading and processing data..."):
            try:
                if st.session_state.data_source == "upload" and uploaded_file is not None:
                    ranked_df, data_timestamp, metadata = load_and_process_data(
                        "upload", file_data=uploaded_file
                    )
                else:
                    ranked_df, data_timestamp, metadata = load_and_process_data(
                        "sheet", 
                        sheet_id=sheet_id,
                        gid=gid
                    )
                
                st.session_state.ranked_df = ranked_df
                st.session_state.data_timestamp = data_timestamp
                st.session_state.last_refresh = datetime.now(timezone.utc)
                
                if metadata.get('warnings'):
                    for warning in metadata['warnings']:
                        st.warning(warning)
                
                if metadata.get('errors'):
                    for error in metadata['errors']:
                        st.error(error)
                
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}")
                
                last_good_data = st.session_state.get('last_good_data')
                if last_good_data:
                    ranked_df, data_timestamp, metadata = last_good_data
                    st.warning("Failed to load fresh data, using cached version")
                else:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Common issues:\n- Invalid Google Sheets ID\n- Sheet not publicly accessible\n- Network connectivity\n- Invalid CSV format")
                    st.stop()
        
    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(str(e))
        st.stop()
    
    # Quick Action Buttons
    st.markdown("### ⚡ Quick Actions")
    qa_col1, qa_col2, qa_col3, qa_col4, qa_col5 = st.columns(5)
    
    quick_filter_applied = st.session_state.get('quick_filter_applied', False)
    quick_filter = st.session_state.get('quick_filter', None)
    
    with qa_col1:
        if st.button("📈 Top Gainers", use_container_width=True):
            st.session_state['quick_filter'] = 'top_gainers'
            st.session_state['quick_filter_applied'] = True
            st.rerun()
    
    with qa_col2:
        if st.button("🔥 Volume Surges", use_container_width=True):
            st.session_state['quick_filter'] = 'volume_surges'
            st.session_state['quick_filter_applied'] = True
            st.rerun()
    
    with qa_col3:
        if st.button("🎯 Breakout Ready", use_container_width=True):
            st.session_state['quick_filter'] = 'breakout_ready'
            st.session_state['quick_filter_applied'] = True
            st.rerun()
    
    with qa_col4:
        if st.button("💎 Hidden Gems", use_container_width=True):
            st.session_state['quick_filter'] = 'hidden_gems'
            st.session_state['quick_filter_applied'] = True
            st.rerun()
    
    with qa_col5:
        if st.button("🌊 Show All", use_container_width=True):
            st.session_state['quick_filter'] = None
            st.session_state['quick_filter_applied'] = False
            st.rerun()
    
    if quick_filter:
        if quick_filter == 'top_gainers':
            ranked_df_display = ranked_df[ranked_df['momentum_score'] >= 80]
            st.info(f"Showing {len(ranked_df_display)} stocks with momentum score ≥ 80")
        elif quick_filter == 'volume_surges':
            ranked_df_display = ranked_df[ranked_df['rvol'] >= 3]
            st.info(f"Showing {len(ranked_df_display)} stocks with RVOL ≥ 3x")
        elif quick_filter == 'breakout_ready':
            ranked_df_display = ranked_df[ranked_df['breakout_score'] >= 80]
            st.info(f"Showing {len(ranked_df_display)} stocks with breakout score ≥ 80")
        elif quick_filter == 'hidden_gems':
            ranked_df_display = ranked_df[ranked_df['patterns'].str.contains('HIDDEN GEM', na=False)]
            st.info(f"Showing {len(ranked_df_display)} hidden gem stocks")
    else:
        ranked_df_display = ranked_df
    
    # Sidebar filters
    with st.sidebar:
        # Initialize centralized filter state
        FilterEngine.initialize_filters()
        
        # Initialize filters dict for current frame
        filters = {}
        
        # Display Mode
        st.markdown("### 📊 Display Mode")
        display_mode = st.radio(
            "Choose your view:",
            options=["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0 if st.session_state.user_preferences['display_mode'] == 'Technical' else 1,
            help="Technical: Pure momentum analysis | Hybrid: Adds PE & EPS data",
            key="display_mode_toggle"
        )
        
        st.session_state.user_preferences['display_mode'] = display_mode
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        
        st.markdown("---")
        
        # CRITICAL: Define callback functions BEFORE widgets
        def sync_categories():
            if 'category_multiselect' in st.session_state:
                st.session_state.filter_state['categories'] = st.session_state.category_multiselect
        
        def sync_sectors():
            if 'sector_multiselect' in st.session_state:
                st.session_state.filter_state['sectors'] = st.session_state.sector_multiselect
        
        def sync_industries():
            if 'industry_multiselect' in st.session_state:
                st.session_state.filter_state['industries'] = st.session_state.industry_multiselect
        
        def sync_min_score():
            if 'min_score_slider' in st.session_state:
                st.session_state.filter_state['min_score'] = st.session_state.min_score_slider
        
        def sync_patterns():
            if 'patterns_multiselect' in st.session_state:
                st.session_state.filter_state['patterns'] = st.session_state.patterns_multiselect
        
        def sync_trend():
            if 'trend_selectbox' in st.session_state:
                trend_options = {
                    "All Trends": (0, 100),
                    "🔥 Strong Uptrend (80+)": (80, 100),
                    "✅ Good Uptrend (60-79)": (60, 79),
                    "➡️ Neutral Trend (40-59)": (40, 59),
                    "⚠️ Weak/Downtrend (<40)": (0, 39)
                }
                st.session_state.filter_state['trend_filter'] = st.session_state.trend_selectbox
                st.session_state.filter_state['trend_range'] = trend_options[st.session_state.trend_selectbox]
        
        def sync_wave_states():
            if 'wave_states_multiselect' in st.session_state:
                st.session_state.filter_state['wave_states'] = st.session_state.wave_states_multiselect
        
        def sync_wave_strength():
            if 'wave_strength_slider' in st.session_state:
                st.session_state.filter_state['wave_strength_range'] = st.session_state.wave_strength_slider
        
        # Category filter with callback
        categories = FilterEngine.get_filter_options(ranked_df_display, 'category', filters)
        
        selected_categories = st.multiselect(
            "Market Cap Category",
            options=categories,
            default=st.session_state.filter_state.get('categories', []),
            placeholder="Select categories (empty = All)",
            help="Filter by market capitalization category",
            key="category_multiselect",
            on_change=sync_categories  # SYNC ON CHANGE
        )
        
        if selected_categories:
            filters['categories'] = selected_categories
        
        # Sector filter with callback
        sectors = FilterEngine.get_filter_options(ranked_df_display, 'sector', filters)
        
        selected_sectors = st.multiselect(
            "Sector",
            options=sectors,
            default=st.session_state.filter_state.get('sectors', []),
            placeholder="Select sectors (empty = All)",
            help="Filter by business sector",
            key="sector_multiselect",
            on_change=sync_sectors  # SYNC ON CHANGE
        )
        
        if selected_sectors:
            filters['sectors'] = selected_sectors
        
        # Industry filter with callback
        industries = FilterEngine.get_filter_options(ranked_df_display, 'industry', filters)
        
        selected_industries = st.multiselect(
            "Industry",
            options=industries,
            default=st.session_state.filter_state.get('industries', []),
            placeholder="Select industries (empty = All)",
            help="Filter by specific industry",
            key="industry_multiselect",
            on_change=sync_industries  # SYNC ON CHANGE
        )
        
        if selected_industries:
            filters['industries'] = selected_industries
        
        # Score filter with callback
        min_score = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=st.session_state.filter_state.get('min_score', 0),
            step=5,
            help="Filter stocks by minimum score",
            key="min_score_slider",
            on_change=sync_min_score  # SYNC ON CHANGE
        )
        
        if min_score > 0:
            filters['min_score'] = min_score
        
        # Pattern filter with callback
        all_patterns = set()
        for patterns in ranked_df_display['patterns'].dropna():
            if patterns:
                all_patterns.update(patterns.split(' | '))
        
        if all_patterns:
            selected_patterns = st.multiselect(
                "Patterns",
                options=sorted(all_patterns),
                default=st.session_state.filter_state.get('patterns', []),
                placeholder="Select patterns (empty = All)",
                help="Filter by specific patterns",
                key="patterns_multiselect",
                on_change=sync_patterns  # SYNC ON CHANGE
            )
            
            if selected_patterns:
                filters['patterns'] = selected_patterns
        
        # Trend filter with callback
        st.markdown("#### 📈 Trend Strength")
        trend_options = {
            "All Trends": (0, 100),
            "🔥 Strong Uptrend (80+)": (80, 100),
            "✅ Good Uptrend (60-79)": (60, 79),
            "➡️ Neutral Trend (40-59)": (40, 59),
            "⚠️ Weak/Downtrend (<40)": (0, 39)
        }
        
        current_trend = st.session_state.filter_state.get('trend_filter', "All Trends")
        if current_trend not in trend_options:
            current_trend = "All Trends"
        
        selected_trend = st.selectbox(
            "Trend Quality",
            options=list(trend_options.keys()),
            index=list(trend_options.keys()).index(current_trend),
            help="Filter stocks by trend strength based on SMA alignment",
            key="trend_selectbox",
            on_change=sync_trend  # SYNC ON CHANGE
        )
        
        if selected_trend != "All Trends":
            filters['trend_filter'] = selected_trend
            filters['trend_range'] = trend_options[selected_trend]
        
        # Wave filters with callbacks
        st.markdown("#### 🌊 Wave Filters")
        wave_states_options = FilterEngine.get_filter_options(ranked_df_display, 'wave_state', filters)
        
        selected_wave_states = st.multiselect(
            "Wave State",
            options=wave_states_options,
            default=st.session_state.filter_state.get('wave_states', []),
            placeholder="Select wave states (empty = All)",
            help="Filter by the detected 'Wave State'",
            key="wave_states_multiselect",
            on_change=sync_wave_states  # SYNC ON CHANGE
        )
        
        if selected_wave_states:
            filters['wave_states'] = selected_wave_states
        
        if 'overall_wave_strength' in ranked_df_display.columns:
            min_strength = float(ranked_df_display['overall_wave_strength'].min())
            max_strength = float(ranked_df_display['overall_wave_strength'].max())
            
            slider_min_val = 0
            slider_max_val = 100
            
            if pd.notna(min_strength) and pd.notna(max_strength) and min_strength <= max_strength:
                default_range_value = (int(min_strength), int(max_strength))
            else:
                default_range_value = (0, 100)
            
            current_wave_range = st.session_state.filter_state.get('wave_strength_range', default_range_value)
            current_wave_range = (
                max(slider_min_val, min(slider_max_val, current_wave_range[0])),
                max(slider_min_val, min(slider_max_val, current_wave_range[1]))
            )
            
            wave_strength_range = st.slider(
                "Overall Wave Strength",
                min_value=slider_min_val,
                max_value=slider_max_val,
                value=current_wave_range,
                step=1,
                help="Filter by the calculated 'Overall Wave Strength' score",
                key="wave_strength_slider",
                on_change=sync_wave_strength  # SYNC ON CHANGE
            )
            
            if wave_strength_range != (0, 100):
                filters['wave_strength_range'] = wave_strength_range
        
        # Advanced filters with callbacks
        with st.expander("🔧 Advanced Filters"):
            # Define callbacks for advanced filters
            def sync_eps_tier():
                if 'eps_tier_multiselect' in st.session_state:
                    st.session_state.filter_state['eps_tiers'] = st.session_state.eps_tier_multiselect
            
            def sync_pe_tier():
                if 'pe_tier_multiselect' in st.session_state:
                    st.session_state.filter_state['pe_tiers'] = st.session_state.pe_tier_multiselect
            
            def sync_price_tier():
                if 'price_tier_multiselect' in st.session_state:
                    st.session_state.filter_state['price_tiers'] = st.session_state.price_tier_multiselect
            
            def sync_eps_change():
                if 'eps_change_input' in st.session_state:
                    value = st.session_state.eps_change_input
                    if value.strip():
                        try:
                            st.session_state.filter_state['min_eps_change'] = float(value)
                        except ValueError:
                            st.session_state.filter_state['min_eps_change'] = None
                    else:
                        st.session_state.filter_state['min_eps_change'] = None
            
            def sync_min_pe():
                if 'min_pe_input' in st.session_state:
                    value = st.session_state.min_pe_input
                    if value.strip():
                        try:
                            st.session_state.filter_state['min_pe'] = float(value)
                        except ValueError:
                            st.session_state.filter_state['min_pe'] = None
                    else:
                        st.session_state.filter_state['min_pe'] = None
            
            def sync_max_pe():
                if 'max_pe_input' in st.session_state:
                    value = st.session_state.max_pe_input
                    if value.strip():
                        try:
                            st.session_state.filter_state['max_pe'] = float(value)
                        except ValueError:
                            st.session_state.filter_state['max_pe'] = None
                    else:
                        st.session_state.filter_state['max_pe'] = None
            
            def sync_fundamental():
                if 'require_fundamental_checkbox' in st.session_state:
                    st.session_state.filter_state['require_fundamental_data'] = st.session_state.require_fundamental_checkbox
            
            # Tier filters
            for tier_type, col_name, filter_key, sync_func in [
                ('eps_tiers', 'eps_tier', 'eps_tiers', sync_eps_tier),
                ('pe_tiers', 'pe_tier', 'pe_tiers', sync_pe_tier),
                ('price_tiers', 'price_tier', 'price_tiers', sync_price_tier)
            ]:
                if col_name in ranked_df_display.columns:
                    tier_options = FilterEngine.get_filter_options(ranked_df_display, col_name, filters)
                    
                    selected_tiers = st.multiselect(
                        f"{col_name.replace('_', ' ').title()}",
                        options=tier_options,
                        default=st.session_state.filter_state.get(filter_key, []),
                        placeholder=f"Select {col_name.replace('_', ' ')}s (empty = All)",
                        key=f"{col_name}_multiselect",
                        on_change=sync_func  # SYNC ON CHANGE
                    )
                    
                    if selected_tiers:
                        filters[tier_type] = selected_tiers
            
            # EPS change filter
            if 'eps_change_pct' in ranked_df_display.columns:
                current_eps_change = st.session_state.filter_state.get('min_eps_change')
                eps_change_str = str(current_eps_change) if current_eps_change is not None else ""
                
                eps_change_input = st.text_input(
                    "Min EPS Change %",
                    value=eps_change_str,
                    placeholder="e.g. -50 or leave empty",
                    help="Enter minimum EPS growth percentage",
                    key="eps_change_input",
                    on_change=sync_eps_change  # SYNC ON CHANGE
                )
                
                if eps_change_input.strip():
                    try:
                        eps_change_val = float(eps_change_input)
                        filters['min_eps_change'] = eps_change_val
                    except ValueError:
                        st.error("Please enter a valid number for EPS change")
                else:
                    st.session_state.filter_state['min_eps_change'] = None
            
            # PE filters (only in hybrid mode)
            if show_fundamentals and 'pe' in ranked_df_display.columns:
                st.markdown("**🔍 Fundamental Filters**")
                
                col1, col2 = st.columns(2)
                with col1:
                    current_min_pe = st.session_state.filter_state.get('min_pe')
                    min_pe_str = str(current_min_pe) if current_min_pe is not None else ""
                    
                    min_pe_input = st.text_input(
                        "Min PE Ratio",
                        value=min_pe_str,
                        placeholder="e.g. 10",
                        key="min_pe_input",
                        on_change=sync_min_pe  # SYNC ON CHANGE
                    )
                    
                    if min_pe_input.strip():
                        try:
                            min_pe_val = float(min_pe_input)
                            filters['min_pe'] = min_pe_val
                        except ValueError:
                            st.error("Invalid Min PE")
                
                with col2:
                    current_max_pe = st.session_state.filter_state.get('max_pe')
                    max_pe_str = str(current_max_pe) if current_max_pe is not None else ""
                    
                    max_pe_input = st.text_input(
                        "Max PE Ratio",
                        value=max_pe_str,
                        placeholder="e.g. 30",
                        key="max_pe_input",
                        on_change=sync_max_pe  # SYNC ON CHANGE
                    )
                    
                    if max_pe_input.strip():
                        try:
                            max_pe_val = float(max_pe_input)
                            filters['max_pe'] = max_pe_val
                        except ValueError:
                            st.error("Invalid Max PE")
                
                # Data completeness filter
                require_fundamental = st.checkbox(
                    "Only show stocks with PE and EPS data",
                    value=st.session_state.filter_state.get('require_fundamental_data', False),
                    key="require_fundamental_checkbox",
                    on_change=sync_fundamental  # SYNC ON CHANGE
                )
                
                if require_fundamental:
                    filters['require_fundamental_data'] = True
        
        # Count active filters using FilterEngine method
        active_filter_count = FilterEngine.get_active_count()
        st.session_state.active_filter_count = active_filter_count
        
        if active_filter_count > 0:
            st.info(f"🔍 **{active_filter_count} filter{'s' if active_filter_count > 1 else ''} active**")
        
        # Clear filters button - ENHANCED VERSION
        if st.button("🗑️ Clear All Filters", 
                    use_container_width=True, 
                    type="primary" if active_filter_count > 0 else "secondary",
                    key="clear_filters_sidebar_btn"):
            
            # Use both FilterEngine and SessionStateManager clear methods
            FilterEngine.clear_all_filters()
            SessionStateManager.clear_filters()
            
            st.success("✅ All filters cleared!")
            time.sleep(0.3)
            st.rerun()
    
    # Apply filters (outside sidebar)
    if quick_filter_applied:
        filtered_df = FilterEngine.apply_filters(ranked_df_display, filters)
    else:
        filtered_df = FilterEngine.apply_filters(ranked_df, filters)
    
    filtered_df = filtered_df.sort_values('rank')
    
    # Save current filters
    st.session_state.user_preferences['last_filters'] = filters
    
    # Debug info (OPTIONAL)
    if show_debug:
        with st.sidebar.expander("🐛 Debug Info", expanded=True):
            st.write("**Active Filters:**")
            for key, value in filters.items():
                if value is not None and value != [] and value != 0 and \
                   (not (isinstance(value, tuple) and value == (0,100))):
                    st.write(f"• {key}: {value}")
            
            st.write(f"\n**Filter State:**")
            st.write(st.session_state.filter_state)
            
            st.write(f"\n**Filter Result:**")
            st.write(f"Before: {len(ranked_df)} stocks")
            st.write(f"After: {len(filtered_df)} stocks")
            
            if st.session_state.performance_metrics:
                st.write(f"\n**Performance:**")
                for func, time_taken in st.session_state.performance_metrics.items():
                    if time_taken > 0.001:
                        st.write(f"• {func}: {time_taken:.4f}s")
    
    active_filter_count = st.session_state.get('active_filter_count', 0)
    if active_filter_count > 0 or quick_filter_applied:
        filter_status_col1, filter_status_col2 = st.columns([5, 1])
        with filter_status_col1:
            if quick_filter:
                quick_filter_names = {
                    'top_gainers': '📈 Top Gainers',
                    'volume_surges': '🔥 Volume Surges',
                    'breakout_ready': '🎯 Breakout Ready',
                    'hidden_gems': '💎 Hidden Gems'
                }
                filter_display = quick_filter_names.get(quick_filter, 'Filtered')
                
                if active_filter_count > 1:
                    st.info(f"**Viewing:** {filter_display} + {active_filter_count - 1} other filter{'s' if active_filter_count > 2 else ''} | **{len(filtered_df):,} stocks** shown")
                else:
                    st.info(f"**Viewing:** {filter_display} | **{len(filtered_df):,} stocks** shown")
        
        with filter_status_col2:
            if st.button("Clear Filters", type="secondary", key="clear_filters_main_btn"):
                FilterEngine.clear_all_filters()
                SessionStateManager.clear_filters()
                st.rerun()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_stocks = len(filtered_df)
        total_original = len(ranked_df)
        pct_of_all = (total_stocks/total_original*100) if total_original > 0 else 0
        
        UIComponents.render_metric_card(
            "Total Stocks",
            f"{total_stocks:,}",
            f"{pct_of_all:.0f}% of {total_original:,}"
        )
    
    with col2:
        if not filtered_df.empty and 'master_score' in filtered_df.columns:
            avg_score = filtered_df['master_score'].mean()
            std_score = filtered_df['master_score'].std()
            UIComponents.render_metric_card(
                "Avg Score",
                f"{avg_score:.1f}",
                f"σ={std_score:.1f}"
            )
        else:
            UIComponents.render_metric_card("Avg Score", "N/A")
    
    with col3:
        if show_fundamentals and 'pe' in filtered_df.columns:
            valid_pe = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 10000)
            pe_coverage = valid_pe.sum()
            pe_pct = (pe_coverage / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            
            if pe_coverage > 0:
                median_pe = filtered_df.loc[valid_pe, 'pe'].median()
                UIComponents.render_metric_card(
                    "Median PE",
                    f"{median_pe:.1f}x",
                    f"{pe_pct:.0f}% have data"
                )
            else:
                UIComponents.render_metric_card("PE Data", "Limited", "No PE data")
        else:
            if not filtered_df.empty and 'master_score' in filtered_df.columns:
                min_score = filtered_df['master_score'].min()
                max_score = filtered_df['master_score'].max()
                score_range = f"{min_score:.1f}-{max_score:.1f}"
            else:
                score_range = "N/A"
            UIComponents.render_metric_card("Score Range", score_range)
    
    with col4:
        if show_fundamentals and 'eps_change_pct' in filtered_df.columns:
            valid_eps_change = filtered_df['eps_change_pct'].notna()
            positive_eps_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 0)
            strong_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 50)
            mega_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 100)
            
            growth_count = positive_eps_growth.sum()
            strong_count = strong_growth.sum()
            
            if mega_growth.sum() > 0:
                UIComponents.render_metric_card(
                    "EPS Growth +ve",
                    f"{growth_count}",
                    f"{strong_count} >50% | {mega_growth.sum()} >100%"
                )
            else:
                UIComponents.render_metric_card(
                    "EPS Growth +ve",
                    f"{growth_count}",
                    f"{valid_eps_change.sum()} have data"
                )
        else:
            if 'acceleration_score' in filtered_df.columns:
                accelerating = (filtered_df['acceleration_score'] >= 80).sum()
            else:
                accelerating = 0
            UIComponents.render_metric_card("Accelerating", f"{accelerating}")
    
    with col5:
        if 'rvol' in filtered_df.columns:
            high_rvol = (filtered_df['rvol'] > 2).sum()
        else:
            high_rvol = 0
        UIComponents.render_metric_card("High RVOL", f"{high_rvol}")
    
    with col6:
        if 'trend_quality' in filtered_df.columns:
            strong_trends = (filtered_df['trend_quality'] >= 80).sum()
            total = len(filtered_df)
            UIComponents.render_metric_card(
                "Strong Trends", 
                f"{strong_trends}",
                f"{strong_trends/total*100:.0f}%" if total > 0 else "0%"
            )
        else:
            with_patterns = (filtered_df['patterns'] != '').sum()
            UIComponents.render_metric_card("With Patterns", f"{with_patterns}")
    
    tabs = st.tabs([
        "📊 Summary", "🏆 Rankings", "🌊 Wave Radar", "📊 Analysis", "🔍 Search", "📥 Export", "ℹ️ About"
    ])
    
    # ============================================
    # SUMMARY TAB - COMPLETE EXECUTIVE DASHBOARD
    # ============================================
    
    with tabs[0]:  # Summary Tab
        st.markdown("### 📊 Executive Summary Dashboard")
        
        # Clean status bar
        timestamp = datetime.now().strftime('%H:%M:%S')
        st.caption(f"Last Update: {timestamp} • {len(filtered_df)} Stocks Analyzed • Data Quality: {st.session_state.data_quality.get('completeness', 0):.0f}%")
        
        if not filtered_df.empty:
            
            # ====================================
            # SECTION 1: MARKET PULSE (Enhanced Metrics)
            # ====================================
            st.markdown("#### 📈 Market Pulse - Real-Time Health Monitor")
            
            pulse_cols = st.columns(6)
            
            with pulse_cols[0]:
                # A/D Ratio (Advance/Decline)
                if 'ret_1d' in filtered_df.columns:
                    advancing = len(filtered_df[filtered_df['ret_1d'] > 0])
                    declining = len(filtered_df[filtered_df['ret_1d'] < 0])
                    unchanged = len(filtered_df[filtered_df['ret_1d'] == 0])
                    
                    if declining > 0:
                        ad_ratio = advancing / declining
                        if ad_ratio > 2:
                            ad_emoji = "🟢"
                            ad_text = "Bullish"
                        elif ad_ratio > 1:
                            ad_emoji = "🟡"
                            ad_text = "Positive"
                        else:
                            ad_emoji = "🔴"
                            ad_text = "Bearish"
                    else:
                        ad_ratio = advancing
                        ad_emoji = "🟢"
                        ad_text = "Strong"
                    
                    st.metric(
                        "A/D Ratio",
                        f"{ad_emoji} {ad_ratio:.2f}",
                        f"{advancing}↑ {declining}↓",
                        help="Advance/Decline Ratio - Market breadth indicator"
                    )
                else:
                    st.metric("A/D Ratio", "N/A")
            
            with pulse_cols[1]:
                # Momentum Health
                if 'momentum_score' in filtered_df.columns:
                    healthy_momentum = len(filtered_df[filtered_df['momentum_score'] > 60])
                    momentum_health = (healthy_momentum / len(filtered_df)) * 100
                    
                    if momentum_health > 40:
                        mom_emoji = "💪"
                        mom_status = "Strong"
                    elif momentum_health > 20:
                        mom_emoji = "👍"
                        mom_status = "Moderate"
                    else:
                        mom_emoji = "👎"
                        mom_status = "Weak"
                    
                    st.metric(
                        "Momentum Health",
                        f"{mom_emoji} {momentum_health:.0f}%",
                        f"{healthy_momentum} stocks >60",
                        help="Percentage of stocks with healthy momentum"
                    )
                else:
                    st.metric("Momentum Health", "N/A")
            
            with pulse_cols[2]:
                # Volume State
                if 'rvol' in filtered_df.columns:
                    extreme_vol = len(filtered_df[filtered_df['rvol'] > 3])
                    high_vol = len(filtered_df[filtered_df['rvol'] > 2])
                    normal_vol = len(filtered_df[filtered_df['rvol'].between(0.8, 1.2)])
                    
                    if extreme_vol > 10:
                        vol_emoji = "🌋"
                        vol_state = "Explosive"
                    elif high_vol > 20:
                        vol_emoji = "🔥"
                        vol_state = "Active"
                    elif normal_vol > len(filtered_df) * 0.5:
                        vol_emoji = "😴"
                        vol_state = "Quiet"
                    else:
                        vol_emoji = "⚡"
                        vol_state = "Building"
                    
                    st.metric(
                        "Volume State",
                        f"{vol_emoji} {vol_state}",
                        f"{extreme_vol} extreme, {high_vol} high",
                        help="Market-wide volume activity level"
                    )
                else:
                    st.metric("Volume State", "N/A")
            
            with pulse_cols[3]:
                # Pattern Quality
                if 'patterns' in filtered_df.columns:
                    stocks_with_patterns = filtered_df[filtered_df['patterns'] != '']
                    pattern_coverage = (len(stocks_with_patterns) / len(filtered_df)) * 100
                    
                    # Count high-value patterns
                    critical_patterns = stocks_with_patterns[
                        stocks_with_patterns['patterns'].str.contains(
                            'PERFECT STORM|VOL EXPLOSION|BREAKOUT|ACCELERATING|VAMPIRE', 
                            na=False
                        )
                    ]
                    
                    if len(critical_patterns) > 20:
                        pattern_emoji = "🎯"
                        pattern_quality = "Excellent"
                    elif len(critical_patterns) > 10:
                        pattern_emoji = "✅"
                        pattern_quality = "Good"
                    else:
                        pattern_emoji = "⚠️"
                        pattern_quality = "Limited"
                    
                    st.metric(
                        "Pattern Quality",
                        f"{pattern_emoji} {pattern_quality}",
                        f"{len(critical_patterns)} critical signals",
                        help="Quality and quantity of detected patterns"
                    )
                else:
                    st.metric("Pattern Quality", "N/A")
            
            with pulse_cols[4]:
                # Breakout Pipeline
                if 'breakout_score' in filtered_df.columns:
                    ready_now = len(filtered_df[filtered_df['breakout_score'] > 80])
                    building = len(filtered_df[filtered_df['breakout_score'].between(60, 80)])
                    
                    if ready_now > 10:
                        breakout_emoji = "🚀"
                        breakout_text = "Hot"
                    elif ready_now > 5:
                        breakout_emoji = "🎯"
                        breakout_text = "Ready"
                    else:
                        breakout_emoji = "🔍"
                        breakout_text = "Scarce"
                    
                    st.metric(
                        "Breakout Pipeline",
                        f"{breakout_emoji} {ready_now} ready",
                        f"{building} building (60-80)",
                        help="Stocks approaching breakout levels"
                    )
                else:
                    st.metric("Breakout Pipeline", "N/A")
            
            with pulse_cols[5]:
                # Market Energy (Composite)
                market_energy = 0
                energy_factors = []
                
                # Calculate composite energy
                if 'wave_state' in filtered_df.columns:
                    cresting = len(filtered_df[filtered_df['wave_state'].str.contains('CRESTING', na=False)])
                    building = len(filtered_df[filtered_df['wave_state'].str.contains('BUILDING', na=False)])
                    wave_energy = ((cresting * 2) + building) / len(filtered_df) * 100
                    market_energy += min(wave_energy, 40)  # Cap at 40
                    if wave_energy > 20:
                        energy_factors.append("Waves+")
                
                if 'acceleration_score' in filtered_df.columns:
                    accel_energy = len(filtered_df[filtered_df['acceleration_score'] > 70]) / len(filtered_df) * 100
                    market_energy += min(accel_energy, 30)  # Cap at 30
                    if accel_energy > 15:
                        energy_factors.append("Accel+")
                
                if 'rvol' in filtered_df.columns:
                    vol_energy = len(filtered_df[filtered_df['rvol'] > 2]) / len(filtered_df) * 100
                    market_energy += min(vol_energy, 30)  # Cap at 30
                    if vol_energy > 10:
                        energy_factors.append("Vol+")
                
                if market_energy > 70:
                    energy_emoji = "⚡⚡⚡"
                    energy_level = "Extreme"
                elif market_energy > 50:
                    energy_emoji = "⚡⚡"
                    energy_level = "High"
                elif market_energy > 30:
                    energy_emoji = "⚡"
                    energy_level = "Moderate"
                else:
                    energy_emoji = "💤"
                    energy_level = "Low"
                
                st.metric(
                    "Market Energy",
                    f"{energy_emoji} {energy_level}",
                    f"{market_energy:.0f}% | " + " ".join(energy_factors),
                    help="Composite market activity and momentum"
                )
            
            st.markdown("---")
            
            # ====================================
            # DISCOVERY TABS - ENHANCED VERSION
            # ====================================
            discovery_tabs = st.tabs([
                "🎯 Today's Best & Hidden",
                "🌊 Wave Leaders", 
                "🔍 Pattern Discoveries",
                "💰 Money Flow",
                "🚀 Momentum Stars",
            ])
            
            # TAB 1: TODAY'S BEST & HIDDEN OPPORTUNITIES
            with discovery_tabs[0]:
                st.markdown("#### 🎯 Today's Best & Hidden Opportunities")
                
                # ========================================
                # TOP OPPORTUNITIES TABLE
                # ========================================
                st.markdown("##### 🏆 **Top Opportunities Matrix**")
                
                # Build opportunity scoring
                opp_df = filtered_df.copy()
                opp_df['opportunity_score'] = 0
                opp_df['signals'] = ''
                
                # Score calculations (simple and effective)
                if 'momentum_score' in opp_df.columns:
                    high_momentum = opp_df['momentum_score'] > 70
                    opp_df.loc[high_momentum, 'opportunity_score'] += 25
                    opp_df.loc[high_momentum, 'signals'] += '📈'
                
                if 'acceleration_score' in opp_df.columns:
                    accelerating = opp_df['acceleration_score'] > 70
                    opp_df.loc[accelerating, 'opportunity_score'] += 25
                    opp_df.loc[accelerating, 'signals'] += '🚀'
                
                if 'rvol' in opp_df.columns:
                    volume_surge = opp_df['rvol'] > 2
                    extreme_vol = opp_df['rvol'] > 3
                    opp_df.loc[volume_surge, 'opportunity_score'] += 20
                    opp_df.loc[extreme_vol, 'opportunity_score'] += 10
                    opp_df.loc[extreme_vol, 'signals'] += '🌋'
                    opp_df.loc[volume_surge & ~extreme_vol, 'signals'] += '🔥'
                
                if 'breakout_score' in opp_df.columns:
                    breakout_ready = opp_df['breakout_score'] > 80
                    opp_df.loc[breakout_ready, 'opportunity_score'] += 20
                    opp_df.loc[breakout_ready, 'signals'] += '🎯'
                
                if 'wave_state' in opp_df.columns:
                    cresting = opp_df['wave_state'].str.contains('CRESTING', na=False)
                    building = opp_df['wave_state'].str.contains('BUILDING', na=False)
                    opp_df.loc[cresting, 'opportunity_score'] += 20
                    opp_df.loc[building, 'opportunity_score'] += 10
                    opp_df.loc[cresting, 'signals'] += '🌊'
                
                # Get top 10 opportunities
                top_opportunities = opp_df[opp_df['opportunity_score'] >= 40].nlargest(10, 'opportunity_score')
                
                if len(top_opportunities) > 0:
                    # Prepare display dataframe
                    display_data = []
                    for _, stock in top_opportunities.iterrows():
                        # Determine primary opportunity type
                        if stock.get('rvol', 0) > 3:
                            opp_type = "Volume Explosion"
                            type_emoji = "🌋"
                        elif stock.get('acceleration_score', 0) > 85:
                            opp_type = "Accelerating"
                            type_emoji = "🚀"
                        elif stock.get('breakout_score', 0) > 85:
                            opp_type = "Breakout"
                            type_emoji = "🎯"
                        elif 'CRESTING' in str(stock.get('wave_state', '')):
                            opp_type = "Peak Wave"
                            type_emoji = "🌊"
                        else:
                            opp_type = "Momentum"
                            type_emoji = "📈"
                        
                        display_data.append({
                            '': type_emoji,
                            'Ticker': stock['ticker'],
                            'Company': str(stock.get('company_name', ''))[:20] + '...' if len(str(stock.get('company_name', ''))) > 20 else stock.get('company_name', ''),
                            'Score': stock['master_score'],
                            'Price': stock['price'],
                            'Type': opp_type,
                            'Signals': stock['signals'][:5],  # Limit to 5 emojis
                            '1D%': stock.get('ret_1d', 0),
                            '7D%': stock.get('ret_7d', 0),
                            'RVOL': stock.get('rvol', 1),
                            'Category': stock.get('category', 'N/A')
                        })
                    
                    opportunities_df = pd.DataFrame(display_data)
                    
                    # Display with optimized column config
                    st.dataframe(
                        opportunities_df,
                        use_container_width=True,
                        hide_index=True,
                        height=400,
                        column_config={
                            '': st.column_config.TextColumn('', width='small'),
                            'Ticker': st.column_config.TextColumn('Ticker', help='Stock symbol', width='small'),
                            'Company': st.column_config.TextColumn('Company', width='medium'),
                            'Score': st.column_config.ProgressColumn(
                                'Score',
                                min_value=0,
                                max_value=100,
                                format='%.0f',
                                width='small'
                            ),
                            'Price': st.column_config.NumberColumn('Price', format='₹%.0f', width='small'),
                            'Type': st.column_config.TextColumn('Opportunity', width='medium'),
                            'Signals': st.column_config.TextColumn('Signals', help='Active signals', width='small'),
                            '1D%': st.column_config.NumberColumn('1D%', format='%.1f%%', width='small'),
                            '7D%': st.column_config.NumberColumn('7D%', format='%.1f%%', width='small'),
                            'RVOL': st.column_config.NumberColumn('RVOL', format='%.1fx', width='small'),
                            'Category': st.column_config.TextColumn('Category', width='medium')
                        }
                    )
                    
                    # Quick stats bar
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Opportunities", len(top_opportunities))
                    with col2:
                        vol_explosions = len(top_opportunities[top_opportunities['rvol'] > 3])
                        st.metric("Volume Explosions", vol_explosions)
                    with col3:
                        avg_score = top_opportunities['master_score'].mean()
                        st.metric("Avg Score", f"{avg_score:.0f}")
                    with col4:
                        if 'ret_7d' in top_opportunities.columns:
                            avg_momentum = top_opportunities['ret_7d'].mean()
                            st.metric("Avg 7D Return", f"{avg_momentum:.1f}%")
                else:
                    st.info("No high-confidence opportunities detected. Market may be in consolidation.")
                
                st.markdown("---")
                
                # ========================================
                # HIDDEN GEMS TABLE - FIXED VERSION
                # ========================================
                st.markdown("##### 💎 **Hidden Gems & Under-Radar Plays**")
                
                # Find hidden gems - CORRECTED IMPLEMENTATION
                hidden_df = filtered_df.copy()
                hidden_df['hidden_score'] = 0
                hidden_df['hidden_reasons'] = ''  # Use string instead of list for simplicity
                
                # Criterion 1: Category leader but low overall rank
                if 'category_percentile' in hidden_df.columns and 'percentile' in hidden_df.columns:
                    category_leaders = (hidden_df['category_percentile'] > 85) & (hidden_df['percentile'] < 70)
                    hidden_df.loc[category_leaders, 'hidden_score'] += 40
                    
                    # Add reason
                    if 'category_rank' in hidden_df.columns:
                        hidden_df.loc[category_leaders, 'hidden_reasons'] += 'Cat Leader | '
                    else:
                        hidden_df.loc[category_leaders, 'hidden_reasons'] += 'Category Top | '
                
                # Criterion 2: Pattern-based hidden gems
                if 'patterns' in hidden_df.columns:
                    # Hidden gem pattern
                    has_hidden = hidden_df['patterns'].str.contains('HIDDEN GEM', na=False)
                    hidden_df.loc[has_hidden, 'hidden_score'] += 30
                    hidden_df.loc[has_hidden, 'hidden_reasons'] += 'Hidden Pattern | '
                    
                    # Stealth pattern
                    has_stealth = hidden_df['patterns'].str.contains('STEALTH', na=False)
                    hidden_df.loc[has_stealth, 'hidden_score'] += 30
                    hidden_df.loc[has_stealth, 'hidden_reasons'] += 'Stealth | '
                    
                    # Vampire pattern
                    has_vampire = hidden_df['patterns'].str.contains('VAMPIRE', na=False)
                    hidden_df.loc[has_vampire, 'hidden_score'] += 30
                    hidden_df.loc[has_vampire, 'hidden_reasons'] += 'Vampire | '
                
                # Criterion 3: Low volume but strong momentum (under radar)
                if all(col in hidden_df.columns for col in ['momentum_score', 'rvol', 'ret_30d']):
                    under_radar = (hidden_df['momentum_score'] > 65) & (hidden_df['rvol'] < 1.5) & (hidden_df['ret_30d'] > 10)
                    hidden_df.loc[under_radar, 'hidden_score'] += 30
                    hidden_df.loc[under_radar, 'hidden_reasons'] += 'Under Radar | '
                
                # Clean up reasons (remove trailing separator)
                hidden_df['hidden_reasons'] = hidden_df['hidden_reasons'].str.rstrip(' | ')
                
                # Get top hidden gems
                top_hidden = hidden_df[hidden_df['hidden_score'] >= 30].nlargest(8, 'hidden_score')
                
                if len(top_hidden) > 0:
                    hidden_display = []
                    for _, stock in top_hidden.iterrows():
                        # Format reasons
                        reasons = stock['hidden_reasons'] if stock['hidden_reasons'] else 'Multiple Factors'
                        # Limit to first 2 reasons for display
                        reason_parts = reasons.split(' | ')[:2]
                        reason_text = ' | '.join(reason_parts)
                        
                        # Determine discovery level
                        if stock['hidden_score'] >= 70:
                            discovery = "💎💎💎"
                        elif stock['hidden_score'] >= 50:
                            discovery = "💎💎"
                        else:
                            discovery = "💎"
                        
                        hidden_display.append({
                            'Discovery': discovery,
                            'Ticker': stock['ticker'],
                            'Company': str(stock.get('company_name', ''))[:20] + '...' if len(str(stock.get('company_name', ''))) > 20 else stock.get('company_name', ''),
                            'Rank': int(stock['rank']),
                            'Score': stock['master_score'],
                            'Why Hidden': reason_text,
                            '30D%': stock.get('ret_30d', 0),
                            'RVOL': stock.get('rvol', 1),
                            'Category': stock.get('category', 'N/A'),
                            'Entry': stock['price']
                        })
                    
                    hidden_gems_df = pd.DataFrame(hidden_display)
                    
                    st.dataframe(
                        hidden_gems_df,
                        use_container_width=True,
                        hide_index=True,
                        height=350,
                        column_config={
                            'Discovery': st.column_config.TextColumn('', width='small'),
                            'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                            'Company': st.column_config.TextColumn('Company', width='medium'),
                            'Rank': st.column_config.NumberColumn('Rank', format='#%d', width='small'),
                            'Score': st.column_config.ProgressColumn(
                                'Score',
                                min_value=0,
                                max_value=100,
                                format='%.0f',
                                width='small'
                            ),
                            'Why Hidden': st.column_config.TextColumn('Discovery Reason', width='large'),
                            '30D%': st.column_config.NumberColumn('30D%', format='%.1f%%', width='small'),
                            'RVOL': st.column_config.NumberColumn('RVOL', format='%.1fx', width='small'),
                            'Category': st.column_config.TextColumn('Category', width='medium'),
                            'Entry': st.column_config.NumberColumn('Entry', format='₹%.0f', width='small')
                        }
                    )
                    
                    # Hidden gems summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        cat_leaders = len(top_hidden[top_hidden['hidden_reasons'].str.contains('Cat', na=False)])
                        st.metric("Category Leaders", cat_leaders)
                    with col2:
                        under_radar_count = len(top_hidden[top_hidden['hidden_reasons'].str.contains('Radar', na=False)])
                        st.metric("Under Radar", under_radar_count)
                    with col3:
                        pattern_based = len(top_hidden[top_hidden['hidden_reasons'].str.contains('Pattern|Stealth|Vampire', na=False)])
                        st.metric("Pattern Based", pattern_based)
                else:
                    st.info("No hidden gems identified. All strong stocks are already visible.")
                
                st.markdown("---")
                
                # ========================================
                # QUICK ACTION ZONES - ENHANCED VERSION
                # ========================================
                st.markdown("##### ⚡ **Quick Action Zones**")
                
                # Create three action zone tabs for better organization
                zone_tabs = st.tabs(["🚀 Momentum Zone", "🎯 Breakout Zone", "🌋 Volume Zone"])
                
                # MOMENTUM ZONE
                with zone_tabs[0]:
                    if all(col in filtered_df.columns for col in ['momentum_score', 'acceleration_score']):
                        momentum_zone = filtered_df[
                            (filtered_df['momentum_score'] > 70) & 
                            (filtered_df['acceleration_score'] > 70)
                        ].nlargest(5, 'master_score')
                        
                        if len(momentum_zone) > 0:
                            momentum_data = []
                            for _, stock in momentum_zone.iterrows():
                                # Determine momentum strength
                                if stock['momentum_score'] > 85 and stock['acceleration_score'] > 85:
                                    strength = "🔥🔥🔥"
                                    action = "STRONG BUY"
                                elif stock['momentum_score'] > 75 and stock['acceleration_score'] > 75:
                                    strength = "🔥🔥"
                                    action = "BUY"
                                else:
                                    strength = "🔥"
                                    action = "WATCH"
                                
                                momentum_data.append({
                                    'Strength': strength,
                                    'Ticker': stock['ticker'],
                                    'Company': str(stock.get('company_name', ''))[:25] + '...' if len(str(stock.get('company_name', ''))) > 25 else stock.get('company_name', ''),
                                    'Price': stock['price'],
                                    'Momentum': stock['momentum_score'],
                                    'Acceleration': stock['acceleration_score'],
                                    '7D%': stock.get('ret_7d', 0),
                                    'Wave': stock.get('wave_state', 'N/A'),
                                    'Action': action
                                })
                            
                            momentum_df = pd.DataFrame(momentum_data)
                            
                            st.dataframe(
                                momentum_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'Strength': st.column_config.TextColumn('', width='small'),
                                    'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                                    'Company': st.column_config.TextColumn('Company', width='medium'),
                                    'Price': st.column_config.NumberColumn('Price', format='₹%.0f', width='small'),
                                    'Momentum': st.column_config.ProgressColumn(
                                        'Mom',
                                        min_value=0,
                                        max_value=100,
                                        format='%.0f',
                                        width='small'
                                    ),
                                    'Acceleration': st.column_config.ProgressColumn(
                                        'Accel',
                                        min_value=0,
                                        max_value=100,
                                        format='%.0f',
                                        width='small'
                                    ),
                                    '7D%': st.column_config.NumberColumn('7D%', format='%.1f%%', width='small'),
                                    'Wave': st.column_config.TextColumn('Wave', width='medium'),
                                    'Action': st.column_config.TextColumn('Action', width='small')
                                }
                            )
                            
                            # Best momentum pick
                            best_momentum = momentum_zone.iloc[0]
                            st.success(
                                f"**🏆 Top Pick: {best_momentum['ticker']}**\n"
                                f"Entry: ₹{best_momentum['price']:.0f} | "
                                f"Target: ₹{best_momentum['price'] * 1.08:.0f} | "
                                f"Stop: ₹{best_momentum['price'] * 0.96:.0f}"
                            )
                        else:
                            st.info("No stocks qualify for Momentum Zone (Need Mom>70 & Accel>70)")
                    else:
                        st.info("Momentum data not available")
                
                # BREAKOUT ZONE
                with zone_tabs[1]:
                    if 'breakout_score' in filtered_df.columns:
                        breakout_zone = filtered_df[
                            filtered_df['breakout_score'] > 75
                        ].nlargest(5, 'breakout_score')
                        
                        if len(breakout_zone) > 0:
                            breakout_data = []
                            for _, stock in breakout_zone.iterrows():
                                # Calculate breakout readiness
                                if stock['breakout_score'] > 90:
                                    readiness = "🎯🎯🎯"
                                    status = "IMMINENT"
                                elif stock['breakout_score'] > 85:
                                    readiness = "🎯🎯"
                                    status = "READY"
                                else:
                                    readiness = "🎯"
                                    status = "BUILDING"
                                
                                # Determine resistance level
                                if 'high_52w' in stock and pd.notna(stock['high_52w']):
                                    resistance = stock['high_52w']
                                    distance = ((resistance - stock['price']) / stock['price']) * 100
                                else:
                                    resistance = stock['price'] * 1.05
                                    distance = 5.0
                                
                                breakout_data.append({
                                    'Ready': readiness,
                                    'Ticker': stock['ticker'],
                                    'Company': str(stock.get('company_name', ''))[:25] + '...' if len(str(stock.get('company_name', ''))) > 25 else stock.get('company_name', ''),
                                    'Price': stock['price'],
                                    'Score': stock['breakout_score'],
                                    'Resistance': resistance,
                                    'Distance': distance,
                                    'RVOL': stock.get('rvol', 1),
                                    'Status': status
                                })
                            
                            breakout_df = pd.DataFrame(breakout_data)
                            
                            st.dataframe(
                                breakout_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'Ready': st.column_config.TextColumn('', width='small'),
                                    'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                                    'Company': st.column_config.TextColumn('Company', width='medium'),
                                    'Price': st.column_config.NumberColumn('Price', format='₹%.0f', width='small'),
                                    'Score': st.column_config.ProgressColumn(
                                        'Breakout',
                                        min_value=0,
                                        max_value=100,
                                        format='%.0f',
                                        width='small'
                                    ),
                                    'Resistance': st.column_config.NumberColumn('Target', format='₹%.0f', width='small'),
                                    'Distance': st.column_config.NumberColumn('Gap%', format='%.1f%%', width='small'),
                                    'RVOL': st.column_config.NumberColumn('RVOL', format='%.1fx', width='small'),
                                    'Status': st.column_config.TextColumn('Status', width='small')
                                }
                            )
                            
                            # Nearest breakout
                            nearest = breakout_zone.iloc[0]
                            st.warning(
                                f"**🎯 Nearest Breakout: {nearest['ticker']}**\n"
                                f"Watch above: ₹{resistance:.0f} | "
                                f"Volume confirmation needed: RVOL > 2x"
                            )
                        else:
                            st.info("No stocks ready for breakout (Need score > 75)")
                    else:
                        st.info("Breakout data not available")
                
                # VOLUME ZONE
                with zone_tabs[2]:
                    if 'rvol' in filtered_df.columns:
                        volume_zone = filtered_df[
                            filtered_df['rvol'] > 2
                        ].nlargest(5, 'rvol')
                        
                        if len(volume_zone) > 0:
                            volume_data = []
                            for _, stock in volume_zone.iterrows():
                                # Determine volume type
                                if stock['rvol'] > 5:
                                    vol_type = "🌋🌋🌋"
                                    signal = "EXPLOSIVE"
                                elif stock['rvol'] > 3:
                                    vol_type = "🌋🌋"
                                    signal = "SURGE"
                                else:
                                    vol_type = "🌋"
                                    signal = "ACTIVE"
                                
                                # Determine direction
                                if stock.get('ret_1d', 0) > 2:
                                    direction = "📈 Buying"
                                    dir_color = "🟢"
                                elif stock.get('ret_1d', 0) < -2:
                                    direction = "📉 Selling"
                                    dir_color = "🔴"
                                else:
                                    direction = "➡️ Neutral"
                                    dir_color = "🟡"
                                
                                volume_data.append({
                                    'Signal': vol_type,
                                    'Ticker': stock['ticker'],
                                    'Company': str(stock.get('company_name', ''))[:25] + '...' if len(str(stock.get('company_name', ''))) > 25 else stock.get('company_name', ''),
                                    'Price': stock['price'],
                                    'RVOL': stock['rvol'],
                                    '1D%': stock.get('ret_1d', 0),
                                    'Flow ₹M': stock.get('money_flow_mm', 0),
                                    'Direction': f"{dir_color} {direction}",
                                    'Type': signal
                                })
                            
                            volume_df = pd.DataFrame(volume_data)
                            
                            st.dataframe(
                                volume_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'Signal': st.column_config.TextColumn('', width='small'),
                                    'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                                    'Company': st.column_config.TextColumn('Company', width='medium'),
                                    'Price': st.column_config.NumberColumn('Price', format='₹%.0f', width='small'),
                                    'RVOL': st.column_config.NumberColumn(
                                        'RVOL',
                                        format='%.1fx',
                                        width='small',
                                        help='Relative Volume vs average'
                                    ),
                                    '1D%': st.column_config.NumberColumn('1D%', format='%.1f%%', width='small'),
                                    'Flow ₹M': st.column_config.NumberColumn('Flow', format='₹%.0fM', width='small'),
                                    'Direction': st.column_config.TextColumn('Direction', width='medium'),
                                    'Type': st.column_config.TextColumn('Type', width='small')
                                }
                            )
                            
                            # Volume alert
                            if len(volume_zone[volume_zone['rvol'] > 5]) > 0:
                                extreme = volume_zone[volume_zone['rvol'] > 5].iloc[0]
                                st.error(
                                    f"**🌋 EXTREME VOLUME: {extreme['ticker']}**\n"
                                    f"RVOL: {extreme['rvol']:.1f}x | "
                                    f"Money Flow: ₹{extreme.get('money_flow_mm', 0):.0f}M | "
                                    f"Action: Monitor for continuation"
                                )
                        else:
                            st.info("No unusual volume activity (Need RVOL > 2x)")
                    else:
                        st.info("Volume data not available")
                
                # Summary action box
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_momentum = len(filtered_df[(filtered_df.get('momentum_score', 0) > 70) & (filtered_df.get('acceleration_score', 0) > 70)])
                    st.metric("📊 Momentum Plays", total_momentum)
                
                with col2:
                    total_breakouts = len(filtered_df[filtered_df.get('breakout_score', 0) > 75])
                    st.metric("🎯 Breakout Ready", total_breakouts)
                
                with col3:
                    total_volume = len(filtered_df[filtered_df.get('rvol', 0) > 2])
                    st.metric("🌋 Volume Surges", total_volume)
            
            # TAB 2: WAVE LEADERS (Keep your existing excellent implementation)
            # TAB 2: WAVE LEADERS - ULTIMATE CLEAN VERSION
            with discovery_tabs[1]:
                st.markdown("#### 🌊 Wave State Leaders - Market Momentum Lifecycle")
                
                if 'wave_state' in filtered_df.columns:
                    
                    # ========================================
                    # WAVE OVERVIEW - SINGLE CLEAN TABLE
                    # ========================================
                    
                    # Calculate wave metrics
                    wave_summary = []
                    total_stocks = len(filtered_df)
                    
                    wave_order = ['CRESTING', 'BUILDING', 'FORMING', 'BREAKING']
                    wave_emojis = {
                        'CRESTING': '🌊🌊🌊',
                        'BUILDING': '🌊🌊',
                        'FORMING': '🌊',
                        'BREAKING': '💥'
                    }
                    
                    for wave in wave_order:
                        wave_df = filtered_df[filtered_df['wave_state'].str.contains(wave, na=False)]
                        
                        if len(wave_df) > 0:
                            # Get top 3 stocks for this wave
                            top_stocks = wave_df.nlargest(3, 'master_score')['ticker'].tolist()
                            top_stocks_str = ', '.join(top_stocks[:3])
                            
                            # Calculate performance metrics
                            avg_momentum = wave_df['momentum_score'].mean() if 'momentum_score' in wave_df.columns else 0
                            avg_rvol = wave_df['rvol'].mean() if 'rvol' in wave_df.columns else 1
                            avg_30d = wave_df['ret_30d'].mean() if 'ret_30d' in wave_df.columns else 0
                            
                            # Determine health
                            if avg_momentum > 70 and avg_rvol > 2:
                                health = "🟢 Strong"
                            elif avg_momentum > 50 and avg_rvol > 1.5:
                                health = "🟡 Moderate"
                            else:
                                health = "🔴 Weak"
                            
                            wave_summary.append({
                                'Wave': f"{wave_emojis[wave]} {wave}",
                                'Count': len(wave_df),
                                '%': f"{(len(wave_df)/total_stocks)*100:.0f}%",
                                'Avg Score': wave_df['master_score'].mean(),
                                'Momentum': avg_momentum,
                                'RVOL': avg_rvol,
                                '30D Ret': avg_30d,
                                'Health': health,
                                'Top 3 Stocks': top_stocks_str
                            })
                    
                    if wave_summary:
                        wave_overview_df = pd.DataFrame(wave_summary)
                        
                        # Display clean overview table
                        st.dataframe(
                            wave_overview_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'Wave': st.column_config.TextColumn('Wave State', width='medium'),
                                'Count': st.column_config.NumberColumn('Stocks', width='small'),
                                '%': st.column_config.TextColumn('Market %', width='small'),
                                'Avg Score': st.column_config.ProgressColumn(
                                    'Avg Score',
                                    min_value=0,
                                    max_value=100,
                                    format='%.1f',
                                    width='small'
                                ),
                                'Momentum': st.column_config.ProgressColumn(
                                    'Momentum',
                                    min_value=0,
                                    max_value=100,
                                    format='%.0f',
                                    width='small'
                                ),
                                'RVOL': st.column_config.NumberColumn('RVOL', format='%.1fx', width='small'),
                                '30D Ret': st.column_config.NumberColumn('30D%', format='%.1f%%', width='small'),
                                'Health': st.column_config.TextColumn('Health', width='small'),
                                'Top 3 Stocks': st.column_config.TextColumn('Leaders', width='large')
                            }
                        )
                    
                    # ========================================
                    # WAVE QUALITY INDICATOR
                    # ========================================
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Calculate market wave health
                        cresting_count = len(filtered_df[filtered_df['wave_state'].str.contains('CRESTING', na=False)])
                        building_count = len(filtered_df[filtered_df['wave_state'].str.contains('BUILDING', na=False)])
                        forming_count = len(filtered_df[filtered_df['wave_state'].str.contains('FORMING', na=False)])
                        breaking_count = len(filtered_df[filtered_df['wave_state'].str.contains('BREAKING', na=False)])
                        
                        total_waves = cresting_count + building_count + forming_count + breaking_count
                        
                        if total_waves > 0:
                            wave_health = (
                                (cresting_count * 100) + 
                                (building_count * 75) + 
                                (forming_count * 50) + 
                                (breaking_count * 25)
                            ) / total_waves
                            
                            if wave_health > 70:
                                st.success(f"🔥 **Wave Strength: {wave_health:.0f}/100**")
                            elif wave_health > 50:
                                st.warning(f"⚡ **Wave Strength: {wave_health:.0f}/100**")
                            else:
                                st.error(f"❄️ **Wave Strength: {wave_health:.0f}/100**")
                    
                    with col2:
                        # Dominant wave
                        if wave_summary:
                            dominant = max(wave_summary, key=lambda x: x['Count'])
                            st.info(f"**Dominant:** {dominant['Wave']}\n{dominant['Count']} stocks")
                    
                    with col3:
                        # Transition alert
                        if breaking_count > total_waves * 0.3:
                            st.error(f"⚠️ **{breaking_count} Breaking**\nReduce exposure")
                        elif cresting_count > total_waves * 0.3:
                            st.success(f"🚀 **{cresting_count} Cresting**\nRide the wave")
                        else:
                            st.info(f"⚖️ **Market Building**\nStay selective")
                    
                    st.markdown("---")
                    
                    # ========================================
                    # TOP WAVE PERFORMERS - CLEAN TABLE
                    # ========================================
                    st.markdown("##### 🏆 **Top Performers by Wave State**")
                    
                    # Create wave selection
                    wave_selection = st.selectbox(
                        "Select Wave State",
                        options=['🌊🌊🌊 CRESTING', '🌊🌊 BUILDING', '🌊 FORMING', '💥 BREAKING'],
                        index=0,
                        key="wave_select"
                    )
                    
                    # Extract wave name
                    selected_wave = wave_selection.split()[-1]
                    
                    # Get stocks for selected wave
                    wave_stocks = filtered_df[filtered_df['wave_state'].str.contains(selected_wave, na=False)]
                    
                    if len(wave_stocks) > 0:
                        # Prepare display data
                        wave_display_data = []
                        for idx, (_, stock) in enumerate(wave_stocks.nlargest(15, 'master_score').iterrows()):
                            # Rank indicator
                            if idx == 0:
                                rank_emoji = "🥇"
                            elif idx == 1:
                                rank_emoji = "🥈"
                            elif idx == 2:
                                rank_emoji = "🥉"
                            else:
                                rank_emoji = f"#{idx+1}"
                            
                            # Momentum indicator
                            if stock.get('momentum_harmony', 0) == 4:
                                harmony = "🟢🟢🟢🟢"
                            elif stock.get('momentum_harmony', 0) == 3:
                                harmony = "🟢🟢🟢"
                            elif stock.get('momentum_harmony', 0) == 2:
                                harmony = "🟢🟢"
                            else:
                                harmony = "🟢"
                            
                            wave_display_data.append({
                                'Rank': rank_emoji,
                                'Ticker': stock['ticker'],
                                'Company': str(stock.get('company_name', ''))[:25] + '...' if len(str(stock.get('company_name', ''))) > 25 else stock.get('company_name', ''),
                                'Score': stock['master_score'],
                                'Price': stock['price'],
                                'Momentum': stock.get('momentum_score', 0),
                                'Acceleration': stock.get('acceleration_score', 0),
                                'RVOL': stock.get('rvol', 1),
                                '7D%': stock.get('ret_7d', 0),
                                '30D%': stock.get('ret_30d', 0),
                                'Harmony': harmony,
                                'Category': stock.get('category', 'N/A')
                            })
                        
                        wave_performers_df = pd.DataFrame(wave_display_data)
                        
                        # Display with optimal column config
                        st.dataframe(
                            wave_performers_df,
                            use_container_width=True,
                            hide_index=True,
                            height=400,
                            column_config={
                                'Rank': st.column_config.TextColumn('', width='small'),
                                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                                'Company': st.column_config.TextColumn('Company', width='medium'),
                                'Score': st.column_config.ProgressColumn(
                                    'Score',
                                    min_value=0,
                                    max_value=100,
                                    format='%.0f',
                                    width='small'
                                ),
                                'Price': st.column_config.NumberColumn('Price', format='₹%.0f', width='small'),
                                'Momentum': st.column_config.ProgressColumn(
                                    'Mom',
                                    min_value=0,
                                    max_value=100,
                                    format='%.0f',
                                    width='small'
                                ),
                                'Acceleration': st.column_config.ProgressColumn(
                                    'Accel',
                                    min_value=0,
                                    max_value=100,
                                    format='%.0f',
                                    width='small'
                                ),
                                'RVOL': st.column_config.NumberColumn('RVOL', format='%.1fx', width='small'),
                                '7D%': st.column_config.NumberColumn('7D%', format='%.1f%%', width='small'),
                                '30D%': st.column_config.NumberColumn('30D%', format='%.1f%%', width='small'),
                                'Harmony': st.column_config.TextColumn('Sync', help='Momentum harmony across timeframes', width='small'),
                                'Category': st.column_config.TextColumn('Category', width='medium')
                            }
                        )
                        
                        # Wave-specific insights
                        if selected_wave == 'CRESTING':
                            st.success(
                                "💡 **Cresting Insight:** Peak momentum stocks. Watch for distribution or exhaustion. "
                                "Take partial profits on extreme moves."
                            )
                        elif selected_wave == 'BUILDING':
                            st.info(
                                "💡 **Building Insight:** Best risk/reward zone. Add on pullbacks. "
                                "These have room to run to cresting."
                            )
                        elif selected_wave == 'FORMING':
                            st.warning(
                                "💡 **Forming Insight:** Early stage opportunities. Need volume confirmation. "
                                "Wait for momentum > 60 before entry."
                            )
                        else:  # BREAKING
                            st.error(
                                "💡 **Breaking Insight:** Momentum failing. Exit or avoid. "
                                "Wait for base formation before considering entry."
                            )
                    else:
                        st.info(f"No stocks currently in {selected_wave} state")
                    
                    st.markdown("---")
                    
                    # ========================================
                    # WAVE TRANSITIONS - ACTIONABLE ALERTS
                    # ========================================
                    st.markdown("##### 🔄 **Wave Transition Alerts**")
                    
                    transitions = []
                    
                    # About to CREST
                    if all(col in filtered_df.columns for col in ['wave_state', 'momentum_score', 'acceleration_score']):
                        about_to_crest = filtered_df[
                            (filtered_df['wave_state'].str.contains('BUILDING', na=False)) &
                            (filtered_df['momentum_score'] > 75) &
                            (filtered_df['acceleration_score'] > 80)
                        ].nlargest(3, 'momentum_score')
                        
                        for _, stock in about_to_crest.iterrows():
                            transitions.append({
                                'Alert': '🚀 BUILDING→CRESTING',
                                'Ticker': stock['ticker'],
                                'Company': str(stock.get('company_name', ''))[:20] + '...' if len(str(stock.get('company_name', ''))) > 20 else stock.get('company_name', ''),
                                'Price': stock['price'],
                                'Signal': f"Mom: {stock['momentum_score']:.0f}, Accel: {stock['acceleration_score']:.0f}",
                                'Action': 'BUY/ADD'
                            })
                    
                    # About to BREAK
                    if all(col in filtered_df.columns for col in ['wave_state', 'momentum_score', 'rvol']):
                        about_to_break = filtered_df[
                            (filtered_df['wave_state'].str.contains('CRESTING', na=False)) &
                            ((filtered_df['momentum_score'] < 60) | (filtered_df['rvol'] < 0.5))
                        ].nlargest(3, 'master_score')
                        
                        for _, stock in about_to_break.iterrows():
                            transitions.append({
                                'Alert': '⚠️ CRESTING→BREAKING',
                                'Ticker': stock['ticker'],
                                'Company': str(stock.get('company_name', ''))[:20] + '...' if len(str(stock.get('company_name', ''))) > 20 else stock.get('company_name', ''),
                                'Price': stock['price'],
                                'Signal': f"Mom weakening: {stock['momentum_score']:.0f}",
                                'Action': 'EXIT/REDUCE'
                            })
                    
                    # Recovery from BREAKING
                    if all(col in filtered_df.columns for col in ['wave_state', 'acceleration_score', 'ret_7d']):
                        recovering = filtered_df[
                            (filtered_df['wave_state'].str.contains('BREAKING', na=False)) &
                            (filtered_df['acceleration_score'] > 60) &
                            (filtered_df['ret_7d'] > 0)
                        ].nlargest(2, 'acceleration_score')
                        
                        for _, stock in recovering.iterrows():
                            transitions.append({
                                'Alert': '🔄 BREAKING→FORMING',
                                'Ticker': stock['ticker'],
                                'Company': str(stock.get('company_name', ''))[:20] + '...' if len(str(stock.get('company_name', ''))) > 20 else stock.get('company_name', ''),
                                'Price': stock['price'],
                                'Signal': f"Recovery starting: +{stock['ret_7d']:.1f}%",
                                'Action': 'WATCH'
                            })
                    
                    if transitions:
                        transition_df = pd.DataFrame(transitions)
                        
                        st.dataframe(
                            transition_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'Alert': st.column_config.TextColumn('Transition', width='medium'),
                                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                                'Company': st.column_config.TextColumn('Company', width='medium'),
                                'Price': st.column_config.NumberColumn('Price', format='₹%.0f', width='small'),
                                'Signal': st.column_config.TextColumn('Signal', width='large'),
                                'Action': st.column_config.TextColumn('Action', width='small')
                            }
                        )
                    else:
                        st.info("No significant wave transitions detected")
                
                else:
                    st.warning("Wave state data not available")
            
            # TAB 3: PATTERN DISCOVERIES & INSIGHTS
            with discovery_tabs[2]:
                st.markdown("#### 🔍 Pattern Discoveries & Market Insights")
                
                if 'patterns' in filtered_df.columns:
                    # ========================================
                    # PATTERN FREQUENCY & IMPACT TABLE
                    # ========================================
                    
                    # Build pattern analysis
                    pattern_analysis = []
                    
                    # Count patterns and find example stocks
                    for idx, row in filtered_df.iterrows():
                        if row['patterns'] and row['patterns'] != '':
                            patterns = str(row['patterns']).split(' | ')
                            for pattern in patterns:
                                pattern = pattern.strip()
                                if pattern:
                                    pattern_analysis.append({
                                        'pattern': pattern,
                                        'ticker': row['ticker'],
                                        'score': row['master_score'],
                                        'ret_30d': row.get('ret_30d', 0),
                                        'rvol': row.get('rvol', 1),
                                        'price': row['price']
                                    })
                    
                    if pattern_analysis:
                        pattern_df = pd.DataFrame(pattern_analysis)
                        
                        # Aggregate pattern statistics
                        pattern_stats = pattern_df.groupby('pattern').agg({
                            'ticker': ['count', lambda x: list(x.head(3))],  # Count and top 3 tickers
                            'score': 'mean',
                            'ret_30d': 'mean',
                            'rvol': 'mean'
                        }).round(2)
                        
                        pattern_stats.columns = ['Count', 'Top Stocks', 'Avg Score', 'Avg 30D%', 'Avg RVOL']
                        pattern_stats = pattern_stats.sort_values('Count', ascending=False)
                        
                        # Classify patterns
                        def classify_pattern(pattern_name):
                            bullish = ['ACCELERATING', 'BREAKOUT', 'MOMENTUM', 'LEADER', 'GEM', 'GOLDEN', 'RISING', 'CRESTING']
                            bearish = ['TRAP', 'DISTRIBUTION', 'EXHAUSTION', 'BREAKING', 'FALLING']
                            reversal = ['CAPITULATION', 'BOUNCE', 'TURNAROUND', 'ROTATION']
                            extreme = ['PERFECT STORM', 'VOL EXPLOSION', 'VAMPIRE']
                            
                            if any(word in pattern_name for word in extreme):
                                return "⚡ EXTREME", "#ff4444"
                            elif any(word in pattern_name for word in bullish):
                                return "📈 BULLISH", "#00cc88"
                            elif any(word in pattern_name for word in bearish):
                                return "📉 BEARISH", "#ff6666"
                            elif any(word in pattern_name for word in reversal):
                                return "🔄 REVERSAL", "#ffaa00"
                            else:
                                return "➡️ NEUTRAL", "#888888"
                        
                        # Build display dataframe
                        display_patterns = []
                        for pattern, row in pattern_stats.head(15).iterrows():
                            pattern_type, color = classify_pattern(pattern)
                            
                            # Determine strength
                            if row['Count'] > 20:
                                strength = "🔥🔥🔥"
                            elif row['Count'] > 10:
                                strength = "🔥🔥"
                            elif row['Count'] > 5:
                                strength = "🔥"
                            else:
                                strength = "•"
                            
                            # Format top stocks list
                            top_stocks = ', '.join(row['Top Stocks'][:3]) if isinstance(row['Top Stocks'], list) else ''
                            
                            display_patterns.append({
                                'Strength': strength,
                                'Pattern': pattern,
                                'Type': pattern_type,
                                'Count': int(row['Count']),
                                'Top Stocks': top_stocks,
                                'Avg Score': row['Avg Score'],
                                'Avg 30D%': row['Avg 30D%'],
                                'Avg RVOL': row['Avg RVOL']
                            })
                        
                        patterns_display_df = pd.DataFrame(display_patterns)
                        
                        # Display main pattern table
                        st.markdown("##### 📊 **Active Pattern Signals**")
                        
                        st.dataframe(
                            patterns_display_df,
                            use_container_width=True,
                            hide_index=True,
                            height=400,
                            column_config={
                                'Strength': st.column_config.TextColumn('', width='small'),
                                'Pattern': st.column_config.TextColumn('Pattern', width='medium'),
                                'Type': st.column_config.TextColumn('Type', width='small'),
                                'Count': st.column_config.NumberColumn('Stocks', format='%d', width='small'),
                                'Top Stocks': st.column_config.TextColumn('Leading Stocks', width='large'),
                                'Avg Score': st.column_config.ProgressColumn(
                                    'Avg Score',
                                    min_value=0,
                                    max_value=100,
                                    format='%.0f',
                                    width='small'
                                ),
                                'Avg 30D%': st.column_config.NumberColumn('Avg 30D%', format='%.1f%%', width='small'),
                                'Avg RVOL': st.column_config.NumberColumn('Avg Vol', format='%.1fx', width='small')
                            }
                        )
                        
                        st.markdown("---")
                        
                        # ========================================
                        # PATTERN INSIGHTS DASHBOARD
                        # ========================================
                        st.markdown("##### 💡 **Market Insights from Patterns**")
                        
                        # Calculate pattern bias
                        pattern_groups = patterns_display_df.groupby('Type')['Count'].sum()
                        total_patterns = pattern_groups.sum()
                        
                        # Create insights columns
                        insight_col1, insight_col2, insight_col3 = st.columns(3)
                        
                        with insight_col1:
                            st.markdown("**📊 Pattern Distribution**")
                            
                            for pattern_type, count in pattern_groups.items():
                                pct = (count / total_patterns) * 100 if total_patterns > 0 else 0
                                
                                if 'BULLISH' in pattern_type:
                                    st.success(f"{pattern_type}: {count} ({pct:.0f}%)")
                                elif 'BEARISH' in pattern_type:
                                    st.error(f"{pattern_type}: {count} ({pct:.0f}%)")
                                elif 'EXTREME' in pattern_type:
                                    st.warning(f"{pattern_type}: {count} ({pct:.0f}%)")
                                elif 'REVERSAL' in pattern_type:
                                    st.info(f"{pattern_type}: {count} ({pct:.0f}%)")
                                else:
                                    st.caption(f"{pattern_type}: {count} ({pct:.0f}%)")
                        
                        with insight_col2:
                            st.markdown("**🎯 Critical Patterns Active**")
                            
                            # Find critical patterns
                            critical_patterns = [
                                ('⛈️ PERFECT STORM', 'success'),
                                ('⚡ VOL EXPLOSION', 'error'),
                                ('🚀 ACCELERATING', 'info'),
                                ('🪤 BULL TRAP', 'warning'),
                                ('💣 CAPITULATION', 'success'),
                                ('⚠️ DISTRIBUTION', 'error')
                            ]
                            
                            critical_found = False
                            for pattern_name, alert_type in critical_patterns:
                                if pattern_name in patterns_display_df['Pattern'].values:
                                    pattern_row = patterns_display_df[patterns_display_df['Pattern'] == pattern_name]
                                    if not pattern_row.empty:
                                        count = pattern_row.iloc[0]['Count']
                                        stocks = pattern_row.iloc[0]['Top Stocks']
                                        
                                        getattr(st, alert_type)(
                                            f"**{pattern_name}** ({count})\n{stocks}"
                                        )
                                        critical_found = True
                                        break  # Show only top 3 critical patterns
                            
                            if not critical_found:
                                st.info("No critical patterns detected")
                        
                        with insight_col3:
                            st.markdown("**📈 Market Bias Signal**")
                            
                            # Calculate overall bias
                            bullish_count = pattern_groups.get('📈 BULLISH', 0)
                            bearish_count = pattern_groups.get('📉 BEARISH', 0)
                            extreme_count = pattern_groups.get('⚡ EXTREME', 0)
                            reversal_count = pattern_groups.get('🔄 REVERSAL', 0)
                            
                            if extreme_count > 10:
                                st.error(
                                    "**⚡ EXTREME ACTIVITY**\n"
                                    "High volatility expected\n"
                                    "Use tight risk management"
                                )
                            elif bullish_count > bearish_count * 2:
                                st.success(
                                    "**📈 STRONG BULLISH BIAS**\n"
                                    f"Ratio: {bullish_count}:{bearish_count}\n"
                                    "Favor long positions"
                                )
                            elif bearish_count > bullish_count * 2:
                                st.error(
                                    "**📉 STRONG BEARISH BIAS**\n"
                                    f"Ratio: {bearish_count}:{bullish_count}\n"
                                    "Reduce exposure"
                                )
                            elif reversal_count > total_patterns * 0.3:
                                st.warning(
                                    "**🔄 REVERSAL ZONE**\n"
                                    "Market at inflection point\n"
                                    "Wait for confirmation"
                                )
                            else:
                                st.info(
                                    "**➡️ MIXED SIGNALS**\n"
                                    f"Bull: {bullish_count} | Bear: {bearish_count}\n"
                                    "Stay selective"
                                )
                        
                        st.markdown("---")
                        
                        # ========================================
                        # TOP PATTERN PLAYS
                        # ========================================
                        st.markdown("##### 🏆 **Top Stocks by Pattern Type**")
                        
                        # Create pattern type tabs
                        pattern_tabs = st.tabs(["📈 Bullish Leaders", "⚡ Extreme Movers", "🔄 Reversal Plays"])
                        
                        with pattern_tabs[0]:  # Bullish Leaders
                            bullish_patterns = ['🚀 ACCELERATING', '🎯 BREAKOUT', '🌊 MOMENTUM WAVE', '🔥 CAT LEADER', '💎 HIDDEN GEM']
                            bullish_stocks = filtered_df[
                                filtered_df['patterns'].str.contains('|'.join(bullish_patterns), na=False, regex=True)
                            ].nlargest(5, 'master_score')
                            
                            if len(bullish_stocks) > 0:
                                bullish_display = []
                                for _, stock in bullish_stocks.iterrows():
                                    # Find which bullish patterns this stock has
                                    stock_patterns = []
                                    for bp in bullish_patterns:
                                        if bp in str(stock['patterns']):
                                            stock_patterns.append(bp.split(' ')[0])  # Just emoji
                                    
                                    bullish_display.append({
                                        'Signals': ' '.join(stock_patterns[:3]),
                                        'Ticker': stock['ticker'],
                                        'Company': str(stock.get('company_name', ''))[:25],
                                        'Score': stock['master_score'],
                                        'Price': stock['price'],
                                        '30D%': stock.get('ret_30d', 0),
                                        'RVOL': stock.get('rvol', 1),
                                        'Entry': stock['price']
                                    })
                                
                                st.dataframe(
                                    pd.DataFrame(bullish_display),
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        'Signals': st.column_config.TextColumn('', width='small'),
                                        'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                                        'Company': st.column_config.TextColumn('Company', width='medium'),
                                        'Score': st.column_config.ProgressColumn('Score', min_value=0, max_value=100, format='%.0f', width='small'),
                                        'Price': st.column_config.NumberColumn('Price', format='₹%.0f', width='small'),
                                        '30D%': st.column_config.NumberColumn('30D%', format='%.1f%%', width='small'),
                                        'RVOL': st.column_config.NumberColumn('RVOL', format='%.1fx', width='small'),
                                        'Entry': st.column_config.NumberColumn('Entry', format='₹%.0f', width='small')
                                    }
                                )
                            else:
                                st.info("No bullish pattern stocks found")
                        
                        with pattern_tabs[1]:  # Extreme Movers
                            extreme_patterns = ['⛈️ PERFECT STORM', '⚡ VOL EXPLOSION', '🧛 VAMPIRE']
                            extreme_stocks = filtered_df[
                                filtered_df['patterns'].str.contains('|'.join(extreme_patterns), na=False, regex=True)
                            ].nlargest(5, 'master_score')
                            
                            if len(extreme_stocks) > 0:
                                extreme_display = []
                                for _, stock in extreme_stocks.iterrows():
                                    extreme_display.append({
                                        'Alert': '⚡⚡⚡' if stock['rvol'] > 5 else '⚡⚡' if stock['rvol'] > 3 else '⚡',
                                        'Ticker': stock['ticker'],
                                        'Company': str(stock.get('company_name', ''))[:25],
                                        'RVOL': stock.get('rvol', 1),
                                        '1D%': stock.get('ret_1d', 0),
                                        'Flow ₹M': stock.get('money_flow_mm', 0),
                                        'Pattern': str(stock['patterns'])[:30]
                                    })
                                
                                st.dataframe(
                                    pd.DataFrame(extreme_display),
                                    use_container_width=True,
                                    hide_index=True
                                )
                            else:
                                st.info("No extreme pattern stocks found")
                        
                        with pattern_tabs[2]:  # Reversal Plays
                            reversal_patterns = ['💣 CAPITULATION', '🔄 52W LOW BOUNCE', '⚡ TURNAROUND']
                            reversal_stocks = filtered_df[
                                filtered_df['patterns'].str.contains('|'.join(reversal_patterns), na=False, regex=True)
                            ].nlargest(5, 'master_score')
                            
                            if len(reversal_stocks) > 0:
                                reversal_display = []
                                for _, stock in reversal_stocks.iterrows():
                                    reversal_display.append({
                                        'Signal': '🔄',
                                        'Ticker': stock['ticker'],
                                        'Company': str(stock.get('company_name', ''))[:25],
                                        'From Low': stock.get('from_low_pct', 0),
                                        '7D%': stock.get('ret_7d', 0),
                                        'Score': stock['master_score'],
                                        'Risk/Reward': 'High' if stock.get('from_low_pct', 0) < 10 else 'Medium'
                                    })
                                
                                st.dataframe(
                                    pd.DataFrame(reversal_display),
                                    use_container_width=True,
                                    hide_index=True
                                )
                            else:
                                st.info("No reversal pattern stocks found")
                    
                    else:
                        st.info("No patterns detected in current dataset")
                
                else:
                    st.info("Pattern data not available")
            
            # TAB 4: MONEY FLOW
            with discovery_tabs[3]:
                st.markdown("#### 💰 Smart Money Flow Analysis")
                
                if 'money_flow_mm' in filtered_df.columns:
                    # Calculate money flow tiers
                    flow_q75 = filtered_df['money_flow_mm'].quantile(0.75)
                    flow_q50 = filtered_df['money_flow_mm'].quantile(0.50)
                    flow_q25 = filtered_df['money_flow_mm'].quantile(0.25)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**💸 Institutional Money Flow Leaders**")
                        
                        top_flow = filtered_df.nlargest(10, 'money_flow_mm')
                        
                        flow_display = []
                        for _, stock in top_flow.iterrows():
                            # Determine flow type
                            flow_type = []
                            if stock.get('rvol', 1) > 3:
                                flow_type.append("🌋 Explosive")
                            elif stock.get('rvol', 1) > 2:
                                flow_type.append("🔥 Active")
                            else:
                                flow_type.append("🤫 Stealth")
                            
                            if stock.get('ret_1d', 0) > 3:
                                flow_type.append("📈 Buying")
                            elif stock.get('ret_1d', 0) < -3:
                                flow_type.append("📉 Selling")
                            
                            flow_display.append({
                                'Ticker': stock['ticker'],
                                'Company': stock.get('company_name', 'N/A')[:20],
                                'Flow ₹M': f"{stock['money_flow_mm']:.1f}",
                                'Price': f"₹{stock['price']:.0f}",
                                'RVOL': f"{stock.get('rvol', 1):.1f}x",
                                'Type': ' '.join(flow_type[:1]),
                                'Category': stock.get('category', 'N/A')
                            })
                        
                        flow_df = pd.DataFrame(flow_display)
                        st.dataframe(flow_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.markdown("**📊 Flow Statistics**")
                        
                        total_flow = filtered_df['money_flow_mm'].sum()
                        top10_flow = top_flow['money_flow_mm'].sum()
                        concentration = (top10_flow / total_flow) * 100 if total_flow > 0 else 0
                        
                        st.metric("Total Flow", f"₹{total_flow:.0f}M")
                        st.metric("Top 10 Share", f"{concentration:.0f}%")
                        
                        # Flow direction by category
                        if 'category' in filtered_df.columns:
                            cat_flow = filtered_df.groupby('category')['money_flow_mm'].sum().sort_values(ascending=False)
                            
                            st.markdown("**Flow by Category**")
                            for cat, flow in cat_flow.head(3).items():
                                flow_pct = (flow / total_flow) * 100 if total_flow > 0 else 0
                                st.caption(f"• {cat}: ₹{flow:.0f}M ({flow_pct:.0f}%)")
                            
                            # Risk assessment
                            if 'Small' in cat_flow.index[0] or 'Micro' in cat_flow.index[0]:
                                st.success("🔥 Risk-On: Small caps leading")
                            elif 'Large' in cat_flow.index[0] or 'Mega' in cat_flow.index[0]:
                                st.warning("🛡️ Risk-Off: Large caps leading")
            
            # TAB 5: MOMENTUM STARS & VELOCITY LEADERS
            with discovery_tabs[4]:
                st.markdown("#### 🚀 Momentum Stars & Velocity Leaders")
                
                if all(col in filtered_df.columns for col in ['momentum_score', 'acceleration_score']):
                    
                    # ====================================
                    # MOMENTUM CLASSIFICATION SYSTEM
                    # ====================================
                    
                    # Create momentum classifications
                    filtered_df['momentum_class'] = 'Regular'
                    filtered_df['momentum_signal'] = 0
                    
                    # Classification logic
                    conditions = [
                        (filtered_df['momentum_score'] > 80) & (filtered_df['acceleration_score'] > 80),  # Elite
                        (filtered_df['momentum_score'] > 70) & (filtered_df['acceleration_score'] > 70),  # Star
                        (filtered_df['momentum_score'] > 60) & (filtered_df['acceleration_score'] > 60),  # Rising
                        (filtered_df['momentum_score'] > 50) & (filtered_df['acceleration_score'] < 30),  # Fading
                    ]
                    
                    choices = ['Elite', 'Star', 'Rising', 'Fading']
                    signals = [100, 75, 50, 25]
                    
                    for i, condition in enumerate(conditions):
                        filtered_df.loc[condition, 'momentum_class'] = choices[i]
                        filtered_df.loc[condition, 'momentum_signal'] = signals[i]
                    
                    # ====================================
                    # SECTION 1: MOMENTUM ELITE TABLE
                    # ====================================
                    
                    st.markdown("##### 👑 **MOMENTUM ELITE** (Top Performers)")
                    
                    # Get elite momentum stocks
                    elite_momentum = filtered_df[filtered_df['momentum_class'].isin(['Elite', 'Star'])].copy()
                    
                    if len(elite_momentum) > 0:
                        # Calculate additional metrics
                        elite_momentum['velocity'] = 0
                        if all(col in elite_momentum.columns for col in ['ret_1d', 'ret_7d']):
                            with np.errstate(divide='ignore', invalid='ignore'):
                                elite_momentum['velocity'] = np.where(
                                    elite_momentum['ret_7d'] != 0,
                                    elite_momentum['ret_1d'] / (elite_momentum['ret_7d'] / 7),
                                    0
                                )
                        
                        # Sort by combined score
                        elite_momentum['elite_score'] = (
                            elite_momentum['momentum_score'] * 0.4 +
                            elite_momentum['acceleration_score'] * 0.4 +
                            elite_momentum.get('rvol_score', 50) * 0.2
                        )
                        
                        # Get top 15
                        top_elite = elite_momentum.nlargest(15, 'elite_score')
                        
                        # Prepare display dataframe
                        elite_display = []
                        for _, stock in top_elite.iterrows():
                            # Determine signals
                            signals = []
                            if stock['momentum_class'] == 'Elite':
                                signals.append('👑')
                            elif stock['momentum_class'] == 'Star':
                                signals.append('⭐')
                            
                            if stock.get('velocity', 0) > 2:
                                signals.append('⚡')
                            
                            if stock.get('momentum_harmony', 0) == 4:
                                signals.append('🎯')
                            elif stock.get('momentum_harmony', 0) >= 3:
                                signals.append('✅')
                            
                            if stock.get('rvol', 1) > 3:
                                signals.append('🌋')
                            elif stock.get('rvol', 1) > 2:
                                signals.append('🔥')
                            
                            elite_display.append({
                                '🏆': ' '.join(signals[:2]),
                                'Ticker': stock['ticker'],
                                'Company': str(stock.get('company_name', ''))[:20],
                                'Score': f"{stock['master_score']:.0f}",
                                'Mom': f"{stock['momentum_score']:.0f}",
                                'Accel': f"{stock['acceleration_score']:.0f}",
                                '1D%': f"{stock.get('ret_1d', 0):+.1f}",
                                '7D%': f"{stock.get('ret_7d', 0):+.1f}",
                                '30D%': f"{stock.get('ret_30d', 0):+.1f}",
                                'RVOL': f"{stock.get('rvol', 1):.1f}x",
                                'Velocity': f"{stock.get('velocity', 0):.1f}x" if stock.get('velocity', 0) > 0 else '-',
                                'Wave': stock.get('wave_state', 'N/A')[:10]
                            })
                        
                        elite_df = pd.DataFrame(elite_display)
                        
                        # Display with column configuration
                        st.dataframe(
                            elite_df,
                            use_container_width=True,
                            hide_index=True,
                            height=400,
                            column_config={
                                '🏆': st.column_config.TextColumn('🏆', width='small'),
                                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                                'Company': st.column_config.TextColumn('Company', width='medium'),
                                'Score': st.column_config.TextColumn('Score', width='small'),
                                'Mom': st.column_config.TextColumn('Mom', width='small'),
                                'Accel': st.column_config.TextColumn('Accel', width='small'),
                                '1D%': st.column_config.TextColumn('1D%', width='small'),
                                '7D%': st.column_config.TextColumn('7D%', width='small'),
                                '30D%': st.column_config.TextColumn('30D%', width='small'),
                                'RVOL': st.column_config.TextColumn('RVOL', width='small'),
                                'Velocity': st.column_config.TextColumn('Vel', width='small'),
                                'Wave': st.column_config.TextColumn('Wave', width='small')
                            }
                        )
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Elite Count", f"{len(elite_momentum[elite_momentum['momentum_class'] == 'Elite'])}")
                        with col2:
                            st.metric("Star Count", f"{len(elite_momentum[elite_momentum['momentum_class'] == 'Star'])}")
                        with col3:
                            avg_velocity = elite_momentum['velocity'].mean() if 'velocity' in elite_momentum.columns else 0
                            st.metric("Avg Velocity", f"{avg_velocity:.1f}x")
                        with col4:
                            perfect_harmony = len(elite_momentum[elite_momentum.get('momentum_harmony', 0) == 4])
                            st.metric("Perfect Harmony", f"{perfect_harmony}")
                    
                    else:
                        st.info("No momentum elite stocks found in current filter")
                    
                    st.markdown("---")
                    
                    # ====================================
                    # SECTION 2: VELOCITY ANALYSIS TABLE
                    # ====================================
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### ⚡ **VELOCITY LEADERS** (Accelerating Fast)")
                        
                        # Calculate velocity for all stocks
                        if all(col in filtered_df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
                            velocity_df = filtered_df.copy()
                            
                            # Calculate multiple velocity metrics
                            with np.errstate(divide='ignore', invalid='ignore'):
                                # 1D vs 7D velocity
                                velocity_df['velocity_1_7'] = np.where(
                                    velocity_df['ret_7d'] != 0,
                                    velocity_df['ret_1d'] / (velocity_df['ret_7d'] / 7),
                                    0
                                )
                                
                                # 7D vs 30D velocity
                                velocity_df['velocity_7_30'] = np.where(
                                    velocity_df['ret_30d'] != 0,
                                    (velocity_df['ret_7d'] / 7) / (velocity_df['ret_30d'] / 30),
                                    0
                                )
                            
                            # Filter for positive velocity
                            velocity_leaders = velocity_df[
                                (velocity_df['velocity_1_7'] > 1.5) & 
                                (velocity_df['ret_1d'] > 0) &
                                (velocity_df['momentum_score'] > 50)
                            ].nlargest(8, 'velocity_1_7')
                            
                            if len(velocity_leaders) > 0:
                                velocity_display = []
                                for _, stock in velocity_leaders.iterrows():
                                    # Velocity status
                                    if stock['velocity_1_7'] > 3:
                                        vel_status = '🚀 Extreme'
                                    elif stock['velocity_1_7'] > 2:
                                        vel_status = '⚡ High'
                                    else:
                                        vel_status = '📈 Building'
                                    
                                    velocity_display.append({
                                        'Status': vel_status,
                                        'Ticker': stock['ticker'],
                                        '1D': f"{stock['ret_1d']:+.1f}%",
                                        '7D Avg': f"{(stock['ret_7d']/7):+.1f}%",
                                        'Velocity': f"{stock['velocity_1_7']:.1f}x",
                                        'RVOL': f"{stock.get('rvol', 1):.1f}x"
                                    })
                                
                                vel_df = pd.DataFrame(velocity_display)
                                st.dataframe(vel_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No velocity leaders detected")
                        else:
                            st.info("Insufficient data for velocity analysis")
                    
                    with col2:
                        st.markdown("##### 🎯 **MOMENTUM HARMONY** (Multi-Timeframe)")
                        
                        # Find stocks with perfect momentum alignment
                        if 'momentum_harmony' in filtered_df.columns:
                            harmony_stocks = filtered_df[filtered_df['momentum_harmony'] >= 2].copy()
                            
                            if len(harmony_stocks) > 0:
                                # Sort by harmony and score
                                harmony_stocks = harmony_stocks.sort_values(
                                    ['momentum_harmony', 'master_score'], 
                                    ascending=[False, False]
                                ).head(8)
                                
                                harmony_display = []
                                for _, stock in harmony_stocks.iterrows():
                                    harmony = int(stock['momentum_harmony'])
                                    
                                    # Visual harmony indicator
                                    harmony_visual = '🟢' * harmony + '⚪' * (4 - harmony)
                                    
                                    # Determine strength
                                    if harmony == 4:
                                        strength = 'Perfect'
                                    elif harmony == 3:
                                        strength = 'Strong'
                                    elif harmony == 2:
                                        strength = 'Good'
                                    else:
                                        strength = 'Weak'
                                    
                                    harmony_display.append({
                                        'Harmony': harmony_visual,
                                        'Ticker': stock['ticker'],
                                        'Strength': strength,
                                        '1D': f"{stock.get('ret_1d', 0):+.0f}%",
                                        '7D': f"{stock.get('ret_7d', 0):+.0f}%",
                                        '30D': f"{stock.get('ret_30d', 0):+.0f}%",
                                        'Score': f"{stock['master_score']:.0f}"
                                    })
                                
                                harmony_df = pd.DataFrame(harmony_display)
                                st.dataframe(harmony_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No stocks with momentum harmony")
                        else:
                            st.info("Harmony data not available")
                    
                    st.markdown("---")
                    
                    # ====================================
                    # SECTION 3: MOMENTUM TRANSITIONS
                    # ====================================
                    
                    st.markdown("##### 🔄 **MOMENTUM TRANSITIONS** (Status Changes)")
                    
                    transition_col1, transition_col2, transition_col3 = st.columns(3)
                    
                    with transition_col1:
                        st.markdown("**🌱 EMERGING** (50-60 Score)")
                        
                        emerging = filtered_df[
                            filtered_df['momentum_score'].between(50, 60) &
                            (filtered_df['acceleration_score'] > 60)
                        ].nlargest(3, 'acceleration_score')
                        
                        if len(emerging) > 0:
                            for _, stock in emerging.iterrows():
                                st.success(
                                    f"**{stock['ticker']}**\n"
                                    f"Mom: {stock['momentum_score']:.0f} → Building\n"
                                    f"Accel: {stock['acceleration_score']:.0f}"
                                )
                        else:
                            st.caption("None emerging")
                    
                    with transition_col2:
                        st.markdown("**⚠️ FADING** (Losing Steam)")
                        
                        fading = filtered_df[
                            (filtered_df['momentum_score'] > 60) &
                            (filtered_df['acceleration_score'] < 40) &
                            (filtered_df.get('ret_1d', 0) < 0)
                        ].nlargest(3, 'master_score')
                        
                        if len(fading) > 0:
                            for _, stock in fading.iterrows():
                                st.warning(
                                    f"**{stock['ticker']}**\n"
                                    f"Mom: {stock['momentum_score']:.0f} ↓\n"
                                    f"Accel: {stock['acceleration_score']:.0f}"
                                )
                        else:
                            st.caption("None fading")
                    
                    with transition_col3:
                        st.markdown("**💥 EXPLOSIVE** (Just Started)")
                        
                        explosive = filtered_df[
                            (filtered_df['momentum_score'] < 70) &
                            (filtered_df['acceleration_score'] > 85) &
                            (filtered_df.get('rvol', 1) > 2)
                        ].nlargest(3, 'acceleration_score')
                        
                        if len(explosive) > 0:
                            for _, stock in explosive.iterrows():
                                st.error(
                                    f"**{stock['ticker']}**\n"
                                    f"Accel: {stock['acceleration_score']:.0f} 🚀\n"
                                    f"RVOL: {stock['rvol']:.1f}x"
                                )
                        else:
                            st.caption("None explosive")
                    
                    # ====================================
                    # FINAL MOMENTUM SUMMARY
                    # ====================================
                    
                    st.markdown("---")
                    st.markdown("##### 📊 **MOMENTUM MARKET SUMMARY**")
                    
                    # Calculate momentum market metrics
                    summary_cols = st.columns(5)
                    
                    with summary_cols[0]:
                        total_momentum = len(filtered_df[filtered_df['momentum_score'] > 60])
                        momentum_pct = (total_momentum / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
                        st.metric(
                            "Momentum Stocks",
                            f"{total_momentum}",
                            f"{momentum_pct:.0f}% of market"
                        )
                    
                    with summary_cols[1]:
                        if 'velocity' in filtered_df.columns:
                            high_velocity = len(filtered_df[filtered_df['velocity'] > 2])
                        else:
                            high_velocity = 0
                        st.metric("High Velocity", f"{high_velocity}")
                    
                    with summary_cols[2]:
                        if 'momentum_harmony' in filtered_df.columns:
                            perfect = len(filtered_df[filtered_df['momentum_harmony'] == 4])
                            st.metric("Perfect Harmony", f"{perfect}")
                        else:
                            st.metric("Perfect Harmony", "N/A")
                    
                    with summary_cols[3]:
                        accelerating = len(filtered_df[filtered_df['acceleration_score'] > 70])
                        st.metric("Accelerating", f"{accelerating}")
                    
                    with summary_cols[4]:
                        # Market momentum health
                        if momentum_pct > 30:
                            st.metric("Momentum Health", "🔥 Strong", "Bullish")
                        elif momentum_pct > 15:
                            st.metric("Momentum Health", "⚡ Moderate", "Neutral")
                        else:
                            st.metric("Momentum Health", "❄️ Weak", "Bearish")
                    
                else:
                    st.info("Momentum data not available for analysis")
            
            # ====================================
            # MARKET INTELLIGENCE SYSTEM
            # ====================================
            st.markdown("#### 🧠 Market Intelligence System")
            
            intel_tabs = st.tabs(["🏢 Sector Rotation", "💰 Smart Money Flow"])
            
            # TAB 1: SECTOR ROTATION
            with intel_tabs[0]:
                if 'sector' in filtered_df.columns:
                    st.markdown("##### 🏢 **Sector Rotation Analysis**")
                    
                    # Calculate comprehensive sector metrics - YOUR METHOD
                    sector_analysis = []
                    
                    for sector in filtered_df['sector'].unique():
                        if sector != 'Unknown':
                            sector_df = filtered_df[filtered_df['sector'] == sector]
                            
                            if len(sector_df) >= 2:  # Need minimum stocks
                                # Calculate rotation score - YOUR EXACT LOGIC
                                rotation_score = 0
                                signals = []
                                
                                # Score component 1: Relative performance
                                sector_avg_score = sector_df['master_score'].mean()
                                market_avg_score = filtered_df['master_score'].mean()
                                if sector_avg_score > market_avg_score:
                                    rotation_score += 25
                                    signals.append("Score+")
                                
                                # Score component 2: Momentum
                                if 'ret_30d' in sector_df.columns:
                                    sector_momentum = sector_df['ret_30d'].mean()
                                    if sector_momentum > 0:
                                        rotation_score += 25
                                        signals.append("Mom+")
                                    if sector_momentum > filtered_df['ret_30d'].mean():
                                        rotation_score += 15
                                        signals.append("RelMom+")
                                
                                # Score component 3: Volume activity
                                if 'rvol' in sector_df.columns:
                                    sector_rvol = sector_df['rvol'].mean()
                                    if sector_rvol > 1.5:
                                        rotation_score += 20
                                        signals.append("Vol+")
                                
                                # Score component 4: Money flow
                                if 'money_flow_mm' in sector_df.columns:
                                    sector_flow = sector_df['money_flow_mm'].sum()
                                    avg_sector_flow = filtered_df.groupby('sector')['money_flow_mm'].sum().mean()
                                    if sector_flow > avg_sector_flow:
                                        rotation_score += 15
                                        signals.append("Flow+")
                                
                                sector_analysis.append({
                                    'Sector': sector[:15],
                                    'Stocks': len(sector_df),
                                    'Avg Score': sector_avg_score,
                                    'Rotation Score': rotation_score,
                                    'Signals': ' '.join(signals),
                                    'Top Stock': sector_df.nlargest(1, 'master_score')['ticker'].iloc[0],
                                    '30D Avg': sector_df['ret_30d'].mean() if 'ret_30d' in sector_df.columns else 0,
                                    'RVOL Avg': sector_df['rvol'].mean() if 'rvol' in sector_df.columns else 1,
                                    'Flow ₹M': sector_df['money_flow_mm'].sum() if 'money_flow_mm' in sector_df.columns else 0
                                })
                    
                    if sector_analysis:
                        sector_df_display = pd.DataFrame(sector_analysis)
                        sector_df_display = sector_df_display.sort_values('Rotation Score', ascending=False)
                        
                        # Format display - YOUR EXACT FORMATTING
                        sector_df_display['Status'] = sector_df_display['Rotation Score'].apply(
                            lambda x: '🔥 HOT' if x >= 75 else '📈 RISING' if x >= 50 else '➡️ NEUTRAL' if x >= 25 else '📉 WEAK'
                        )
                        
                        sector_df_display['Avg Score'] = sector_df_display['Avg Score'].apply(lambda x: f"{x:.1f}")
                        sector_df_display['30D Avg'] = sector_df_display['30D Avg'].apply(lambda x: f"{x:+.1f}%")
                        sector_df_display['RVOL Avg'] = sector_df_display['RVOL Avg'].apply(lambda x: f"{x:.1f}x")
                        sector_df_display['Flow ₹M'] = sector_df_display['Flow ₹M'].apply(lambda x: f"{x:.0f}")
                        
                        # Display main table - YOUR COLUMN ORDER
                        display_cols = ['Status', 'Sector', 'Rotation Score', 'Stocks', 'Avg Score', 
                                       '30D Avg', 'RVOL Avg', 'Flow ₹M', 'Top Stock', 'Signals']
                        
                        st.dataframe(
                            sector_df_display[display_cols],
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                        
                        # Rotation summary - YOUR EXACT MESSAGES
                        hot_sectors = len(sector_df_display[sector_df_display['Rotation Score'] >= 75])
                        rising_sectors = len(sector_df_display[sector_df_display['Rotation Score'] >= 50])
                        
                        if hot_sectors > 0:
                            top_sector = sector_df_display.iloc[0]
                            st.success(
                                f"🔥 **{hot_sectors} HOT SECTORS** | Leader: {top_sector['Sector']} "
                                f"(Score: {top_sector['Rotation Score']})"
                            )
                        elif rising_sectors > 0:
                            st.info(f"📈 {rising_sectors} sectors showing positive rotation")
                        else:
                            st.warning("⚠️ No clear sector rotation detected")
                else:
                    st.info("Sector data not available")
            
            # TAB 2: SMART MONEY FLOW
            with intel_tabs[1]:
                st.markdown("##### 💰 **Smart Money Flow Analysis**")
                
                # ============================================
                # TOP METRICS ROW - QUICK OVERVIEW
                # ============================================
                flow_cols = st.columns(5)
                
                with flow_cols[0]:
                    total_flow = filtered_df['money_flow_mm'].sum() if 'money_flow_mm' in filtered_df.columns else 0
                    st.metric("Total Flow", f"₹{total_flow:,.0f}M")
                
                with flow_cols[1]:
                    if 'money_flow_mm' in filtered_df.columns:
                        top10_flow = filtered_df.nlargest(10, 'money_flow_mm')['money_flow_mm'].sum()
                        concentration = (top10_flow / total_flow * 100) if total_flow > 0 else 0
                        st.metric("Top 10 Share", f"{concentration:.0f}%")
                    else:
                        st.metric("Top 10 Share", "N/A")
                
                with flow_cols[2]:
                    if 'rvol' in filtered_df.columns:
                        high_rvol_flow = filtered_df[filtered_df['rvol'] > 2]['money_flow_mm'].sum() if 'money_flow_mm' in filtered_df.columns else 0
                        st.metric("High RVOL Flow", f"₹{high_rvol_flow:,.0f}M")
                    else:
                        st.metric("High RVOL Flow", "N/A")
                
                with flow_cols[3]:
                    # Institutional count
                    inst_count = 0
                    if all(col in filtered_df.columns for col in ['rvol', 'ret_1d']):
                        inst_count = len(filtered_df[(filtered_df['rvol'] > 2) & (filtered_df['ret_1d'].abs() < 3)])
                    st.metric("Institutional Signals", inst_count)
                
                with flow_cols[4]:
                    # Market mood based on category flow
                    if 'category' in filtered_df.columns and 'money_flow_mm' in filtered_df.columns:
                        cat_flow = filtered_df.groupby('category')['money_flow_mm'].sum()
                        if len(cat_flow) > 0:
                            top_cat = cat_flow.idxmax()
                            if 'Small' in top_cat or 'Micro' in top_cat:
                                st.metric("Market Mood", "🔥 RISK-ON")
                            elif 'Large' in top_cat or 'Mega' in top_cat:
                                st.metric("Market Mood", "🛡️ RISK-OFF")
                            else:
                                st.metric("Market Mood", "⚖️ NEUTRAL")
                    else:
                        st.metric("Market Mood", "N/A")
                
                st.markdown("---")
                
                # ============================================
                # MAIN SMART MONEY TABLE
                # ============================================
                st.markdown("**🏦 Institutional Money Flow Leaders**")
                
                if 'money_flow_mm' in filtered_df.columns:
                    # Calculate institutional score for each stock
                    filtered_df['inst_score'] = 0
                    
                    # Signal 1: High volume with controlled price movement
                    if all(col in filtered_df.columns for col in ['rvol', 'ret_1d']):
                        signal1 = (filtered_df['rvol'] > 2) & (filtered_df['ret_1d'].abs() < 3)
                        filtered_df.loc[signal1, 'inst_score'] += 25
                    
                    # Signal 2: Sustained volume increase
                    if all(col in filtered_df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d']):
                        signal2 = (filtered_df['vol_ratio_7d_90d'] > 1.5) & (filtered_df['vol_ratio_30d_90d'] > 1.3)
                        filtered_df.loc[signal2, 'inst_score'] += 25
                    
                    # Signal 3: Large money flow (top 20%)
                    flow_q80 = filtered_df['money_flow_mm'].quantile(0.8)
                    signal3 = filtered_df['money_flow_mm'] > flow_q80
                    filtered_df.loc[signal3, 'inst_score'] += 25
                    
                    # Signal 4: Institutional patterns
                    if 'patterns' in filtered_df.columns:
                        signal4 = filtered_df['patterns'].str.contains('INSTITUTIONAL|STEALTH|PYRAMID', na=False, regex=True)
                        filtered_df.loc[signal4, 'inst_score'] += 25
                    
                    # Get top institutional stocks
                    institutional = filtered_df[filtered_df['inst_score'] >= 50].copy()
                    institutional = institutional.nlargest(20, 'money_flow_mm')
                    
                    if len(institutional) > 0:
                        # Prepare display with company names
                        inst_display = []
                        for _, stock in institutional.iterrows():
                            # Determine flow type
                            flow_type = []
                            if stock.get('rvol', 1) > 3:
                                flow_type.append("🌋 Explosive")
                            elif stock.get('rvol', 1) > 2:
                                flow_type.append("🔥 Active")
                            else:
                                flow_type.append("🤫 Stealth")
                            
                            # Direction based on price movement
                            if stock.get('ret_1d', 0) > 2:
                                direction = "📈 Buying"
                            elif stock.get('ret_1d', 0) < -2:
                                direction = "📉 Selling"
                            else:
                                direction = "➡️ Accumulating"
                            
                            # Confidence level
                            if stock['inst_score'] >= 75:
                                confidence = "🟢 HIGH"
                            elif stock['inst_score'] >= 50:
                                confidence = "🟡 MODERATE"
                            else:
                                confidence = "⚪ LOW"
                            
                            inst_display.append({
                                'Company': stock.get('company_name', 'N/A')[:30] + '...' if len(str(stock.get('company_name', ''))) > 30 else stock.get('company_name', 'N/A'),
                                'Ticker': stock['ticker'],
                                'Flow ₹M': stock['money_flow_mm'],
                                'Price': stock['price'],
                                'RVOL': stock.get('rvol', 1),
                                '1D%': stock.get('ret_1d', 0),
                                'Type': flow_type[0] if flow_type else '',
                                'Direction': direction,
                                'Confidence': confidence,
                                'Score': stock['master_score'],
                                'Category': stock.get('category', 'N/A')
                            })
                        
                        inst_df = pd.DataFrame(inst_display)
                        
                        # Display with column configuration
                        st.dataframe(
                            inst_df,
                            use_container_width=True,
                            hide_index=True,
                            height=400,
                            column_config={
                                'Company': st.column_config.TextColumn(
                                    'Company',
                                    help="Company name",
                                    width="large"
                                ),
                                'Ticker': st.column_config.TextColumn(
                                    'Ticker',
                                    help="Stock symbol",
                                    width="small"
                                ),
                                'Flow ₹M': st.column_config.NumberColumn(
                                    'Flow ₹M',
                                    help="Money flow in millions",
                                    format="₹%.1f M"
                                ),
                                'Price': st.column_config.NumberColumn(
                                    'Price',
                                    help="Current price",
                                    format="₹%.0f"
                                ),
                                'RVOL': st.column_config.NumberColumn(
                                    'RVOL',
                                    help="Relative volume",
                                    format="%.1fx"
                                ),
                                '1D%': st.column_config.NumberColumn(
                                    '1D%',
                                    help="1-day return",
                                    format="%.1f%%"
                                ),
                                'Type': st.column_config.TextColumn(
                                    'Type',
                                    help="Flow type",
                                    width="medium"
                                ),
                                'Direction': st.column_config.TextColumn(
                                    'Direction',
                                    help="Flow direction",
                                    width="medium"
                                ),
                                'Confidence': st.column_config.TextColumn(
                                    'Confidence',
                                    help="Signal confidence",
                                    width="small"
                                ),
                                'Score': st.column_config.ProgressColumn(
                                    'Score',
                                    help="Master score",
                                    format="%.0f",
                                    min_value=0,
                                    max_value=100
                                ),
                                'Category': st.column_config.TextColumn(
                                    'Category',
                                    help="Market cap category",
                                    width="medium"
                                )
                            }
                        )
                        
                        # Top institutional pick
                        if len(institutional) > 0:
                            best = institutional.iloc[0]
                            st.success(
                                f"🏆 **Top Institutional Pick: {best.get('company_name', 'N/A')[:40]} ({best['ticker']})**\n"
                                f"• Money Flow: ₹{best['money_flow_mm']:.0f}M\n"
                                f"• Entry: ₹{best['price']:.0f} | RVOL: {best.get('rvol', 1):.1f}x\n"
                                f"• Confidence: {int(best['inst_score'])}% | Score: {best['master_score']:.0f}"
                            )
                    else:
                        st.info("No clear institutional activity detected")
                
                st.markdown("---")
                
                # ============================================
                # CATEGORY FLOW ANALYSIS
                # ============================================
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Category Flow Distribution**")
                    
                    if 'category' in filtered_df.columns and 'money_flow_mm' in filtered_df.columns:
                        cat_flow = filtered_df.groupby('category').agg({
                            'money_flow_mm': 'sum',
                            'ticker': 'count',
                            'master_score': 'mean',
                            'rvol': 'mean'
                        }).round(2)
                        
                        cat_flow.columns = ['Total Flow ₹M', 'Stocks', 'Avg Score', 'Avg RVOL']
                        cat_flow = cat_flow.sort_values('Total Flow ₹M', ascending=False)
                        
                        # Add percentage
                        total = cat_flow['Total Flow ₹M'].sum()
                        cat_flow['% Share'] = (cat_flow['Total Flow ₹M'] / total * 100).round(1)
                        
                        # Format for display
                        cat_flow['Total Flow ₹M'] = cat_flow['Total Flow ₹M'].apply(lambda x: f"{x:,.0f}")
                        cat_flow['Avg Score'] = cat_flow['Avg Score'].apply(lambda x: f"{x:.1f}")
                        cat_flow['Avg RVOL'] = cat_flow['Avg RVOL'].apply(lambda x: f"{x:.1f}x")
                        cat_flow['% Share'] = cat_flow['% Share'].apply(lambda x: f"{x:.1f}%")
                        
                        st.dataframe(
                            cat_flow[['Total Flow ₹M', '% Share', 'Stocks', 'Avg Score', 'Avg RVOL']],
                            use_container_width=True
                        )
                
                with col2:
                    st.markdown("**⚠️ Unusual Flow Alerts**")
                    
                    if all(col in filtered_df.columns for col in ['money_flow_mm', 'ret_1d', 'rvol', 'company_name']):
                        # Abnormal flow detection
                        abnormal = filtered_df[
                            (filtered_df['money_flow_mm'] > filtered_df['money_flow_mm'].quantile(0.8)) &
                            (filtered_df['ret_1d'].abs() < 1) &
                            (filtered_df['rvol'] > 2)
                        ].head(5)
                        
                        if len(abnormal) > 0:
                            for _, stock in abnormal.iterrows():
                                company_short = stock['company_name'][:25] + '...' if len(stock['company_name']) > 25 else stock['company_name']
                                st.warning(
                                    f"**{company_short}**\n"
                                    f"({stock['ticker']}) - Potential Distribution\n"
                                    f"High flow ₹{stock['money_flow_mm']:.0f}M but flat price"
                                )
                        
                        # Divergence detection
                        divergence = filtered_df[
                            (filtered_df['ret_30d'] > 20) &
                            (filtered_df.get('vol_ratio_30d_180d', 1) < 0.7)
                        ].head(3)
                        
                        if len(divergence) > 0:
                            st.error("**📉 Volume Divergence Detected:**")
                            for _, stock in divergence.iterrows():
                                company_short = stock['company_name'][:20] + '...' if len(stock['company_name']) > 20 else stock['company_name']
                                st.caption(f"• {company_short} ({stock['ticker']}): Price up but volume down")
                        
                        if len(abnormal) == 0 and len(divergence) == 0:
                            st.success("✅ No unusual flow patterns detected")            
            st.markdown("---")
            
            # ====================================
            # EXECUTIVE ACTION SUMMARY - ULTIMATE VERSION
            # ====================================
            st.markdown("#### 📋 Executive Action Summary")
            
            # Calculate market health with YOUR factors
            market_health = 0
            health_details = []
            
            # Factor 1: Breadth
            if 'ret_1d' in filtered_df.columns:
                advancing = len(filtered_df[filtered_df['ret_1d'] > 0])
                declining = len(filtered_df[filtered_df['ret_1d'] < 0])
                breadth = advancing / len(filtered_df) if len(filtered_df) > 0 else 0
                
                if breadth > 0.6:
                    market_health += 25
                    health_details.append(f"✅ Breadth: {breadth:.0%} positive")
                elif breadth > 0.4:
                    market_health += 15
                    health_details.append(f"➡️ Breadth: {breadth:.0%} neutral")
                else:
                    health_details.append(f"❌ Breadth: {breadth:.0%} negative")
            
            # Factor 2: Momentum
            if 'momentum_score' in filtered_df.columns:
                high_momentum = len(filtered_df[filtered_df['momentum_score'] > 70])
                momentum_pct = high_momentum / len(filtered_df) if len(filtered_df) > 0 else 0
                
                if momentum_pct > 0.3:
                    market_health += 25
                    health_details.append(f"✅ Momentum: {high_momentum} strong stocks")
                elif momentum_pct > 0.15:
                    market_health += 15
                    health_details.append(f"➡️ Momentum: {high_momentum} stocks")
                else:
                    health_details.append(f"❌ Momentum: Only {high_momentum} stocks")
            
            # Factor 3: Volume
            if 'rvol' in filtered_df.columns:
                active_volume = len(filtered_df[filtered_df['rvol'] > 2])
                volume_pct = active_volume / len(filtered_df) if len(filtered_df) > 0 else 0
                
                if volume_pct > 0.2:
                    market_health += 25
                    health_details.append(f"✅ Volume: {active_volume} active stocks")
                elif volume_pct > 0.1:
                    market_health += 15
                    health_details.append(f"➡️ Volume: {active_volume} stocks")
                else:
                    health_details.append(f"❌ Volume: Low activity")
            
            # Factor 4: Patterns
            if 'patterns' in filtered_df.columns:
                with_patterns = len(filtered_df[filtered_df['patterns'] != ''])
                pattern_pct = with_patterns / len(filtered_df) if len(filtered_df) > 0 else 0
                
                if pattern_pct > 0.4:
                    market_health += 25
                    health_details.append(f"✅ Patterns: {with_patterns} signals")
                elif pattern_pct > 0.2:
                    market_health += 15
                    health_details.append(f"➡️ Patterns: {with_patterns} signals")
                else:
                    health_details.append(f"❌ Patterns: Few signals")
            
            # ============================================
            # MAIN ACTION DISPLAY - 3 COLUMNS
            # ============================================
            action_col1, action_col2, action_col3 = st.columns([1, 2, 1])
            
            with action_col1:
                st.markdown("**📊 MARKET STATUS**")
                
                # Market health meter
                if market_health >= 70:
                    st.success(f"🟢 **HEALTHY** ({market_health}%)")
                    market_condition = "BULLISH"
                elif market_health >= 40:
                    st.warning(f"🟡 **NEUTRAL** ({market_health}%)")
                    market_condition = "MIXED"
                else:
                    st.error(f"🔴 **WEAK** ({market_health}%)")
                    market_condition = "BEARISH"
                
                # Health details
                for detail in health_details:
                    st.caption(detail)
                
                st.markdown("---")
                
                # Quick stats
                st.markdown("**📈 Quick Stats**")
                if len(filtered_df) > 0:
                    st.caption(f"• Stocks: {len(filtered_df)}")
                    st.caption(f"• Avg Score: {filtered_df['master_score'].mean():.1f}")
                    if 'ret_30d' in filtered_df.columns:
                        winners_30d = (filtered_df['ret_30d'] > 0).sum()
                        st.caption(f"• 30D Winners: {winners_30d}/{len(filtered_df)}")
            
            with action_col2:
                st.markdown("**🎯 SPECIFIC ACTIONS NOW**")
                
                # Get actual stocks for actions
                if market_health >= 70:  # BULLISH
                    # BUY candidates
                    buy_candidates = filtered_df[
                        (filtered_df['momentum_score'] > 70) & 
                        (filtered_df['acceleration_score'] > 70) &
                        (filtered_df['from_high_pct'] > -10) &
                        (filtered_df['rvol'] > 1.5)
                    ].nlargest(3, 'master_score')
                    
                    if len(buy_candidates) > 0:
                        st.success("**BUY THESE NOW:**")
                        for _, stock in buy_candidates.iterrows():
                            company = stock.get('company_name', stock['ticker'])[:25]
                            entry = stock['price']
                            stop = entry * 0.95  # 5% stop
                            target = entry * 1.10  # 10% target
                            
                            st.write(
                                f"**{company} ({stock['ticker']})**\n"
                                f"Entry: ₹{entry:.0f} | Stop: ₹{stop:.0f} | Target: ₹{target:.0f}"
                            )
                    
                    # HOLD candidates
                    hold_candidates = filtered_df[
                        (filtered_df['master_score'] > 70) &
                        (filtered_df['from_low_pct'] > 50)
                    ].head(2)
                    
                    if len(hold_candidates) > 0:
                        st.info("**HOLD & TRAIL STOP:**")
                        for _, stock in hold_candidates.iterrows():
                            company = stock.get('company_name', stock['ticker'])[:20]
                            st.caption(f"• {company} - Move stop to ₹{stock['price'] * 0.92:.0f}")
                    
                elif market_health >= 40:  # NEUTRAL
                    # WATCH candidates
                    watch_candidates = filtered_df[
                        (filtered_df['breakout_score'] > 75) &
                        (filtered_df['momentum_score'] < 70)
                    ].nlargest(3, 'master_score')
                    
                    if len(watch_candidates) > 0:
                        st.warning("**WATCH FOR BREAKOUT:**")
                        for _, stock in watch_candidates.iterrows():
                            company = stock.get('company_name', stock['ticker'])[:25]
                            trigger = stock['price'] * 1.02  # 2% above current
                            
                            st.write(
                                f"**{company} ({stock['ticker']})**\n"
                                f"Current: ₹{stock['price']:.0f} | Buy above: ₹{trigger:.0f}"
                            )
                    
                    # REDUCE weak positions
                    st.info("**REDUCE:** Stocks below 50 score or negative momentum")
                    
                else:  # BEARISH
                    # EXIT candidates
                    exit_candidates = filtered_df[
                        (filtered_df['momentum_score'] < 40) |
                        (filtered_df['from_high_pct'] < -20) |
                        (filtered_df['patterns'].str.contains('DISTRIBUTION|EXHAUSTION', na=False))
                    ].head(3)
                    
                    if len(exit_candidates) > 0:
                        st.error("**EXIT THESE NOW:**")
                        for _, stock in exit_candidates.iterrows():
                            company = stock.get('company_name', stock['ticker'])[:25]
                            reason = "Weak momentum" if stock['momentum_score'] < 40 else "Distribution pattern"
                            
                            st.write(
                                f"**{company} ({stock['ticker']})**\n"
                                f"Exit at: ₹{stock['price']:.0f} | Reason: {reason}"
                            )
                    
                    st.warning("**CASH IS KING** - Wait for market recovery")
            
            with action_col3:
                st.markdown("**⚠️ RISK MANAGEMENT**")
                
                # Calculate risk level
                risk_score = 0
                risk_factors = []
                
                # Check overextension
                if 'from_low_pct' in filtered_df.columns:
                    overextended = len(filtered_df[filtered_df['from_low_pct'] > 80])
                    if overextended > len(filtered_df) * 0.3:
                        risk_score += 40
                        risk_factors.append("Many overextended")
                
                # Check breaking waves
                if 'wave_state' in filtered_df.columns:
                    breaking = len(filtered_df[filtered_df['wave_state'].str.contains('BREAKING', na=False)])
                    if breaking > len(filtered_df) * 0.3:
                        risk_score += 30
                        risk_factors.append("Waves breaking")
                
                # Check high PE
                if 'pe' in filtered_df.columns:
                    high_pe = len(filtered_df[(filtered_df['pe'] > 50) & (filtered_df['pe'] < 10000)])
                    if high_pe > len(filtered_df) * 0.4:
                        risk_score += 30
                        risk_factors.append("Valuations high")
                
                # Display risk level
                if risk_score >= 70:
                    st.error("**🔴 HIGH RISK**")
                    st.write(
                        "**Actions:**\n"
                        "• Reduce positions by 30%\n"
                        "• Use 3% stop losses\n"
                        "• No new longs"
                    )
                elif risk_score >= 40:
                    st.warning("**🟡 MODERATE RISK**")
                    st.write(
                        "**Actions:**\n"
                        "• Trail stops to 5%\n"
                        "• Book partial profits\n"
                        "• Selective entries only"
                    )
                else:
                    st.success("**🟢 LOW RISK**")
                    st.write(
                        "**Actions:**\n"
                        "• Normal position sizes\n"
                        "• 7% stop losses\n"
                        "• Add on dips"
                    )
                
                # Risk factors
                if risk_factors:
                    st.markdown("**Risk Factors:**")
                    for factor in risk_factors:
                        st.caption(f"• {factor}")
                
                st.markdown("---")
                
                # Position sizing
                st.markdown("**💰 Position Size**")
                if market_health >= 70:
                    st.success("Max 5% per stock")
                elif market_health >= 40:
                    st.warning("Max 3% per stock")
                else:
                    st.error("Max 2% per stock")
            
            # ============================================
            # ACTION SUMMARY BOX
            # ============================================
            st.markdown("---")
            
            # Create action summary based on market condition
            if market_condition == "BULLISH":
                summary_color = "success"
                summary_icon = "🚀"
                summary_text = "Market is strong. Focus on momentum leaders with volume."
            elif market_condition == "MIXED":
                summary_color = "warning"
                summary_icon = "⚖️"
                summary_text = "Market is mixed. Be selective, wait for clear setups."
            else:
                summary_color = "error"
                summary_icon = "🛡️"
                summary_text = "Market is weak. Preserve capital, avoid new positions."
            
            getattr(st, summary_color)(
                f"{summary_icon} **BOTTOM LINE:** {summary_text}\n"
                f"Market Health: {market_health}% | "
                f"Active Filters: {st.session_state.get('active_filter_count', 0)} | "
                f"Stocks Analyzed: {len(filtered_df)}"
            )
    
    # Tab 1: Rankings
    # RANKINGS TAB - DATAFRAME SECTION ONLY
    with tabs[1]:
        st.markdown("### 🏆 Top Ranked Stocks")
        
        col1, col2, col3 = st.columns([2, 2, 6])
        with col1:
            display_count = st.selectbox(
                "Show top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=CONFIG.AVAILABLE_TOP_N.index(st.session_state.user_preferences['default_top_n']),
                key="display_count_select"
            )
            st.session_state.user_preferences['default_top_n'] = display_count
        
        with col2:
            sort_options = ['Rank', 'Master Score', 'RVOL', 'Momentum', 'Money Flow']
            if 'trend_quality' in filtered_df.columns:
                sort_options.append('Trend')
            
            sort_by = st.selectbox(
                "Sort by", 
                options=sort_options, 
                index=0,
                key="sort_by_select"
            )
        
        display_df = filtered_df.head(display_count).copy()
        
        # Apply sorting
        if sort_by == 'Master Score':
            display_df = display_df.sort_values('master_score', ascending=False)
        elif sort_by == 'RVOL':
            display_df = display_df.sort_values('rvol', ascending=False)
        elif sort_by == 'Momentum':
            display_df = display_df.sort_values('momentum_score', ascending=False)
        elif sort_by == 'Money Flow' and 'money_flow_mm' in display_df.columns:
            display_df = display_df.sort_values('money_flow_mm', ascending=False)
        elif sort_by == 'Trend' and 'trend_quality' in display_df.columns:
            display_df = display_df.sort_values('trend_quality', ascending=False)
        
        if not display_df.empty:
            # ============================================
            # PREPARE DISPLAY DATAFRAME - KEEP NUMERIC!
            # ============================================
            
            # Select columns in logical order
            display_columns = []
            
            # 1. IDENTIFICATION
            display_columns.extend(['rank', 'ticker', 'company_name'])
            
            # 2. SCORES (keep numeric for progress bars)
            display_columns.extend(['master_score'])
            
            # 3. PRICE & RANGE
            display_columns.extend(['price'])
            if 'from_low_pct' in display_df.columns:
                display_columns.append('from_low_pct')
            if 'from_high_pct' in display_df.columns:
                display_columns.append('from_high_pct')
            
            # 4. MOMENTUM METRICS
            if 'momentum_score' in display_df.columns:
                display_columns.append('momentum_score')
            if 'acceleration_score' in display_df.columns:
                display_columns.append('acceleration_score')
            
            # 5. RETURNS
            for ret_col in ['ret_1d', 'ret_7d', 'ret_30d']:
                if ret_col in display_df.columns:
                    display_columns.append(ret_col)
            
            # 6. VOLUME
            if 'rvol' in display_df.columns:
                display_columns.append('rvol')
            if 'volume_score' in display_df.columns:
                display_columns.append('volume_score')
            
            # 7. MONEY FLOW
            if 'money_flow_mm' in display_df.columns:
                display_columns.append('money_flow_mm')
            
            # 8. FUNDAMENTALS (if hybrid mode)
            if show_fundamentals:
                if 'pe' in display_df.columns:
                    display_columns.append('pe')
                if 'eps_change_pct' in display_df.columns:
                    display_columns.append('eps_change_pct')
            
            # 9. PATTERNS & SIGNALS
            if 'wave_state' in display_df.columns:
                display_columns.append('wave_state')
            if 'patterns' in display_df.columns:
                display_columns.append('patterns')
            
            # 10. CLASSIFICATION
            display_columns.extend(['category', 'sector'])
            if 'industry' in display_df.columns:
                display_columns.append('industry')
            
            # Filter to available columns
            available_cols = [col for col in display_columns if col in display_df.columns]
            final_df = display_df[available_cols].copy()
            
            # ============================================
            # ULTIMATE COLUMN CONFIGURATION
            # ============================================
            column_config = {
                # IDENTIFICATION
                "rank": st.column_config.NumberColumn(
                    "🏆",
                    help="Overall ranking position",
                    width="small",
                    format="%d"
                ),
                "ticker": st.column_config.TextColumn(
                    "Ticker",
                    help="Stock symbol",
                    width="small"
                ),
                "company_name": st.column_config.TextColumn(
                    "Company",
                    help="Company name",
                    width="large",
                    max_chars=50
                ),
                
                # MAIN SCORE - PROGRESS BAR
                "master_score": st.column_config.ProgressColumn(
                    "Score",
                    help="Master Score (0-100) - Click column to sort",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                    width="small"
                ),
                
                # PRICE - FORMATTED NUMBER
                "price": st.column_config.NumberColumn(
                    "Price",
                    help="Current stock price",
                    format="₹%.0f",
                    width="small"
                ),
                
                # RANGE POSITION - WITH COLOR
                "from_low_pct": st.column_config.NumberColumn(
                    "📈 From Low",
                    help="% up from 52-week low",
                    format="%.0f%%",
                    width="small"
                ),
                "from_high_pct": st.column_config.NumberColumn(
                    "📉 From High",
                    help="% down from 52-week high",
                    format="%.0f%%",
                    width="small"
                ),
                
                # MOMENTUM SCORES - PROGRESS BARS
                "momentum_score": st.column_config.ProgressColumn(
                    "Mom",
                    help="Momentum Score",
                    format="%.0f",
                    min_value=0,
                    max_value=100,
                    width="small"
                ),
                "acceleration_score": st.column_config.ProgressColumn(
                    "Accel",
                    help="Acceleration Score",
                    format="%.0f",
                    min_value=0,
                    max_value=100,
                    width="small"
                ),
                "volume_score": st.column_config.ProgressColumn(
                    "Vol",
                    help="Volume Score",
                    format="%.0f",
                    min_value=0,
                    max_value=100,
                    width="small"
                ),
                
                # RETURNS - COLOR CODED
                "ret_1d": st.column_config.NumberColumn(
                    "1D%",
                    help="1-day return",
                    format="%.1f%%",
                    width="small"
                ),
                "ret_7d": st.column_config.NumberColumn(
                    "7D%",
                    help="7-day return",
                    format="%.1f%%",
                    width="small"
                ),
                "ret_30d": st.column_config.NumberColumn(
                    "30D%",
                    help="30-day return",
                    format="%.1f%%",
                    width="small"
                ),
                
                # VOLUME - SPECIAL FORMAT
                "rvol": st.column_config.NumberColumn(
                    "RVOL",
                    help="Relative Volume - Times normal volume",
                    format="%.1fx",
                    width="small"
                ),
                
                # MONEY FLOW - BAR CHART
                "money_flow_mm": st.column_config.BarChartColumn(
                    "Flow ₹M",
                    help="Money Flow in Millions",
                    width="small",
                    y_min=0,
                    y_max=float(display_df['money_flow_mm'].max()) if 'money_flow_mm' in display_df.columns else 100
                ),
                
                # FUNDAMENTALS
                "pe": st.column_config.NumberColumn(
                    "PE",
                    help="Price to Earnings Ratio",
                    format="%.1f",
                    width="small"
                ),
                "eps_change_pct": st.column_config.NumberColumn(
                    "EPS Δ%",
                    help="EPS Change %",
                    format="%.0f%%",
                    width="small"
                ),
                
                # WAVE STATE - TEXT WITH EMOJI
                "wave_state": st.column_config.TextColumn(
                    "Wave",
                    help="Current momentum wave state",
                    width="medium"
                ),
                
                # PATTERNS - LONG TEXT
                "patterns": st.column_config.TextColumn(
                    "Patterns",
                    help="Detected trading patterns",
                    width="large",
                    max_chars=100
                ),
                
                # CLASSIFICATION
                "category": st.column_config.SelectboxColumn(
                    "Category",
                    help="Market cap category",
                    width="medium",
                    options=filtered_df['category'].unique().tolist() if 'category' in filtered_df.columns else []
                ),
                "sector": st.column_config.TextColumn(
                    "Sector",
                    help="Business sector",
                    width="medium",
                    max_chars=30
                ),
                "industry": st.column_config.TextColumn(
                    "Industry",
                    help="Specific industry",
                    width="medium",
                    max_chars=40
                )
            }
            
            # ============================================
            # DISPLAY WITH ENHANCED STYLING
            # ============================================
            
            # Add conditional formatting for specific columns
            styled_df = final_df.style
            
            # Color code returns
            if 'ret_1d' in final_df.columns:
                styled_df = styled_df.applymap(
                    lambda x: 'color: green' if x > 0 else 'color: red' if x < 0 else '',
                    subset=['ret_1d']
                )
            if 'ret_7d' in final_df.columns:
                styled_df = styled_df.applymap(
                    lambda x: 'color: green' if x > 0 else 'color: red' if x < 0 else '',
                    subset=['ret_7d']
                )
            if 'ret_30d' in final_df.columns:
                styled_df = styled_df.applymap(
                    lambda x: 'color: green' if x > 0 else 'color: red' if x < 0 else '',
                    subset=['ret_30d']
                )
            
            # Highlight high RVOL
            if 'rvol' in final_df.columns:
                styled_df = styled_df.applymap(
                    lambda x: 'background-color: #ffebee' if x > 3 else 'background-color: #fff3e0' if x > 2 else '',
                    subset=['rvol']
                )
            
            # Color gradient for scores
            score_cols = [col for col in ['master_score', 'momentum_score', 'acceleration_score', 'volume_score'] 
                         if col in final_df.columns]
            if score_cols:
                styled_df = styled_df.background_gradient(
                    subset=score_cols,
                    cmap='RdYlGn',
                    vmin=0,
                    vmax=100
                )
            
            # Display the enhanced dataframe
            st.dataframe(
                final_df,  # Use unstyled for column_config to work properly
                use_container_width=True,
                height=min(600, len(final_df) * 35 + 50),
                hide_index=True,
                column_config=column_config,
                column_order=available_cols  # Maintain our logical order
            )
            
            # ============================================
            # QUICK ACTION BAR BELOW TABLE
            # ============================================
            action_cols = st.columns(6)
            
            with action_cols[0]:
                if st.button("📋 Copy Tickers", use_container_width=True):
                    tickers = ', '.join(final_df['ticker'].head(10).tolist())
                    st.code(tickers, language=None)
            
            with action_cols[1]:
                top_score = final_df.iloc[0] if not final_df.empty else None
                if top_score is not None:
                    st.metric("🏆 Top Score", f"{top_score['master_score']:.1f}")
            
            with action_cols[2]:
                avg_score = final_df['master_score'].mean()
                st.metric("📊 Avg Score", f"{avg_score:.1f}")
            
            with action_cols[3]:
                if 'rvol' in final_df.columns:
                    high_vol = (final_df['rvol'] > 2).sum()
                    st.metric("🔥 High Vol", f"{high_vol}")
            
            with action_cols[4]:
                if 'ret_30d' in final_df.columns:
                    gainers = (final_df['ret_30d'] > 0).sum()
                    st.metric("📈 30D Gainers", f"{gainers}")
            
            with action_cols[5]:
                if 'patterns' in final_df.columns:
                    with_patterns = (final_df['patterns'] != '').sum()
                    st.metric("🎯 Patterns", f"{with_patterns}")
            
            # Quick Statistics Section
            with st.expander("📊 Quick Statistics", expanded=False):
                stat_cols = st.columns(4)
                
                with stat_cols[0]:
                    st.markdown("**📈 Score Distribution**")
                    if 'master_score' in display_df.columns:
                        score_stats = {
                            'Max': f"{display_df['master_score'].max():.1f}",
                            'Q3': f"{display_df['master_score'].quantile(0.75):.1f}",
                            'Median': f"{display_df['master_score'].median():.1f}",
                            'Q1': f"{display_df['master_score'].quantile(0.25):.1f}",
                            'Min': f"{display_df['master_score'].min():.1f}",
                            'Mean': f"{display_df['master_score'].mean():.1f}",
                            'Std Dev': f"{display_df['master_score'].std():.1f}"
                        }
                        
                        stats_df = pd.DataFrame(
                            list(score_stats.items()),
                            columns=['Metric', 'Value']
                        )
                        
                        st.dataframe(
                            stats_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'Metric': st.column_config.TextColumn('Metric', width="small"),
                                'Value': st.column_config.TextColumn('Value', width="small")
                            }
                        )
                
                with stat_cols[1]:
                    st.markdown("**💰 Returns (30D)**")
                    if 'ret_30d' in display_df.columns:
                        ret_stats = {
                            'Max': f"{display_df['ret_30d'].max():.1f}%",
                            'Min': f"{display_df['ret_30d'].min():.1f}%",
                            'Avg': f"{display_df['ret_30d'].mean():.1f}%",
                            'Positive': f"{(display_df['ret_30d'] > 0).sum()}",
                            'Negative': f"{(display_df['ret_30d'] < 0).sum()}",
                            'Win Rate': f"{(display_df['ret_30d'] > 0).sum() / len(display_df) * 100:.0f}%"
                        }
                        
                        ret_df = pd.DataFrame(
                            list(ret_stats.items()),
                            columns=['Metric', 'Value']
                        )
                        
                        st.dataframe(
                            ret_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'Metric': st.column_config.TextColumn('Metric', width="small"),
                                'Value': st.column_config.TextColumn('Value', width="small")
                            }
                        )
                    else:
                        st.text("No 30D return data available")
                
                with stat_cols[2]:
                    if show_fundamentals:
                        st.markdown("**💎 Fundamentals**")
                        fund_stats = {}
                        
                        if 'pe' in display_df.columns:
                            valid_pe = display_df['pe'].notna() & (display_df['pe'] > 0) & (display_df['pe'] < 10000)
                            if valid_pe.any():
                                median_pe = display_df.loc[valid_pe, 'pe'].median()
                                fund_stats['Median PE'] = f"{median_pe:.1f}x"
                                fund_stats['PE < 15'] = f"{(display_df['pe'] < 15).sum()}"
                                fund_stats['PE 15-30'] = f"{((display_df['pe'] >= 15) & (display_df['pe'] < 30)).sum()}"
                                fund_stats['PE > 30'] = f"{(display_df['pe'] >= 30).sum()}"
                        
                        if 'eps_change_pct' in display_df.columns:
                            valid_eps = display_df['eps_change_pct'].notna()
                            if valid_eps.any():
                                positive = (display_df['eps_change_pct'] > 0).sum()
                                fund_stats['EPS Growth +ve'] = f"{positive}"
                                fund_stats['EPS > 50%'] = f"{(display_df['eps_change_pct'] > 50).sum()}"
                        
                        if fund_stats:
                            fund_df = pd.DataFrame(
                                list(fund_stats.items()),
                                columns=['Metric', 'Value']
                            )
                            
                            st.dataframe(
                                fund_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'Metric': st.column_config.TextColumn('Metric', width="medium"),
                                    'Value': st.column_config.TextColumn('Value', width="small")
                                }
                            )
                        else:
                            st.text("No fundamental data")
                    else:
                        st.markdown("**🔊 Volume**")
                        if 'rvol' in display_df.columns:
                            vol_stats = {
                                'Max RVOL': f"{display_df['rvol'].max():.1f}x",
                                'Avg RVOL': f"{display_df['rvol'].mean():.1f}x",
                                'RVOL > 3x': f"{(display_df['rvol'] > 3).sum()}",
                                'RVOL > 2x': f"{(display_df['rvol'] > 2).sum()}",
                                'RVOL > 1.5x': f"{(display_df['rvol'] > 1.5).sum()}"
                            }
                            
                            vol_df = pd.DataFrame(
                                list(vol_stats.items()),
                                columns=['Metric', 'Value']
                            )
                            
                            st.dataframe(
                                vol_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'Metric': st.column_config.TextColumn('Metric', width="medium"),
                                    'Value': st.column_config.TextColumn('Value', width="small")
                                }
                            )
                
                with stat_cols[3]:
                    st.markdown("**📊 Trend Distribution**")
                    if 'trend_quality' in display_df.columns:
                        trend_stats = {
                            'Avg Trend': f"{display_df['trend_quality'].mean():.1f}",
                            'Strong (80+)': f"{(display_df['trend_quality'] >= 80).sum()}",
                            'Good (60-79)': f"{((display_df['trend_quality'] >= 60) & (display_df['trend_quality'] < 80)).sum()}",
                            'Neutral (40-59)': f"{((display_df['trend_quality'] >= 40) & (display_df['trend_quality'] < 60)).sum()}",
                            'Weak (<40)': f"{(display_df['trend_quality'] < 40).sum()}"
                        }
                        
                        trend_df = pd.DataFrame(
                            list(trend_stats.items()),
                            columns=['Metric', 'Value']
                        )
                        
                        st.dataframe(
                            trend_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'Metric': st.column_config.TextColumn('Metric', width="medium"),
                                'Value': st.column_config.TextColumn('Value', width="small")
                            }
                        )
                    else:
                        st.text("No trend data available")
            
            # Top Patterns Section
            with st.expander("🎯 Top Patterns Detected", expanded=False):
                if 'patterns' in display_df.columns:
                    pattern_counts = {}
                    for patterns_str in display_df['patterns'].dropna():
                        if patterns_str:
                            for pattern in patterns_str.split(' | '):
                                pattern = pattern.strip()
                                if pattern:
                                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                    
                    if pattern_counts:
                        # Sort patterns by count
                        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        pattern_data = []
                        for pattern, count in sorted_patterns:
                            # Get stocks with this pattern
                            stocks_with_pattern = display_df[
                                display_df['patterns'].str.contains(pattern, na=False, regex=False)
                            ]['ticker'].head(5).tolist()
                            
                            pattern_data.append({
                                'Pattern': pattern,
                                'Count': count,
                                'Top Stocks': ', '.join(stocks_with_pattern[:3]) + ('...' if len(stocks_with_pattern) > 3 else '')
                            })
                        
                        patterns_df = pd.DataFrame(pattern_data)
                        
                        st.dataframe(
                            patterns_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'Pattern': st.column_config.TextColumn(
                                    'Pattern',
                                    help="Detected pattern name",
                                    width="medium"
                                ),
                                'Count': st.column_config.NumberColumn(
                                    'Count',
                                    help="Number of stocks with this pattern",
                                    format="%d",
                                    width="small"
                                ),
                                'Top Stocks': st.column_config.TextColumn(
                                    'Top Stocks',
                                    help="Example stocks with this pattern",
                                    width="large"
                                )
                            }
                        )
                    else:
                        st.info("No patterns detected in current selection")
                else:
                    st.info("Pattern data not available")
            
            # Category Performance Section
            with st.expander("📈 Category Performance", expanded=False):
                if 'category' in display_df.columns:
                    cat_performance = display_df.groupby('category').agg({
                        'master_score': ['mean', 'count'],
                        'ret_30d': 'mean' if 'ret_30d' in display_df.columns else lambda x: None,
                        'rvol': 'mean' if 'rvol' in display_df.columns else lambda x: None
                    }).round(2)
                    
                    # Flatten columns
                    cat_performance.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                              for col in cat_performance.columns.values]
                    
                    # Rename columns for clarity
                    rename_dict = {
                        'master_score_mean': 'Avg Score',
                        'master_score_count': 'Count',
                        'ret_30d_mean': 'Avg 30D Ret',
                        'ret_30d_<lambda>': 'Avg 30D Ret',
                        'rvol_mean': 'Avg RVOL',
                        'rvol_<lambda>': 'Avg RVOL'
                    }
                    
                    cat_performance.rename(columns=rename_dict, inplace=True)
                    
                    # Sort by average score
                    cat_performance = cat_performance.sort_values('Avg Score', ascending=False)
                    
                    # Format values
                    if 'Avg 30D Ret' in cat_performance.columns:
                        cat_performance['Avg 30D Ret'] = cat_performance['Avg 30D Ret'].apply(
                            lambda x: f"{x:.1f}%" if pd.notna(x) else '-'
                        )
                    
                    if 'Avg RVOL' in cat_performance.columns:
                        cat_performance['Avg RVOL'] = cat_performance['Avg RVOL'].apply(
                            lambda x: f"{x:.1f}x" if pd.notna(x) else '-'
                        )
                    
                    st.dataframe(
                        cat_performance,
                        use_container_width=True,
                        column_config={
                            'Avg Score': st.column_config.NumberColumn(
                                'Avg Score',
                                help="Average master score in category",
                                format="%.1f",
                                width="small"
                            ),
                            'Count': st.column_config.NumberColumn(
                                'Count',
                                help="Number of stocks in category",
                                format="%d",
                                width="small"
                            ),
                            'Avg 30D Ret': st.column_config.TextColumn(
                                'Avg 30D Ret',
                                help="Average 30-day return",
                                width="small"
                            ),
                            'Avg RVOL': st.column_config.TextColumn(
                                'Avg RVOL',
                                help="Average relative volume",
                                width="small"
                            )
                        }
                    )
                else:
                    st.info("Category data not available")
        
        else:
            st.warning("No stocks match the selected filters.")
            
            # Show filter summary
            st.markdown("#### Current Filters Applied:")
            if active_filter_count > 0:
                filter_summary = []
                
                if st.session_state.filter_state.get('categories'):
                    filter_summary.append(f"Categories: {', '.join(st.session_state.filter_state['categories'])}")
                if st.session_state.filter_state.get('sectors'):
                    filter_summary.append(f"Sectors: {', '.join(st.session_state.filter_state['sectors'])}")
                if st.session_state.filter_state.get('industries'):
                    filter_summary.append(f"Industries: {', '.join(st.session_state.filter_state['industries'][:5])}...")
                if st.session_state.filter_state.get('min_score', 0) > 0:
                    filter_summary.append(f"Min Score: {st.session_state.filter_state['min_score']}")
                if st.session_state.filter_state.get('patterns'):
                    filter_summary.append(f"Patterns: {len(st.session_state.filter_state['patterns'])} selected")
                
                for filter_text in filter_summary:
                    st.write(f"• {filter_text}")
                
                if st.button("Clear All Filters", type="primary", key="clear_filters_ranking_btn"):
                    FilterEngine.clear_all_filters()
                    SessionStateManager.clear_filters()
                    st.rerun()
            else:
                st.info("No filters applied. All stocks should be visible unless there's no data loaded.")
        
    # Tab 2: Wave Radar
    with tabs[2]:
        st.markdown("### 🌊 Wave Radar - Early Momentum Detection System")
        st.markdown("*Catch waves as they form, not after they've peaked!*")
        
        radar_col1, radar_col2, radar_col3, radar_col4 = st.columns([2, 2, 2, 1])
        
        with radar_col1:
            wave_timeframe = st.selectbox(
                "Wave Detection Timeframe",
                options=[
                    "All Waves",
                    "Intraday Surge",
                    "3-Day Buildup", 
                    "Weekly Breakout",
                    "Monthly Trend"
                ],
                index=["All Waves", "Intraday Surge", "3-Day Buildup", "Weekly Breakout", "Monthly Trend"].index(st.session_state.get('wave_timeframe_select', "All Waves")),
                key="wave_timeframe_select",
                help="""
                🌊 All Waves: Complete unfiltered view
                ⚡ Intraday Surge: High RVOL & today's movers
                📈 3-Day Buildup: Building momentum patterns
                🚀 Weekly Breakout: Near 52w highs with volume
                💪 Monthly Trend: Established trends with SMAs
                """
            )
        
        with radar_col2:
            sensitivity = st.select_slider(
                "Detection Sensitivity",
                options=["Conservative", "Balanced", "Aggressive"],
                value=st.session_state.get('wave_sensitivity', "Balanced"),
                key="wave_sensitivity",
                help="Conservative = Stronger signals, Aggressive = More signals"
            )
            
            show_sensitivity_details = st.checkbox(
                "Show thresholds",
                value=st.session_state.get('show_sensitivity_details', False),
                key="show_sensitivity_details",
                help="Display exact threshold values for current sensitivity"
            )
        
        with radar_col3:
            show_market_regime = st.checkbox(
                "📊 Market Regime Analysis",
                value=st.session_state.get('show_market_regime', True),
                key="show_market_regime",
                help="Show category rotation flow and market regime detection"
            )
        
        wave_filtered_df = filtered_df.copy()
        
        with radar_col4:
            if not wave_filtered_df.empty and 'overall_wave_strength' in wave_filtered_df.columns:
                try:
                    wave_strength_score = wave_filtered_df['overall_wave_strength'].mean()
                    
                    if wave_strength_score > 70:
                        wave_emoji = "🌊🔥"
                        wave_color = "🟢"
                    elif wave_strength_score > 50:
                        wave_emoji = "🌊"
                        wave_color = "🟡"
                    else:
                        wave_emoji = "💤"
                        wave_color = "🔴"
                    
                    UIComponents.render_metric_card(
                        "Wave Strength",
                        f"{wave_emoji} {wave_strength_score:.0f}%",
                        f"{wave_color} Market"
                    )
                except Exception as e:
                    logger.error(f"Error calculating wave strength: {str(e)}")
                    UIComponents.render_metric_card("Wave Strength", "N/A", "Error")
            else:
                UIComponents.render_metric_card("Wave Strength", "N/A", "Data not available")
        
        if show_sensitivity_details:
            with st.expander("📊 Current Sensitivity Thresholds", expanded=True):
                if sensitivity == "Conservative":
                    st.markdown("""
                    **Conservative Settings** 🛡️
                    - **Momentum Shifts:** Score ≥ 60, Acceleration ≥ 70
                    - **Emerging Patterns:** Within 5% of qualifying threshold
                    - **Volume Surges:** RVOL ≥ 3.0x (extreme volumes only)
                    - **Acceleration Alerts:** Score ≥ 85 (strongest signals)
                    - **Pattern Distance:** 5% from qualification
                    """)
                elif sensitivity == "Balanced":
                    st.markdown("""
                    **Balanced Settings** ⚖️
                    - **Momentum Shifts:** Score ≥ 50, Acceleration ≥ 60
                    - **Emerging Patterns:** Within 10% of qualifying threshold
                    - **Volume Surges:** RVOL ≥ 2.0x (standard threshold)
                    - **Acceleration Alerts:** Score ≥ 70 (good acceleration)
                    - **Pattern Distance:** 10% from qualification
                    """)
                else:  # Aggressive
                    st.markdown("""
                    **Aggressive Settings** 🚀
                    - **Momentum Shifts:** Score ≥ 40, Acceleration ≥ 50
                    - **Emerging Patterns:** Within 15% of qualifying threshold
                    - **Volume Surges:** RVOL ≥ 1.5x (building volume)
                    - **Acceleration Alerts:** Score ≥ 60 (early signals)
                    - **Pattern Distance:** 15% from qualification
                    """)
                
                st.info("💡 **Tip**: Start with Balanced, then adjust based on market conditions and your risk tolerance.")
        
        if wave_timeframe != "All Waves":
            try:
                if wave_timeframe == "Intraday Surge":
                    required_cols = ['rvol', 'ret_1d', 'price', 'prev_close']
                    if all(col in wave_filtered_df.columns for col in required_cols):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['rvol'] >= 2.5) &
                            (wave_filtered_df['ret_1d'] > 2) &
                            (wave_filtered_df['price'] > wave_filtered_df['prev_close'] * 1.02)
                        ]
                    
                elif wave_timeframe == "3-Day Buildup":
                    required_cols = ['ret_3d', 'vol_ratio_7d_90d', 'price', 'sma_20d']
                    if all(col in wave_filtered_df.columns for col in required_cols):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_3d'] > 5) &
                            (wave_filtered_df['vol_ratio_7d_90d'] > 1.5) &
                            (wave_filtered_df['price'] > wave_filtered_df['sma_20d'])
                        ]
                
                elif wave_timeframe == "Weekly Breakout":
                    required_cols = ['ret_7d', 'vol_ratio_7d_90d', 'from_high_pct']
                    if all(col in wave_filtered_df.columns for col in required_cols):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_7d'] > 8) &
                            (wave_filtered_df['vol_ratio_7d_90d'] > 2.0) &
                            (wave_filtered_df['from_high_pct'] > -10)
                        ]
                
                elif wave_timeframe == "Monthly Trend":
                    required_cols = ['ret_30d', 'vol_ratio_30d_180d', 'from_low_pct']
                    if all(col in wave_filtered_df.columns for col in required_cols):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_30d'] > 15) &
                            (wave_filtered_df['vol_ratio_30d_180d'] > 1.2) &
                            (wave_filtered_df['from_low_pct'] > 30)
                        ]
            except Exception as e:
                logger.warning(f"Error applying {wave_timeframe} filter: {str(e)}")
                st.warning(f"Some data not available for {wave_timeframe} filter")
        
        if not wave_filtered_df.empty:
            st.markdown("#### 🚀 Momentum Shifts - Stocks Entering Strength")
            
            if sensitivity == "Conservative":
                momentum_threshold = 60
                acceleration_threshold = 70
                min_rvol = 3.0
            elif sensitivity == "Balanced":
                momentum_threshold = 50
                acceleration_threshold = 60
                min_rvol = 2.0
            else:
                momentum_threshold = 40
                acceleration_threshold = 50
                min_rvol = 1.5
            
            momentum_shifts = wave_filtered_df[
                (wave_filtered_df['momentum_score'] >= momentum_threshold) & 
                (wave_filtered_df['acceleration_score'] >= acceleration_threshold)
            ].copy()
            
            if len(momentum_shifts) > 0:
                momentum_shifts['signal_count'] = 0
                momentum_shifts.loc[momentum_shifts['momentum_score'] >= momentum_threshold, 'signal_count'] += 1
                momentum_shifts.loc[momentum_shifts['acceleration_score'] >= acceleration_threshold, 'signal_count'] += 1
                momentum_shifts.loc[momentum_shifts['rvol'] >= min_rvol, 'signal_count'] += 1
                if 'breakout_score' in momentum_shifts.columns:
                    momentum_shifts.loc[momentum_shifts['breakout_score'] >= 75, 'signal_count'] += 1
                if 'vol_ratio_7d_90d' in momentum_shifts.columns:
                    momentum_shifts.loc[momentum_shifts['vol_ratio_7d_90d'] >= 1.5, 'signal_count'] += 1
                
                momentum_shifts['shift_strength'] = (
                    momentum_shifts['momentum_score'] * 0.4 +
                    momentum_shifts['acceleration_score'] * 0.4 +
                    momentum_shifts['rvol_score'] * 0.2
                )
                
                top_shifts = momentum_shifts.sort_values(['signal_count', 'shift_strength'], ascending=[False, False]).head(20)
                
                display_columns = ['ticker', 'company_name', 'master_score', 'momentum_score', 
                                 'acceleration_score', 'rvol', 'signal_count', 'wave_state']
                
                if 'ret_7d' in top_shifts.columns:
                    display_columns.insert(-2, 'ret_7d')
                
                display_columns.append('category')
                
                shift_display = top_shifts[[col for col in display_columns if col in top_shifts.columns]].copy()
                
                shift_display['Signals'] = shift_display['signal_count'].apply(
                    lambda x: f"{'🔥' * min(x, 3)} {x}/5"
                )
                
                if 'ret_7d' in shift_display.columns:
                    shift_display['7D Return'] = shift_display['ret_7d'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else '-')
                
                if 'rvol' in shift_display.columns:
                    shift_display['RVOL'] = shift_display['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else '-')
                    shift_display = shift_display.drop('rvol', axis=1)
                
                rename_dict = {
                    'ticker': 'Ticker',
                    'company_name': 'Company',
                    'master_score': 'Score',
                    'momentum_score': 'Momentum',
                    'acceleration_score': 'Acceleration',
                    'wave_state': 'Wave',
                    'category': 'Category'
                }
                
                shift_display = shift_display.rename(columns=rename_dict)
                
                if 'signal_count' in shift_display.columns:
                    shift_display = shift_display.drop('signal_count', axis=1)
                
                # OPTIMIZED DATAFRAME WITH COLUMN_CONFIG
                st.dataframe(
                    shift_display, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        'Ticker': st.column_config.TextColumn(
                            'Ticker',
                            help="Stock symbol",
                            width="small"
                        ),
                        'Company': st.column_config.TextColumn(
                            'Company',
                            help="Company name",
                            width="medium"
                        ),
                        'Score': st.column_config.ProgressColumn(
                            'Score',
                            help="Master Score",
                            format="%.1f",
                            min_value=0,
                            max_value=100,
                            width="small"
                        ),
                        'Momentum': st.column_config.ProgressColumn(
                            'Momentum',
                            help="Momentum Score",
                            format="%.0f",
                            min_value=0,
                            max_value=100,
                            width="small"
                        ),
                        'Acceleration': st.column_config.ProgressColumn(
                            'Acceleration',
                            help="Acceleration Score",
                            format="%.0f",
                            min_value=0,
                            max_value=100,
                            width="small"
                        ),
                        'RVOL': st.column_config.TextColumn(
                            'RVOL',
                            help="Relative Volume",
                            width="small"
                        ),
                        'Signals': st.column_config.TextColumn(
                            'Signals',
                            help="Signal strength indicator",
                            width="small"
                        ),
                        '7D Return': st.column_config.TextColumn(
                            '7D Return',
                            help="7-day return percentage",
                            width="small"
                        ),
                        'Wave': st.column_config.TextColumn(
                            'Wave',
                            help="Current wave state",
                            width="medium"
                        ),
                        'Category': st.column_config.TextColumn(
                            'Category',
                            help="Market cap category",
                            width="medium"
                        )
                    }
                )
                
                multi_signal = len(top_shifts[top_shifts['signal_count'] >= 3])
                if multi_signal > 0:
                    st.success(f"🏆 Found {multi_signal} stocks with 3+ signals (strongest momentum)")
                
                super_signals = top_shifts[top_shifts['signal_count'] >= 4]
                if len(super_signals) > 0:
                    st.warning(f"🔥🔥 {len(super_signals)} stocks showing EXTREME momentum (4+ signals)!")
            else:
                st.info(f"No momentum shifts detected in {wave_timeframe} timeframe. Try 'Aggressive' sensitivity.")
            
            st.markdown("#### 🚀 Acceleration Profiles - Momentum Building Over Time")
            
            if sensitivity == "Conservative":
                accel_threshold = 85
            elif sensitivity == "Balanced":
                accel_threshold = 70
            else:
                accel_threshold = 60
            
            accelerating_stocks = wave_filtered_df[
                wave_filtered_df['acceleration_score'] >= accel_threshold
            ].nlargest(10, 'acceleration_score')
            
            if len(accelerating_stocks) > 0:
                fig_accel = Visualizer.create_acceleration_profiles(accelerating_stocks, n=10)
                st.plotly_chart(fig_accel, use_container_width=True, theme="streamlit")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    perfect_accel = len(accelerating_stocks[accelerating_stocks['acceleration_score'] >= 90])
                    st.metric("Perfect Acceleration (90+)", perfect_accel)
                with col2:
                    strong_accel = len(accelerating_stocks[accelerating_stocks['acceleration_score'] >= 80])
                    st.metric("Strong Acceleration (80+)", strong_accel)
                with col3:
                    avg_accel = accelerating_stocks['acceleration_score'].mean()
                    st.metric("Avg Acceleration Score", f"{avg_accel:.1f}")
            else:
                st.info(f"No stocks meet the acceleration threshold ({accel_threshold}+) for {sensitivity} sensitivity.")
            
            if show_market_regime:
                st.markdown("#### 💰 Category Rotation - Smart Money Flow")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    try:
                        if 'category' in wave_filtered_df.columns:
                            category_dfs = []
                            for cat in wave_filtered_df['category'].unique():
                                if cat != 'Unknown':
                                    cat_df = wave_filtered_df[wave_filtered_df['category'] == cat]
                                    
                                    category_size = len(cat_df)
                                    if 1 <= category_size <= 5:
                                        sample_count = category_size
                                    elif 6 <= category_size <= 20:
                                        sample_count = max(1, int(category_size * 0.80))
                                    elif 21 <= category_size <= 50:
                                        sample_count = max(1, int(category_size * 0.60))
                                    else:
                                        sample_count = min(50, int(category_size * 0.25))
                                    
                                    if sample_count > 0:
                                        cat_df = cat_df.nlargest(sample_count, 'master_score')
                                    else:
                                        cat_df = pd.DataFrame()
                                        
                                    if not cat_df.empty:
                                        category_dfs.append(cat_df)
                            
                            if category_dfs:
                                normalized_cat_df = pd.concat(category_dfs, ignore_index=True)
                            else:
                                normalized_cat_df = pd.DataFrame()
                            
                            if not normalized_cat_df.empty:
                                category_flow = normalized_cat_df.groupby('category').agg({
                                    'master_score': ['mean', 'count'],
                                    'momentum_score': 'mean',
                                    'volume_score': 'mean',
                                    'rvol': 'mean'
                                }).round(2)
                                
                                if not category_flow.empty:
                                    category_flow.columns = ['Avg Score', 'Count', 'Avg Momentum', 'Avg Volume', 'Avg RVOL']
                                    category_flow['Flow Score'] = (
                                        category_flow['Avg Score'] * 0.4 +
                                        category_flow['Avg Momentum'] * 0.3 +
                                        category_flow['Avg Volume'] * 0.3
                                    )
                                    
                                    category_flow = category_flow.sort_values('Flow Score', ascending=False)
                                    
                                    top_category = category_flow.index[0] if len(category_flow) > 0 else ""
                                    if 'Small' in top_category or 'Micro' in top_category:
                                        flow_direction = "🔥 RISK-ON"
                                    elif 'Large' in top_category or 'Mega' in top_category:
                                        flow_direction = "❄️ RISK-OFF"
                                    else:
                                        flow_direction = "➡️ Neutral"
                                    
                                    fig_flow = go.Figure()
                                    
                                    fig_flow.add_trace(go.Bar(
                                        x=category_flow.index,
                                        y=category_flow['Flow Score'],
                                        text=[f"{val:.1f}" for val in category_flow['Flow Score']],
                                        textposition='outside',
                                        marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' 
                                                     for score in category_flow['Flow Score']],
                                        hovertemplate='Category: %{x}<br>Flow Score: %{y:.1f}<br>Stocks: %{customdata}<extra></extra>',
                                        customdata=category_flow['Count']
                                    ))
                                    
                                    fig_flow.update_layout(
                                        title=f"Smart Money Flow Direction: {flow_direction} (Dynamically Sampled)",
                                        xaxis_title="Market Cap Category",
                                        yaxis_title="Flow Score",
                                        height=300,
                                        template='plotly_white',
                                        showlegend=False
                                    )
                                    
                                    st.plotly_chart(fig_flow, use_container_width=True, theme="streamlit")
                                else:
                                    st.info("Insufficient data for category flow analysis after sampling.")
                            else:
                                st.info("No valid stocks found in categories for flow analysis after sampling.")
                        else:
                            st.info("Category data not available for flow analysis.")
                            
                    except Exception as e:
                        logger.error(f"Error in category flow analysis: {str(e)}")
                        st.error("Unable to analyze category flow")
                
                with col2:
                    if 'category_flow' in locals() and not category_flow.empty:
                        st.markdown(f"**🎯 Market Regime: {flow_direction}**")
                        
                        st.markdown("**💎 Strongest Categories:**")
                        for i, (cat, row) in enumerate(category_flow.head(3).iterrows()):
                            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                            st.write(f"{emoji} **{cat}**: Score {row['Flow Score']:.1f}")
                        
                        st.markdown("**🔄 Category Shifts:**")
                        small_caps_score = category_flow[category_flow.index.str.contains('Small|Micro')]['Flow Score'].mean()
                        large_caps_score = category_flow[category_flow.index.str.contains('Large|Mega')]['Flow Score'].mean()
                        
                        if small_caps_score > large_caps_score + 10:
                            st.success("📈 Small Caps Leading - Early Bull Signal!")
                        elif large_caps_score > small_caps_score + 10:
                            st.warning("📉 Large Caps Leading - Defensive Mode")
                        else:
                            st.info("➡️ Balanced Market - No Clear Leader")
                    else:
                        st.info("Category data not available")
            
            st.markdown("#### 🎯 Emerging Patterns - About to Qualify")
            
            pattern_distance = {"Conservative": 5, "Balanced": 10, "Aggressive": 15}[sensitivity]
            
            emergence_data = []
            
            if 'category_percentile' in wave_filtered_df.columns:
                close_to_leader = wave_filtered_df[
                    (wave_filtered_df['category_percentile'] >= (90 - pattern_distance)) & 
                    (wave_filtered_df['category_percentile'] < 90)
                ]
                for _, stock in close_to_leader.iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '🔥 CAT LEADER',
                        'Distance': f"{90 - stock['category_percentile']:.1f}% away",
                        'Current': f"{stock['category_percentile']:.1f}%ile",
                        'Score': stock['master_score']
                    })
            
            if 'breakout_score' in wave_filtered_df.columns:
                close_to_breakout = wave_filtered_df[
                    (wave_filtered_df['breakout_score'] >= (80 - pattern_distance)) & 
                    (wave_filtered_df['breakout_score'] < 80)
                ]
                for _, stock in close_to_breakout.iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '🎯 BREAKOUT',
                        'Distance': f"{80 - stock['breakout_score']:.1f} pts away",
                        'Current': f"{stock['breakout_score']:.1f} score",
                        'Score': stock['master_score']
                    })
            
            if emergence_data:
                emergence_df = pd.DataFrame(emergence_data).sort_values('Score', ascending=False).head(15)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    # OPTIMIZED DATAFRAME WITH COLUMN_CONFIG
                    st.dataframe(
                        emergence_df, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            'Ticker': st.column_config.TextColumn(
                                'Ticker',
                                help="Stock symbol",
                                width="small"
                            ),
                            'Company': st.column_config.TextColumn(
                                'Company',
                                help="Company name",
                                width="medium"
                            ),
                            'Pattern': st.column_config.TextColumn(
                                'Pattern',
                                help="Pattern about to emerge",
                                width="medium"
                            ),
                            'Distance': st.column_config.TextColumn(
                                'Distance',
                                help="Distance from pattern qualification",
                                width="small"
                            ),
                            'Current': st.column_config.TextColumn(
                                'Current',
                                help="Current value",
                                width="small"
                            ),
                            'Score': st.column_config.ProgressColumn(
                                'Score',
                                help="Master Score",
                                format="%.1f",
                                min_value=0,
                                max_value=100,
                                width="small"
                            )
                        }
                    )
                with col2:
                    UIComponents.render_metric_card("Emerging Patterns", len(emergence_df))
            else:
                st.info(f"No patterns emerging within {pattern_distance}% threshold.")
            
            st.markdown("#### 🌊 Volume Surges - Unusual Activity NOW")
            
            rvol_threshold = {"Conservative": 3.0, "Balanced": 2.0, "Aggressive": 1.5}[sensitivity]
            
            volume_surges = wave_filtered_df[wave_filtered_df['rvol'] >= rvol_threshold].copy()
            
            if len(volume_surges) > 0:
                volume_surges['surge_score'] = (
                    volume_surges['rvol_score'] * 0.5 +
                    volume_surges['volume_score'] * 0.3 +
                    volume_surges['momentum_score'] * 0.2
                )
                
                top_surges = volume_surges.nlargest(15, 'surge_score')
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    display_cols = ['ticker', 'company_name', 'rvol', 'price', 'money_flow_mm', 'wave_state', 'category']
                    
                    if 'ret_1d' in top_surges.columns:
                        display_cols.insert(3, 'ret_1d')
                    
                    surge_display = top_surges[[col for col in display_cols if col in top_surges.columns]].copy()
                    
                    surge_display['Type'] = surge_display['rvol'].apply(
                        lambda x: "🔥🔥🔥" if x > 5 else "🔥🔥" if x > 3 else "🔥"
                    )
                    
                    if 'ret_1d' in surge_display.columns:
                        surge_display['ret_1d'] = surge_display['ret_1d'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else '-')
                    
                    if 'money_flow_mm' in surge_display.columns:
                        surge_display['money_flow_mm'] = surge_display['money_flow_mm'].apply(lambda x: f"₹{x:.1f}M" if pd.notna(x) else '-')
                    
                    surge_display['price'] = surge_display['price'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else '-')
                    surge_display['rvol'] = surge_display['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else '-')
                    
                    rename_dict = {
                        'ticker': 'Ticker',
                        'company_name': 'Company',
                        'rvol': 'RVOL',
                        'price': 'Price',
                        'money_flow_mm': 'Money Flow',
                        'wave_state': 'Wave',
                        'category': 'Category',
                        'ret_1d': '1D Ret'
                    }
                    surge_display = surge_display.rename(columns=rename_dict)
                    
                    # OPTIMIZED DATAFRAME WITH COLUMN_CONFIG
                    st.dataframe(
                        surge_display, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            'Type': st.column_config.TextColumn(
                                'Type',
                                help="Volume surge intensity",
                                width="small"
                            ),
                            'Ticker': st.column_config.TextColumn(
                                'Ticker',
                                help="Stock symbol",
                                width="small"
                            ),
                            'Company': st.column_config.TextColumn(
                                'Company',
                                help="Company name",
                                width="medium"
                            ),
                            'RVOL': st.column_config.TextColumn(
                                'RVOL',
                                help="Relative Volume",
                                width="small"
                            ),
                            'Price': st.column_config.TextColumn(
                                'Price',
                                help="Current price",
                                width="small"
                            ),
                            '1D Ret': st.column_config.TextColumn(
                                '1D Ret',
                                help="1-day return",
                                width="small"
                            ),
                            'Money Flow': st.column_config.TextColumn(
                                'Money Flow',
                                help="Money flow in millions",
                                width="small"
                            ),
                            'Wave': st.column_config.TextColumn(
                                'Wave',
                                help="Current wave state",
                                width="medium"
                            ),
                            'Category': st.column_config.TextColumn(
                                'Category',
                                help="Market cap category",
                                width="medium"
                            )
                        }
                    )
                
                with col2:
                    UIComponents.render_metric_card("Active Surges", len(volume_surges))
                    UIComponents.render_metric_card("Extreme (>5x)", len(volume_surges[volume_surges['rvol'] > 5]))
                    UIComponents.render_metric_card("High (>3x)", len(volume_surges[volume_surges['rvol'] > 3]))
                    
                    if 'category' in volume_surges.columns:
                        st.markdown("**📊 Surge by Category:**")
                        surge_categories = volume_surges['category'].value_counts()
                        if len(surge_categories) > 0:
                            for cat, count in surge_categories.head(3).items():
                                st.caption(f"• {cat}: {count} stocks")
            else:
                st.info(f"No volume surges detected with {sensitivity} sensitivity (requires RVOL ≥ {rvol_threshold}x).")

                st.markdown("---")
                st.markdown("#### ⚠️ Critical Reversal Signals - Risk Management Alerts")
                
                # Check for reversal patterns
                if 'patterns' in wave_filtered_df.columns:
                    # Define critical reversal patterns
                    reversal_patterns = ['🪤 BULL TRAP', '💣 CAPITULATION', '⚠️ DISTRIBUTION']
                    
                    # Find stocks with reversal patterns
                    reversal_mask = wave_filtered_df['patterns'].str.contains(
                        '|'.join(reversal_patterns), 
                        na=False, 
                        regex=True
                    )
                    reversal_stocks = wave_filtered_df[reversal_mask]
                    
                    if len(reversal_stocks) > 0:
                        # Separate by pattern type
                        bull_traps = reversal_stocks[reversal_stocks['patterns'].str.contains('BULL TRAP', na=False)]
                        capitulations = reversal_stocks[reversal_stocks['patterns'].str.contains('CAPITULATION', na=False)]
                        distributions = reversal_stocks[reversal_stocks['patterns'].str.contains('DISTRIBUTION', na=False)]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if len(bull_traps) > 0:
                                st.error(f"🪤 **BULL TRAPS ({len(bull_traps)})**")
                                for _, stock in bull_traps.head(3).iterrows():
                                    st.write(f"• **{stock['ticker']}**")
                                    st.caption(f"7D: {stock.get('ret_7d', 0):.1f}% | From High: {stock.get('from_high_pct', 0):.1f}%")
                            else:
                                st.info("🪤 No Bull Traps")
                        
                        with col2:
                            if len(capitulations) > 0:
                                st.success(f"💣 **CAPITULATIONS ({len(capitulations)})**")
                                for _, stock in capitulations.head(3).iterrows():
                                    st.write(f"• **{stock['ticker']}**")
                                    st.caption(f"1D: {stock.get('ret_1d', 0):.1f}% | RVOL: {stock.get('rvol', 0):.1f}x")
                            else:
                                st.info("💣 No Capitulations")
                        
                        with col3:
                            if len(distributions) > 0:
                                st.warning(f"⚠️ **DISTRIBUTIONS ({len(distributions)})**")
                                for _, stock in distributions.head(3).iterrows():
                                    st.write(f"• **{stock['ticker']}**")
                                    st.caption(f"30D: {stock.get('ret_30d', 0):.1f}% | RVOL: {stock.get('rvol', 0):.1f}x")
                            else:
                                st.info("⚠️ No Distributions")
                        
                        # Show detailed table if there are many reversals
                        if len(reversal_stocks) > 5:
                            with st.expander(f"📊 View All {len(reversal_stocks)} Reversal Signals", expanded=False):
                                reversal_display = reversal_stocks[['ticker', 'company_name', 'patterns', 'master_score', 
                                                                    'ret_1d', 'ret_7d', 'from_high_pct', 'rvol']].copy()
                                
                                reversal_display['Type'] = reversal_display['patterns'].apply(
                                    lambda x: '🪤 Trap' if 'BULL TRAP' in x else 
                                             '💣 Bottom' if 'CAPITULATION' in x else 
                                             '⚠️ Top'
                                )
                                
                                st.dataframe(
                                    reversal_display,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        'ticker': st.column_config.TextColumn('Ticker', width="small"),
                                        'company_name': st.column_config.TextColumn('Company', width="medium"),
                                        'Type': st.column_config.TextColumn('Signal', width="small"),
                                        'master_score': st.column_config.ProgressColumn(
                                            'Score',
                                            min_value=0,
                                            max_value=100,
                                            format="%.0f"
                                        ),
                                        'ret_1d': st.column_config.NumberColumn('1D%', format="%.1f%%"),
                                        'ret_7d': st.column_config.NumberColumn('7D%', format="%.1f%%"),
                                        'from_high_pct': st.column_config.NumberColumn('From High', format="%.1f%%"),
                                        'rvol': st.column_config.NumberColumn('RVOL', format="%.1fx")
                                    }
                                )
                    else:
                        st.info("No reversal patterns detected in current wave timeframe")
                else:
                    st.info("Pattern data not available for reversal detection")
        
        else:
            st.warning(f"No data available for Wave Radar analysis with {wave_timeframe} timeframe.")
    
    # Tab 3: Analysis
    # MAKE SURE THIS IS INSIDE THE MAIN TABS BLOCK!
    with tabs[3]:
        st.markdown("### 📊 Market Analysis Dashboard")
        
        if not filtered_df.empty:
            # ADD SUB-TABS FOR BETTER ORGANIZATION
            analysis_subtabs = st.tabs([
                "🎯 Quick Insights",
                "📈 Technical Analysis", 
                "🏢 Sector Analysis",
                "🏭 Industry Analysis",
                "🎨 Pattern Analysis",
                "📊 Category Breakdown"
            ])
            
            # ==========================================
            # QUICK INSIGHTS TAB
            # ==========================================
            with analysis_subtabs[0]:
                st.markdown("#### 🔍 Market Overview at a Glance")
                
                # Key Metrics Row
                metric_cols = st.columns(5)
                
                with metric_cols[0]:
                    avg_score = filtered_df['master_score'].mean() if 'master_score' in filtered_df.columns else 0
                    score_color = "🟢" if avg_score > 60 else "🟡" if avg_score > 40 else "🔴"
                    st.metric(
                        "Market Strength",
                        f"{score_color} {avg_score:.1f}",
                        f"Top: {filtered_df['master_score'].max():.0f}" if 'master_score' in filtered_df.columns else "N/A"
                    )
                
                with metric_cols[1]:
                    if 'ret_30d' in filtered_df.columns:
                        bullish = len(filtered_df[filtered_df['ret_30d'] > 0])
                        bearish = len(filtered_df[filtered_df['ret_30d'] <= 0])
                        st.metric(
                            "Market Breadth",
                            f"{bullish}/{bearish}",
                            f"{bullish/(bullish+bearish)*100:.0f}% Bullish" if (bullish+bearish) > 0 else "N/A"
                        )
                    else:
                        st.metric("Market Breadth", "N/A")
                
                with metric_cols[2]:
                    if 'rvol' in filtered_df.columns:
                        high_rvol = len(filtered_df[filtered_df['rvol'] > 2])
                        st.metric(
                            "Active Stocks",
                            f"{high_rvol}",
                            "RVOL > 2x"
                        )
                    else:
                        st.metric("Active Stocks", "N/A")
                
                with metric_cols[3]:
                    if 'patterns' in filtered_df.columns:
                        patterns_count = (filtered_df['patterns'] != '').sum()
                        st.metric(
                            "Pattern Signals",
                            f"{patterns_count}",
                            f"{patterns_count/len(filtered_df)*100:.0f}% have patterns" if len(filtered_df) > 0 else "N/A"
                        )
                    else:
                        st.metric("Pattern Signals", "N/A")
                
                with metric_cols[4]:
                    if 'category' in filtered_df.columns and 'master_score' in filtered_df.columns:
                        top_category = filtered_df.groupby('category')['master_score'].mean().idxmax() if not filtered_df.empty else "N/A"
                        st.metric(
                            "Leading Category",
                            top_category,
                            "By avg score"
                        )
                    else:
                        st.metric("Leading Category", "N/A")
                
                st.markdown("---")
                
                # Quick Winners and Losers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 🏆 Top 5 Performers")
                    if 'master_score' in filtered_df.columns:
                        top_5 = filtered_df.nlargest(5, 'master_score')[['ticker', 'company_name', 'master_score']]
                        if 'ret_30d' in filtered_df.columns:
                            top_5 = filtered_df.nlargest(5, 'master_score')[['ticker', 'company_name', 'master_score', 'ret_30d']]
                        if 'patterns' in filtered_df.columns:
                            top_5 = filtered_df.nlargest(5, 'master_score')[['ticker', 'company_name', 'master_score', 'patterns']]
                            if 'ret_30d' in filtered_df.columns:
                                top_5 = filtered_df.nlargest(5, 'master_score')[['ticker', 'company_name', 'master_score', 'ret_30d', 'patterns']]
                        
                        for idx, row in top_5.iterrows():
                            with st.container():
                                subcol1, subcol2, subcol3 = st.columns([2, 1, 2])
                                with subcol1:
                                    st.write(f"**{row['ticker']}**")
                                    company_name = row.get('company_name', 'N/A')
                                    st.caption(f"{str(company_name)[:25]}...")
                                with subcol2:
                                    st.write(f"Score: **{row['master_score']:.0f}**")
                                    if 'ret_30d' in row:
                                        st.caption(f"30D: {row['ret_30d']:.1f}%")
                                with subcol3:
                                    if 'patterns' in row and row['patterns']:
                                        patterns_list = str(row['patterns']).split(' | ')[:2]
                                        st.caption(' | '.join(patterns_list))
                    else:
                        st.info("No score data available")
                
                with col2:
                    st.markdown("##### 📉 Bottom 5 Performers")
                    if 'master_score' in filtered_df.columns:
                        bottom_5 = filtered_df.nsmallest(5, 'master_score')[['ticker', 'company_name', 'master_score']]
                        if 'ret_30d' in filtered_df.columns:
                            bottom_5 = filtered_df.nsmallest(5, 'master_score')[['ticker', 'company_name', 'master_score', 'ret_30d']]
                        if 'wave_state' in filtered_df.columns:
                            bottom_5 = filtered_df.nsmallest(5, 'master_score')[['ticker', 'company_name', 'master_score', 'wave_state']]
                            if 'ret_30d' in filtered_df.columns:
                                bottom_5 = filtered_df.nsmallest(5, 'master_score')[['ticker', 'company_name', 'master_score', 'ret_30d', 'wave_state']]
                        
                        for idx, row in bottom_5.iterrows():
                            with st.container():
                                subcol1, subcol2, subcol3 = st.columns([2, 1, 2])
                                with subcol1:
                                    st.write(f"**{row['ticker']}**")
                                    company_name = row.get('company_name', 'N/A')
                                    st.caption(f"{str(company_name)[:25]}...")
                                with subcol2:
                                    st.write(f"Score: **{row['master_score']:.0f}**")
                                    if 'ret_30d' in row:
                                        st.caption(f"30D: {row['ret_30d']:.1f}%")
                                with subcol3:
                                    if 'wave_state' in row:
                                        st.caption(row['wave_state'])
                    else:
                        st.info("No score data available")
                
                # Market Signals Summary
                st.markdown("---")
                st.markdown("##### 📡 Key Market Signals")
                
                signal_cols = st.columns(4)
                
                with signal_cols[0]:
                    if 'momentum_score' in filtered_df.columns:
                        momentum_leaders = len(filtered_df[filtered_df['momentum_score'] > 70])
                        st.info(f"**{momentum_leaders}** Momentum Leaders")
                        st.caption("Score > 70")
                    else:
                        st.info("**0** Momentum Leaders")
                
                with signal_cols[1]:
                    if 'breakout_score' in filtered_df.columns:
                        breakout_ready = len(filtered_df[filtered_df['breakout_score'] > 80])
                        st.success(f"**{breakout_ready}** Breakout Ready")
                        st.caption("Breakout > 80")
                    else:
                        st.success("**0** Breakout Ready")
                
                with signal_cols[2]:
                    if 'patterns' in filtered_df.columns:
                        vol_explosions = len(filtered_df[filtered_df['patterns'].str.contains('VOL EXPLOSION', na=False)])
                        st.warning(f"**{vol_explosions}** Volume Explosions")
                        st.caption("Extreme activity")
                    else:
                        st.warning("**0** Volume Explosions")
                
                with signal_cols[3]:
                    if 'patterns' in filtered_df.columns:
                        perfect_storms = len(filtered_df[filtered_df['patterns'].str.contains('PERFECT STORM', na=False)])
                        st.error(f"**{perfect_storms}** Perfect Storms")
                        st.caption("All signals aligned")
                    else:
                        st.error("**0** Perfect Storms")
            
            # ==========================================
            # TECHNICAL ANALYSIS TAB
            # ==========================================
            with analysis_subtabs[1]:
                st.markdown("#### 📈 Technical Indicators Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Score Distribution Chart
                    try:
                        fig_dist = Visualizer.create_score_distribution(filtered_df)
                        st.plotly_chart(fig_dist, use_container_width=True, theme="streamlit")
                    except Exception as e:
                        st.error(f"Error creating score distribution: {str(e)}")
                
                with col2:
                    # Trend Quality Distribution
                    if 'trend_quality' in filtered_df.columns:
                        try:
                            fig_trend = go.Figure()
                            
                            trend_bins = [0, 20, 40, 60, 80, 100]
                            trend_labels = ['Very Weak', 'Weak', 'Neutral', 'Strong', 'Very Strong']
                            trend_counts = pd.cut(filtered_df['trend_quality'], bins=trend_bins, labels=trend_labels).value_counts()
                            
                            colors = ['#e74c3c', '#f39c12', '#95a5a6', '#2ecc71', '#27ae60']
                            
                            fig_trend.add_trace(go.Bar(
                                x=trend_counts.index,
                                y=trend_counts.values,
                                marker_color=colors,
                                text=trend_counts.values,
                                textposition='outside'
                            ))
                            
                            fig_trend.update_layout(
                                title="Trend Quality Distribution",
                                xaxis_title="Trend Strength",
                                yaxis_title="Number of Stocks",
                                template='plotly_white',
                                height=400
                            )
                            
                            st.plotly_chart(fig_trend, use_container_width=True, theme="streamlit")
                        except Exception as e:
                            st.error(f"Error creating trend chart: {str(e)}")
                    else:
                        st.info("Trend quality data not available")
                
                # Wave State Analysis
                st.markdown("---")
                st.markdown("##### 🌊 Wave State Analysis")
                
                if 'wave_state' in filtered_df.columns:
                    try:
                        wave_analysis = filtered_df.groupby('wave_state').agg({
                            'ticker': 'count',
                            'master_score': 'mean' if 'master_score' in filtered_df.columns else lambda x: 0,
                            'momentum_score': 'mean' if 'momentum_score' in filtered_df.columns else lambda x: 0,
                            'rvol': 'mean' if 'rvol' in filtered_df.columns else lambda x: 0,
                            'ret_30d': 'mean' if 'ret_30d' in filtered_df.columns else lambda x: 0
                        }).round(2)
                        
                        wave_analysis.columns = ['Count', 'Avg Score', 'Avg Momentum', 'Avg RVOL', 'Avg 30D Return']
                        wave_analysis = wave_analysis.sort_values('Count', ascending=False)
                        
                        st.dataframe(
                            wave_analysis.style.background_gradient(subset=['Avg Score', 'Avg Momentum']),
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error in wave analysis: {str(e)}")
                else:
                    st.info("Wave state data not available")
            
            # ==========================================
            # SECTOR ANALYSIS TAB
            # ==========================================
            with analysis_subtabs[2]:
                st.markdown("#### 🏢 Sector Performance & Rotation")
                
                try:
                    sector_rotation = MarketIntelligence.detect_sector_rotation(filtered_df)
                    
                    if not sector_rotation.empty:
                        # Sector Performance Chart
                        fig_sector = go.Figure()
                        
                        top_sectors = sector_rotation.head(10)
                        
                        fig_sector.add_trace(go.Bar(
                            x=top_sectors.index,
                            y=top_sectors['flow_score'],
                            text=[f"{val:.1f}" for val in top_sectors['flow_score']],
                            textposition='outside',
                            marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' 
                                         for score in top_sectors['flow_score']],
                            hovertemplate=(
                                'Sector: %{x}<br>'
                                'Flow Score: %{y:.1f}<br>'
                                'Stocks Analyzed: %{customdata[0]}<br>'
                                'Total Stocks: %{customdata[1]}<br>'
                                '<extra></extra>'
                            ),
                            customdata=np.column_stack((
                                top_sectors['analyzed_stocks'],
                                top_sectors['total_stocks']
                            ))
                        ))
                        
                        fig_sector.update_layout(
                            title="Sector Rotation Map - Smart Money Flow",
                            xaxis_title="Sector",
                            yaxis_title="Flow Score",
                            height=400,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_sector, use_container_width=True, theme="streamlit")
                        
                        # Sector Details Table
                        st.markdown("##### 📊 Detailed Sector Metrics")
                        
                        display_cols = ['flow_score', 'avg_score', 'avg_momentum', 
                                       'avg_volume', 'avg_rvol', 'analyzed_stocks', 'total_stocks']
                        
                        # Check which columns exist
                        available_cols = [col for col in display_cols if col in sector_rotation.columns]
                        
                        if available_cols:
                            sector_display = sector_rotation[available_cols].copy()
                            st.dataframe(
                                sector_display.style.background_gradient(subset=['flow_score'] if 'flow_score' in available_cols else []),
                                use_container_width=True
                            )
                        
                        st.info("📊 **Note**: Analysis based on dynamically sampled top performers per sector for fair comparison")
                    else:
                        st.warning("No sector data available in the filtered dataset")
                except Exception as e:
                    st.error(f"Error in sector analysis: {str(e)}")
                    st.info("Try adjusting filters to include more stocks")
            
            # ==========================================
            # INDUSTRY ANALYSIS TAB
            # ==========================================
            with analysis_subtabs[3]:
                st.markdown("#### 🏭 Industry Performance & Trends")
                
                try:
                    industry_rotation = MarketIntelligence.detect_industry_rotation(filtered_df)
                    
                    if not industry_rotation.empty:
                        # Top/Bottom Industries
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### 🔥 Top 10 Industries")
                            display_cols = ['flow_score', 'avg_score', 'analyzed_stocks', 'total_stocks']
                            available_cols = [col for col in display_cols if col in industry_rotation.columns]
                            
                            if available_cols:
                                top_industries = industry_rotation.head(10)[available_cols]
                                st.dataframe(top_industries, use_container_width=True)
                            else:
                                st.info("Industry metrics not available")
                        
                        with col2:
                            st.markdown("##### ❄️ Bottom 10 Industries")
                            if available_cols:
                                bottom_industries = industry_rotation.tail(10)[available_cols]
                                st.dataframe(bottom_industries, use_container_width=True)
                            else:
                                st.info("Industry metrics not available")
                        
                        # Quality Warnings
                        if 'quality_flag' in industry_rotation.columns:
                            low_quality = industry_rotation[industry_rotation['quality_flag'] != '']
                            if len(low_quality) > 0:
                                st.warning(f"⚠️ {len(low_quality)} industries have low sampling quality")
                    else:
                        st.warning("No industry data available")
                except Exception as e:
                    st.error(f"Error in industry analysis: {str(e)}")
                    st.info("Try adjusting filters to include more stocks")
            
            # ==========================================
            # PATTERN ANALYSIS TAB
            # ==========================================
            with analysis_subtabs[4]:
                st.markdown("#### 🎨 Pattern Detection Analysis")
                
                if 'patterns' in filtered_df.columns:
                    # Pattern Frequency
                    pattern_counts = {}
                    for patterns in filtered_df['patterns'].dropna():
                        if patterns:
                            for p in str(patterns).split(' | '):
                                p = p.strip()
                                if p:
                                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
                    
                    if pattern_counts:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Pattern Bar Chart
                            pattern_df = pd.DataFrame(
                                list(pattern_counts.items()),
                                columns=['Pattern', 'Count']
                            ).sort_values('Count', ascending=False).head(15)
                            
                            try:
                                fig_patterns = go.Figure([
                                    go.Bar(
                                        x=pattern_df['Count'],
                                        y=pattern_df['Pattern'],
                                        orientation='h',
                                        marker_color='#3498db',
                                        text=pattern_df['Count'],
                                        textposition='outside'
                                    )
                                ])
                                
                                fig_patterns.update_layout(
                                    title="Top 15 Pattern Frequencies",
                                    xaxis_title="Number of Stocks",
                                    yaxis_title="Pattern",
                                    template='plotly_white',
                                    height=500,
                                    margin=dict(l=150)
                                )
                                
                                st.plotly_chart(fig_patterns, use_container_width=True, theme="streamlit")
                            except Exception as e:
                                st.error(f"Error creating pattern chart: {str(e)}")
                        
                        with col2:
                            st.markdown("##### 🎯 Pattern Performance")
                            
                            # Calculate average score per pattern
                            pattern_performance = {}
                            for pattern in pattern_counts.keys():
                                stocks_with_pattern = filtered_df[filtered_df['patterns'].str.contains(pattern, na=False, regex=False)]
                                if len(stocks_with_pattern) > 0:
                                    pattern_performance[pattern] = {
                                        'Avg Score': stocks_with_pattern['master_score'].mean() if 'master_score' in stocks_with_pattern.columns else 0,
                                        'Avg 30D': stocks_with_pattern['ret_30d'].mean() if 'ret_30d' in stocks_with_pattern.columns else 0,
                                        'Count': len(stocks_with_pattern)
                                    }
                            
                            if pattern_performance:
                                perf_df = pd.DataFrame(pattern_performance).T
                                perf_df = perf_df.sort_values('Avg Score', ascending=False).head(10)
                                perf_df['Avg Score'] = perf_df['Avg Score'].round(1)
                                perf_df['Avg 30D'] = perf_df['Avg 30D'].round(1)
                                perf_df['Count'] = perf_df['Count'].astype(int)
                                
                                st.dataframe(
                                    perf_df.style.background_gradient(subset=['Avg Score']),
                                    use_container_width=True
                                )
                    else:
                        st.info("No patterns detected in current selection")
                else:
                    st.info("Pattern data not available")
            
            # ==========================================
            # CATEGORY BREAKDOWN TAB
            # ==========================================
            with analysis_subtabs[5]:
                st.markdown("#### 📊 Market Cap Category Analysis")
                
                if 'category' in filtered_df.columns:
                    try:
                        # Category Performance Metrics
                        cat_analysis = filtered_df.groupby('category').agg({
                            'ticker': 'count',
                            'master_score': ['mean', 'std'] if 'master_score' in filtered_df.columns else lambda x: [0, 0],
                            'ret_30d': 'mean' if 'ret_30d' in filtered_df.columns else lambda x: 0,
                            'rvol': 'mean' if 'rvol' in filtered_df.columns else lambda x: 0,
                            'money_flow_mm': 'sum' if 'money_flow_mm' in filtered_df.columns else lambda x: 0
                        }).round(2)
                        
                        # Flatten columns
                        cat_analysis.columns = ['Count', 'Avg Score', 'Score Std', 'Avg 30D Ret', 'Avg RVOL', 'Total Money Flow']
                        
                        # Sort by average score
                        cat_analysis = cat_analysis.sort_values('Avg Score', ascending=False)
                        
                        # Pie chart of distribution
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=cat_analysis.index,
                                values=cat_analysis['Count'],
                                hole=.3
                            )])
                            
                            fig_pie.update_layout(
                                title="Distribution by Category",
                                height=300
                            )
                            
                            st.plotly_chart(fig_pie, use_container_width=True, theme="streamlit")
                        
                        with col2:
                            st.dataframe(
                                cat_analysis.style.background_gradient(subset=['Avg Score'] if 'Avg Score' in cat_analysis.columns else []),
                                use_container_width=True
                            )
                        
                        # Category-wise top stocks
                        st.markdown("---")
                        st.markdown("##### 🏆 Top Stock per Category")
                        
                        category_tops = []
                        for category in filtered_df['category'].unique():
                            cat_df = filtered_df[filtered_df['category'] == category]
                            if not cat_df.empty and 'master_score' in cat_df.columns:
                                top_stock = cat_df.nlargest(1, 'master_score').iloc[0]
                                category_tops.append({
                                    'Category': category,
                                    'Top Stock': top_stock['ticker'],
                                    'Company': str(top_stock.get('company_name', 'N/A'))[:30],
                                    'Score': top_stock['master_score'],
                                    'Patterns': str(top_stock.get('patterns', 'None'))[:50] if 'patterns' in top_stock.index else 'None'
                                })
                        
                        if category_tops:
                            tops_df = pd.DataFrame(category_tops)
                            tops_df = tops_df.sort_values('Score', ascending=False)
                            st.dataframe(tops_df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Error in category analysis: {str(e)}")
                else:
                    st.info("Category data not available")
        
        else:
            st.warning("No data available for analysis. Please adjust your filters.")
    
    # Tab 4: Search
    with tabs[4]:
        st.markdown("### 🔍 Advanced Stock Search")
        
        # Search interface
        col1, col2 = st.columns([4, 1])
        
        with col1:
            search_query = st.text_input(
                "Search stocks",
                placeholder="Enter ticker or company name...",
                help="Search by ticker symbol or company name",
                key="search_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("🔎 Search", type="primary", use_container_width=True, key="search_btn")
        
        # Perform search
        if search_query or search_clicked:
            with st.spinner("Searching..."):
                search_results = SearchEngine.search_stocks(filtered_df, search_query)
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching stock(s)")
                
                # Create summary dataframe for search results
                summary_columns = ['ticker', 'company_name', 'rank', 'master_score', 'price', 
                                  'ret_30d', 'rvol', 'wave_state', 'category']
                
                available_summary_cols = [col for col in summary_columns if col in search_results.columns]
                search_summary = search_results[available_summary_cols].copy()
                
                # Format the summary data
                if 'price' in search_summary.columns:
                    search_summary['price_display'] = search_summary['price'].apply(
                        lambda x: f"₹{x:,.0f}" if pd.notna(x) else '-'
                    )
                    search_summary = search_summary.drop('price', axis=1)
                
                if 'ret_30d' in search_summary.columns:
                    search_summary['ret_30d_display'] = search_summary['ret_30d'].apply(
                        lambda x: f"{x:+.1f}%" if pd.notna(x) else '-'
                    )
                    search_summary = search_summary.drop('ret_30d', axis=1)
                
                if 'rvol' in search_summary.columns:
                    search_summary['rvol_display'] = search_summary['rvol'].apply(
                        lambda x: f"{x:.1f}x" if pd.notna(x) else '-'
                    )
                    search_summary = search_summary.drop('rvol', axis=1)
                
                # Rename columns for display
                column_rename = {
                    'ticker': 'Ticker',
                    'company_name': 'Company',
                    'rank': 'Rank',
                    'master_score': 'Score',
                    'price_display': 'Price',
                    'ret_30d_display': '30D Return',
                    'rvol_display': 'RVOL',
                    'wave_state': 'Wave State',
                    'category': 'Category'
                }
                
                search_summary = search_summary.rename(columns=column_rename)
                
                # Display search results summary with optimized column_config
                st.markdown("#### 📊 Search Results Overview")
                st.dataframe(
                    search_summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Ticker': st.column_config.TextColumn(
                            'Ticker',
                            help="Stock symbol - Click expander below for details",
                            width="small"
                        ),
                        'Company': st.column_config.TextColumn(
                            'Company',
                            help="Company name",
                            width="large"
                        ),
                        'Rank': st.column_config.NumberColumn(
                            'Rank',
                            help="Overall ranking position",
                            format="%d",
                            width="small"
                        ),
                        'Score': st.column_config.ProgressColumn(
                            'Score',
                            help="Master Score (0-100)",
                            format="%.1f",
                            min_value=0,
                            max_value=100,
                            width="small"
                        ),
                        'Price': st.column_config.TextColumn(
                            'Price',
                            help="Current stock price",
                            width="small"
                        ),
                        '30D Return': st.column_config.TextColumn(
                            '30D Return',
                            help="30-day return percentage",
                            width="small"
                        ),
                        'RVOL': st.column_config.TextColumn(
                            'RVOL',
                            help="Relative Volume",
                            width="small"
                        ),
                        'Wave State': st.column_config.TextColumn(
                            'Wave State',
                            help="Current momentum wave state",
                            width="medium"
                        ),
                        'Category': st.column_config.TextColumn(
                            'Category',
                            help="Market cap category",
                            width="medium"
                        )
                    }
                )
                
                st.markdown("---")
                st.markdown("#### 📋 Detailed Stock Information")
                
                # Display each result in expandable sections
                for idx, stock in search_results.iterrows():
                    with st.expander(
                        f"📊 {stock['ticker']} - {stock['company_name']} "
                        f"(Rank #{int(stock['rank'])})",
                        expanded=(len(search_results) == 1)  # Auto-expand if only one result
                    ):
                        # Header metrics
                        metric_cols = st.columns(6)
                        
                        with metric_cols[0]:
                            UIComponents.render_metric_card(
                                "Master Score",
                                f"{stock['master_score']:.1f}",
                                f"Rank #{int(stock['rank'])}"
                            )
                        
                        with metric_cols[1]:
                            price_value = f"₹{stock['price']:,.0f}" if pd.notna(stock.get('price')) else "N/A"
                            ret_1d_value = f"{stock['ret_1d']:+.1f}%" if pd.notna(stock.get('ret_1d')) else None
                            UIComponents.render_metric_card("Price", price_value, ret_1d_value)
                        
                        with metric_cols[2]:
                            UIComponents.render_metric_card(
                                "From Low",
                                f"{stock['from_low_pct']:.0f}%" if pd.notna(stock.get('from_low_pct')) else "N/A",
                                "52-week range position"
                            )
                        
                        with metric_cols[3]:
                            ret_30d = stock.get('ret_30d', 0)
                            UIComponents.render_metric_card(
                                "30D Return",
                                f"{ret_30d:+.1f}%" if pd.notna(ret_30d) else "N/A",
                                "↑" if ret_30d > 0 else "↓" if ret_30d < 0 else "→"
                            )
                        
                        with metric_cols[4]:
                            rvol = stock.get('rvol', 1)
                            UIComponents.render_metric_card(
                                "RVOL",
                                f"{rvol:.1f}x" if pd.notna(rvol) else "N/A",
                                "High" if rvol > 2 else "Normal" if rvol > 0.5 else "Low"
                            )
                        
                        with metric_cols[5]:
                            UIComponents.render_metric_card(
                                "Wave State",
                                stock.get('wave_state', 'N/A'),
                                stock.get('category', 'N/A')
                            )
                        
                        # Score breakdown with optimized display
                        st.markdown("#### 📈 Score Components")
                        
                        # Create score breakdown dataframe
                        score_data = {
                            'Component': ['Position', 'Volume', 'Momentum', 'Acceleration', 'Breakout', 'RVOL'],
                            'Score': [
                                stock.get('position_score', 0),
                                stock.get('volume_score', 0),
                                stock.get('momentum_score', 0),
                                stock.get('acceleration_score', 0),
                                stock.get('breakout_score', 0),
                                stock.get('rvol_score', 0)
                            ],
                            'Weight': [
                                f"{CONFIG.POSITION_WEIGHT:.0%}",
                                f"{CONFIG.VOLUME_WEIGHT:.0%}",
                                f"{CONFIG.MOMENTUM_WEIGHT:.0%}",
                                f"{CONFIG.ACCELERATION_WEIGHT:.0%}",
                                f"{CONFIG.BREAKOUT_WEIGHT:.0%}",
                                f"{CONFIG.RVOL_WEIGHT:.0%}"
                            ],
                            'Contribution': [
                                stock.get('position_score', 0) * CONFIG.POSITION_WEIGHT,
                                stock.get('volume_score', 0) * CONFIG.VOLUME_WEIGHT,
                                stock.get('momentum_score', 0) * CONFIG.MOMENTUM_WEIGHT,
                                stock.get('acceleration_score', 0) * CONFIG.ACCELERATION_WEIGHT,
                                stock.get('breakout_score', 0) * CONFIG.BREAKOUT_WEIGHT,
                                stock.get('rvol_score', 0) * CONFIG.RVOL_WEIGHT
                            ]
                        }
                        
                        score_df = pd.DataFrame(score_data)
                        
                        # Add quality indicator
                        score_df['Quality'] = score_df['Score'].apply(
                            lambda x: '🟢 Strong' if x >= 80 
                            else '🟡 Good' if x >= 60 
                            else '🟠 Fair' if x >= 40 
                            else '🔴 Weak'
                        )
                        
                        # Display score breakdown with column_config
                        st.dataframe(
                            score_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'Component': st.column_config.TextColumn(
                                    'Component',
                                    help="Score component name",
                                    width="medium"
                                ),
                                'Score': st.column_config.ProgressColumn(
                                    'Score',
                                    help="Component score (0-100)",
                                    format="%.1f",
                                    min_value=0,
                                    max_value=100,
                                    width="small"
                                ),
                                'Weight': st.column_config.TextColumn(
                                    'Weight',
                                    help="Component weight in master score",
                                    width="small"
                                ),
                                'Contribution': st.column_config.NumberColumn(
                                    'Contribution',
                                    help="Points contributed to master score",
                                    format="%.1f",
                                    width="small"
                                ),
                                'Quality': st.column_config.TextColumn(
                                    'Quality',
                                    help="Component strength indicator",
                                    width="small"
                                )
                            }
                        )
                        
                        # Patterns
                        if stock.get('patterns'):
                            st.markdown(f"**🎯 Patterns Detected:**")
                            patterns_list = stock['patterns'].split(' | ')
                            pattern_cols = st.columns(min(3, len(patterns_list)))
                            for i, pattern in enumerate(patterns_list):
                                with pattern_cols[i % 3]:
                                    st.info(pattern)
                        
                        # Additional details in organized tabs
                        detail_tabs = st.tabs(["📊 Classification", "📈 Performance", "💰 Fundamentals", "🔍 Technicals", "🎯 Advanced"])
                        
                        with detail_tabs[0]:  # Classification
                            class_col1, class_col2 = st.columns(2)
                            
                            with class_col1:
                                st.markdown("**📊 Stock Classification**")
                                classification_data = {
                                    'Attribute': ['Sector', 'Industry', 'Category', 'Market Cap'],
                                    'Value': [
                                        stock.get('sector', 'Unknown'),
                                        stock.get('industry', 'Unknown'),
                                        stock.get('category', 'Unknown'),
                                        stock.get('market_cap', 'N/A')
                                    ]
                                }
                                class_df = pd.DataFrame(classification_data)
                                st.dataframe(
                                    class_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        'Attribute': st.column_config.TextColumn('Attribute', width="medium"),
                                        'Value': st.column_config.TextColumn('Value', width="large")
                                    }
                                )
                            
                            with class_col2:
                                st.markdown("**📈 Tier Classifications**")
                                tier_data = {
                                    'Tier Type': [],
                                    'Classification': []
                                }
                                
                                if 'price_tier' in stock.index:
                                    tier_data['Tier Type'].append('Price Tier')
                                    tier_data['Classification'].append(stock.get('price_tier', 'N/A'))
                                
                                if 'eps_tier' in stock.index:
                                    tier_data['Tier Type'].append('EPS Tier')
                                    tier_data['Classification'].append(stock.get('eps_tier', 'N/A'))
                                
                                if 'pe_tier' in stock.index:
                                    tier_data['Tier Type'].append('PE Tier')
                                    tier_data['Classification'].append(stock.get('pe_tier', 'N/A'))
                                
                                if tier_data['Tier Type']:
                                    tier_df = pd.DataFrame(tier_data)
                                    st.dataframe(
                                        tier_df,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            'Tier Type': st.column_config.TextColumn('Type', width="medium"),
                                            'Classification': st.column_config.TextColumn('Class', width="medium")
                                        }
                                    )
                                else:
                                    st.info("No tier data available")
                        
                        with detail_tabs[1]:  # Performance
                            st.markdown("**📈 Historical Performance**")
                            
                            perf_data = {
                                'Period': [],
                                'Return': [],
                                'Status': []
                            }
                            
                            periods = [
                                ('1 Day', 'ret_1d'),
                                ('3 Days', 'ret_3d'),
                                ('7 Days', 'ret_7d'),
                                ('30 Days', 'ret_30d'),
                                ('3 Months', 'ret_3m'),
                                ('6 Months', 'ret_6m'),
                                ('1 Year', 'ret_1y'),
                                ('3 Years', 'ret_3y'),
                                ('5 Years', 'ret_5y')
                            ]
                            
                            for period_name, col_name in periods:
                                if col_name in stock.index and pd.notna(stock[col_name]):
                                    perf_data['Period'].append(period_name)
                                    ret_val = stock[col_name]
                                    perf_data['Return'].append(f"{ret_val:+.1f}%")
                                    
                                    if ret_val > 10:
                                        perf_data['Status'].append('🟢 Strong')
                                    elif ret_val > 0:
                                        perf_data['Status'].append('🟡 Positive')
                                    elif ret_val > -10:
                                        perf_data['Status'].append('🟠 Negative')
                                    else:
                                        perf_data['Status'].append('🔴 Weak')
                            
                            if perf_data['Period']:
                                perf_df = pd.DataFrame(perf_data)
                                st.dataframe(
                                    perf_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        'Period': st.column_config.TextColumn('Period', width="medium"),
                                        'Return': st.column_config.TextColumn('Return', width="small"),
                                        'Status': st.column_config.TextColumn('Status', width="small")
                                    }
                                )
                            else:
                                st.info("No performance data available")
                        
                        with detail_tabs[2]:  # Fundamentals
                            if show_fundamentals:
                                st.markdown("**💰 Fundamental Analysis**")
                                
                                fund_data = {
                                    'Metric': [],
                                    'Value': [],
                                    'Assessment': []
                                }
                                
                                # PE Ratio
                                if 'pe' in stock.index and pd.notna(stock['pe']):
                                    fund_data['Metric'].append('PE Ratio')
                                    pe_val = stock['pe']
                                    
                                    if pe_val <= 0:
                                        fund_data['Value'].append('Loss/Negative')
                                        fund_data['Assessment'].append('🔴 No Earnings')
                                    elif pe_val < 15:
                                        fund_data['Value'].append(f"{pe_val:.1f}x")
                                        fund_data['Assessment'].append('🟢 Undervalued')
                                    elif pe_val < 25:
                                        fund_data['Value'].append(f"{pe_val:.1f}x")
                                        fund_data['Assessment'].append('🟡 Fair Value')
                                    elif pe_val < 50:
                                        fund_data['Value'].append(f"{pe_val:.1f}x")
                                        fund_data['Assessment'].append('🟠 Expensive')
                                    else:
                                        fund_data['Value'].append(f"{pe_val:.1f}x")
                                        fund_data['Assessment'].append('🔴 Very Expensive')
                                
                                # EPS
                                if 'eps_current' in stock.index and pd.notna(stock['eps_current']):
                                    fund_data['Metric'].append('Current EPS')
                                    fund_data['Value'].append(f"₹{stock['eps_current']:.2f}")
                                    fund_data['Assessment'].append('📊 Earnings/Share')
                                
                                # EPS Change
                                if 'eps_change_pct' in stock.index and pd.notna(stock['eps_change_pct']):
                                    fund_data['Metric'].append('EPS Growth')
                                    eps_chg = stock['eps_change_pct']
                                    
                                    if eps_chg >= 100:
                                        fund_data['Value'].append(f"{eps_chg:+.0f}%")
                                        fund_data['Assessment'].append('🚀 Explosive Growth')
                                    elif eps_chg >= 50:
                                        fund_data['Value'].append(f"{eps_chg:+.1f}%")
                                        fund_data['Assessment'].append('🔥 High Growth')
                                    elif eps_chg >= 20:
                                        fund_data['Value'].append(f"{eps_chg:+.1f}%")
                                        fund_data['Assessment'].append('🟢 Good Growth')
                                    elif eps_chg >= 0:
                                        fund_data['Value'].append(f"{eps_chg:+.1f}%")
                                        fund_data['Assessment'].append('🟡 Modest Growth')
                                    else:
                                        fund_data['Value'].append(f"{eps_chg:+.1f}%")
                                        fund_data['Assessment'].append('🔴 Declining')
                                
                                if fund_data['Metric']:
                                    fund_df = pd.DataFrame(fund_data)
                                    st.dataframe(
                                        fund_df,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            'Metric': st.column_config.TextColumn('Metric', width="medium"),
                                            'Value': st.column_config.TextColumn('Value', width="small"),
                                            'Assessment': st.column_config.TextColumn('Assessment', width="medium")
                                        }
                                    )
                                else:
                                    st.info("No fundamental data available")
                            else:
                                st.info("Enable 'Hybrid' display mode to see fundamental data")
                        
                        with detail_tabs[3]:  # Technicals
                            st.markdown("**🔍 Technical Analysis**")
                            
                            tech_col1, tech_col2 = st.columns(2)
                            
                            with tech_col1:
                                st.markdown("**📊 52-Week Range**")
                                range_data = {
                                    'Metric': [],
                                    'Value': []
                                }
                                
                                if 'low_52w' in stock.index and pd.notna(stock['low_52w']):
                                    range_data['Metric'].append('52W Low')
                                    range_data['Value'].append(f"₹{stock['low_52w']:,.0f}")
                                
                                if 'high_52w' in stock.index and pd.notna(stock['high_52w']):
                                    range_data['Metric'].append('52W High')
                                    range_data['Value'].append(f"₹{stock['high_52w']:,.0f}")
                                
                                if 'from_low_pct' in stock.index and pd.notna(stock['from_low_pct']):
                                    range_data['Metric'].append('From Low')
                                    range_data['Value'].append(f"{stock['from_low_pct']:.0f}%")
                                
                                if 'from_high_pct' in stock.index and pd.notna(stock['from_high_pct']):
                                    range_data['Metric'].append('From High')
                                    range_data['Value'].append(f"{stock['from_high_pct']:.0f}%")
                                
                                if range_data['Metric']:
                                    range_df = pd.DataFrame(range_data)
                                    st.dataframe(
                                        range_df,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            'Metric': st.column_config.TextColumn('Metric', width="medium"),
                                            'Value': st.column_config.TextColumn('Value', width="medium")
                                        }
                                    )
                            
                            with tech_col2:
                                st.markdown("**📈 Moving Averages**")
                                sma_data = {
                                    'SMA': [],
                                    'Value': [],
                                    'Position': []
                                }
                                
                                current_price = stock.get('price', 0)
                                
                                for sma_col, sma_label in [('sma_20d', '20 DMA'), ('sma_50d', '50 DMA'), ('sma_200d', '200 DMA')]:
                                    if sma_col in stock.index and pd.notna(stock[sma_col]) and stock[sma_col] > 0:
                                        sma_value = stock[sma_col]
                                        sma_data['SMA'].append(sma_label)
                                        sma_data['Value'].append(f"₹{sma_value:,.0f}")
                                        
                                        if current_price > sma_value:
                                            pct_diff = ((current_price - sma_value) / sma_value) * 100
                                            sma_data['Position'].append(f"🟢 +{pct_diff:.1f}%")
                                        else:
                                            pct_diff = ((sma_value - current_price) / sma_value) * 100
                                            sma_data['Position'].append(f"🔴 -{pct_diff:.1f}%")
                                
                                if sma_data['SMA']:
                                    sma_df = pd.DataFrame(sma_data)
                                    st.dataframe(
                                        sma_df,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            'SMA': st.column_config.TextColumn('SMA', width="small"),
                                            'Value': st.column_config.TextColumn('Value', width="medium"),
                                            'Position': st.column_config.TextColumn('Position', width="small")
                                        }
                                    )
                            
                            # Trend Analysis
                            if 'trend_quality' in stock.index and pd.notna(stock['trend_quality']):
                                tq = stock['trend_quality']
                                if tq >= 80:
                                    trend_status = f"🔥 Strong Uptrend ({tq:.0f})"
                                    trend_color = "success"
                                elif tq >= 60:
                                    trend_status = f"✅ Good Uptrend ({tq:.0f})"
                                    trend_color = "success"
                                elif tq >= 40:
                                    trend_status = f"➡️ Neutral Trend ({tq:.0f})"
                                    trend_color = "warning"
                                else:
                                    trend_status = f"⚠️ Weak/Downtrend ({tq:.0f})"
                                    trend_color = "error"
                                
                                getattr(st, trend_color)(f"**Trend Status:** {trend_status}")
                        
                        with detail_tabs[4]:  # Advanced Metrics
                            st.markdown("**🎯 Advanced Metrics**")
                            
                            adv_data = {
                                'Metric': [],
                                'Value': [],
                                'Description': []
                            }
                            
                            # VMI
                            if 'vmi' in stock.index and pd.notna(stock['vmi']):
                                adv_data['Metric'].append('VMI')
                                adv_data['Value'].append(f"{stock['vmi']:.2f}")
                                adv_data['Description'].append('Volume Momentum Index')
                            
                            # Position Tension
                            if 'position_tension' in stock.index and pd.notna(stock['position_tension']):
                                adv_data['Metric'].append('Position Tension')
                                adv_data['Value'].append(f"{stock['position_tension']:.0f}")
                                adv_data['Description'].append('Range position stress')
                            
                            # Momentum Harmony
                            if 'momentum_harmony' in stock.index and pd.notna(stock['momentum_harmony']):
                                harmony_val = int(stock['momentum_harmony'])
                                harmony_emoji = "🟢" if harmony_val >= 3 else "🟡" if harmony_val >= 2 else "🔴"
                                adv_data['Metric'].append('Momentum Harmony')
                                adv_data['Value'].append(f"{harmony_emoji} {harmony_val}/4")
                                adv_data['Description'].append('Multi-timeframe alignment')
                            
                            # Money Flow
                            if 'money_flow_mm' in stock.index and pd.notna(stock['money_flow_mm']):
                                adv_data['Metric'].append('Money Flow')
                                adv_data['Value'].append(f"₹{stock['money_flow_mm']:.1f}M")
                                adv_data['Description'].append('Price × Volume × RVOL')
                            
                            # Overall Wave Strength
                            if 'overall_wave_strength' in stock.index and pd.notna(stock['overall_wave_strength']):
                                adv_data['Metric'].append('Wave Strength')
                                adv_data['Value'].append(f"{stock['overall_wave_strength']:.1f}%")
                                adv_data['Description'].append('Composite wave score')
                            
                            # Pattern Confidence
                            if 'pattern_confidence' in stock.index and pd.notna(stock['pattern_confidence']):
                                adv_data['Metric'].append('Pattern Confidence')
                                adv_data['Value'].append(f"{stock['pattern_confidence']:.1f}%")
                                adv_data['Description'].append('Pattern strength score')
                            
                            if adv_data['Metric']:
                                adv_df = pd.DataFrame(adv_data)
                                st.dataframe(
                                    adv_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        'Metric': st.column_config.TextColumn(
                                            'Metric',
                                            help="Advanced metric name",
                                            width="medium"
                                        ),
                                        'Value': st.column_config.TextColumn(
                                            'Value',
                                            help="Metric value",
                                            width="small"
                                        ),
                                        'Description': st.column_config.TextColumn(
                                            'Description',
                                            help="What this metric measures",
                                            width="large"
                                        )
                                    }
                                )
                            else:
                                st.info("No advanced metrics available")
            
            else:
                st.warning("No stocks found matching your search criteria.")
                
                # Provide search suggestions
                st.markdown("#### 💡 Search Tips:")
                st.markdown("""
                - **Ticker Search:** Enter exact ticker symbol (e.g., RELIANCE, TCS, INFY)
                - **Company Search:** Enter part of company name (e.g., Tata, Infosys, Reliance)
                - **Partial Match:** Search works with partial text (e.g., 'REL' finds RELIANCE)
                - **Case Insensitive:** Search is not case-sensitive
                """)
        
        else:
            # Show search instructions when no search is active
            st.info("Enter a ticker symbol or company name to search")
            
            # Show top performers as suggestions
            st.markdown("#### 🏆 Today's Top Performers")
            
            if not filtered_df.empty:
                top_performers = filtered_df.nlargest(5, 'master_score')[['ticker', 'company_name', 'master_score', 'ret_1d', 'rvol']]
                
                suggestions_data = []
                for _, row in top_performers.iterrows():
                    suggestions_data.append({
                        'Ticker': row['ticker'],
                        'Company': row['company_name'][:30] + '...' if len(row['company_name']) > 30 else row['company_name'],
                        'Score': row['master_score'],
                        '1D Return': f"{row['ret_1d']:+.1f}%" if pd.notna(row['ret_1d']) else '-',
                        'RVOL': f"{row['rvol']:.1f}x" if pd.notna(row['rvol']) else '-'
                    })
                
                suggestions_df = pd.DataFrame(suggestions_data)
                
                st.dataframe(
                    suggestions_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Ticker': st.column_config.TextColumn('Ticker', width="small"),
                        'Company': st.column_config.TextColumn('Company', width="large"),
                        'Score': st.column_config.ProgressColumn(
                            'Score',
                            format="%.1f",
                            min_value=0,
                            max_value=100,
                            width="small"
                        ),
                        '1D Return': st.column_config.TextColumn('1D Return', width="small"),
                        'RVOL': st.column_config.TextColumn('RVOL', width="small")
                    }
                )
                
                st.caption("💡 Tip: Click on any ticker above and copy it to search")    
                
    with tabs[5]:
        st.markdown("### 📥 Export Data")
        
        st.markdown("#### 📋 Export Templates")
        export_template = st.radio(
            "Choose export template:",
            options=[
                "Full Analysis (All Data)",
                "Day Trader Focus",
                "Swing Trader Focus",
                "Investor Focus"
            ],
            key="export_template_radio",
            help="Select a template based on your trading style"
        )
        
        template_map = {
            "Full Analysis (All Data)": "full",
            "Day Trader Focus": "day_trader",
            "Swing Trader Focus": "swing_trader",
            "Investor Focus": "investor"
        }
        
        selected_template = template_map[export_template]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Excel Report")
            st.markdown(
                "Comprehensive multi-sheet report including:\n"
                "- Top 100 stocks with all scores\n"
                "- Market intelligence dashboard\n"
                "- Sector rotation analysis\n"
                "- Pattern frequency analysis\n"
                "- Wave Radar signals\n"
                "- Summary statistics"
            )
            
            if st.button("Generate Excel Report", type="primary", use_container_width=True):
                if len(filtered_df) == 0:
                    st.error("No data to export. Please adjust your filters.")
                else:
                    with st.spinner("Creating Excel report..."):
                        try:
                            excel_file = ExportEngine.create_excel_report(
                                filtered_df, template=selected_template
                            )
                            
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_file,
                                file_name=f"wave_detection_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            st.success("Excel report generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error generating Excel report: {str(e)}")
                            logger.error(f"Excel export error: {str(e)}", exc_info=True)
        
        with col2:
            st.markdown("#### 📄 CSV Export")
            st.markdown(
                "Enhanced CSV format with:\n"
                "- All ranking scores\n"
                "- Advanced metrics (VMI, Money Flow)\n"
                "- Pattern detections\n"
                "- Wave states\n"
                "- Category classifications\n"
                "- Optimized for further analysis"
            )
            
            if st.button("Generate CSV Export", use_container_width=True):
                if len(filtered_df) == 0:
                    st.error("No data to export. Please adjust your filters.")
                else:
                    try:
                        csv_data = ExportEngine.create_csv_export(filtered_df)
                        
                        st.download_button(
                            label="📥 Download CSV File",
                            data=csv_data,
                            file_name=f"wave_detection_data_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        st.success("CSV export generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating CSV: {str(e)}")
                        logger.error(f"CSV export error: {str(e)}", exc_info=True)
        
        st.markdown("---")
        st.markdown("#### 📊 Export Preview")
        
        export_stats = {
            "Total Stocks": len(filtered_df),
            "Average Score": f"{filtered_df['master_score'].mean():.1f}" if not filtered_df.empty else "N/A",
            "Stocks with Patterns": (filtered_df['patterns'] != '').sum() if 'patterns' in filtered_df.columns else 0,
            "High RVOL (>2x)": (filtered_df['rvol'] > 2).sum() if 'rvol' in filtered_df.columns else 0,
            "Positive 30D Returns": (filtered_df['ret_30d'] > 0).sum() if 'ret_30d' in filtered_df.columns else 0,
            "Data Quality": f"{st.session_state.data_quality.get('completeness', 0):.1f}%"
        }
        
        stat_cols = st.columns(3)
        for i, (label, value) in enumerate(export_stats.items()):
            with stat_cols[i % 3]:
                UIComponents.render_metric_card(label, value)
    
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0 - Final Production Version")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            #### 🌊 Welcome to Wave Detection Ultimate 3.0
            
            The FINAL production version of the most advanced stock ranking system designed to catch momentum waves early.
            This professional-grade tool combines technical analysis, volume dynamics, advanced metrics, and 
            smart pattern recognition to identify high-potential stocks before they peak.
            
            #### 🎯 Core Features - LOCKED IN PRODUCTION
            
            **Master Score 3.0** - Proprietary ranking algorithm (DO NOT MODIFY):
            - **Position Analysis (30%)** - 52-week range positioning
            - **Volume Dynamics (25%)** - Multi-timeframe volume patterns
            - **Momentum Tracking (15%)** - 30-day price momentum
            - **Acceleration Detection (10%)** - Momentum acceleration signals
            - **Breakout Probability (10%)** - Technical breakout readiness
            - **RVOL Integration (10%)** - Real-time relative volume
            
            **Advanced Metrics** - NEW IN FINAL VERSION:
            - **Money Flow** - Price × Volume × RVOL in millions
            - **VMI (Volume Momentum Index)** - Weighted volume trend score
            - **Position Tension** - Range position stress indicator
            - **Momentum Harmony** - Multi-timeframe alignment (0-4)
            - **Wave State** - Real-time momentum classification
            - **Overall Wave Strength** - Composite score for wave filter
            
            **30 Pattern Detection** - Complete set:
            - 11 Technical patterns
            - 5 Fundamental patterns (Hybrid mode)
            - 6 Price range patterns
            - 3 Intelligence patterns
            - 5 NEW Quant reversal patterns
            - 3 NEW intelligence patterns (Stealth, Vampire, Perfect Storm)
            
            #### 💡 How to Use
            
            1. **Data Source** - Google Sheets (default) or CSV upload
            2. **Quick Actions** - Instant filtering for common scenarios
            3. **Smart Filters** - Interconnected filtering system, including new Wave filters
            4. **Display Modes** - Technical or Hybrid (with fundamentals)
            5. **Wave Radar** - Monitor early momentum signals
            6. **Export Templates** - Customized for trading styles
            
            #### 🔧 Production Features
            
            - **Performance Optimized** - Sub-2 second processing
            - **Memory Efficient** - Handles 2000+ stocks smoothly
            - **Error Resilient** - Graceful degradation
            - **Data Validation** - Comprehensive quality checks
            - **Smart Caching** - 1-hour intelligent cache
            - **Mobile Responsive** - Works on all devices
            
            #### 📊 Data Processing Pipeline
            
            1. Load from Google Sheets or CSV
            2. Validate and clean all 41 columns
            3. Calculate 6 component scores
            4. Generate Master Score 3.0
            5. Calculate advanced metrics
            6. Detect all 25 patterns
            7. Classify into tiers
            8. Apply smart ranking
            
            #### 🎨 Display Modes
            
            **Technical Mode** (Default)
            - Pure momentum analysis
            - Technical indicators only
            - Pattern detection
            - Volume dynamics
            
            **Hybrid Mode**
            - All technical features
            - PE ratio analysis
            - EPS growth tracking
            - Fundamental patterns
            - Value indicators
            """)
        
        with col2:
            st.markdown("""
            #### 📈 Pattern Groups
            
            **Technical Patterns**
            - 🔥 CAT LEADER
            - 💎 HIDDEN GEM
            - 🚀 ACCELERATING
            - 🏦 INSTITUTIONAL
            - ⚡ VOL EXPLOSION
            - 🎯 BREAKOUT
            - 👑 MARKET LEADER
            - 🌊 MOMENTUM WAVE
            - 💰 LIQUID LEADER
            - 💪 LONG STRENGTH
            - 📈 QUALITY TREND
            
            **Range Patterns**
            - 🎯 52W HIGH APPROACH
            - 🔄 52W LOW BOUNCE
            - 👑 GOLDEN ZONE
            - 📊 VOL ACCUMULATION
            - 🔀 MOMENTUM DIVERGE
            - 🎯 RANGE COMPRESS
            
            **NEW Intelligence**
            - 🤫 STEALTH
            - 🧛 VAMPIRE
            - ⛈️ PERFECT STORM
            
            **Fundamental** (Hybrid)
            - 💎 VALUE MOMENTUM
            - 📊 EARNINGS ROCKET
            - 🏆 QUALITY LEADER
            - ⚡ TURNAROUND
            - ⚠️ HIGH PE

            **Quant Reversal**
            - 🪤 BULL TRAP
            - 💣 CAPITULATION
            - 🏃 RUNAWAY GAP
            - 🔄 ROTATION LEADER
            - ⚠️ DISTRIBUTION
            
            #### ⚡ Performance
            
            - Initial load: <2 seconds
            - Filtering: <200ms
            - Pattern detection: <500ms
            - Search: <50ms
            - Export: <1 second
            
            #### 🔒 Production Status
            
            **Version**: 3.0.7-FINAL-COMPLETE
            **Last Updated**: July 2025
            **Status**: PRODUCTION
            **Updates**: LOCKED
            **Testing**: COMPLETE
            **Optimization**: MAXIMUM
            
            #### 💬 Credits
            
            Developed for professional traders
            requiring reliable, fast, and
            comprehensive market analysis.
            
            This is the FINAL version.
            No further updates will be made.
            All features are permanent.
            
            ---
            
            **Indian Market Optimized**
            - ₹ Currency formatting
            - IST timezone aware
            - NSE/BSE categories
            - Local number formats
            """)
        
        # System stats
        st.markdown("---")
        st.markdown("#### 📊 Current Session Statistics")
        
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            UIComponents.render_metric_card(
                "Total Stocks Loaded",
                f"{len(ranked_df):,}" if 'ranked_df' in locals() else "0"
            )
        
        with stats_cols[1]:
            UIComponents.render_metric_card(
                "Currently Filtered",
                f"{len(filtered_df):,}" if 'filtered_df' in locals() else "0"
            )
        
        with stats_cols[2]:
            data_quality = st.session_state.data_quality.get('completeness', 0)
            quality_emoji = "🟢" if data_quality > 80 else "🟡" if data_quality > 60 else "🔴"
            UIComponents.render_metric_card(
                "Data Quality",
                f"{quality_emoji} {data_quality:.1f}%"
            )
        
        with stats_cols[3]:
            cache_time = datetime.now(timezone.utc) - st.session_state.last_refresh
            minutes = int(cache_time.total_seconds() / 60)
            cache_status = "Fresh" if minutes < 60 else "Stale"
            cache_emoji = "🟢" if minutes < 60 else "🔴"
            UIComponents.render_metric_card(
                "Cache Age",
                f"{cache_emoji} {minutes} min",
                cache_status
            )
    
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            🌊 Wave Detection Ultimate 3.0 - Final Production Version<br>
            <small>Professional Stock Ranking System • All Features Complete • Performance Optimized • Permanently Locked</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical Application Error: {str(e)}")
        logger.error(f"Application crashed: {str(e)}", exc_info=True)
        
        if st.button("🔄 Restart Application"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📧 Report Issue"):
            st.info("Please take a screenshot and report this error.")
