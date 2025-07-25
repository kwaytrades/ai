import redis
import json
import os
import logging
from datetime import datetime, time, timedelta
import pytz
import holidays
from typing import Optional, Dict, Any
from config.popular_tickers import POPULAR_TICKERS

logger = logging.getLogger(__name__)

class MarketScheduler:
    def __init__(self):
        self.timezone = pytz.timezone(os.getenv('MARKET_TIMEZONE', 'US/Eastern'))
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        self.holidays = holidays.UnitedStates()
    
    def is_market_hours(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(self.timezone)
        
        # Weekend check
        if now.weekday() >= 5:
            return False
        
        # Holiday check
        if now.date() in self.holidays:
            return False
        
        # Time check
        current_time = now.time()
        return self.market_open <= current_time <= self.market_close
    
    def next_market_open(self) -> datetime:
        """Calculate next market opening time."""
        now = datetime.now(self.timezone)
        
        # Start with today at market open
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # If market already closed today, move to next day
        if now.time() > self.market_close:
            next_open += timedelta(days=1)
        
        # Skip weekends and holidays
        while (next_open.weekday() >= 5 or next_open.date() in self.holidays):
            next_open += timedelta(days=1)
        
        return next_open
    
    def is_trading_day(self, date: datetime = None) -> bool:
        """Check if given date (or today) is a trading day."""
        if date is None:
            date = datetime.now(self.timezone)
        
        return (date.weekday() < 5 and date.date() not in self.holidays)

class CacheManager:
    def __init__(self):
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,  # Auto-decode responses
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except (redis.ConnectionError, Exception) as e:
            logger.warning(f"Redis connection failed, running without cache: {e}")
            self.redis_client = None
        
        self.scheduler = MarketScheduler()
        
        # Cache TTL settings (in seconds)
        self.cache_times = {
            "popular": int(os.getenv('CACHE_POPULAR_TTL', 1800)),      # 30 min
            "ondemand": int(os.getenv('CACHE_ONDEMAND_TTL', 300)),     # 5 min
            "afterhours": int(os.getenv('CACHE_AFTERHOURS_TTL', 3600)), # 1 hour
        }
    
    def get_cache_key(self, ticker: str, interval: str, period: str) -> str:
        """Generate Redis cache key."""
        return f"ta:{ticker.upper()}:{interval}:{period}"
    
    def get_cache_ttl(self, ticker: str) -> int:
        """Determine cache TTL based on market hours and ticker popularity."""
        # Longer cache during after hours
        if not self.scheduler.is_market_hours():
            return self.cache_times["afterhours"]
        
        # Popular tickers get longer cache during market hours
        if ticker.upper() in POPULAR_TICKERS:
            return self.cache_times["popular"]
        else:
            return self.cache_times["ondemand"]
    
    def get_cached_data(self, ticker: str, interval: str, period: str) -> Optional[Dict[Any, Any]]:
        """Get cached data synchronously."""
        if not self.redis_client:
            return None
            
        try:
            key = self.get_cache_key(ticker, interval, period)
            cached = self.redis_client.get(key)
            
            if cached:
                data = json.loads(cached)
                logger.debug(f"Cache HIT for {key}")
                return data
            else:
                logger.debug(f"Cache MISS for {key}")
                return None
                
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache get error for {ticker}: {e}")
            return None
    
    def set_cached_data(self, ticker: str, interval: str, period: str, data: Dict[Any, Any]) -> bool:
        """Set cached data synchronously."""
        if not self.redis_client:
            return False
            
        try:
            key = self.get_cache_key(ticker, interval, period)
            ttl = self.get_cache_ttl(ticker)
            
            # Serialize data with proper datetime handling
            serialized_data = json.dumps(data, default=self._json_serializer)
            
            # Set main data
            self.redis_client.setex(key, ttl, serialized_data)
            
            # Set metadata
            meta_key = f"meta:{key}"
            meta_data = {
                "last_updated": datetime.now().isoformat(),
                "ttl": ttl,
                "is_popular": ticker.upper() in POPULAR_TICKERS,
                "market_hours": self.scheduler.is_market_hours(),
                "ticker": ticker.upper()
            }
            self.redis_client.setex(meta_key, ttl, json.dumps(meta_data))
            
            logger.debug(f"Cached {key} with TTL {ttl}s")
            return True
            
        except (redis.RedisError, TypeError) as e:
            logger.error(f"Cache set error for {ticker}: {e}")
            return False
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for datetime objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def invalidate_cache(self, ticker: str, interval: str = None, period: str = None) -> bool:
        """Invalidate cache for specific ticker or data type."""
        if not self.redis_client:
            return False
            
        try:
            if interval and period:
                # Invalidate specific cache
                key = self.get_cache_key(ticker, interval, period)
                keys_to_delete = [key, f"meta:{key}"]
            else:
                # Invalidate all cache for ticker
                pattern = f"ta:{ticker.upper()}:*"
                keys_to_delete = self.redis_client.keys(pattern)
                meta_keys = [f"meta:{key}" for key in keys_to_delete]
                keys_to_delete.extend(meta_keys)
            
            if keys_to_delete:
                deleted = self.redis_client.delete(*keys_to_delete)
                logger.info(f"Invalidated {deleted} cache keys for {ticker}")
                return deleted > 0
            
            return False
            
        except redis.RedisError as e:
            logger.error(f"Cache invalidation error: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            if not self.redis_client:
                return {"status": "redis_unavailable"}
                
            # Basic Redis stats
            info = self.redis_client.info()
            
            # Count different types of keys
            ta_keys = len(self.redis_client.keys("ta:*"))
            meta_keys = len(self.redis_client.keys("meta:*"))
            
            # Popular ticker cache status
            popular_cached = 0
            for ticker in POPULAR_TICKERS[:10]:  # Check first 10
                pattern = f"ta:{ticker}:*"
                if self.redis_client.keys(pattern):
                    popular_cached += 1
            
            return {
                "redis_status": "connected",
                "total_keys": info.get('db0', {}).get('keys', 0) if 'db0' in info else 0,
                "ta_cache_keys": ta_keys,
                "meta_keys": meta_keys,
                "memory_used": info.get('used_memory_human', 'N/A'),
                "memory_peak": info.get('used_memory_peak_human', 'N/A'),
                "connected_clients": info.get('connected_clients', 0),
                "total_commands_processed": info.get('total_commands_processed', 0),
                "cache_config": {
                    "popular_ttl": self.cache_times["popular"],
                    "ondemand_ttl": self.cache_times["ondemand"], 
                    "afterhours_ttl": self.cache_times["afterhours"]
                },
                "market_info": {
                    "is_market_hours": self.scheduler.is_market_hours(),
                    "next_market_open": self.scheduler.next_market_open().isoformat(),
                    "timezone": str(self.scheduler.timezone)
                },
                "popular_tickers": {
                    "total_count": len(POPULAR_TICKERS),
                    "cached_count": popular_cached,
                    "sample": POPULAR_TICKERS[:5]
                }
            }
            
        except redis.RedisError as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "status": "error",
                "error": str(e),
                "market_hours": self.scheduler.is_market_hours()
            }

# Singleton instance
cache_manager = CacheManager()

# Convenience functions for easy import
def get_cached_analysis(ticker: str, interval: str = '1d', period: str = '1mo') -> Optional[Dict]:
    """Get cached technical analysis data."""
    return cache_manager.get_cached_data(ticker, interval, period)

def cache_analysis(ticker: str, interval: str, period: str, data: Dict) -> bool:
    """Cache technical analysis data."""
    return cache_manager.set_cached_data(ticker, interval, period, data)

def invalidate_ticker_cache(ticker: str) -> bool:
    """Invalidate all cache for a ticker."""
    return cache_manager.invalidate_cache(ticker)
