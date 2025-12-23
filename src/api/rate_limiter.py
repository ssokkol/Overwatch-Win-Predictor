"""Rate limiting implementation using Redis with fallback to in-memory cache."""

import time
from typing import Dict, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RateLimiter:
    """Rate limiter using token bucket algorithm."""

    def __init__(
        self,
        redis_client: any = None,
        max_calls: int = 100,
        period: int = 60,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            redis_client: Redis client instance (optional)
            max_calls: Maximum number of calls allowed
            period: Time period in seconds
        """
        self.redis_client = redis_client
        self.max_calls = max_calls
        self.period = period
        self._memory_cache: Dict[str, list[float]] = {}

    def is_allowed(self, identifier: str) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed under rate limit.

        Args:
            identifier: Unique identifier (e.g., IP address, API key)

        Returns:
            Tuple of (is_allowed, remaining_calls)
        """
        current_time = time.time()

        if self.redis_client:
            return self._check_redis(identifier, current_time)
        else:
            return self._check_memory(identifier, current_time)

    def _check_redis(self, identifier: str, current_time: float) -> tuple[bool, Optional[int]]:
        """
        Check rate limit using Redis.

        Args:
            identifier: Unique identifier
            current_time: Current timestamp

        Returns:
            Tuple of (is_allowed, remaining_calls)
        """
        try:
            key = f"rate_limit:{identifier}"
            pipe = self.redis_client.pipeline()
            
            # Get current timestamps
            pipe.zrangebyscore(key, current_time - self.period, current_time)
            pipe.zcard(key)
            pipe.zremrangebyscore(key, 0, current_time - self.period)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            pipe.expire(key, self.period)
            
            results = pipe.execute()
            current_count = results[1] + 1  # +1 for current request
            
            is_allowed = current_count <= self.max_calls
            remaining = max(0, self.max_calls - current_count) if is_allowed else 0
            
            return is_allowed, remaining
            
        except Exception as e:
            logger.warning(f"Redis rate limit check failed: {e}, falling back to memory")
            return self._check_memory(identifier, current_time)

    def _check_memory(self, identifier: str, current_time: float) -> tuple[bool, Optional[int]]:
        """
        Check rate limit using in-memory cache.

        Args:
            identifier: Unique identifier
            current_time: Current timestamp

        Returns:
            Tuple of (is_allowed, remaining_calls)
        """
        if identifier not in self._memory_cache:
            self._memory_cache[identifier] = []

        # Clean old entries
        timestamps = self._memory_cache[identifier]
        timestamps[:] = [
            t for t in timestamps if t > current_time - self.period
        ]

        # Check limit
        current_count = len(timestamps)
        if current_count >= self.max_calls:
            return False, 0

        # Add current request
        timestamps.append(current_time)
        self._memory_cache[identifier] = timestamps

        remaining = self.max_calls - current_count - 1
        return True, remaining

