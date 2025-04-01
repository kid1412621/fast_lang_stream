# utils/rate_limiter.py
import time
from typing import Dict
from fastapi import Request, HTTPException
import logging

logger = logging.getLogger("rate_limiter")

class RateLimiter:
    def __init__(self, requests_per_minute: int = 100, cleanup_interval: int = 300):
        self.rate_limit_dict: Dict[str, Dict] = {}
        self.requests_per_minute = requests_per_minute
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
    
    async def check_rate_limit(self, request: Request) -> bool:
        """Rate limiting dependency"""
        client_ip = request.client.host
        current_time = time.time()
        
        # Perform cleanup if needed
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self.cleanup_rate_limit()
            self.last_cleanup = current_time

        if client_ip in self.rate_limit_dict:
            if current_time - self.rate_limit_dict[client_ip]["timestamp"] > 60:
                self.rate_limit_dict[client_ip] = {"count": 1, "timestamp": current_time}
            else:
                self.rate_limit_dict[client_ip]["count"] += 1
                if self.rate_limit_dict[client_ip]["count"] > self.requests_per_minute:
                    logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                    raise HTTPException(
                        status_code=429, detail="Too many requests. Please try again later."
                    )
        else:
            self.rate_limit_dict[client_ip] = {"count": 1, "timestamp": current_time}

        return True
    
    async def cleanup_rate_limit(self):
        """Clean up old rate limit entries"""
        current_time = time.time()
        expired_ips = [
            ip
            for ip, data in self.rate_limit_dict.items()
            if current_time - data["timestamp"] > 60
        ]
        for ip in expired_ips:
            del self.rate_limit_dict[ip]
        if expired_ips:
            logger.info(f"Cleaned up {len(expired_ips)} expired rate limit entries")