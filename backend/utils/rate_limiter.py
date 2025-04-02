# middleware/rate_limiter.py
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
import asyncio
from typing import Dict

logger = logging.getLogger("rate_limiter")

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app,
        requests_per_minute: int = 100,
        cleanup_interval: int = 300,
        exclude_paths: list = None
    ):
        super().__init__(app)
        self.rate_limit_dict: Dict[str, Dict] = {}
        self.requests_per_minute = requests_per_minute
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        self.exclude_paths = exclude_paths or ["/docs", "/redoc", "/openapi.json", "/health"]
        
        # Start background cleanup task
        asyncio.create_task(self._periodic_cleanup())
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process each request through rate limiting"""
        
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Check rate limit
        client_ip = request.client.host
        current_time = time.time()
        
        # Check if we need to clean up
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_rate_limit()
            self.last_cleanup = current_time
        
        # Check rate limit for this IP
        if client_ip in self.rate_limit_dict:
            if current_time - self.rate_limit_dict[client_ip]["timestamp"] > 60:
                # Reset if it's been more than a minute
                self.rate_limit_dict[client_ip] = {"count": 1, "timestamp": current_time}
            else:
                # Increment count
                self.rate_limit_dict[client_ip]["count"] += 1
                
                # Check if limit exceeded
                if self.rate_limit_dict[client_ip]["count"] > self.requests_per_minute:
                    logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Too many requests. Please try again later."}
                    )
        else:
            # First request from this IP
            self.rate_limit_dict[client_ip] = {"count": 1, "timestamp": current_time}
        
        # Process the request
        return await call_next(request)
    
    async def _cleanup_rate_limit(self):
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
    
    async def _periodic_cleanup(self):
        """Run cleanup periodically in the background"""
        while True:
            await asyncio.sleep(self.cleanup_interval)
            await self._cleanup_rate_limit()