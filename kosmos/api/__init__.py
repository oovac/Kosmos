"""
API module for Kosmos AI Scientist.

Provides:
- FastAPI web server for production deployment
- Health check endpoints for monitoring
- REST API for research operations
"""

from kosmos.api.health import (
    get_basic_health,
    get_readiness_check,
    get_metrics,
    HealthChecker,
    get_health_checker
)

__all__ = [
    "get_basic_health",
    "get_readiness_check",
    "get_metrics",
    "HealthChecker",
    "get_health_checker"
]

# Import app lazily to avoid circular imports
def get_app():
    """Get the FastAPI application instance."""
    from kosmos.api.server import app
    return app
