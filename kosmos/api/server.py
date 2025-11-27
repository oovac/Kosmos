"""
Kosmos AI Scientist - FastAPI Web Server

Production-ready API server for deployment on Northflank and other platforms.

Provides:
- Health check endpoints (liveness, readiness, metrics)
- Research API endpoints
- WebSocket support for real-time updates
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from kosmos.api.health import (
    get_basic_health,
    get_readiness_check,
    get_metrics,
    get_health_checker,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class ResearchRequest(BaseModel):
    """Request model for starting a research task."""
    question: str = Field(..., description="The research question to investigate")
    domain: Optional[str] = Field(None, description="Scientific domain (e.g., biology, materials)")
    max_iterations: int = Field(5, ge=1, le=20, description="Maximum research iterations")
    enable_execution: bool = Field(False, description="Enable code execution")


class ResearchResponse(BaseModel):
    """Response model for research task."""
    task_id: str
    status: str
    message: str
    created_at: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    uptime_seconds: float
    service: str
    version: str


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None
    timestamp: str


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    logger.info("Starting Kosmos API server...")
    
    # Initialize database connection
    try:
        from kosmos.db import init_from_config
        init_from_config()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed (may be expected): {e}")
    
    # Initialize health checker
    get_health_checker()
    logger.info("Health checker initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Kosmos API server...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Kosmos AI Scientist",
    description="Autonomous scientific research system powered by Claude",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Basic health check (liveness)",
    description="Returns basic health status for liveness probes"
)
async def health():
    """Basic health check endpoint for liveness probes."""
    return get_basic_health()


@app.get(
    "/health/ready",
    tags=["Health"],
    summary="Readiness check",
    description="Returns detailed readiness status including all dependencies"
)
async def health_ready():
    """Readiness check endpoint that verifies all dependencies."""
    result = get_readiness_check()
    
    # Return 503 if not ready
    if result.get("status") != "ready":
        return JSONResponse(status_code=503, content=result)
    
    return result


@app.get(
    "/health/metrics",
    tags=["Health"],
    summary="System metrics",
    description="Returns detailed system metrics including CPU, memory, and disk usage"
)
async def health_metrics():
    """System metrics endpoint."""
    return get_metrics()


@app.get(
    "/",
    tags=["Info"],
    summary="API information",
    description="Returns basic API information and available endpoints"
)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "kosmos-ai-scientist",
        "version": "0.2.0",
        "description": "Autonomous scientific research system powered by Claude",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "health": {
                "liveness": "/health",
                "readiness": "/health/ready",
                "metrics": "/health/metrics"
            },
            "research": {
                "start": "POST /api/v1/research",
                "status": "GET /api/v1/research/{task_id}",
                "list": "GET /api/v1/research"
            }
        }
    }


# =============================================================================
# Research API Endpoints
# =============================================================================

# In-memory task storage (replace with database in production)
_research_tasks: Dict[str, Dict[str, Any]] = {}


@app.post(
    "/api/v1/research",
    response_model=ResearchResponse,
    tags=["Research"],
    summary="Start a research task",
    description="Submit a research question to be investigated by the AI scientist"
)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks
):
    """Start a new research task."""
    import uuid
    
    task_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    
    # Store task
    _research_tasks[task_id] = {
        "task_id": task_id,
        "question": request.question,
        "domain": request.domain,
        "max_iterations": request.max_iterations,
        "enable_execution": request.enable_execution,
        "status": "pending",
        "created_at": created_at,
        "updated_at": created_at,
        "result": None,
        "error": None
    }
    
    # Start research in background
    background_tasks.add_task(run_research_task, task_id, request)
    
    return ResearchResponse(
        task_id=task_id,
        status="pending",
        message="Research task submitted successfully",
        created_at=created_at
    )


@app.get(
    "/api/v1/research/{task_id}",
    tags=["Research"],
    summary="Get research task status",
    description="Get the current status and results of a research task"
)
async def get_research_status(task_id: str):
    """Get the status of a research task."""
    if task_id not in _research_tasks:
        raise HTTPException(status_code=404, detail="Research task not found")
    
    return _research_tasks[task_id]


@app.get(
    "/api/v1/research",
    tags=["Research"],
    summary="List research tasks",
    description="List all research tasks with optional filtering"
)
async def list_research_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip")
):
    """List all research tasks."""
    tasks = list(_research_tasks.values())
    
    # Filter by status if provided
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    
    # Sort by created_at descending
    tasks.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Apply pagination
    total = len(tasks)
    tasks = tasks[offset:offset + limit]
    
    return {
        "tasks": tasks,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.delete(
    "/api/v1/research/{task_id}",
    tags=["Research"],
    summary="Cancel research task",
    description="Cancel a running research task"
)
async def cancel_research(task_id: str):
    """Cancel a research task."""
    if task_id not in _research_tasks:
        raise HTTPException(status_code=404, detail="Research task not found")
    
    task = _research_tasks[task_id]
    if task["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel task with status: {task['status']}"
        )
    
    _research_tasks[task_id]["status"] = "cancelled"
    _research_tasks[task_id]["updated_at"] = datetime.utcnow().isoformat()
    
    return {"message": "Research task cancelled", "task_id": task_id}


# =============================================================================
# Background Tasks
# =============================================================================

async def run_research_task(task_id: str, request: ResearchRequest):
    """Run a research task in the background."""
    try:
        _research_tasks[task_id]["status"] = "running"
        _research_tasks[task_id]["updated_at"] = datetime.utcnow().isoformat()
        
        # Import research components
        from kosmos.orchestration.coordinator import ResearchCoordinator
        from kosmos.config import get_config
        
        config = get_config()
        
        # Create coordinator and run research
        coordinator = ResearchCoordinator(config)
        result = await coordinator.run_research_async(
            question=request.question,
            domain=request.domain,
            max_iterations=request.max_iterations
        )
        
        _research_tasks[task_id]["status"] = "completed"
        _research_tasks[task_id]["result"] = result
        _research_tasks[task_id]["updated_at"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        logger.error(f"Research task {task_id} failed: {e}")
        _research_tasks[task_id]["status"] = "failed"
        _research_tasks[task_id]["error"] = str(e)
        _research_tasks[task_id]["updated_at"] = datetime.utcnow().isoformat()


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the server using uvicorn."""
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    uvicorn.run(
        "kosmos.api.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=log_level
    )


if __name__ == "__main__":
    main()

