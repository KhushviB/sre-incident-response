"""
server.py — FastAPI HTTP server wrapping SREEnvironment

This is the ONLY file that knows about HTTP.
It is a thin wrapper around environment.py.
All real logic lives in environment.py.

ENDPOINTS:
  POST /reset        → starts a new episode, returns first observation
  POST /step         → agent sends action, gets observation + reward back
  GET  /state        → returns current server state snapshot
  GET  /health       → simple liveness check (judges ping this)
  GET  /tasks        → lists all 3 tasks with metadata

JUDGES WILL:
  1. POST /reset          to start the environment
  2. POST /step (repeat)  to run the agent
  3. GET  /state          to inspect what happened
  4. GET  /health         to verify the Space is live

RUN LOCALLY:
  pip install fastapi uvicorn
  python server.py
  → http://localhost:7860
"""

import os
import traceback
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import SREEnvironment, TASKS
from env.models import Action

app = FastAPI(
    title="SRE Incident Response — OpenEnv",
    description=(
        "An OpenEnv environment where an AI agent acts as an SRE engineer, "
        "diagnosing and fixing cascading server failures of increasing complexity."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SREEnvironment()

class ResetRequest(BaseModel):
    task_id: int = 1

class StepRequest(BaseModel):
    action_type: str
    pid: Optional[int] = None
    service: Optional[str] = None
    config_key: Optional[str] = None
    config_value: Optional[str] = None
    env_key: Optional[str] = None
    env_value: Optional[str] = None

def _obs_to_dict(obs) -> Dict[str, Any]:
    return {
        "message":      obs.message,
        "nginx_status": obs.nginx_status,
        "memory_usage": obs.memory_usage,
        "disk_usage":   obs.disk_usage,
        "db_status":    obs.db_status,
        "http_status":  obs.http_status,
        "processes": [
            {
                "pid":         p.pid,
                "name":        p.name,
                "cpu_percent": p.cpu_percent,
                "mem_percent": p.mem_percent,
                "status":      p.status,
            }
            for p in obs.processes
        ],
        "logs":        obs.logs,
        "env_vars":    obs.env_vars,
        "config":      obs.config,
        "step_number": obs.step_number,
        "task_id":     obs.task_id,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness check — must return 200 for validation to pass."""
    return {"status": "ok", "environment": "sre-incident-response"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """Lists all 3 tasks with metadata for judges."""
    return {
        "tasks": [
            {
                "task_id":     tid,
                "name":        cfg["name"],
                "description": cfg["description"],
                "difficulty":  cfg["difficulty"],
                "max_steps":   cfg["max_steps"],
            }
            for tid, cfg in TASKS.items()
        ]
    }


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    """
    Starts a new episode. Wipes old state, builds fresh broken server.
    Body (optional): { "task_id": 1 }   1=easy, 2=medium, 3=hard
    """
    try:
        result = env.reset(task_id=request.task_id)
        return {
            "task_id":          result.task_id,
            "task_name":        result.task_name,
            "task_description": result.task_description,
            "max_steps":        result.max_steps,
            "observation":      _obs_to_dict(result.observation),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    """
    Agent sends one action, gets observation + reward + done back.

    Body examples:
      { "action_type": "read_logs" }
      { "action_type": "list_processes" }
      { "action_type": "kill_process", "pid": 1042 }
      { "action_type": "restart_service", "service": "nginx" }
      { "action_type": "fix_config", "config_key": "port", "config_value": "80" }
      { "action_type": "clear_disk" }
      { "action_type": "set_env_var", "env_key": "DB_PORT", "env_value": "5432" }
      { "action_type": "check_health" }
    """
    try:
        action = Action(
            action_type=request.action_type,
            pid=request.pid,
            service=request.service,
            config_key=request.config_key,
            config_value=request.config_value,
            env_key=request.env_key,
            env_value=request.env_value,
        )
        result = env.step(action)
        return {
            "observation": _obs_to_dict(result.observation),
            "reward": {
                "value":          result.reward.value,
                "reason":         result.reward.reason,
                "breakdown":      result.reward.breakdown,
                "is_improvement": result.reward.is_improvement,
            },
            "done": result.done,
            "info": result.info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/state")
def state() -> Dict[str, Any]:
    """Full snapshot of current environment state for judges."""
    try:
        return env.state()
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )