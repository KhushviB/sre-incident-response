"""
models.py — OpenEnv typed models for SRE Incident Response environment.

Three core models that every other file imports:
  Action      → what the agent sends IN to the environment
  Observation → what the agent gets BACK after each step
  Reward      → the score + explanation after each action

Action types the agent can use:
  read_logs         → inspect current log output (no state change)
  list_processes    → see all running processes and memory usage
  kill_process      → terminate a process by pid
  restart_service   → restart nginx, database, or app
  fix_config        → change a config key to a new value
  clear_disk        → delete old log files to free disk space
  set_env_var       → set an environment variable to a correct value
  check_health      → request a health check (returns HTTP status)
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Action(BaseModel):
    """
    Structured action the agent takes each step.
    action_type must be one of the 8 valid types above.
    """
    action_type: str = Field(
        ...,
        description=(
            "One of: read_logs, list_processes, kill_process, "
            "restart_service, fix_config, clear_disk, set_env_var, check_health"
        )
    )
    pid: Optional[int] = Field(
        None,
        description="Process ID to kill. Required when action_type=kill_process."
    )
    service: Optional[str] = Field(
        None,
        description="Service name to restart (nginx, database, app). Required when action_type=restart_service."
    )
    config_key: Optional[str] = Field(
        None,
        description="Config key to change. Required when action_type=fix_config."
    )
    config_value: Optional[str] = Field(
        None,
        description="New value for config key. Required when action_type=fix_config."
    )
    env_key: Optional[str] = Field(
        None,
        description="Environment variable name. Required when action_type=set_env_var."
    )
    env_value: Optional[str] = Field(
        None,
        description="Value to set for the env variable. Required when action_type=set_env_var."
    )


class Process(BaseModel):
    """Represents one running process on the fake server."""
    pid: int
    name: str
    cpu_percent: float
    mem_percent: float
    status: str = "running"

class Observation(BaseModel):
    """
    Everything the agent can see about the current server state.
    Returned by reset() and step().
    """
    message: str = Field(
        ...,
        description="Human-readable description of the current situation."
    )

    nginx_status: str = Field(
        ...,
        description="Current nginx status: running | stopped | crashed | misconfigured"
    )
    memory_usage: float = Field(
        ...,
        description="Overall server memory usage as a percentage (0–100)."
    )
    disk_usage: float = Field(
        ...,
        description="Disk usage as a percentage (0–100)."
    )
    db_status: str = Field(
        ...,
        description="Database status: running | stopped | connection_failed"
    )
    http_status: Optional[int] = Field(
        None,
        description="Last HTTP health check response code. None if not yet checked."
    )

    processes: List[Process] = Field(
        default_factory=list,
        description="List of currently running processes."
    )
    logs: str = Field(
        ...,
        description="Most recent server log output the agent can read."
    )
    env_vars: Dict[str, str] = Field(
        default_factory=dict,
        description="Current environment variables on the server."
    )
    config: Dict[str, str] = Field(
        default_factory=dict,
        description="Current server config values (e.g. port, worker_count)."
    )

    step_number: int = Field(
        0,
        description="Which step of the episode this observation is from."
    )
    task_id: int = Field(
        ...,
        description="Which task is being run: 1 (easy), 2 (medium), 3 (hard)."
    )

class Reward(BaseModel):
    """
    Score the agent receives after each action.
    value is always between 0.0 and 1.0.
    breakdown shows exactly which fixes earned which points.
    """
    value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score for this step. 0.0 = nothing fixed, 1.0 = fully resolved."
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of what contributed to this score."
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-criterion scores. E.g. {'nginx_running': 0.5, 'memory_ok': 0.3, 'http_200': 0.2}"
        )
    )
    is_improvement: bool = Field(
        False,
        description="True if this action improved the score compared to the previous step."
    )

class StepResult(BaseModel):
    """
    Full result returned by environment.step().
    Wraps observation + reward + episode status together.
    """
    observation: Observation
    reward: Reward
    done: bool = Field(
        False,
        description="True when the episode is complete (server fully healthy or max steps reached)."
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata (steps taken, task id, score history, etc)."
    )

class ResetResult(BaseModel):
    """
    Result returned by environment.reset().
    Contains the first observation and task metadata.
    """
    observation: Observation
    task_id: int
    task_name: str
    task_description: str
    max_steps: int = 10