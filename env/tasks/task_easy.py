"""
task_easy.py — Task 1: Single Failure (Easy)

SCENARIO:
  The on-call alert fires at 2am.
  One rogue process called "memory_hog" has consumed 95% of server memory.
  As a result nginx has crashed and the server is returning no response.

WHAT THE AGENT MUST DO:
  1. read_logs       → understand what is wrong
  2. list_processes  → find the bad process and its PID
  3. kill_process    → kill pid 1042 (memory_hog)
  4. restart_service → restart nginx
  5. check_health    → verify server returns HTTP 200

GRADER CRITERIA (grader_easy.py will check these):
  nginx_status == "running"     → +0.40
  memory_usage < 80             → +0.30
  http_status == 200            → +0.20
  leaky process is gone         → +0.10
  MAX TOTAL                     → 1.00

DIFFICULTY: Easy
  - Only 1 root cause
  - Bad process is clearly named "memory_hog"
  - Logs directly mention the process
  - Can be solved in 3-5 steps
  - Target agent score: ~0.8
"""

from typing import Dict, Any
from env.models import Process

GROUND_TRUTH = {
    "bad_pid": 1042,
    "bad_process_name": "memory_hog",
    "required_nginx_status": "running",
    "required_memory_below": 80.0,
    "required_http_status": 200,
}

def build() -> Dict[str, Any]:
    """
    Creates and returns the initial fake broken server state for Task 1.
    This is a plain Python dict — no real server, no Docker, no OS calls.
    environment.py stores this dict and mutates it as the agent acts.
    """
    return {

        "nginx_status": "crashed",
        "memory_usage": 95.0,     
        "disk_usage": 42.0,      
        "db_status": "running",    
        "http_status": None,   

        "processes": [
            Process(pid=1001, name="systemd",     cpu_percent=0.1, mem_percent=0.5,  status="running"),
            Process(pid=1012, name="sshd",        cpu_percent=0.0, mem_percent=0.2,  status="running"),
            Process(pid=1021, name="postgres",    cpu_percent=1.2, mem_percent=2.1,  status="running"),
            Process(pid=1042, name="memory_hog",  cpu_percent=4.5, mem_percent=92.0, status="running"),
            Process(pid=1055, name="cron",        cpu_percent=0.0, mem_percent=0.1,  status="running"),
        ],
        "logs": (
            "[CRITICAL] nginx: worker process crashed (signal 9)\n"
            "[ERROR]    nginx: failed to allocate memory for worker\n"
            "[ERROR]    system memory exhausted: usage at 95%\n"
            "[WARNING]  process memory_hog (pid 1042) consuming 92% RAM\n"
            "[WARNING]  memory_hog has been running for 6h with no owner\n"
            "[INFO]     postgres: connection pool healthy\n"
            "[INFO]     sshd: accepting connections\n"
            "[INFO]     last nginx restart: 6 hours ago\n"
        ),

        "env_vars": {
            "PORT":        "80",
            "ENVIRONMENT": "production",
            "DB_HOST":     "localhost",
            "DB_PORT":     "5432",
            "LOG_LEVEL":   "info",
        },

        "config": {
            "nginx_port":     "80",
            "worker_count":   "4",
            "timeout":        "30",
            "max_connections": "1000",
        },

        "_task_id": 1,
        "_task_name": "memory-leak-fix",
        "_task_description": (
            "A rogue process has consumed all server memory causing nginx to crash. "
            "Identify the bad process, kill it, restart nginx, and verify the server is healthy."
        ),
        "_max_steps": 10,
        "_solved": False,
    }

def apply_action(state: Dict[str, Any], action) -> Dict[str, Any]:
    """
    Takes the current server state and an Action.
    Returns the updated server state after the action is applied.
    This is the fake "operating system" responding to agent commands.
    """

    atype = action.action_type

    if atype == "read_logs":
        pass 

    elif atype == "list_processes":
        pass 

    elif atype == "kill_process":
        if action.pid is not None:
            before_count = len(state["processes"])
            state["processes"] = [
                p for p in state["processes"]
                if p.pid != action.pid
            ]
            killed = len(state["processes"]) < before_count

            if killed and action.pid == GROUND_TRUTH["bad_pid"]:
                state["memory_usage"] = 18.0   # back to healthy level
                state["logs"] += (
                    f"[INFO]  process {action.pid} (memory_hog) terminated\n"
                    f"[INFO]  memory usage dropped to 18%\n"
                )
            elif killed:
                state["logs"] += f"[INFO]  process {action.pid} terminated\n"
            else:
                state["logs"] += f"[WARNING] kill: no process found with pid {action.pid}\n"

    elif atype == "restart_service":
        svc = (action.service or "").lower()

        if svc == "nginx":
            if state["memory_usage"] < 80:
                state["nginx_status"] = "running"
                state["http_status"] = 200
                state["logs"] += (
                    "[INFO]  nginx: started successfully\n"
                    "[INFO]  nginx: accepting connections on port 80\n"
                )
            else:
                state["nginx_status"] = "crashed"
                state["http_status"] = 503
                state["logs"] += (
                    "[ERROR] nginx: failed to start — insufficient memory\n"
                    "[ERROR] nginx: worker crashed immediately on startup\n"
                )

        elif svc == "database":
            state["db_status"] = "running"
            state["logs"] += "[INFO]  postgres: already running, restart skipped\n"

        else:
            state["logs"] += f"[WARNING] restart: unknown service '{svc}'\n"

    elif atype == "fix_config":
        if action.config_key and action.config_value:
            state["config"][action.config_key] = action.config_value
            state["logs"] += (
                f"[INFO]  config: {action.config_key} set to {action.config_value}\n"
            )

    elif atype == "clear_disk":
        old = state["disk_usage"]
        state["disk_usage"] = max(old - 5.0, 10.0)
        state["logs"] += f"[INFO]  disk: old logs cleared, usage {old}% → {state['disk_usage']}%\n"

    elif atype == "set_env_var":
        if action.env_key and action.env_value:
            state["env_vars"][action.env_key] = action.env_value
            state["logs"] += (
                f"[INFO]  env: {action.env_key} set to {action.env_value}\n"
            )

    elif atype == "check_health":
        if state["nginx_status"] == "running":
            state["http_status"] = 200
            state["logs"] += "[INFO]  health check: HTTP 200 OK\n"
        else:
            state["http_status"] = 503
            state["logs"] += "[ERROR] health check: HTTP 503 Service Unavailable\n"

    else:
        state["logs"] += f"[WARNING] unknown action type: '{atype}'\n"

    return state