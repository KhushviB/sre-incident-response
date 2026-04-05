"""
task_medium.py — Task 2: Cascading 500 Errors (Medium)

SCENARIO:
  The server is throwing HTTP 500 errors in production.
  Three things are wrong simultaneously:
    1. A memory-leaking app process (app_worker, pid 2031) consuming 78% RAM
    2. The nginx config has the wrong port (9999 instead of 8080)
    3. The app environment variable APP_MODE is set to "debug" not "production"

  Unlike Task 1, the agent must:
    - Read logs carefully to find multiple root causes
    - Fix issues in the right order (config before restart)
    - Know which service to restart after which fix

WHAT THE AGENT MUST DO:
  1. read_logs          → see 500 errors and memory warnings
  2. list_processes     → find app_worker pid 2031 eating 78% RAM
  3. kill_process 2031  → kill the leaking worker
  4. fix_config         → set nginx_port from 9999 to 8080
  5. set_env_var        → set APP_MODE to production
  6. restart_service    → restart nginx
  7. check_health       → verify HTTP 200

GRADER CRITERIA (grader_medium.py will check these):
  memory under 70%              +0.20
  app_worker process gone       +0.10
  nginx_port config == 8080     +0.20
  APP_MODE env == production    +0.15
  nginx_status == running       +0.20
  http_status == 200            +0.15
  MAX TOTAL                     1.00

DIFFICULTY: Medium
  - 3 simultaneous root causes
  - Agent must diagnose before acting
  - Wrong order = partial results
  - Target agent score: ~0.5
"""

from typing import Dict, Any
from env.models import Process


GROUND_TRUTH = {
    "bad_pid":               2031,
    "bad_process_name":      "app_worker",
    "required_memory_below": 70.0,
    "required_nginx_port":   "8080",
    "required_app_mode":     "production",
    "required_nginx_status": "running",
    "required_http_status":  200,
}


def build() -> Dict[str, Any]:
    """
    Creates the initial broken server state for Task 2.
    Three simultaneous failures the agent must find and fix.
    """
    return {
        "nginx_status": "running",   
        "memory_usage": 78.0,        
        "disk_usage":   38.0,       
        "db_status":    "running",   
        "http_status":  500,      

        "processes": [
            Process(pid=2001, name="systemd",    cpu_percent=0.1, mem_percent=0.4,  status="running"),
            Process(pid=2011, name="sshd",       cpu_percent=0.0, mem_percent=0.2,  status="running"),
            Process(pid=2021, name="postgres",   cpu_percent=1.1, mem_percent=2.0,  status="running"),
            Process(pid=2031, name="app_worker", cpu_percent=8.2, mem_percent=78.0, status="running"),
            Process(pid=2041, name="nginx",      cpu_percent=0.3, mem_percent=0.5,  status="running"),
            Process(pid=2051, name="cron",       cpu_percent=0.0, mem_percent=0.1,  status="running"),
        ],

        "logs": (
            "[ERROR]   HTTP 500: upstream connect error on port 9999\n"
            "[ERROR]   nginx: no response from upstream app:9999\n"
            "[WARNING] app_worker (pid 2031) memory usage at 78% and growing\n"
            "[WARNING] app_worker has not released memory in 4 hours\n"
            "[ERROR]   app: running in DEBUG mode — not safe for production\n"
            "[ERROR]   app: APP_MODE=debug disables connection pooling\n"
            "[INFO]    nginx: configured to proxy to localhost:9999\n"
            "[INFO]    postgres: connection pool healthy (32 connections)\n"
        ),

        "env_vars": {
            "PORT":        "8080",
            "ENVIRONMENT": "production",
            "DB_HOST":     "localhost",
            "DB_PORT":     "5432",
            "APP_MODE":    "debug",       
            "LOG_LEVEL":   "debug",
        },

        "config": {
            "nginx_port":      "9999",   
            "worker_count":    "4",
            "timeout":         "30",
            "max_connections": "1000",
            "proxy_pass":      "localhost:9999",
        },

        "_task_id":          2,
        "_task_name":        "cascading-500-errors",
        "_task_description": (
            "The server is throwing HTTP 500 errors. "
            "A memory-leaking worker, wrong nginx port config, and wrong APP_MODE "
            "env variable are causing the outage. Fix all three to restore service."
        ),
        "_max_steps":  15,
        "_solved":     False,
    }

def apply_action(state: Dict[str, Any], action) -> Dict[str, Any]:
    """
    Applies the agent's action to the medium task server state.
    Three bugs to fix — each requiring a different action type.
    """
    atype = action.action_type

    if atype == "read_logs":
        pass

    elif atype == "list_processes":
        pass

    elif atype == "kill_process":
        if action.pid is not None:
            before = len(state["processes"])
            state["processes"] = [
                p for p in state["processes"] if p.pid != action.pid
            ]
            killed = len(state["processes"]) < before

            if killed and action.pid == GROUND_TRUTH["bad_pid"]:
                state["memory_usage"] = 22.0
                state["logs"] += (
                    f"[INFO]  app_worker (pid {action.pid}) terminated\n"
                    f"[INFO]  memory usage dropped to 22%\n"
                )
            elif killed:
                state["logs"] += f"[INFO]  process {action.pid} terminated\n"
            else:
                state["logs"] += f"[WARNING] kill: no process with pid {action.pid}\n"

    elif atype == "restart_service":
        svc = (action.service or "").lower()

        if svc == "nginx":
            port_ok = state["config"].get("nginx_port") == "8080"
            app_ok  = state["env_vars"].get("APP_MODE") == "production"
            mem_ok  = state["memory_usage"] < 70.0

            state["nginx_status"] = "running"

            if port_ok and app_ok and mem_ok:
                state["http_status"] = 200
                state["logs"] += (
                    "[INFO]  nginx: restarted on port 8080\n"
                    "[INFO]  app: production mode active\n"
                    "[INFO]  HTTP 200: service restored\n"
                )
            elif port_ok:
                state["http_status"] = 500
                state["logs"] += (
                    "[INFO]  nginx: restarted on port 8080\n"
                    "[ERROR] app: still returning 500 — "
                    "check APP_MODE and memory\n"
                )
            else:
                state["http_status"] = 500
                state["logs"] += (
                    f"[INFO]  nginx: restarted but still on port "
                    f"{state['config'].get('nginx_port')}\n"
                    "[ERROR] fix nginx_port config first\n"
                )

        elif svc == "app":
            app_ok  = state["env_vars"].get("APP_MODE") == "production"
            port_ok = state["config"].get("nginx_port") == "8080"
            mem_ok  = state["memory_usage"] < 70.0

            if app_ok and port_ok and mem_ok:
                state["http_status"] = 200
                state["logs"] += (
                    "[INFO]  app: restarted in production mode\n"
                    "[INFO]  HTTP 200: service restored\n"
                )
            else:
                state["logs"] += (
                    "[WARNING] app: restarted but issues remain\n"
                )

        elif svc == "database":
            state["db_status"] = "running"
            state["logs"] += "[INFO]  postgres: already running\n"

        else:
            state["logs"] += f"[WARNING] restart: unknown service '{svc}'\n"

    elif atype == "fix_config":
        if action.config_key and action.config_value:
            old = state["config"].get(action.config_key, "not set")
            state["config"][action.config_key] = action.config_value
            state["logs"] += (
                f"[INFO]  config: {action.config_key} "
                f"changed {old} → {action.config_value}\n"
            )
            if action.config_key == "nginx_port":
                state["config"]["proxy_pass"] = f"localhost:{action.config_value}"

    elif atype == "set_env_var":
        if action.env_key and action.env_value:
            old = state["env_vars"].get(action.env_key, "not set")
            state["env_vars"][action.env_key] = action.env_value
            state["logs"] += (
                f"[INFO]  env: {action.env_key} "
                f"changed {old} → {action.env_value}\n"
            )

    elif atype == "clear_disk":
        old = state["disk_usage"]
        state["disk_usage"] = max(old - 5.0, 10.0)
        state["logs"] += (
            f"[INFO]  disk: cleared, "
            f"{old}% → {state['disk_usage']}%\n"
        )

    elif atype == "check_health":
        if state["nginx_status"] == "running" and state["http_status"] == 200:
            state["logs"] += "[INFO]  health check: HTTP 200 OK\n"
        else:
            state["http_status"] = 500
            state["logs"] += "[ERROR] health check: HTTP 500\n"

    else:
        state["logs"] += f"[WARNING] unknown action: '{atype}'\n"

    return state