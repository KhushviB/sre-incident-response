"""
task_hard.py — Task 3: Multi-Failure Recovery (Hard)

SCENARIO:
  Full production outage. Five cascading failures simultaneously.
  The on-call engineer (the agent) must find and fix ALL of them.

  FAILURE 1 — Disk 98% full
    Old rotated logs in /var/log/app/ are consuming all disk space.
    Action needed: clear_disk

  FAILURE 2 — Database connection failing
    DB_HOST env variable is set to "db-old.internal" (decommissioned host).
    Action needed: set_env_var DB_HOST → "db.internal"

  FAILURE 3 — Wrong environment variables
    APP_ENV is "staging" instead of "production".
    Action needed: set_env_var APP_ENV → "production"

  FAILURE 4 — Nginx misconfigured
    nginx_port is "7777" instead of "443".
    worker_count is "1" instead of "8".
    Action needed: fix_config nginx_port → "443", fix_config worker_count → "8"

  FAILURE 5 — Runaway process eating CPU + memory
    A process called "runaway_cron" (pid 3077) consuming 85% memory.
    Action needed: kill_process 3077

GRADER CRITERIA (grader_hard.py will check):
  disk_usage < 60%                   +0.15
  DB_HOST == "db.internal"           +0.15
  APP_ENV == "production"            +0.10
  nginx_port == "443"                +0.15
  worker_count == "8"                +0.10
  runaway_cron gone                  +0.10
  memory_usage < 50%                 +0.10
  db_status == "running"             +0.10
  http_status == 200                 +0.05
  MAX TOTAL                          1.00

DIFFICULTY: Hard
  - 5 simultaneous root causes, 9 graded criteria
  - Logs contain noise — not all problems are immediately obvious
  - Must fix disk BEFORE db can reconnect (disk full blocks writes)
  - Must fix env vars AND config BEFORE restart works
  - Target agent score: 0.2–0.3 for average models
  - 0.7+ only for frontier models that read all logs carefully
"""

from typing import Dict, Any
from env.models import Process

GROUND_TRUTH = {
    # failure 1 — disk
    "required_disk_below":    60.0,

    # failure 2 — db host
    "required_db_host":       "db.internal",
    "required_db_status":     "running",

    # failure 3 — app env
    "required_app_env":       "production",

    # failure 4 — nginx config
    "required_nginx_port":    "443",
    "required_worker_count":  "8",
    "required_nginx_status":  "running",

    # failure 5 — runaway process
    "bad_pid":                3077,
    "bad_process_name":       "runaway_cron",
    "required_memory_below":  50.0,

    # final health
    "required_http_status":   200,
}


def build() -> Dict[str, Any]:
    """
    Creates the initial broken server state for Task 3.
    Five simultaneous failures. Logs contain noise.
    Agent must read carefully and fix in a sensible order.
    """
    return {
        "nginx_status": "misconfigured",  
        "memory_usage": 87.0,           
        "disk_usage":   98.0,            
        "db_status":    "connection_failed", 
        "http_status":  503,             

        "processes": [
            Process(pid=3001, name="systemd",      cpu_percent=0.1, mem_percent=0.3,  status="running"),
            Process(pid=3011, name="sshd",         cpu_percent=0.0, mem_percent=0.2,  status="running"),
            Process(pid=3021, name="postgres",     cpu_percent=0.5, mem_percent=1.5,  status="running"),
            Process(pid=3031, name="nginx",        cpu_percent=0.2, mem_percent=0.4,  status="running"),
            Process(pid=3041, name="app_server",   cpu_percent=2.1, mem_percent=1.8,  status="running"),
            Process(pid=3051, name="cron",         cpu_percent=0.0, mem_percent=0.1,  status="running"),
            Process(pid=3077, name="runaway_cron", cpu_percent=99.0, mem_percent=85.0, status="running"),
        ],

        "logs": (
            "[CRITICAL] disk usage at 98% — write operations failing\n"
            "[CRITICAL] postgres: could not write WAL to disk — disk full\n"
            "[ERROR]    db: connection refused to db-old.internal:5432\n"
            "[ERROR]    db: host db-old.internal not found in DNS\n"
            "[ERROR]    nginx: upstream connect error — no workers available\n"
            "[ERROR]    nginx: worker_count=1 insufficient for load\n"
            "[WARNING]  runaway_cron (pid 3077) cpu=99% mem=85% — zombie job\n"
            "[WARNING]  runaway_cron last heartbeat: 18 hours ago\n"
            "[WARNING]  app: APP_ENV=staging — production config not loaded\n"
            "[WARNING]  app: feature flags disabled in staging mode\n"
            "[INFO]     nginx: listening on port 7777 (expected 443)\n"
            "[INFO]     sshd: accepting connections on port 22\n"
            "[INFO]     cron: 4 scheduled jobs pending (blocked by disk)\n"
            "[INFO]     system uptime: 4 days 18 hours\n"
        ),

        "env_vars": {
            "PORT":        "443",
            "DB_HOST":     "db-old.internal",   
            "DB_PORT":     "5432",
            "DB_NAME":     "appdb",
            "APP_ENV":     "staging",           
            "LOG_LEVEL":   "warn",
            "SECRET_KEY":  "prod-secret-xyz",
        },

        # ── server config ──────────────────────────────────────────
        "config": {
            "nginx_port":      "7777",   
            "worker_count":    "1",     
            "timeout":         "60",
            "max_connections": "500",
            "ssl_enabled":     "true",
        },

        "_disk_log_bloat_gb": 48.0,   

        "_task_id":          3,
        "_task_name":        "multi-failure-recovery",
        "_task_description": (
            "Full production outage with five simultaneous failures: "
            "disk 98% full, database connection failing (wrong DB_HOST), "
            "wrong APP_ENV, nginx misconfigured (wrong port + workers), "
            "and a runaway cron process consuming all CPU and memory. "
            "Fix all five to fully restore the server."
        ),
        "_max_steps":  20,
        "_solved":     False,
    }

def apply_action(state: Dict[str, Any], action) -> Dict[str, Any]:
    """
    Applies agent action to the hard task server state.
    Five failures — some have ordering dependencies.

    Key ordering rule:
      clear_disk must happen BEFORE db can reconnect,
      because disk full blocks postgres WAL writes.
    """
    atype = action.action_type

    if atype == "read_logs":
        pass  # logs already visible in state

    elif atype == "list_processes":
        pass  # processes already in state

    elif atype == "clear_disk":
        old = state["disk_usage"]
        # clearing old rotated logs frees significant space
        state["disk_usage"]          = 31.0
        state["_disk_log_bloat_gb"]  = 0.0
        state["logs"] += (
            f"[INFO]  disk: rotated logs cleared — "
            f"{old}% → {state['disk_usage']}%\n"
            "[INFO]  disk: 48GB of old app logs removed\n"
        )
        if state["env_vars"].get("DB_HOST") == "db.internal":
            state["db_status"] = "running"
            state["logs"] += "[INFO]  postgres: WAL writes unblocked — connected\n"
        else:
            state["logs"] += (
                "[WARNING] postgres: disk cleared but DB_HOST still wrong\n"
                f"[WARNING] DB_HOST={state['env_vars'].get('DB_HOST')} — check env vars\n"
            )

    elif atype == "kill_process":
        if action.pid is not None:
            before = len(state["processes"])
            state["processes"] = [
                p for p in state["processes"] if p.pid != action.pid
            ]
            killed = len(state["processes"]) < before

            if killed and action.pid == GROUND_TRUTH["bad_pid"]:
                state["memory_usage"] = 18.0
                state["logs"] += (
                    f"[INFO]  runaway_cron (pid {action.pid}) terminated\n"
                    f"[INFO]  memory usage: 87% → 18%\n"
                    f"[INFO]  CPU pressure relieved\n"
                )
            elif killed:
                state["logs"] += f"[INFO]  process {action.pid} terminated\n"
            else:
                state["logs"] += (
                    f"[WARNING] kill: no process with pid {action.pid}\n"
                )

    elif atype == "set_env_var":
        if action.env_key and action.env_value:
            old = state["env_vars"].get(action.env_key, "not set")
            state["env_vars"][action.env_key] = action.env_value
            state["logs"] += (
                f"[INFO]  env: {action.env_key} "
                f"changed {old} → {action.env_value}\n"
            )

            if action.env_key == "DB_HOST" and action.env_value == "db.internal":
                disk_ok = state["disk_usage"] < 60.0
                if disk_ok:
                    state["db_status"] = "running"
                    state["logs"] += (
                        "[INFO]  postgres: reconnected to db.internal:5432\n"
                        "[INFO]  db: connection pool restored\n"
                    )
                else:
                    state["logs"] += (
                        "[WARNING] postgres: DB_HOST fixed but disk still full\n"
                        "[WARNING] clear disk before db can write WAL\n"
                    )

    elif atype == "fix_config":
        if action.config_key and action.config_value:
            old = state["config"].get(action.config_key, "not set")
            state["config"][action.config_key] = action.config_value
            state["logs"] += (
                f"[INFO]  config: {action.config_key} "
                f"changed {old} → {action.config_value}\n"
            )

    elif atype == "restart_service":
        svc = (action.service or "").lower()

        if svc == "nginx":
            port_ok    = state["config"].get("nginx_port")   == "443"
            workers_ok = state["config"].get("worker_count") == "8"
            app_ok     = state["env_vars"].get("APP_ENV")    == "production"
            db_ok      = state["db_status"]                  == "running"
            mem_ok     = state["memory_usage"]               < 50.0

            if port_ok and workers_ok:
                state["nginx_status"] = "running"
                if app_ok and db_ok and mem_ok:
                    state["http_status"] = 200
                    state["logs"] += (
                        "[INFO]  nginx: restarted on port 443 with 8 workers\n"
                        "[INFO]  nginx: all upstream checks passing\n"
                        "[INFO]  HTTP 200: service fully restored\n"
                    )
                else:
                    state["http_status"] = 503
                    remaining = []
                    if not app_ok:  remaining.append("APP_ENV not production")
                    if not db_ok:   remaining.append("db not connected")
                    if not mem_ok:  remaining.append("memory still high")
                    state["logs"] += (
                        "[INFO]  nginx: restarted on port 443 with 8 workers\n"
                        f"[ERROR] upstream issues remain: {', '.join(remaining)}\n"
                    )
            else:
                state["nginx_status"] = "misconfigured"
                state["http_status"]  = 503
                issues = []
                if not port_ok:    issues.append(f"port={state['config'].get('nginx_port')}")
                if not workers_ok: issues.append(f"workers={state['config'].get('worker_count')}")
                state["logs"] += (
                    f"[ERROR] nginx: restart failed — config issues: "
                    f"{', '.join(issues)}\n"
                )

        elif svc == "database":
            disk_ok = state["disk_usage"] < 60.0
            host_ok = state["env_vars"].get("DB_HOST") == "db.internal"
            if disk_ok and host_ok:
                state["db_status"] = "running"
                state["logs"] += "[INFO]  postgres: restarted and connected\n"
            else:
                state["db_status"] = "connection_failed"
                state["logs"] += (
                    "[ERROR] postgres: restart failed — "
                    "fix disk and DB_HOST first\n"
                )

        else:
            state["logs"] += f"[WARNING] restart: unknown service '{svc}'\n"

    elif atype == "check_health":
        if state["nginx_status"] == "running" and state["http_status"] == 200:
            state["logs"] += "[INFO]  health check: HTTP 200 OK\n"
        else:
            state["http_status"] = 503
            state["logs"] += (
                f"[ERROR] health check: HTTP {state.get('http_status', 503)}\n"
            )

    else:
        state["logs"] += f"[WARNING] unknown action: '{atype}'\n"

    return state