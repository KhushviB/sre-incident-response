"""
grader_hard.py — Grader for Task 3 (Hard)

SCORING BREAKDOWN (must total exactly 1.0):
  +0.15  disk_usage < 60%
  +0.15  DB_HOST env == "db.internal"
  +0.10  db_status == "running"
  +0.10  APP_ENV env == "production"
  +0.15  nginx_port config == "443"
  +0.10  worker_count config == "8"
  +0.10  runaway_cron process gone
  +0.10  memory_usage < 50%
  +0.05  http_status == 200

DESIGN:
  - 9 independent criteria — agent earns partial credit per fix
  - Partial credit on disk and memory (proportional)
  - Score at start is near zero — all 5 failures active
  - Score of 1.0 requires fixing ALL failures AND restarting nginx
  - Target agent score: 0.2–0.3 (average model), 0.7+ (frontier model)
"""

from typing import Dict, Any
from env.models import Reward
from env.tasks.task_hard import GROUND_TRUTH


WEIGHTS = {
    "disk_cleared":       0.15,
    "db_host_fixed":      0.15,
    "db_connected":       0.10,
    "app_env_fixed":      0.10,
    "nginx_port_fixed":   0.15,
    "worker_count_fixed": 0.10,
    "runaway_gone":       0.10,
    "memory_normal":      0.10,
    "http_ok":            0.05,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

def score(state: Dict[str, Any], previous_score: float = 0.0) -> Reward:
    """
    Scores current server state for Task 3.
    9 criteria, each independent, each earns its weight when met.

    Args:
        state:          Current server state dict
        previous_score: Score from last step (for is_improvement)

    Returns:
        Reward with value, reason, breakdown, is_improvement
    """
    breakdown = {}
    reasons   = []

    # ── criterion 1: disk below 60% ──────────────────────────────
    required_disk = GROUND_TRUTH["required_disk_below"]
    disk          = state["disk_usage"]

    if disk < required_disk:
        breakdown["disk_cleared"] = WEIGHTS["disk_cleared"]
        reasons.append(f"disk {disk:.0f}% — cleared (+0.15)")
    else:
        # proportional partial: at 98% → 0.0, at 60% → full
        partial = max(0.0, (100.0 - disk) / (100.0 - required_disk))
        partial = round(partial * WEIGHTS["disk_cleared"], 4)
        breakdown["disk_cleared"] = partial
        reasons.append(f"disk {disk:.0f}% — still full (+{partial:.2f})")

    actual_host   = state["env_vars"].get("DB_HOST", "")
    required_host = GROUND_TRUTH["required_db_host"]

    if actual_host == required_host:
        breakdown["db_host_fixed"] = WEIGHTS["db_host_fixed"]
        reasons.append(f"DB_HOST={actual_host} correct (+0.15)")
    else:
        breakdown["db_host_fixed"] = 0.0
        reasons.append(f"DB_HOST={actual_host} wrong (0.00)")

    db_status = state["db_status"]

    if db_status == GROUND_TRUTH["required_db_status"]:
        breakdown["db_connected"] = WEIGHTS["db_connected"]
        reasons.append("db connected (+0.10)")
    else:
        breakdown["db_connected"] = 0.0
        reasons.append(f"db {db_status} (0.00)")

    actual_env   = state["env_vars"].get("APP_ENV", "")
    required_env = GROUND_TRUTH["required_app_env"]

    if actual_env == required_env:
        breakdown["app_env_fixed"] = WEIGHTS["app_env_fixed"]
        reasons.append(f"APP_ENV={actual_env} correct (+0.10)")
    else:
        breakdown["app_env_fixed"] = 0.0
        reasons.append(f"APP_ENV={actual_env} wrong (0.00)")

    actual_port   = state["config"].get("nginx_port", "")
    required_port = GROUND_TRUTH["required_nginx_port"]

    if actual_port == required_port:
        breakdown["nginx_port_fixed"] = WEIGHTS["nginx_port_fixed"]
        reasons.append(f"nginx_port={actual_port} correct (+0.15)")
    else:
        breakdown["nginx_port_fixed"] = 0.0
        reasons.append(f"nginx_port={actual_port} wrong (0.00)")

    actual_workers   = state["config"].get("worker_count", "")
    required_workers = GROUND_TRUTH["required_worker_count"]

    if actual_workers == required_workers:
        breakdown["worker_count_fixed"] = WEIGHTS["worker_count_fixed"]
        reasons.append(f"worker_count={actual_workers} correct (+0.10)")
    else:
        breakdown["worker_count_fixed"] = 0.0
        reasons.append(f"worker_count={actual_workers} wrong (0.00)")

    bad_pid   = GROUND_TRUTH["bad_pid"]
    bad_name  = GROUND_TRUTH["bad_process_name"]
    pids      = {p.pid  for p in state["processes"]}
    names     = {p.name for p in state["processes"]}

    if bad_pid not in pids and bad_name not in names:
        breakdown["runaway_gone"] = WEIGHTS["runaway_gone"]
        reasons.append("runaway_cron terminated (+0.10)")
    else:
        breakdown["runaway_gone"] = 0.0
        reasons.append("runaway_cron still running (0.00)")

    required_mem = GROUND_TRUTH["required_memory_below"]
    mem          = state["memory_usage"]

    if mem < required_mem:
        breakdown["memory_normal"] = WEIGHTS["memory_normal"]
        reasons.append(f"memory {mem:.0f}% — normal (+0.10)")
    else:
        partial = max(0.0, (100.0 - mem) / (100.0 - required_mem))
        partial = round(partial * WEIGHTS["memory_normal"], 4)
        breakdown["memory_normal"] = partial
        reasons.append(f"memory {mem:.0f}% — high (+{partial:.2f})")

    http_status = state.get("http_status")

    if http_status == GROUND_TRUTH["required_http_status"]:
        breakdown["http_ok"] = WEIGHTS["http_ok"]
        reasons.append("HTTP 200 (+0.05)")
    elif http_status == 503:
        breakdown["http_ok"] = 0.0
        reasons.append("HTTP 503 (0.00)")
    else:
        breakdown["http_ok"] = 0.0
        reasons.append(f"HTTP {http_status} (0.00)")

    total = round(sum(breakdown.values()), 4)
    total = max(0.0, min(1.0, total))

    return Reward(
        value=total,
        reason=" | ".join(reasons),
        breakdown=breakdown,
        is_improvement=total > previous_score,
    )

def is_done(reward: Reward) -> bool:
    """Task 3 complete only when all 9 criteria are met (score == 1.0)."""
    return reward.value >= 1.0