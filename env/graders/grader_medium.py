"""
grader_medium.py — Grader for Task 2 (Medium)

SCORING BREAKDOWN (must total exactly 1.0):
  +0.20  memory usage below 70%
  +0.10  app_worker process is gone
  +0.20  nginx_port config == "8080"
  +0.15  APP_MODE env == "production"
  +0.20  nginx_status == "running"
  +0.15  http_status == 200

DESIGN:
  - Partial credit on memory (proportional)
  - Each criterion fully independent
  - Same state always returns same score (deterministic)
  - Score only reaches 1.0 when ALL six criteria are met
"""

from typing import Dict, Any
from env.models import Reward
from env.tasks.task_medium import GROUND_TRUTH

WEIGHTS = {
    "memory_normal":    0.20,
    "bad_process_gone": 0.10,
    "nginx_port_fixed": 0.20,
    "app_mode_fixed":   0.15,
    "nginx_running":    0.20,
    "http_ok":          0.15,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

def score(state: Dict[str, Any], previous_score: float = 0.0) -> Reward:
    """
    Scores current server state for Task 2.

    Args:
        state:          Current server state dict
        previous_score: Score from last step (for is_improvement)

    Returns:
        Reward with value, reason, breakdown, is_improvement
    """
    breakdown = {}
    reasons   = []
    threshold = GROUND_TRUTH["required_memory_below"]
    mem       = state["memory_usage"]

    if mem < threshold:
        breakdown["memory_normal"] = WEIGHTS["memory_normal"]
        reasons.append(f"memory {mem:.1f}% normal (+0.20)")
    else:
        partial = max(0.0, (100.0 - mem) / (100.0 - threshold)) * WEIGHTS["memory_normal"]
        breakdown["memory_normal"] = round(partial, 4)
        reasons.append(f"memory {mem:.1f}% high (+{partial:.2f})")

    bad_pid  = GROUND_TRUTH["bad_pid"]
    bad_name = GROUND_TRUTH["bad_process_name"]
    pids     = {p.pid  for p in state["processes"]}
    names    = {p.name for p in state["processes"]}

    if bad_pid not in pids and bad_name not in names:
        breakdown["bad_process_gone"] = WEIGHTS["bad_process_gone"]
        reasons.append("app_worker terminated (+0.10)")
    else:
        breakdown["bad_process_gone"] = 0.0
        reasons.append("app_worker still running (0.00)")

    actual_port   = state["config"].get("nginx_port", "")
    required_port = GROUND_TRUTH["required_nginx_port"]

    if actual_port == required_port:
        breakdown["nginx_port_fixed"] = WEIGHTS["nginx_port_fixed"]
        reasons.append(f"nginx_port={actual_port} correct (+0.20)")
    else:
        breakdown["nginx_port_fixed"] = 0.0
        reasons.append(f"nginx_port={actual_port} wrong, need {required_port} (0.00)")

    actual_mode   = state["env_vars"].get("APP_MODE", "")
    required_mode = GROUND_TRUTH["required_app_mode"]

    if actual_mode == required_mode:
        breakdown["app_mode_fixed"] = WEIGHTS["app_mode_fixed"]
        reasons.append(f"APP_MODE={actual_mode} correct (+0.15)")
    else:
        breakdown["app_mode_fixed"] = 0.0
        reasons.append(f"APP_MODE={actual_mode} wrong, need {required_mode} (0.00)")

    nginx_status   = state["nginx_status"]
    required_nginx = GROUND_TRUTH["required_nginx_status"]

    if nginx_status == required_nginx:
        breakdown["nginx_running"] = WEIGHTS["nginx_running"]
        reasons.append("nginx running (+0.20)")
    else:
        breakdown["nginx_running"] = 0.0
        reasons.append(f"nginx {nginx_status} (0.00)")

    http_status = state.get("http_status")

    if http_status == GROUND_TRUTH["required_http_status"]:
        breakdown["http_ok"] = WEIGHTS["http_ok"]
        reasons.append("HTTP 200 (+0.15)")
    elif http_status == 500:
        breakdown["http_ok"] = 0.0
        reasons.append("HTTP 500 (0.00)")
    else:
        breakdown["http_ok"] = 0.0
        reasons.append(f"HTTP {http_status} not checked (0.00)")

    total = round(sum(breakdown.values()), 4)
    total = max(0.0, min(1.0, total))

    return Reward(
        value=total,
        reason=" | ".join(reasons),
        breakdown=breakdown,
        is_improvement=total > previous_score,
    )

def is_done(reward: Reward) -> bool:
    """Task 2 complete when all six criteria met (score == 1.0)."""
    return reward.value >= 1.0