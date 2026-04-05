"""
grader_easy.py — Grader for Task 1 (Easy)

WHAT THIS FILE DOES:
  Takes the current server state dict and returns a Reward object.
  Called by environment.py after every single step the agent takes.
  Always deterministic — same state always returns same score.
  Always gives partial credit — agent is rewarded for each fix.

SCORING BREAKDOWN (must total exactly 1.0):
  +0.10  bad process is gone from process list
  +0.30  memory usage is below 80%
  +0.40  nginx status is "running"
  +0.20  http_status is 200

  0.0  = nothing fixed yet
  0.5  = memory fixed + process killed, nginx still down
  1.0  = everything resolved, server fully healthy

DESIGN RULES:
  - Never returns the same score regardless of state (disqualification rule)
  - Never requires exact string matching on agent text — checks state dict only
  - Scores are additive — each criterion is independent
  - is_improvement=True when score went up from last step
"""

from typing import Dict, Any
from env.models import Reward
from env.tasks.task_easy import GROUND_TRUTH


WEIGHTS = {
    "bad_process_gone": 0.10,
    "memory_normal":    0.30,
    "nginx_running":    0.40,
    "http_ok":          0.20,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

def score(state: Dict[str, Any], previous_score: float = 0.0) -> Reward:
    """
    Scores the current server state for Task 1.

    Args:
        state:          Current server state dict from environment.py
        previous_score: Score from the last step (used to set is_improvement)

    Returns:
        Reward object with value, reason, breakdown, and is_improvement
    """

    breakdown = {}
    reasons = []

    bad_pid       = GROUND_TRUTH["bad_pid"]
    bad_name      = GROUND_TRUTH["bad_process_name"]
    running_pids  = {p.pid for p in state["processes"]}
    running_names = {p.name for p in state["processes"]}

    if bad_pid not in running_pids and bad_name not in running_names:
        breakdown["bad_process_gone"] = WEIGHTS["bad_process_gone"]
        reasons.append("bad process terminated (+0.10)")
    else:
        breakdown["bad_process_gone"] = 0.0
        reasons.append("bad process still running (0.00)")

    threshold = GROUND_TRUTH["required_memory_below"]
    mem        = state["memory_usage"]

    if mem < threshold:
        breakdown["memory_normal"] = WEIGHTS["memory_normal"]
        reasons.append(f"memory at {mem:.1f}% — normal (+0.30)")
    else:
        partial = max(0.0, (100.0 - mem) / (100.0 - threshold)) * WEIGHTS["memory_normal"]
        partial = round(partial, 4)
        breakdown["memory_normal"] = partial
        reasons.append(f"memory at {mem:.1f}% — still high (+{partial:.2f})")

    required_nginx = GROUND_TRUTH["required_nginx_status"]
    nginx_status   = state["nginx_status"]

    if nginx_status == required_nginx:
        breakdown["nginx_running"] = WEIGHTS["nginx_running"]
        reasons.append("nginx running (+0.40)")
    elif nginx_status == "stopped":
        breakdown["nginx_running"] = 0.0
        reasons.append("nginx stopped — needs restart (0.00)")
    elif nginx_status == "crashed":
        breakdown["nginx_running"] = 0.0
        reasons.append("nginx crashed — fix memory first (0.00)")
    elif nginx_status == "misconfigured":
        # misconfigured gets a tiny partial — at least it tried to start
        breakdown["nginx_running"] = round(WEIGHTS["nginx_running"] * 0.1, 4)
        reasons.append("nginx misconfigured — fix config (+0.04)")
    else:
        breakdown["nginx_running"] = 0.0
        reasons.append(f"nginx status unknown: {nginx_status} (0.00)")

    http_status      = state.get("http_status")
    required_http    = GROUND_TRUTH["required_http_status"]

    if http_status == required_http:
        breakdown["http_ok"] = WEIGHTS["http_ok"]
        reasons.append("HTTP 200 confirmed (+0.20)")
    elif http_status == 503:
        breakdown["http_ok"] = 0.0
        reasons.append("HTTP 503 — server unavailable (0.00)")
    elif http_status is None:
        breakdown["http_ok"] = 0.0
        reasons.append("health check not run yet (0.00)")
    else:
        breakdown["http_ok"] = 0.0
        reasons.append(f"HTTP {http_status} — unexpected response (0.00)")

    total = round(sum(breakdown.values()), 4)
    total = max(0.0, min(1.0, total))   # clamp to [0.0, 1.0]

    reason = " | ".join(reasons)
    is_improvement = total > previous_score

    return Reward(
        value=total,
        reason=reason,
        breakdown=breakdown,
        is_improvement=is_improvement,
    )

def is_done(reward: Reward) -> bool:
    """Returns True when the agent has fully resolved Task 1."""
    return reward.value >= 1.0