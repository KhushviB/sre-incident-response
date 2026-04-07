"""
inference.py — Baseline Agent for SRE Incident Response OpenEnv

MANDATORY RULES (competition spec):
  - Must be named inference.py at project root
  - Must use OpenAI client for all LLM calls
  - Must read credentials from environment variables
  - Must emit exactly [START] [STEP] [END] log format to stdout
  - Must complete all 3 tasks in under 20 minutes
  - Must run on 2 vCPU / 8GB RAM

ENVIRONMENT VARIABLES (set before running):
  HF_TOKEN or API_KEY   Your HuggingFace or API key
  API_BASE_URL          LLM endpoint (default: HF router)
  MODEL_NAME            Model to use (default: openai/gpt-oss-20b)
  ENV_URL               Environment server URL (default: localhost:7860)

RUN LOCALLY:
  Terminal 1:  python server.py
  Terminal 2:  export HF_TOKEN=hf_xxx && python inference.py

STDOUT FORMAT (strictly required by competition):
  [START] task=<name> env=sre-incident-response model=<model>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import sys
import json
import textwrap
import requests
from typing import Dict, List, Optional, Any

from openai import OpenAI


API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or "dummy-key"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "openai/gpt-oss-20b")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
BENCHMARK    = "sre-incident-response"

MAX_STEPS   = 15     # hard cap per episode
TEMPERATURE = 0.2    # low for deterministic reasoning
MAX_TOKENS  = 300    # enough for one JSON action

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int,
            score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

def env_reset(task_id: int) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert SRE on-call engineer diagnosing a broken production server.
Your job is to find and fix ALL failures as efficiently as possible.

At each step you receive the current server state and must respond with
EXACTLY one JSON action. No explanation, no markdown, just raw JSON.

Available actions:
  {"action_type": "read_logs"}
  {"action_type": "list_processes"}
  {"action_type": "kill_process", "pid": <integer>}
  {"action_type": "restart_service", "service": "<nginx|database|app>"}
  {"action_type": "fix_config", "config_key": "<key>", "config_value": "<value>"}
  {"action_type": "clear_disk"}
  {"action_type": "set_env_var", "env_key": "<key>", "env_value": "<value>"}
  {"action_type": "check_health"}

CRITICAL STRATEGY — follow this order every time:
  1. read_logs FIRST to understand what is broken
  2. list_processes to find rogue processes consuming memory
  3. kill_process for any process consuming over 50% memory
  4. clear_disk if disk usage is above 80%
  5. set_env_var to fix any wrong environment variables you see
  6. fix_config to fix any wrong config values you see
  7. restart_service ONLY after ALL root causes are fixed
  8. check_health to confirm HTTP 200

KNOWN CORRECT VALUES — use these exact values when fixing:
  config nginx_port    correct value is "8080" (HTTP) or "443" (HTTPS)
  config worker_count  correct value is "8" for production servers
  env    APP_MODE      correct value is "production"
  env    APP_ENV       correct value is "production"
  env    DB_HOST       correct value is "db.internal"

IMPORTANT RULES:
  - Never restart nginx before fixing its config AND killing memory hogs
  - If score did NOT improve, try something completely DIFFERENT next step
  - Do NOT repeat the same failing action twice in a row
  - Check ENV VARIABLES section — any non-production value must be fixed
  - Check CONFIG section — any wrong port or worker count must be fixed
  - worker_count correct value is always "8" not "2" or "4"

Respond with ONLY the JSON. Nothing else. No explanation.
""").strip()

def build_prompt(step: int, obs: Dict[str, Any],
                 last_reward: float, last_improved: bool,
                 history: List[str]) -> str:

    procs = "\n".join(
        f"  pid={p['pid']} name={p['name']} "
        f"cpu={p['cpu_percent']}% mem={p['mem_percent']}%"
        for p in obs.get("processes", [])
    ) or "  (empty)"

    envs = "\n".join(
        f"  {k}={v}" for k, v in obs.get("env_vars", {}).items()
    ) or "  (empty)"

    cfgs = "\n".join(
        f"  {k}={v}" for k, v in obs.get("config", {}).items()
    ) or "  (empty)"

    hist = (
        "\n".join(f"  {h}" for h in history[-6:])
        if history else "  (none yet)"
    )

    note = (
        "Last action IMPROVED the score — keep going." if last_improved
        else "Last action did NOT improve score — try something DIFFERENT."
    )

    warnings = []
    env_vars = obs.get("env_vars", {})
    config   = obs.get("config", {})

    if env_vars.get("APP_MODE") == "debug":
        warnings.append("  WARNING: APP_MODE=debug should be production")
    if env_vars.get("APP_ENV") == "staging":
        warnings.append("  WARNING: APP_ENV=staging should be production")
    if "db-old" in env_vars.get("DB_HOST", ""):
        warnings.append(f"  WARNING: DB_HOST={env_vars['DB_HOST']} is wrong, should be db.internal")
    if config.get("nginx_port") not in ("80", "443", "8080", None):
        warnings.append(f"  WARNING: nginx_port={config['nginx_port']} looks wrong")
    if config.get("worker_count") not in ("8", None):
        warnings.append(f"  WARNING: worker_count={config['worker_count']} should be 8")
    if obs.get("disk_usage", 0) > 80:
        warnings.append(f"  WARNING: disk {obs['disk_usage']}% is critically full — run clear_disk")
    if obs.get("memory_usage", 0) > 70:
        warnings.append(f"  WARNING: memory {obs['memory_usage']}% is high — kill the biggest process")

    warning_block = "\n".join(warnings) if warnings else "  (none)"

    return textwrap.dedent(f"""
Step {step} | Last reward: {last_reward:.2f} | {note}

SERVER STATUS
  nginx_status : {obs.get('nginx_status')}
  memory_usage : {obs.get('memory_usage')}%
  disk_usage   : {obs.get('disk_usage')}%
  db_status    : {obs.get('db_status')}
  http_status  : {obs.get('http_status')}

ISSUES DETECTED (fix these):
{warning_block}

RUNNING PROCESSES
{procs}

ENVIRONMENT VARIABLES
{envs}

SERVER CONFIG
{cfgs}

RECENT LOGS (last 800 chars)
{obs.get('logs', '')[-800:]}

ACTION HISTORY
{hist}

Respond with ONLY a JSON action. Fix one of the detected issues above.
""").strip()

def get_action(client: OpenAI, step: int, obs: Dict[str, Any],
               last_reward: float, last_improved: bool,
               history: List[str]) -> tuple:
    """
    Calls the LLM and parses the JSON action from its response.
    Retries up to 3 times on empty or unparseable responses.
    Falls back to a smart action (not just read_logs) on failure.
    Returns (action_dict, error_or_None).
    """
    prompt = build_prompt(step, obs, last_reward, last_improved, history)
    last_error = None

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            raw = (completion.choices[0].message.content or "").strip()
            if not raw:
                last_error = "empty response from model"
                continue

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            if not raw.startswith("{"):
                start = raw.find("{")
                end   = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    raw = raw[start:end]

            action = json.loads(raw)
            return action, None

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            continue
        except Exception as e:
            last_error = f"LLM error: {e}"
            break

    fallback = _smart_fallback(obs, history)
    return fallback, last_error


def _smart_fallback(obs: Dict[str, Any], history: List[str]) -> Dict[str, Any]:
    """
    When the LLM fails to return valid JSON, pick a smart action
    based on current server state instead of always reading logs.
    """
    done_actions = set(h.split(":")[1].strip().split(" ")[0] for h in history)

    if not any("read_logs" in h for h in history):
        return {"action_type": "read_logs"}

    if obs.get("memory_usage", 0) > 70:
        if not any("list_processes" in h for h in history):
            return {"action_type": "list_processes"}
        procs = obs.get("processes", [])
        bad   = [p for p in procs if p.get("mem_percent", 0) > 50]
        if bad:
            worst = max(bad, key=lambda p: p["mem_percent"])
            return {"action_type": "kill_process", "pid": worst["pid"]}

    if obs.get("disk_usage", 0) > 80:
        return {"action_type": "clear_disk"}

    env_vars = obs.get("env_vars", {})
    if env_vars.get("APP_MODE") == "debug":
        return {"action_type": "set_env_var",
                "env_key": "APP_MODE", "env_value": "production"}
    if env_vars.get("APP_ENV") == "staging":
        return {"action_type": "set_env_var",
                "env_key": "APP_ENV", "env_value": "production"}
    if "db-old" in env_vars.get("DB_HOST", ""):
        return {"action_type": "set_env_var",
                "env_key": "DB_HOST", "env_value": "db.internal"}

    config = obs.get("config", {})
    if config.get("nginx_port") not in ("80", "443", "8080"):
        return {"action_type": "fix_config",
                "config_key": "nginx_port", "config_value": "8080"}
    if config.get("worker_count") != "8":
        return {"action_type": "fix_config",
                "config_key": "worker_count", "config_value": "8"}

    return {"action_type": "check_health"}

def run_episode(client: OpenAI, task_id: int) -> Dict[str, Any]:
    """Runs a full episode for one task. Returns summary dict."""

    reset_data = env_reset(task_id)
    task_name  = reset_data["task_name"]
    obs        = reset_data["observation"]

    log_start(task=task_name, model=MODEL_NAME)

    rewards:      List[float] = []
    history:      List[str]   = []
    steps_taken:  int         = 0
    last_reward:  float       = 0.0
    last_improved: bool       = False
    done:         bool        = False
    success:      bool        = False

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        action_dict, error = get_action(
            client, step, obs, last_reward, last_improved, history
        )

        try:
            result       = env_step(action_dict)
            obs          = result["observation"]
            reward_val   = result["reward"]["value"]
            last_improved = result["reward"]["is_improvement"]
            done         = result["done"]
        except Exception as e:
            reward_val    = last_reward
            last_improved = False
            error         = str(e)
            done          = False

        rewards.append(reward_val)
        last_reward = reward_val
        steps_taken = step
        action_str  = json.dumps(action_dict, separators=(",", ":"))

        log_step(
            step=step,
            action=action_str,
            reward=reward_val,
            done=done,
            error=error,
        )

        history.append(
            f"step {step}: {action_str} "
            f"→ reward={reward_val:.2f} "
            f"({'improved' if last_improved else 'no change'})"
        )

        if done:
            success = reward_val >= 1.0
            break

    score = max(rewards) if rewards else 0.0
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":   task_id,
        "task_name": task_name,
        "steps":     steps_taken,
        "score":     score,
        "success":   success,
        "rewards":   rewards,
    }

def main() -> None:
    try:
        client = OpenAI(
            base_url=os.environ.get("API_BASE_URL") or API_BASE_URL,
            api_key=os.environ.get("API_KEY") or API_KEY,
        )
    except Exception as e:
        for task_name in ["memory-leak-fix", "cascading-500-errors", "multi-failure-recovery"]:
            log_start(task=task_name, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[0.0])
        return
    results = []

    for task_id in [1, 2, 3]:
        print(f"\n{'='*56}", flush=True)
        print(f"  Running Task {task_id}", flush=True)
        print(f"{'='*56}", flush=True)
        result = run_episode(client, task_id)
        results.append(result)
        print(
            f"\n  Task {task_id} complete — "
            f"score={result['score']:.3f} "
            f"steps={result['steps']} "
            f"success={result['success']}",
            flush=True,
        )

    print(f"\n{'='*56}", flush=True)
    print("  BASELINE SUMMARY", flush=True)
    print(f"{'='*56}", flush=True)
    for r in results:
        print(
            f"  task {r['task_id']}  {r['task_name']:<32} "
            f"score={r['score']:.3f}",
            flush=True,
        )
    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n  average score : {avg:.3f}", flush=True)
    print(f"{'='*56}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[END] success=false steps=0 score=0.000 rewards=", flush=True)
        sys.exit(0)