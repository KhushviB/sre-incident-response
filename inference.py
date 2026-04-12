"""
inference.py — Baseline Agent for SRE Incident Response OpenEnv
"""

import os
import sys
import json
import textwrap
from typing import Dict, List, Optional, Any

from openai import OpenAI

# Direct import bypasses the need for local server running during inference
from env.environment import SREEnvironment
from env.models import Action

# =======================================================================
# EXACT BASE FORMAT AS EXPECTED BY THE HACKATHON VALIDATOR
# =======================================================================
# The validator injects these specific environment variables
api_base   = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
api_key    = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or "dummy-key"
model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


# =======================================================================

BENCHMARK = "sre-incident-response"
MAX_STEPS = 15
TEMPERATURE = 0.2
MAX_TOKENS = 300

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

Respond with ONLY the JSON. Nothing else. No explanation.
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def _obs_to_dict(obs) -> Dict[str, Any]:
    return {
        "message": getattr(obs, "message", ""),
        "nginx_status": getattr(obs, "nginx_status", ""),
        "memory_usage": getattr(obs, "memory_usage", 0.0),
        "disk_usage": getattr(obs, "disk_usage", 0.0),
        "db_status": getattr(obs, "db_status", ""),
        "http_status": getattr(obs, "http_status", None),
        "processes": [{"pid": getattr(p, "pid", 0), "name": getattr(p, "name", ""), "cpu_percent": getattr(p, "cpu_percent", 0.0), "mem_percent": getattr(p, "mem_percent", 0.0)} for p in getattr(obs, "processes", [])],
        "logs": getattr(obs, "logs", ""),
        "env_vars": getattr(obs, "env_vars", {}),
        "config": getattr(obs, "config", {}),
    }

def build_prompt(step: int, obs: Dict[str, Any], last_reward: float, last_improved: bool, history: List[str]) -> str:
    procs = "\n".join(f"  pid={p['pid']} name={p['name']} cpu={p['cpu_percent']}% mem={p['mem_percent']}%" for p in obs.get("processes", [])) or "  (empty)"
    envs = "\n".join(f"  {k}={v}" for k, v in obs.get("env_vars", {}).items()) or "  (empty)"
    cfgs = "\n".join(f"  {k}={v}" for k, v in obs.get("config", {}).items()) or "  (empty)"
    hist = "\n".join(f"  {h}" for h in history[-6:]) if history else "  (none yet)"

    note = "Last action IMPROVED the score — keep going." if last_improved else "Last action did NOT improve score — try something DIFFERENT."

    warnings = []
    if obs.get("env_vars", {}).get("APP_MODE") == "debug": warnings.append("  WARNING: APP_MODE=debug should be production")
    if obs.get("env_vars", {}).get("APP_ENV") == "staging": warnings.append("  WARNING: APP_ENV=staging should be production")
    if "db-old" in obs.get("env_vars", {}).get("DB_HOST", ""): warnings.append("  WARNING: DB_HOST is wrong, should be db.internal")
    if str(obs.get("config", {}).get("nginx_port")) not in ("80", "443", "8080", "None"): warnings.append("  WARNING: nginx_port looks wrong")
    if str(obs.get("config", {}).get("worker_count")) not in ("8", "None"): warnings.append("  WARNING: worker_count should be 8")
    if obs.get("disk_usage", 0) > 80: warnings.append(f"  WARNING: disk {obs['disk_usage']}% is full — run clear_disk")
    if obs.get("memory_usage", 0) > 70: warnings.append(f"  WARNING: memory {obs['memory_usage']}% is high — kill the biggest process")

    warning_block = "\n".join(warnings) if warnings else "  (none)"

    return textwrap.dedent(f"""
Step {step} | Last reward: {last_reward:.2f} | {note}

SERVER STATUS
  nginx_status : {obs.get('nginx_status')}
  memory_usage : {obs.get('memory_usage')}%
  disk_usage   : {obs.get('disk_usage')}%
  db_status    : {obs.get('db_status')}
  http_status  : {obs.get('http_status')}

ISSUES DETECTED:
{warning_block}

RUNNING PROCESSES
{procs}

ENVIRONMENT VARIABLES
{envs}

SERVER CONFIG
{cfgs}

RECENT LOGS (last 800 chars)
{str(obs.get('logs', ''))[-800:]}

ACTION HISTORY
{hist}

Respond with ONLY a JSON action.
""").strip()

def get_action(client: OpenAI, step: int, obs: Dict[str, Any], last_reward: float, last_improved: bool, history: List[str]) -> tuple:
    prompt = build_prompt(step, obs, last_reward, last_improved, history)
    
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
        
        if not raw.startswith("{"):
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start: raw = raw[start:end]

        return json.loads(raw), None

    except Exception as e:
        # We catch LLM errors here ONLY so the loop doesn't completely die on a timeout
        return {"action_type": "read_logs"}, f"LLM Error: {str(e)}"

def main() -> None:
    try:
        client = OpenAI(base_url=api_base, api_key=api_key)
    except Exception:
        try:
            client = OpenAI(base_url=api_base, api_key=api_key, http_client=None)
        except Exception:
            client = OpenAI(api_key=api_key)
    env = SREEnvironment()
    
    for task_id in [1, 2, 3]:
        task_names = {1: "memory-leak-fix", 2: "cascading-500-errors", 3: "multi-failure-recovery"}
        task_name = task_names.get(task_id, f"task-{task_id}")
        
        log_start(task=task_name, env=BENCHMARK, model=model_name)
        
        result = env.reset(task_id=task_id)
        obs_dict = _obs_to_dict(result.observation)

        rewards = []
        history = []
        last_reward = 0.0
        last_improved = False
        done = False
        success = False
        steps_taken = 0

        for step in range(1, MAX_STEPS + 1):
            if done: break

            action_dict, error = get_action(client, step, obs_dict, last_reward, last_improved, history)

            action_model = Action(
                action_type=action_dict.get("action_type", "read_logs"),
                pid=action_dict.get("pid"),
                service=action_dict.get("service"),
                config_key=action_dict.get("config_key"),
                config_value=action_dict.get("config_value"),
                env_key=action_dict.get("env_key"),
                env_value=action_dict.get("env_value"),
            )
            
            res = env.step(action_model)
            obs_dict = _obs_to_dict(res.observation)
            
            reward_val = res.reward.value
            last_improved = getattr(res.reward, "is_improvement", False)
            done = res.done

            rewards.append(reward_val)
            last_reward = reward_val
            steps_taken = step
            action_str = json.dumps(action_dict, separators=(",", ":"))

            log_step(step=step, action=action_str, reward=reward_val, done=done, error=error)
            history.append(f"step {step}: {action_str} -> reward {reward_val:.2f}")

            if done:
                success = reward_val >= 1.0
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(1e-6, min(score, 1 - 1e-6))
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()