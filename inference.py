"""
inference.py — Baseline Agent for SRE Incident Response OpenEnv
"""

import os
import sys
import json
import textwrap
from typing import Dict, List, Optional, Any

from openai import OpenAI

# 1. DIRECT IMPORT: Bypass HTTP requests completely to prevent Connection Refused errors!
from env.environment import SREEnvironment
from env.models import Action

# 2. EXACT MATCH WITH SAMPLE SCRIPT ENV VARS
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

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

# 3. STRICT STDOUT FORMATTING
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # ADDED score= field which was missing in your original!
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def _obs_to_dict(obs) -> Dict[str, Any]:
    return {
        "message": obs.message,
        "nginx_status": obs.nginx_status,
        "memory_usage": obs.memory_usage,
        "disk_usage": obs.disk_usage,
        "db_status": obs.db_status,
        "http_status": obs.http_status,
        "processes": [{"pid": p.pid, "name": p.name, "cpu_percent": p.cpu_percent, "mem_percent": p.mem_percent, "status": p.status} for p in obs.processes],
        "logs": obs.logs,
        "env_vars": obs.env_vars,
        "config": obs.config,
        "step_number": obs.step_number,
        "task_id": obs.task_id,
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
    if obs.get("config", {}).get("nginx_port") not in ("80", "443", "8080", None): warnings.append("  WARNING: nginx_port looks wrong")
    if obs.get("config", {}).get("worker_count") not in ("8", None): warnings.append("  WARNING: worker_count should be 8")
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
{obs.get('logs', '')[-800:]}

ACTION HISTORY
{hist}

Respond with ONLY a JSON action.
""").strip()

def get_action(client: OpenAI, step: int, obs: Dict[str, Any], last_reward: float, last_improved: bool, history: List[str]) -> tuple:
    prompt = build_prompt(step, obs, last_reward, last_improved, history)
    last_error = None

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
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
            last_error = str(e)

    # Fallback if parsing fails heavily
    return {"action_type": "read_logs"}, last_error

def run_episode(client: OpenAI, task_id: int) -> None:
    env = SREEnvironment() # Instantiated locally!
    task_names = {1: "memory-leak-fix", 2: "cascading-500-errors", 3: "multi-failure-recovery"}
    task_name = task_names.get(task_id, f"task-{task_id}")
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        result = env.reset(task_id=task_id)
        obs_dict = _obs_to_dict(result.observation)
    except Exception as e:
        log_step(1, '{"action_type":"read_logs"}', 0.0, True, str(e))
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        return

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

        try:
            # Parse dict into Pydantic model natively
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
            last_improved = res.reward.is_improvement
            done = res.done
        except Exception as e:
            reward_val = last_reward
            last_improved = False
            error = str(e)
            done = False

        rewards.append(reward_val)
        last_reward = reward_val
        steps_taken = step
        action_str = json.dumps(action_dict, separators=(",", ":"))

        log_step(step=step, action=action_str, reward=reward_val, done=done, error=error)
        history.append(f"step {step}: {action_str} -> reward {reward_val:.2f}")

        if done:
            success = reward_val >= 1.0
            break

    score = max(rewards) if rewards else 0.0
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

def main() -> None:
    if not API_KEY:
        print("[FATAL ERROR] API_KEY or HF_TOKEN environment variable is required", file=sys.stderr)
        sys.exit(1)
        
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    for task_id in [1, 2, 3]:
        run_episode(client, task_id)

if __name__ == "__main__":
    main()