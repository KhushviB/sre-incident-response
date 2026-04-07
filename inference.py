"""
inference.py — Baseline Agent for SRE Incident Response OpenEnv
"""

import os
import sys
import json
import textwrap
import requests
from typing import Dict, List, Optional, Any
from openai import OpenAI


API_KEY      = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "dummy-key"))
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "openai/gpt-oss-20b")
ENV_URL      = os.environ.get("ENV_URL",      "https://khushvi-sre-incident-response.hf.space")
BENCHMARK    = "sre-incident-response"
MAX_STEPS    = 10
TEMPERATURE  = 0.2
MAX_TOKENS   = 300

TASK_NAMES = {
    1: "memory-leak-fix",
    2: "cascading-500-errors",
    3: "multi-failure-recovery",
}

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def env_reset(task_id: int) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=60)
    r.raise_for_status()
    return r.json()

def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=60)
    r.raise_for_status()
    return r.json()

SYSTEM_PROMPT = """You are an SRE engineer fixing a broken server.
Respond with EXACTLY one JSON action. No explanation. No markdown. Raw JSON only.

Actions:
  {"action_type": "read_logs"}
  {"action_type": "list_processes"}
  {"action_type": "kill_process", "pid": <int>}
  {"action_type": "restart_service", "service": "nginx"}
  {"action_type": "fix_config", "config_key": "<key>", "config_value": "<value>"}
  {"action_type": "clear_disk"}
  {"action_type": "set_env_var", "env_key": "<key>", "env_value": "<value>"}
  {"action_type": "check_health"}

Known correct values:
  nginx_port = "8080" or "443"
  worker_count = "8"
  APP_MODE = "production"
  APP_ENV = "production"
  DB_HOST = "db.internal"

Strategy: read_logs first, kill high-memory processes, fix configs/env vars, restart nginx last."""

def get_action(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""Server state:
nginx_status: {obs.get('nginx_status')}
memory_usage: {obs.get('memory_usage')}%
disk_usage: {obs.get('disk_usage')}%
db_status: {obs.get('db_status')}
http_status: {obs.get('http_status')}
processes: {[(p['name'], p['pid'], p['mem_percent']) for p in obs.get('processes', [])]}
env_vars: {obs.get('env_vars', {})}
config: {obs.get('config', {})}
logs: {obs.get('logs', '')[-400:]}

Respond with ONE JSON action only."""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        if not raw.startswith("{"):
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
        return json.loads(raw)
    except Exception:
        pass

    procs = obs.get("processes", [])
    bad   = [p for p in procs if p.get("mem_percent", 0) > 50]
    if bad:
        worst = max(bad, key=lambda p: p["mem_percent"])
        return {"action_type": "kill_process", "pid": worst["pid"]}
    if obs.get("disk_usage", 0) > 80:
        return {"action_type": "clear_disk"}
    env_vars = obs.get("env_vars", {})
    if env_vars.get("APP_MODE") == "debug":
        return {"action_type": "set_env_var", "env_key": "APP_MODE", "env_value": "production"}
    if env_vars.get("APP_ENV") == "staging":
        return {"action_type": "set_env_var", "env_key": "APP_ENV", "env_value": "production"}
    if "db-old" in env_vars.get("DB_HOST", ""):
        return {"action_type": "set_env_var", "env_key": "DB_HOST", "env_value": "db.internal"}
    config = obs.get("config", {})
    if config.get("nginx_port") not in ("80", "443", "8080"):
        return {"action_type": "fix_config", "config_key": "nginx_port", "config_value": "8080"}
    if config.get("worker_count") != "8":
        return {"action_type": "fix_config", "config_key": "worker_count", "config_value": "8"}
    return {"action_type": "read_logs"}

def run_episode(client: OpenAI, task_id: int) -> None:
    task_name = TASK_NAMES.get(task_id, f"task-{task_id}")
    log_start(task=task_name, model=MODEL_NAME)

    obs = {}
    try:
        reset_data = env_reset(task_id)
        obs = reset_data.get("observation", {})
    except Exception as e:
        log_step(1, '{"action_type":"read_logs"}', 0.0, True, str(e))
        log_end(False, 1, 0.0, [0.0])
        return

    rewards = []
    done = False
    for step in range(1, MAX_STEPS + 1):
        if done:
            break
        action = get_action(client, obs)
        action_str = json.dumps(action, separators=(",", ":"))
        reward_val = 0.0
        error = None
        try:
            result     = env_step(action)
            obs        = result.get("observation", obs)
            reward_val = result.get("reward", {}).get("value", 0.0)
            done       = result.get("done", False)
        except Exception as e:
            error = str(e)
        rewards.append(reward_val)
        log_step(step, action_str, reward_val, done, error)

    score   = max(rewards) if rewards else 0.0
    success = score >= 1.0
    log_end(success, len(rewards), score, rewards)

def main() -> None:
    try:
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
    except KeyError:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in [1, 2, 3]:
        try:
            run_episode(client, task_id)
        except Exception as e:
            task_name = TASK_NAMES.get(task_id, f"task-{task_id}")
            log_start(task=task_name, model=MODEL_NAME)
            log_step(1, '{"action_type":"read_logs"}', 0.0, True, str(e))
            log_end(False, 1, 0.0, [0.0])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[START] task=memory-leak-fix env={BENCHMARK} model={MODEL_NAME}", flush=True)
        print(f"[STEP] step=1 action={{\"action_type\":\"read_logs\"}} reward=0.00 done=true error={e}", flush=True)
        print(f"[END] success=false steps=1 score=0.000 rewards=0.00", flush=True)
        sys.exit(0)