---
title: SRE Incident Response
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - sre
  - reinforcement-learning
---

# SRE Incident Response — OpenEnv

An [OpenEnv](https://huggingface.co/openenv) environment where an AI agent acts
as an on-call SRE engineer, diagnosing and fixing cascading production server
failures of increasing complexity.

The agent interacts with a **simulated Linux server** through structured actions
(kill processes, fix configs, set env vars, restart services) and receives
**partial-credit rewards** after every step based on how healthy the server is.

---

## Why This Environment

Site Reliability Engineering is one of the highest-stakes, highest-skill jobs
in software. When a production server goes down at 2am, an SRE must:

- Read noisy logs and find the real root cause
- Fix issues in the right order (some fixes depend on others)
- Restart only after root causes are addressed
- Verify the fix actually worked

This environment tests exactly that reasoning chain — not pattern matching,
not retrieval, but genuine multi-step causal diagnosis under pressure.

---

## Environment Description

### Domain
SRE incident response — diagnosing and fixing broken production servers.

### Fake Server
The environment maintains a **Python dictionary** that represents a server's
state. No real OS, no real Docker-in-Docker. The agent interacts with it
through a FastAPI HTTP API that returns structured observations.

### Action Space

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `read_logs` | — | Read current server log output |
| `list_processes` | — | List running processes with CPU/memory |
| `kill_process` | `pid` | Terminate a process by PID |
| `restart_service` | `service` | Restart nginx, database, or app |
| `fix_config` | `config_key`, `config_value` | Change a server config value |
| `clear_disk` | — | Delete old log files to free disk space |
| `set_env_var` | `env_key`, `env_value` | Set an environment variable |
| `check_health` | — | Run HTTP health check, get response code |

**Action format** (JSON sent to `/step`):
```json
{ "action_type": "kill_process", "pid": 1042 }
{ "action_type": "fix_config", "config_key": "nginx_port", "config_value": "8080" }
{ "action_type": "restart_service", "service": "nginx" }
```

### Observation Space

Every step returns a structured observation:

```json
{
  "message":      "Alert received. Server is unhealthy.",
  "nginx_status": "crashed",
  "memory_usage": 95.0,
  "disk_usage":   42.0,
  "db_status":    "running",
  "http_status":  null,
  "processes": [
    { "pid": 1042, "name": "memory_hog", "cpu_percent": 4.5, "mem_percent": 92.0 }
  ],
  "logs":     "[CRITICAL] nginx crashed\n[WARNING] memory_hog pid 1042...",
  "env_vars": { "PORT": "80", "APP_MODE": "debug" },
  "config":   { "nginx_port": "80", "worker_count": "4" },
  "step_number": 1,
  "task_id":     1
}
```

### Reward Space

- Type: `float` in `[0.0, 1.0]`
- **Partial credit after every step** — never binary
- Each criterion scored independently — fixing one thing always helps
- `breakdown` dict shows exactly which fix earned which points
- `is_improvement` flag tells the agent if the last action helped

---

## Tasks

### Task 1 — Memory Leak Fix (Easy)

**Scenario:** A rogue process (`memory_hog`, pid 1042) has consumed 95% of
server memory, causing nginx to crash. Server returns no response.

**What the agent must do:**
1. `read_logs` — see nginx crashed + memory_hog warning
2. `list_processes` — confirm pid 1042 at 92% memory
3. `kill_process` (pid 1042) — kill the rogue process
4. `restart_service` (nginx) — restart now that memory is free
5. `check_health` — verify HTTP 200

**Grader criteria:**

| Criterion | Points |
|-----------|--------|
| bad process gone | +0.10 |
| memory below 80% | +0.30 |
| nginx running | +0.40 |
| HTTP 200 | +0.20 |

**Target score:** ~0.8 | **Max steps:** 10

---

### Task 2 — Cascading 500 Errors (Medium)

**Scenario:** HTTP 500 errors in production. Three simultaneous failures:
a memory-leaking `app_worker` process, nginx proxying to the wrong port
(9999 instead of 8080), and `APP_MODE` set to `debug` instead of `production`.

**What the agent must do:**
1. Read logs to identify all three root causes
2. `kill_process` (app_worker) — fix memory
3. `fix_config` (nginx_port → 8080) — fix routing
4. `set_env_var` (APP_MODE → production) — fix app mode
5. `restart_service` (nginx) — only works after all three fixes

**Grader criteria:**

| Criterion | Points |
|-----------|--------|
| memory below 70% | +0.20 |
| app_worker gone | +0.10 |
| nginx_port == 8080 | +0.20 |
| APP_MODE == production | +0.15 |
| nginx running | +0.20 |
| HTTP 200 | +0.15 |

**Target score:** ~0.5 | **Max steps:** 15

---

### Task 3 — Multi-Failure Recovery (Hard)

**Scenario:** Full production outage. Five simultaneous cascading failures:

1. **Disk 98% full** — old rotated logs blocking all disk writes
2. **DB_HOST wrong** — points to decommissioned host `db-old.internal`
3. **APP_ENV = staging** — production config not loaded
4. **Nginx misconfigured** — wrong port (7777→443) and too few workers (1→8)
5. **Runaway cron** (pid 3077) — consuming 85% memory and 99% CPU

**Key ordering dependency:** disk must be cleared **before** the database
can reconnect, because a full disk blocks postgres WAL writes.

**What the agent must do:**
1. `read_logs` — parse noisy logs with 14 lines, find 5 root causes
2. `clear_disk` — unblock postgres WAL writes
3. `set_env_var` (DB_HOST → db.internal) — reconnect database
4. `set_env_var` (APP_ENV → production) — load production config
5. `fix_config` (nginx_port → 443) — fix nginx port
6. `fix_config` (worker_count → 8) — fix worker count
7. `kill_process` (pid 3077) — kill runaway cron
8. `restart_service` (nginx) — restart after all fixes
9. `check_health` — verify HTTP 200

**Grader criteria:**

| Criterion | Points |
|-----------|--------|
| disk below 60% | +0.15 |
| DB_HOST correct | +0.15 |
| db connected | +0.10 |
| APP_ENV production | +0.10 |
| nginx_port 443 | +0.15 |
| worker_count 8 | +0.10 |
| runaway_cron gone | +0.10 |
| memory below 50% | +0.10 |
| HTTP 200 | +0.05 |

**Target score:** ~0.25 | **Max steps:** 20

---

## Project Structure

```
sre-incident-response/
├── env/
│   ├── __init__.py
│   ├── models.py           ← Pydantic models: Action / Observation / Reward
│   ├── environment.py      ← Core env: reset() / step() / state()
│   ├── tasks/
│   │   ├── task_easy.py    ← Task 1 fake world + apply_action()
│   │   ├── task_medium.py  ← Task 2 fake world + apply_action()
│   │   └── task_hard.py    ← Task 3 fake world + apply_action()
│   └── graders/
│       ├── grader_easy.py  ← Task 1 scorer → 0.0–1.0
│       ├── grader_medium.py← Task 2 scorer → 0.0–1.0
│       └── grader_hard.py  ← Task 3 scorer → 0.0–1.0
├── server.py               ← FastAPI: /reset /step /state /health /tasks
├── inference.py            ← Baseline agent (OpenAI client)
├── test_local.py           ← Local test suite (50 checks)
├── openenv.yaml            ← OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Prerequisites

```bash
python 3.11+
pip install fastapi uvicorn pydantic openai requests pyyaml
```

### Run Locally

```bash
# Terminal 1 — start the environment server
python server.py
# → http://localhost:7860

# Terminal 2 — run the test suite (no API key needed)
python test_local.py --offline

# Terminal 3 — run the baseline agent (needs API key)
export HF_TOKEN=hf_your_token_here
python inference.py
```

### Manual API Testing

```bash
# health check
curl http://localhost:7860/health

# list tasks
curl http://localhost:7860/tasks

# start task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1}'

# send an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "kill_process", "pid": 1042}'

# check state
curl http://localhost:7860/state
```

### Docker

```bash
docker build -t sre-incident-response .
docker run -p 7860:7860 sre-incident-response
```

### Pre-Submission Validation

```bash
./validate-submission.sh https://your-space.hf.space ./sre-incident-response
```

---

## Environment Variables for inference.py

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` or `API_KEY` | — | Your HuggingFace / API key (required) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | `openai/gpt-oss-20b` | Model identifier |
| `ENV_URL` | `http://localhost:7860` | Environment server URL |

---

## Baseline Scores

Baseline run using `openai/gpt-oss-20b` via HuggingFace router:

| Task | Name | Score |
|------|------|-------|
| 1 | memory-leak-fix | — |
| 2 | cascading-500-errors | — |
| 3 | multi-failure-recovery | — |
| | **Average** | **—** |

> Run `python inference.py` to reproduce. Scores update in `openenv.yaml`
> after the baseline run completes.

---

## Reward Design

The reward function is designed to provide a **meaningful signal at every
step** — not just at the end of the episode.

Key properties:

- **Always partial** — fixing one thing always improves the score
- **Deterministic** — same server state always returns same score
- **Independent criteria** — each fix scored separately
- **Ordering awareness** — some fixes only work after prerequisites
  (e.g. nginx restart only succeeds after root causes are fixed)
- **No always-same score** — guaranteed variance across episode steps

Score progression examples:

```
Task 1 (easy):   0.075 → 0.400 → 1.000   (3 steps)
Task 2 (medium): 0.347 → 0.500 → 0.700 → 0.850 → 1.000   (4 steps)
Task 3 (hard):   0.034 → 0.176 → 0.426 → ... → 1.000   (7+ steps)
```

---

## Runtime Constraints

- Max inference time: 20 minutes
- Max vCPU: 2
- Max memory: 8 GB
- Python: 3.11
- Port: 7860

---

## License

MIT
