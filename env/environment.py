"""
environment.py — Core SRE Incident Response Environment

This is the brain of the entire project.
It wires together tasks/ and graders/ and exposes the
three OpenEnv API methods: reset(), step(), state()

HOW IT WORKS:
  1. reset(task_id) loads a task, builds the fake broken server world
  2. step(action)  applies the action, scores the result, returns observation
  3. state()       returns a full snapshot of current server state

WHAT IT DOES NOT KNOW ABOUT:
  - FastAPI / HTTP (that is server.py's job)
  - The LLM agent (that is inference.py's job)
  - Docker or HuggingFace (irrelevant here)

It is pure Python. A dict goes in, a dict comes out.
"""

from typing import Any, Dict, Optional

from env.models import (
    Action,
    Observation,
    Process,
    ResetResult,
    Reward,
    StepResult,
)

from env.tasks import task_easy, task_medium, task_hard
from env.graders import grader_easy, grader_medium, grader_hard


TASKS = {
    1: {
        "module":       task_easy,
        "grader":       grader_easy,
        "name":         "memory-leak-fix",
        "description":  (
            "A rogue process has consumed all server memory causing nginx to crash. "
            "Identify the bad process, kill it, restart nginx, and verify the server is healthy."
        ),
        "difficulty":   "easy",
        "max_steps":    10,
    },
    2: {
        "module":       task_medium,
        "grader":       grader_medium,
        "name":         "cascading-500-errors",
        "description":  (
            "The server is throwing 500 errors. A memory leak exists, the app config has "
            "the wrong port, and nginx needs reconfiguring. Fix all issues to restore service."
        ),
        "difficulty":   "medium",
        "max_steps":    15,
    },
    3: {
        "module":       task_hard,
        "grader":       grader_hard,
        "name":         "multi-failure-recovery",
        "description":  (
            "Multiple cascading failures: disk is 98% full, database connection failing, "
            "wrong environment variables, nginx misconfigured, and a runaway process. "
            "Diagnose and fix all five issues to fully restore the server."
        ),
        "difficulty":   "hard",
        "max_steps":    20,
    },
}

class SREEnvironment:
    """
    SRE Incident Response OpenEnv Environment.

    Usage:
        env = SREEnvironment()
        result = env.reset(task_id=1)
        step_result = env.step(action)
        snapshot = env.state()
    """

    def __init__(self):
        self._server_state: Dict[str, Any] = {}
        self._task_id: Optional[int] = None
        self._task_cfg: Optional[Dict] = None
        self._step_number: int = 0
        self._done: bool = False
        self._previous_score: float = 0.0
        self._score_history: list = []

    def reset(self, task_id: int = 1) -> ResetResult:
        """
        Wipes the old server state and builds a fresh broken server
        for the specified task. Returns the first observation.

        Args:
            task_id: 1 (easy), 2 (medium), or 3 (hard)

        Returns:
            ResetResult with first Observation and task metadata
        """
        if task_id not in TASKS:
            raise ValueError(f"task_id must be 1, 2, or 3. Got: {task_id}")

        # load task config
        self._task_id  = task_id
        self._task_cfg = TASKS[task_id]

        # build fresh fake broken server world
        self._server_state = self._task_cfg["module"].build()

        # reset episode tracking
        self._step_number    = 0
        self._done           = False
        self._previous_score = 0.0
        self._score_history  = []

        # build and return first observation
        observation = self._build_observation(
            message=(
                f"[ALERT] Incident detected on production server.\n"
                f"Task: {self._task_cfg['description']}\n"
                f"You have {self._task_cfg['max_steps']} steps to resolve the incident."
            )
        )

        return ResetResult(
            observation=observation,
            task_id=task_id,
            task_name=self._task_cfg["name"],
            task_description=self._task_cfg["description"],
            max_steps=self._task_cfg["max_steps"],
        )

    def step(self, action: Action) -> StepResult:
        """
        Applies the agent's action to the fake server state.
        Scores the result. Returns new observation + reward + done flag.

        Args:
            action: Action object from the agent

        Returns:
            StepResult with updated Observation, Reward, done, info
        """
        if self._task_cfg is None:
            raise RuntimeError("Call reset() before step().")

        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step_number += 1

        self._server_state = self._task_cfg["module"].apply_action(
            self._server_state, action
        )

        reward: Reward = self._task_cfg["grader"].score(
            self._server_state,
            self._previous_score,
        )

        self._score_history.append(reward.value)
        self._previous_score = reward.value

        max_steps_reached = self._step_number >= self._task_cfg["max_steps"]
        task_solved       = self._task_cfg["grader"].is_done(reward)
        self._done        = task_solved or max_steps_reached

        if task_solved:
            message = "Server is fully healthy. Incident resolved."
        elif max_steps_reached:
            message = f"Max steps ({self._task_cfg['max_steps']}) reached. Episode over."
        else:
            steps_left = self._task_cfg["max_steps"] - self._step_number
            message = (
                f"Action applied. Score: {reward.value:.2f}. "
                f"{steps_left} steps remaining."
            )

        observation = self._build_observation(message=message)

        return StepResult(
            observation=observation,
            reward=reward,
            done=self._done,
            info={
                "task_id":        self._task_id,
                "task_name":      self._task_cfg["name"],
                "step_number":    self._step_number,
                "max_steps":      self._task_cfg["max_steps"],
                "score_history":  self._score_history,
                "task_solved":    task_solved,
                "max_steps_hit":  max_steps_reached,
            },
        )

    def state(self) -> Dict[str, Any]:
        """
        Returns a full snapshot of the current environment state.
        Used by judges and the /state endpoint on server.py.
        """
        if self._task_cfg is None:
            return {"status": "not_started", "message": "Call reset() first."}

        return {
            "task_id":        self._task_id,
            "task_name":      self._task_cfg["name"],
            "difficulty":     self._task_cfg["difficulty"],
            "step_number":    self._step_number,
            "max_steps":      self._task_cfg["max_steps"],
            "done":           self._done,
            "previous_score": self._previous_score,
            "score_history":  self._score_history,
            "server": {
                "nginx_status":  self._server_state.get("nginx_status"),
                "memory_usage":  self._server_state.get("memory_usage"),
                "disk_usage":    self._server_state.get("disk_usage"),
                "db_status":     self._server_state.get("db_status"),
                "http_status":   self._server_state.get("http_status"),
                "process_count": len(self._server_state.get("processes", [])),
                "env_vars":      self._server_state.get("env_vars", {}),
                "config":        self._server_state.get("config", {}),
            },
        }

    def _build_observation(self, message: str) -> Observation:
        """
        Converts the raw server state dictionary into a typed
        Observation model that the agent receives.
        """
        s = self._server_state
        return Observation(
            message=message,
            nginx_status=s.get("nginx_status", "unknown"),
            memory_usage=s.get("memory_usage", 0.0),
            disk_usage=s.get("disk_usage", 0.0),
            db_status=s.get("db_status", "unknown"),
            http_status=s.get("http_status", None),
            processes=s.get("processes", []),
            logs=s.get("logs", ""),
            env_vars=s.get("env_vars", {}),
            config=s.get("config", {}),
            step_number=self._step_number,
            task_id=self._task_id or 0,
        )