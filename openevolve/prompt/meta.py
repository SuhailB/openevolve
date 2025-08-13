"""
Meta prompt evolution utilities.

This module implements a simple bandit-style meta-prompt evolution loop:
- Maintains a small population of candidate system prompts
- Selects one per iteration for program generation
- Updates each prompt's fitness from the child program's metrics
- Periodically generates new candidates using the LLM itself

We evolve ONLY the system prompt. The user prompt stays as-is.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openevolve.llm.ensemble import LLMEnsemble

logger = logging.getLogger(__name__)


@dataclass
class MetaPrompt:
    prompt_id: str
    content: str
    trials: int = 0
    cumulative_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_used_at: float = 0.0

    @property
    def average_score(self) -> float:
        return self.cumulative_score / self.trials if self.trials > 0 else 0.0


class MetaPromptEvolver:
    """
    Manages system-prompt candidates and their fitness.
    """

    def __init__(
        self,
        storage_dir: str,
        initial_prompt: str,
        use_meta_prompting: bool = False,
        evolution_interval: int = 50,
        max_population: int = 8,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.state_path = os.path.join(self.storage_dir, "meta_prompts.json")
        self.use_meta_prompting = use_meta_prompting
        self.evolution_interval = max(1, int(evolution_interval))
        self.max_population = max(2, int(max_population))
        self.rng = random.Random(rng_seed)

        self.prompts: Dict[str, MetaPrompt] = {}
        self.total_trials: int = 0

        # Initialize from disk or seed with the initial prompt
        self._load_state()
        if not self.prompts:
            self._add_prompt("base", initial_prompt)

    def _add_prompt(self, prompt_id: str, content: str) -> None:
        self.prompts[prompt_id] = MetaPrompt(prompt_id=prompt_id, content=content)
        self._trim_population()
        logger.info(
            f"MetaPrompt: added new candidate '{prompt_id}' (chars={len(content)}). "
            f"Population size={len(self.prompts)}"
        )

    def _trim_population(self) -> None:
        if len(self.prompts) <= self.max_population:
            return
        # Keep top by average; drop worst(s)
        ranked = sorted(self.prompts.values(), key=lambda p: p.average_score, reverse=True)
        keep = {p.prompt_id for p in ranked[: self.max_population]}
        self.prompts = {pid: self.prompts[pid] for pid in keep}

    def _save_state(self) -> None:
        try:
            serializable = {
                "total_trials": self.total_trials,
                "prompts": [
                    {
                        "prompt_id": p.prompt_id,
                        "content": p.content,
                        "trials": p.trials,
                        "cumulative_score": p.cumulative_score,
                        "created_at": p.created_at,
                        "last_used_at": p.last_used_at,
                    }
                    for p in self.prompts.values()
                ],
            }
            with open(self.state_path, "w") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save meta-prompt state: {e}")

    def _load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r") as f:
                data = json.load(f)
            self.total_trials = int(data.get("total_trials", 0))
            self.prompts = {}
            for item in data.get("prompts", []):
                mp = MetaPrompt(
                    prompt_id=item["prompt_id"],
                    content=item["content"],
                    trials=int(item.get("trials", 0)),
                    cumulative_score=float(item.get("cumulative_score", 0.0)),
                    created_at=float(item.get("created_at", time.time())),
                    last_used_at=float(item.get("last_used_at", 0.0)),
                )
                self.prompts[mp.prompt_id] = mp
        except Exception as e:
            logger.warning(f"Failed to load meta-prompt state: {e}")

    def select_prompt_id(self) -> str:
        """
        Thompson sampling over prompt averages; explore untried candidates.
        """
        if not self.use_meta_prompting or not self.prompts:
            # Use the best known or the only one
            if not self.prompts:
                return "base"
            best = max(self.prompts.values(), key=lambda p: p.average_score)
            logger.info(
                f"MetaPrompt: selected (static) '{best.prompt_id}' avg={best.average_score:.4f} "
                f"trials={best.trials}"
            )
            return best.prompt_id

        # Prioritize unseen prompts occasionally
        unseen = [p for p in self.prompts.values() if p.trials == 0]
        if unseen and self.rng.random() < 0.3:
            chosen = self.rng.choice(unseen)
            chosen.last_used_at = time.time()
            logger.info(
                f"MetaPrompt: selected unseen '{chosen.prompt_id}' (no trials yet)"
            )
            return chosen.prompt_id

        # Thompson-like sampling by perturbing averages with small noise
        scored = []
        for p in self.prompts.values():
            noise = self.rng.gauss(0, 0.05)
            scored.append((p.average_score + noise, p.prompt_id))
        scored.sort(reverse=True)
        pid = scored[0][1]
        mp = self.prompts[pid]
        mp.last_used_at = time.time()
        logger.info(
            f"MetaPrompt: selected '{pid}' avg={mp.average_score:.4f} trials={mp.trials}"
        )
        return pid

    def get_prompt(self, prompt_id: str) -> str:
        return self.prompts[prompt_id].content

    def update_fitness(self, prompt_id: str, metrics: Dict[str, float]) -> None:
        # Use combined_score if present; else average of numeric metrics
        if not metrics:
            score = 0.0
        else:
            if "combined_score" in metrics and isinstance(metrics["combined_score"], (int, float)):
                score = float(metrics["combined_score"])
            else:
                numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
                score = sum(numeric) / len(numeric) if numeric else 0.0

        mp = self.prompts.get(prompt_id)
        if mp is None:
            return
        mp.trials += 1
        mp.cumulative_score += score
        self.total_trials += 1
        self._save_state()

    async def maybe_evolve(self, llm_ensemble: Optional[LLMEnsemble], iteration: int) -> None:
        if not self.use_meta_prompting:
            return
        if llm_ensemble is None:
            return
        if iteration <= 0 or iteration % self.evolution_interval != 0:
            return

        try:
            # Build a compact meta-prompt asking the LLM to propose improved system prompts
            baseline = self._best_prompt_text()
            meta_user = (
                "You are improving the system prompt that guides a code-evolution agent.\n"
                "Given the current system prompt, propose 3 alternative system prompts that could\n"
                "lead to better program quality and faster improvement. Each should be concise (<= 8 lines),\n"
                "actionable, and emphasize evaluation-driven iteration.\n\n"
                f"Current system prompt:\n---\n{baseline}\n---\n\n"
                "Return only a JSON object with key 'candidates' as a list of strings."
            )
            system = "You are a helpful AI that writes high-quality system prompts."

            # Use the existing event loop from the caller context
            resp = await llm_ensemble.generate_with_context(
                system, [{"role": "user", "content": meta_user}]
            )

            # Extract JSON candidates
            import json as _json

            start = resp.find("{")
            end = resp.rfind("}")
            if start >= 0 and end > start:
                payload = _json.loads(resp[start : end + 1])
                candidates = payload.get("candidates", [])
                logger.info(
                    f"MetaPrompt: LLM proposed {len(candidates)} candidate(s) at iteration {iteration}"
                )
                for idx, c in enumerate(candidates):
                    if isinstance(c, str) and c.strip():
                        pid = f"gen_{iteration}_{idx}_{int(time.time())}"
                        self._add_prompt(pid, c.strip())
                self._save_state()
        except Exception as e:
            logger.warning(f"Meta-prompt evolution attempt failed: {e}")

    def _best_prompt_text(self) -> str:
        if not self.prompts:
            return ""
        best = max(self.prompts.values(), key=lambda p: p.average_score)
        return best.content


