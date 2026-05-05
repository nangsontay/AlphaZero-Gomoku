---
title: 'Task 7 Temperature Schedule'
type: 'feature'
created: '2026-05-05T04:40:00Z'
status: 'done'
route: 'one-shot'
---

# Task 7 Temperature Schedule

## Intent

**Problem:** Self-play used one fixed temperature for an entire game and kept the Gomoku root Dirichlet alpha default at the old 0.03 value. This made opening exploration less configurable and under-explored the 15x15 Gomoku root compared with the optimization guide recommendation.

**Approach:** Add a backward-compatible temperature schedule to `Game.start_self_play`, default training entrypoints to `temperature_moves=8`, and change AlphaZero MCTS self-play Dirichlet alpha defaults to 0.05. Preserve `temp` as the fixed-temperature fallback when `temperature_moves` is not supplied.

## Suggested Review Order

1. [`game.py`](../../game.py:196) — Review the `start_self_play` API compatibility and schedule semantics first; this is the high-impact shared call site.
2. [`train_gpu_evaluator.py`](../../train_gpu_evaluator.py:228) — Review the primary GPU-evaluator training path and worker argument propagation.
3. [`mcts_alphaZero.py`](../../mcts_alphaZero.py:171) — Review the new `MCTSPlayer` default for root Dirichlet noise.
4. [`train.py`](../../train.py:23) and [`train_mp.py`](../../train_mp.py:196) — Review secondary entrypoints for CLI/default parity.
5. [`_bmad-output/implementation-artifacts/deferred-work.md`](deferred-work.md:1) — Confirm deferred P0 items are tracked for the next implementation slice.

## Verification

**Commands:**
- `git diff --check && python3 -m py_compile game.py mcts_alphaZero.py train_gpu_evaluator.py train.py train_mp.py` — passed with exit code 0.
- `npx gitnexus analyze` — refreshed the stale index before impact analysis.
- `gitnexus_impact` on `start_self_play` — reported HIGH risk due to shared training callers; user explicitly approved a backward-compatible implementation before edits.
- `gitnexus_detect_changes(scope="all")` — confirmed modified execution flows are self-play/training flows; reported critical blast radius as expected for the shared training API.

**Notes:**
- A dynamic runtime smoke test could not run in the current shell because `numpy` is unavailable in the active Python environment. Syntax validation still passed because `py_compile` does not import third-party modules.
- Adversarial review produced one patch item: clarify `temperature_moves=0` semantics in the `Game.start_self_play` docstring. That patch was applied and revalidated.
