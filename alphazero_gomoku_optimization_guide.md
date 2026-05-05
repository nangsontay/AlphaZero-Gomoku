# AlphaZero Gomoku 15×15 — Training Pipeline Optimization Guide

> **Audience.** This document is a hand-off spec for an implementation agent. It assumes
> familiarity with the existing codebase (`mcts_alphaZero.py`, `train_gpu_evaluator.py`,
> `policy_value_net_pytorch.py`, `game.py`) and with the AlphaGo Zero paper (Silver et al.,
> *Nature* 550, 2017).
>
> **Goal.** Bring the Gomoku 15×15 self-play training pipeline closer to the asynchronous,
> batched, parallel architecture that AlphaGo Zero actually uses, while adapting every choice
> to Gomoku's specific game characteristics (short games, dense terminals, forced-move
> tactical sequences).
>
> **Out of scope.** Network architecture changes (the current 10×128 ResNet is fine for
> 15×15). Distributed multi-GPU training. Game rule changes.

---

## 0. How to Use This Document

The document is structured as:

1. **Context & gap analysis** — what the paper does vs. what the codebase does, with
   Gomoku-specific differences called out explicitly.
2. **Implementation tasks** — eleven discrete, mostly-independent tasks, ordered by
   priority. Each task has: rationale, files touched, design, code skeleton, validation.
3. **Configuration recommendations** — concrete starting values for hyperparameters that
   need to change for Gomoku 15×15.
4. **Benchmarking protocol** — how to measure that each task actually helped.
5. **Pitfalls** — things specific to Gomoku that bit us in design and will bite the
   implementation if not handled.

For the agent: implement tasks in the order listed in §3.0 unless explicitly noted. Each
task is self-contained enough to be a separate PR/commit. Always benchmark before/after
using the protocol in §5.

---

## 1. Context

### 1.1 Codebase summary

The current AlphaZero Gomoku codebase implements:

- **Game**: Gomoku 15×15, 5-in-a-row, in `game.py` (`Board`, `Game` classes).
- **Network** (`policy_value_net_pytorch.py`): 10-block × 128-channel ResNet with a fully
  convolutional policy head (1 logit per cell) and an FC value head. AMP-enabled training,
  SGD with Nesterov momentum, gradient clipping, weight decay 1e-4. Inputs are 4 binary
  feature planes (current player stones, opponent stones, last move, color-to-play).
- **MCTS** (`mcts_alphaZero.py`): single-threaded PUCT MCTS with running-mean Q updates,
  Dirichlet noise at root for self-play, tree reuse via `update_with_move`.
- **Training pipeline** (`train_gpu_evaluator.py`): a *central batched GPU evaluator*
  pattern. The main process trains the network. A dedicated GPU evaluator process holds
  the network on GPU and serves batched inference to multiple CPU self-play workers via
  `multiprocessing.Queue`. Workers run MCTS and call back to the evaluator for each leaf.

The codebase is *already past* the naïve single-process AlphaZero implementation — central
batched inference is in place. The optimizations below take it the next step closer to the
paper's actual production architecture.

### 1.2 Paper grounding (AlphaGo Zero, Methods)

The optimizations in this document are anchored in three sections of the paper:

- **Search algorithm**: "Multiple simulations are executed in parallel on separate search
  threads." "Positions in the queue are evaluated by the neural network using a mini-batch
  size of 8; the search thread is locked until evaluation completes." "We use virtual loss
  to ensure each thread evaluates different nodes." "Expand and evaluate ... where d_i is a
  dihedral reflection or rotation selected uniformly at random from i in [1..8]."
- **Self-play training pipeline**: "AlphaGo Zero's self-play training pipeline consists of
  three main components, all executed asynchronously in parallel": optimizer, evaluator
  (best-vs-current gating), and self-play.
- **Self-play / Resignation**: "the resignation threshold v_resign is selected automatically
  to keep the fraction of false positives ... below 5%."
- **Self-play / Temperature**: "For the first 30 moves of each game, the temperature is set
  to τ=1 ... For the remainder of the game, an infinitesimal temperature is used, τ→0."

### 1.3 Gomoku 15×15 vs Go 19×19 — the differences that matter for design

| Property | Go 19×19 (paper) | Gomoku 15×15 (this codebase) | Design implication |
|---|---|---|---|
| Average game length | 150–300 plies | 25–50 plies | Per-game overhead amortizes over far fewer NN evals → spawn cost is much more harmful here |
| Terminal check cost | Expensive (Tromp–Taylor scoring, eyes, captures) | Cheap — O(1) check 4 directions around last stone | Many leaves are terminal and should be detected before any NN call |
| Markov property | No (ko rule, repetition forbidden) → needs 8-ply history | Yes — current board fully observable | `in_channels=4` is correct; do not add history planes |
| Average branching at midgame | ~250 / 361 | ~150–200 / 225 | Vectorized PUCT selection still pays off |
| Tactical density | Low; long-horizon strategy | High; forced sequences (open three, four-threat, five-row) | Virtual loss must be tuned **lower** than paper to avoid pulling collectors off the only winning line |
| D4 symmetry | Yes (8-way) | Yes (8-way) | Augmentation strategy carries over identically |
| Outcome variance | Smooth (point-margin gradient) | Step-function (one missed move flips win/loss) | Replay buffer needs to be fresher (sampling window in *games*, not just positions) |
| Resignation savings (paper estimate) | 30–50% compute | 15–20% (games already short) | Implement, but lower priority |

Key consequence: any optimization that assumes "most leaves need NN eval" (true for Go) needs
to be revisited for Gomoku, where a meaningful fraction of leaves are terminal — especially
late in training when the network plays tactically.

---

## 2. Architecture: Current vs Target

### 2.1 Current pipeline (synchronous, spawn-per-batch)

```
main (trainer)
  for game_batch in 1..N:
    spawn GPU evaluator process
    spawn M self-play worker processes
    workers run games to completion (blocking)
    main collects buffer
    kill workers + evaluator
    if buffer big enough:
      main runs policy_update() on the network
      save current_policy.model
    if checkpoint cycle:
      run policy_evaluate() (current vs MCTS_pure)
```

Bottlenecks:
- **Spawn cost** (CUDA init + model load) repeated every batch.
- **MCTS is single-leaf-per-eval**: one worker emits one request, blocks, gets one response.
  The evaluator's effective batch size is bounded by `num_workers`.
- **Self-play and training serialize**: GPU is mostly idle during self-play (workers are
  CPU-bound between NN calls), then workers are idle during training.
- **Evaluation baseline is MCTS_pure**: too weak after a few iterations; gives no
  meaningful gating signal on whether the new network is actually better.

### 2.2 Target pipeline (asynchronous, persistent, leaf-parallel)

```
main (trainer)
  one-time spawn:
    persistent GPU evaluator (sleeps when idle, hot-reloads weights)
    M persistent self-play workers (loop forever, never restart)
  workers continuously stream (state, π, z) tuples → replay queue
  trainer loop:
    drain replay queue into buffer
    if buffer big enough:
      policy_update()
      every K updates: save weights, signal evaluator + workers to reload
    every E updates: run best-vs-current gating evaluation
       if new wins ≥ 55%: promote new → best, broadcast best to workers
       else: keep best, discard or shadow new
    every C updates: save checkpoint
```

Each MCTS within a worker uses **leaf-parallel virtual loss**: per move, instead of 800
sequential single-leaf evaluations, do ~100–200 batched calls of 4–8 leaves each. The
GPU evaluator now sees batches of (M_workers × vl_k) inflight requests, not just M_workers.

---

## 3. Implementation Tasks

### 3.0 Priority ordering and dependency graph

```
Task 1 (Virtual-loss MCTS) ─── depends on ──> Task 2 (Board.copy_fast)
                          └─── enables ────> Task 3 (Numpy children arrays)

Task 4 (Persistent async pipeline) ── independent of 1–3, but bigger win when combined

Task 5 (GPU evaluator opts: CUDA Graphs, FP16, channels_last)
       └── orthogonal to 1–4

Task 6 (Markov state caching) ── refines Task 2

Task 7 (Temperature + Dirichlet tuning) ── independent, quick win

Task 8 (Win-in-1 leaf shortcut) ── builds on Task 1, OPTIONAL (purity tradeoff)

Task 9 (Resignation threshold) ── independent

Task 10 (Shared memory state transfer) ── refines Task 4

Task 11 (Adaptive batch timeout) ── refines Task 4
```

**Recommended execution order:** 7 → 1+2+3 (single PR) → 4 → 5 → 6 → 9 → 8 → 10 → 11.

Task 7 first because it's a 1-hour change that improves data quality and gives a baseline
to compare against. Tasks 1+2+3 are inseparable in practice (changing TreeNode storage,
adding leaf-parallel batching, and vectorizing PUCT all touch `mcts_alphaZero.py`).

---

### Task 1 — Virtual-Loss Leaf-Parallel MCTS with Terminal Oversampling

**Priority:** P0. Largest single throughput improvement.

**Files touched:**
- `mcts_alphaZero.py` (TreeNode, MCTS class)
- `train_gpu_evaluator.py` (`RemotePolicyValueClient` — needs batched request API)
- `train_gpu_evaluator.py` (`gpu_evaluator_loop` — verify it already accepts batched inputs; it does)

**Paper reference:** Methods, *Search algorithm* — "Multiple simulations are executed in
parallel on separate search threads … We use virtual loss to ensure each thread evaluates
different nodes." Mini-batch of 8 per evaluation queue.

**Problem.** The current `MCTS._playout` does one selection → one network call → one
backup, sequentially. With `n_playout=800`, each worker emits 800 sequential requests
per move; the evaluator's batch is limited to ~`num_workers` simultaneous requests.

**Design.** Implement leaf parallelism *within* a single MCTS instance (single-threaded
in Python, no need for actual threads). Per outer iteration:

1. Run K successive selections from the root, applying virtual loss as we descend.
   Each selection lands on a leaf.
2. Terminal leaves: back up exact value immediately, reverting virtual loss as we go.
3. Non-terminal leaves: collect into a batch, send batch to GPU evaluator, receive batched
   priors and values, then expand + back up each.

Because Gomoku has *high terminal density*, plain "K leaves per batch" produces small
GPU batches when many of the K are terminal. Mitigate with **oversampling**: keep
selecting until we have K *non-terminal* leaves, capped at `K × max_oversample`.

**Required changes — TreeNode.** Switch from running-mean Q to (W, N) storage so virtual
loss is reversible:

```python
class TreeNode:
    __slots__ = ('_parent', '_children', '_N', '_W', '_P', '_vl')

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._N = 0           # visit count
        self._W = 0.0         # total action value (sum, not mean)
        self._P = prior_p     # prior probability from network
        self._vl = 0          # cumulative virtual loss applied (for invariants)

    def Q(self):
        return self._W / self._N if self._N > 0 else 0.0

    def get_value(self, c_puct):
        # Standard PUCT. parent._N is the "sum_b N(s,b) + 1" used in paper variant.
        u = c_puct * self._P * (self._parent._N ** 0.5) / (1 + self._N)
        return self.Q() + u

    def add_virtual_loss(self, n_vl):
        self._N += n_vl
        self._W -= n_vl   # virtual loss = pretend we lost n_vl times
        self._vl += n_vl

    def revert_virtual_loss(self, n_vl):
        self._N -= n_vl
        self._W += n_vl
        self._vl -= n_vl

    def update(self, leaf_value):
        self._N += 1
        self._W += leaf_value

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def is_leaf(self):
        return not self._children

    def is_root(self):
        return self._parent is None
```

**Required changes — MCTS.** Replace `_playout` with a batched search loop:

```python
class MCTS:
    def __init__(self, policy_value_fn, policy_value_batch_fn,
                 c_puct=3.0, n_playout=800, vl_k=4, n_vl=1.0,
                 max_oversample=3):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn         # legacy single-state, used for non-batched paths
        self._policy_batch = policy_value_batch_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._vl_k = vl_k
        self._n_vl = n_vl
        self._max_oversample = max_oversample

    def get_move_probs(self, state, temp=1e-3):
        completed = 0
        while completed < self._n_playout:
            target_nn = min(self._vl_k, self._n_playout - completed)
            nn_leaves = []           # list of (leaf_node, sim_state, path)
            terminals_done = 0
            attempts = 0
            attempt_cap = max(target_nn * self._max_oversample, target_nn + 4)

            # 1. Select up to target_nn non-terminal leaves; terminals are backed up in place.
            while len(nn_leaves) < target_nn and attempts < attempt_cap:
                attempts += 1
                node = self._root
                sim_state = state.copy_fast()    # see Task 2
                path = []

                # Descend with virtual loss
                while not node.is_leaf():
                    action, node = self._select_child(node)
                    sim_state.do_move(action)
                    node.add_virtual_loss(self._n_vl)
                    path.append(node)

                end, winner = sim_state.game_end()
                if end:
                    cur = sim_state.get_current_player()
                    leaf_value = 0.0 if winner == -1 else (
                        1.0 if winner == cur else -1.0
                    )
                    # backup expects value from the leaf player's perspective; flip sign as needed
                    self._backup_and_revert(path, -leaf_value)
                    terminals_done += 1
                else:
                    nn_leaves.append((node, sim_state, path))

            # 2. Batched NN eval for non-terminal leaves
            if nn_leaves:
                states_np = np.stack([
                    np.ascontiguousarray(L[1].current_state().astype(np.float32))
                    for L in nn_leaves
                ])
                priors_batch, values_batch = self._policy_batch(states_np)
                # priors_batch: (B, board_size), values_batch: (B,) or (B,1)

                for (leaf, sim, path), priors, value in zip(
                        nn_leaves, priors_batch, values_batch):
                    if leaf.is_leaf():
                        legal = sim.availables
                        leaf.expand([(a, float(priors[a])) for a in legal])
                    self._backup_and_revert(path, -float(np.asarray(value).reshape(-1)[0]))

            completed += len(nn_leaves) + terminals_done

            if attempts >= attempt_cap and not nn_leaves:
                # Tree is fully expanded around root; rest of n_playout pointless
                break

        # Return action probabilities from root visit counts
        act_visits = [(a, n._N) for a, n in self._root._children.items()]
        if not act_visits:
            return [], []
        acts, visits = zip(*act_visits)
        probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, probs

    def _select_child(self, node):
        return max(node._children.items(),
                   key=lambda kv: kv[1].get_value(self._c_puct))

    def _backup_and_revert(self, path, leaf_value):
        # path[0] is the deepest direct child of root, path[-1] is the leaf.
        # Wait — actually in the loop above, path[0] is the FIRST descent step
        # (root's child) and path[-1] is the leaf. Iterate in reverse to backup.
        # leaf_value is already negated to reflect "from the leaf player's perspective".
        # As we unwind, alternate sign because perspective flips per ply.
        sign = 1.0
        for node in reversed(path):
            node.revert_virtual_loss(self._n_vl)
            node.update(sign * leaf_value)
            sign = -sign
```

⚠ **Sign convention**. Sign-flipping in MCTS backup is the most common bug source in
AlphaZero implementations. Validate by comparing root Q against the existing (pre-change)
implementation on a fixed synthetic policy_value_fn that returns deterministic outputs.
The Q values at root for each action should match within float tolerance after running
the same deterministic playouts.

**Required changes — RemotePolicyValueClient.** Add a batched API:

```python
class RemotePolicyValueClient:
    def policy_value_fn(self, board):
        # Existing single-state path, kept for backwards compat (e.g., in policy_evaluate).
        ...

    def policy_value_batch_fn(self, states_np):
        """
        states_np: ndarray of shape (B, in_channels, H, W), float32, contiguous.
        Returns:
          priors_batch: ndarray (B, board_size)
          values_batch: ndarray (B,)
        Sends one batch request and waits for one batched response, OR sends B
        individual requests and gathers them. The latter is simpler initially —
        the GPU evaluator naturally batches across all inflight requests anyway.
        """
        B = states_np.shape[0]
        rids = []
        for i in range(B):
            self.request_id += 1
            rid = self.request_id
            rids.append(rid)
            self.request_queue.put({
                "type": "eval",
                "worker_id": self.worker_id,
                "request_id": rid,
                "state": np.ascontiguousarray(states_np[i]),
            })

        priors_out = [None] * B
        values_out = [None] * B
        rid_to_idx = {rid: i for i, rid in enumerate(rids)}
        received = 0
        while received < B:
            try:
                resp = self.response_queue.get(timeout=self.response_timeout)
            except queue.Empty:
                raise RuntimeError(f"worker {self.worker_id} timed out on batch")
            if resp.get("type") == "error":
                raise RuntimeError(f"GPU evaluator error: {resp.get('error')}")
            rid = resp.get("request_id")
            if rid not in rid_to_idx:
                continue
            idx = rid_to_idx[rid]
            priors_out[idx] = resp["act_probs"]
            values_out[idx] = float(resp["value"])
            received += 1

        return np.stack(priors_out), np.array(values_out, dtype=np.float32)
```

The simpler implementation above sends B individual requests and gathers them. The GPU
evaluator already coalesces across workers — it will see all B from this worker plus
inflight from other workers, batch them, and respond in any order. The client must handle
out-of-order responses (note `rid_to_idx` lookup, ignoring others' results).

**Note on `_select_child`.** The current code uses `max(_children.items(), key=...)` —
this iterates Python dicts and computes PUCT in Python. For Gomoku 15×15 with ~150 legal
moves at each tree level, this is the dominant CPU cost in MCTS. Task 3 vectorizes it.

**Validation.**

1. **Correctness on toy game**: implement a tiny game (e.g., 3×3 tic-tac-toe with a
   uniform policy_value_fn). Run baseline MCTS and new MCTS with `vl_k=1, n_vl=0`. Visit
   counts at root must match exactly.
2. **Sign flip test**: same as above but a deterministic policy that prefers a known
   winning line. Both MCTS variants must converge to the same best action.
3. **Throughput**: log average GPU batch size in `gpu_evaluator_loop`. With 10 workers,
   `vl_k=4`, expect avg_batch from ~10 → ~40. With `vl_k=8`, expect ~80.
4. **Strength regression**: 50-game match `new vs old (vl_k=1, n_vl=0)`. Win rate must
   be ≥ 50% ± noise. If new loses materially, the bug is almost certainly in sign or in
   Dirichlet noise interaction (must still be applied at root after `update_with_move`).

**Recommended starting hyperparameters for Gomoku 15×15:**

- `vl_k = 4` initially. Tune up to 8 if avg batch size is healthy and strength is unaffected.
- `n_vl = 1.0`. **Do not start at 3 (paper)** — too aggressive for Gomoku tactics.
- `max_oversample = 3`.

---

### Task 2 — `Board.copy_fast()`

**Priority:** P0 (blocker for Task 1).

**Files touched:** `game.py` (`Board` class).

**Problem.** `mcts_alphaZero.py` currently does `state_copy = copy.deepcopy(state)` on
each playout. With `n_playout=800` and ~30 plies per game, that's 24,000 deepcopy calls
per game per worker. `deepcopy` of a `Board` with Python lists/dicts/sets is the dominant
CPU cost between NN calls.

**Design.** Implement `Board.copy_fast()` that does a shallow copy of the `Board` object
plus targeted clones of the mutable fields that MCTS will modify (the move-tracking
collections). Do not deepcopy the rule constants (`width`, `height`, `n_in_row`).

```python
class Board:
    def copy_fast(self):
        new = Board.__new__(Board)
        # Constants — share by reference
        new.width = self.width
        new.height = self.height
        new.n_in_row = self.n_in_row
        new.players = self.players          # tuple, immutable
        # Mutable state — shallow copy
        new.states = dict(self.states)      # {move_loc: player_id}
        new.availables = list(self.availables)
        new.current_player = self.current_player
        new.last_move = self.last_move
        # If Board caches anything else (e.g., a feature plane cache), copy it too.
        return new
```

Audit `Board.__init__` and `do_move` to ensure `copy_fast` covers every mutable attribute.
Run a unit test:

```python
b = Board(...); b.init_board(); b.do_move(some_move)
b2 = b.copy_fast()
b2.do_move(other_move)
assert b.states != b2.states                # b unchanged
assert b.last_move != b2.last_move
```

**Validation.** Profile with `cProfile` over a 1-game self-play run before and after.
`copy.deepcopy` should drop from a top-3 hot function to absent. Expected speedup of MCTS
CPU portion: 5–10× (deepcopy on a 15×15 Board with ~50 stones is genuinely slow).

**Optional follow-up (low priority): `do_move`/`undo_move`.** For even faster MCTS,
implement `Board.undo_move()` and have MCTS push moves and undo them on the way back up.
This avoids any per-leaf copy. Only worth doing if profiling after Task 6 still shows
`copy_fast` as a hot spot.

---

### Task 3 — Vectorize PUCT selection with NumPy children arrays

**Priority:** P1.

**Files touched:** `mcts_alphaZero.py`.

**Problem.** `_select_child` does `max(_children.items(), key=...)` — Python loop over
~150 children per tree level, computing PUCT in Python. At `n_playout=800` with effective
tree depth ~15, this is ~12,000 PUCT evaluations per move per worker, all in pure Python.

**Design.** Maintain per-node parallel NumPy arrays for children stats. PUCT becomes a
single vectorized argmax.

```python
class TreeNode:
    """
    Parent node holds arrays for its children. A child is a 'slot' (index into
    its parent's arrays). The child also holds its own children arrays once expanded.
    """
    __slots__ = ('_parent', '_parent_idx', '_children_actions',
                 '_priors', '_N_arr', '_W_arr', '_child_nodes', '_total_N')

    def __init__(self, parent, parent_idx, prior_p_for_this_node):
        self._parent = parent
        self._parent_idx = parent_idx     # index into parent's arrays
        # Own children — populated on expand()
        self._children_actions = None     # np.ndarray (n_legal,) int32
        self._priors = None               # np.ndarray (n_legal,) float32
        self._N_arr = None                # np.ndarray (n_legal,) float32 (float to allow fractional vl)
        self._W_arr = None                # np.ndarray (n_legal,) float32
        self._child_nodes = None          # list[TreeNode | None] of length n_legal
        self._total_N = 0.0               # sum of N over children (≈ visits to this node from above)

    def expand(self, actions, priors_for_actions):
        n = len(actions)
        self._children_actions = np.asarray(actions, dtype=np.int32)
        self._priors = np.asarray(priors_for_actions, dtype=np.float32)
        self._N_arr = np.zeros(n, dtype=np.float32)
        self._W_arr = np.zeros(n, dtype=np.float32)
        self._child_nodes = [None] * n

    def is_leaf(self):
        return self._children_actions is None

    def select(self, c_puct):
        # Vectorized PUCT
        sqrt_total = (self._total_N + 1e-8) ** 0.5
        Q = np.where(self._N_arr > 0, self._W_arr / np.maximum(self._N_arr, 1.0), 0.0)
        U = c_puct * self._priors * sqrt_total / (1.0 + self._N_arr)
        score = Q + U
        idx = int(np.argmax(score))
        action = int(self._children_actions[idx])
        child = self._child_nodes[idx]
        if child is None:
            child = TreeNode(self, idx, float(self._priors[idx]))
            self._child_nodes[idx] = child
        return action, child, idx

    # Virtual loss applies to the *parent's* arrays at the slot of this node:
    def add_virtual_loss(self, n_vl):
        self._parent._N_arr[self._parent_idx] += n_vl
        self._parent._W_arr[self._parent_idx] -= n_vl
        self._parent._total_N += n_vl

    def revert_virtual_loss(self, n_vl):
        self._parent._N_arr[self._parent_idx] -= n_vl
        self._parent._W_arr[self._parent_idx] += n_vl
        self._parent._total_N -= n_vl

    def update(self, leaf_value):
        self._parent._N_arr[self._parent_idx] += 1.0
        self._parent._W_arr[self._parent_idx] += leaf_value
        self._parent._total_N += 1.0
```

Notes on this design:
- A node's *own visit count* is tracked in its parent's `_N_arr[_parent_idx]`. The root
  is special — it has no parent, so use `self._total_N` as its visit count.
- Backup uses *parent.update on the slot* rather than self.update; the path stored during
  selection can be a list of `(parent, idx)` tuples or a list of child nodes — pick one.
- Dirichlet noise application at root: it must be added to `_priors`, not to a separate
  field. After applying noise, all subsequent PUCT computations use the noised priors —
  this is the correct paper behavior.

**Validation.** Same as Task 1: visit-count match against the dict-based version on a
deterministic toy. Profile: `_select_child` Python time should drop ~3–5×.

**Implementation strategy.** Land Tasks 1, 2, 3 as a single PR. The TreeNode rewrite
unifies the storage change (W/N), virtual loss support, and array vectorization. Splitting
them creates a transitional broken state that's hard to validate.

---

### Task 4 — Persistent Async Self-Play Pipeline

**Priority:** P0 (largest pipeline-level win).

**Files touched:** `train_gpu_evaluator.py` (major refactor).

**Paper reference:** *Self-play training pipeline* — "three main components, all executed
asynchronously in parallel": optimizer, evaluator, self-play.

**Problem.** The current `collect_selfplay_data_remote_gpu` spawns workers and the
evaluator from scratch every game batch. Each spawn pays:
- Python interpreter + import cost
- CUDA context init for the evaluator (~1–2 s on a fresh process)
- Model file load + state_dict load
- Worker MCTSPlayer + Board init

For Gomoku where each game only takes ~30 plies × ~800 playouts × ~1 ms NN call ≈ 24
seconds of useful work, spawning 11 processes that each take 2–5 s of init means ~30 s
of overhead per batch — comparable to the work itself.

**Design.** Three persistent process groups, all started once at training start.

```
main process (trainer):
  - Loads PolicyValueNet on GPU
  - Owns the replay buffer (deque)
  - Reads a multiprocessing Queue for new self-play tuples
  - Periodically:
      drain queue into buffer
      if buffer big enough: policy_update()
      every K updates: write weights to /dev/shm or shared tensor; trigger reload event
      every E updates: trigger best-vs-current evaluation
      every C updates: persist current_policy.model and best_policy.model

GPU evaluator process:
  - Loads PolicyValueNet on GPU
  - Loops: pull from request_queue, batch up to eval_batch_size or eval_timeout_ms,
    forward, dispatch responses to per-worker response_queues
  - On weight reload event: load new state_dict from /dev/shm or pipe

self-play worker processes (M of them):
  - Each owns a Board, Game, MCTSPlayer, RemotePolicyValueClient
  - Loop forever: play one game, push tuples to replay_queue
  - On weight reload event: workers don't need to reload anything (NN lives only
    in the GPU evaluator); MCTSPlayer continues calling the evaluator
```

A subtle and important point: in the current architecture, only the *evaluator* holds the
network on GPU. Workers don't need to reload anything when weights change — the evaluator
serves responses using the latest weights it has. So weight reload is a one-process event,
not an M+1-process event.

**Communication primitives:**
- `request_queue` (multi-producer, single-consumer): ctx.Queue, holds eval requests.
- `response_queues` (one per worker, single-producer single-consumer): ctx.Queue.
- `replay_queue` (multi-producer, single-consumer): ctx.Queue, holds finished games. Items
  are full game's tuples list (one game per put — keeps queue ops bounded).
- `weight_event` (mp.Event): set by trainer when new weights are written; cleared by
  evaluator after reload. The path to the latest weights is fixed (e.g.,
  `/dev/shm/policy_latest.pt`).
- `stats_queue` (single-producer, single-consumer): periodic stats from evaluator.

**Skeleton for the trainer loop:**

```python
class AsyncTrainPipeline:
    def __init__(self, ...):
        ctx = mp.get_context("spawn")
        self.request_queue = ctx.Queue(maxsize=4096)
        self.replay_queue = ctx.Queue(maxsize=64)        # queued games, not positions
        self.response_queues = [ctx.Queue(maxsize=64) for _ in range(self.num_workers)]
        self.stats_queue = ctx.Queue()
        self.weight_event = ctx.Event()
        self.shutdown_event = ctx.Event()
        self.weights_path = "/dev/shm/policy_latest.pt"

        # Initial weights snapshot for evaluator + workers
        self.policy_value_net.save_model(self.weights_path)

        self.evaluator_proc = ctx.Process(
            target=persistent_evaluator_loop,
            args=(self.weights_path, self.board_width, self.board_height,
                  self.request_queue, self.response_queues,
                  self.stats_queue, self.weight_event, self.shutdown_event,
                  self.eval_batch_size, self.eval_timeout_ms, self.use_gpu),
            name="gpu-evaluator",
        )
        self.worker_procs = [
            ctx.Process(
                target=persistent_worker_loop,
                args=(wid, self.board_width, self.board_height, self.n_in_row,
                      self.n_playout, self.c_puct, self.dirichlet_alpha,
                      self.noise_eps, self.vl_k, self.n_vl,
                      self.request_queue, self.response_queues[wid],
                      self.replay_queue, self.shutdown_event,
                      self.response_timeout),
                name=f"selfplay-worker-{wid}",
            )
            for wid in range(self.num_workers)
        ]
        self.evaluator_proc.start()
        for p in self.worker_procs:
            p.start()

    def run(self):
        try:
            update_count = 0
            target_updates = self.game_batch_num   # reinterpret as # updates
            while update_count < target_updates:
                # Drain any new games from the replay queue
                drained_games = 0
                drain_deadline = time.time() + 30.0
                while time.time() < drain_deadline and drained_games < 100:
                    try:
                        game_tuples = self.replay_queue.get(timeout=1.0)
                    except queue.Empty:
                        break
                    self.data_buffer.extend(game_tuples)
                    drained_games += 1

                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
                    update_count += 1

                    if update_count % self.weight_push_every == 0:
                        self.policy_value_net.save_model(self.weights_path)
                        self.weight_event.set()

                    if update_count % self.checkpoint_every == 0:
                        self.policy_value_net.save_model("./current_policy.model")

                    if update_count % self.eval_every == 0:
                        self.run_best_vs_current_gating()
        finally:
            self.shutdown_event.set()
            for p in self.worker_procs + [self.evaluator_proc]:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()
```

**Skeleton for `persistent_evaluator_loop`:**

```python
def persistent_evaluator_loop(weights_path, bw, bh, request_queue, response_queues,
                               stats_queue, weight_event, shutdown_event,
                               eval_batch_size, eval_timeout_ms, use_gpu):
    set_cpu_threads(1)
    net = PolicyValueNet(bw, bh, model_file=weights_path, use_gpu=use_gpu)
    timeout_sec = max(0.001, eval_timeout_ms / 1000.0)

    while not shutdown_event.is_set():
        # Hot-reload on signal
        if weight_event.is_set():
            try:
                state = torch.load(weights_path, map_location=net.device)
                net.policy_value_net.load_state_dict(state)
                net.policy_value_net.eval()
            except Exception as e:
                # Log but keep serving old weights — corruption recovery
                print(f"[evaluator] reload failed: {e}", flush=True)
            weight_event.clear()

        # Collect a batch
        pending = []
        try:
            msg = request_queue.get(timeout=0.1)
            if msg.get("type") == "shutdown":
                break
            pending.append(msg)
        except queue.Empty:
            continue

        deadline = time.time() + timeout_sec
        while len(pending) < eval_batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                msg = request_queue.get(timeout=remaining)
                if msg.get("type") == "shutdown":
                    shutdown_event.set()
                    break
                pending.append(msg)
            except queue.Empty:
                break

        if not pending:
            continue

        states = np.stack([p["state"] for p in pending]).astype(np.float32)
        priors_b, values_b = net.policy_value(states)

        for item, priors, value in zip(pending, priors_b, values_b):
            wid = int(item["worker_id"])
            response_queues[wid].put({
                "type": "eval_result",
                "request_id": item["request_id"],
                "act_probs": priors.astype(np.float32, copy=False),
                "value": float(np.asarray(value).reshape(-1)[0]),
            })
```

**Skeleton for `persistent_worker_loop`:**

```python
def persistent_worker_loop(wid, bw, bh, n_in_row, n_playout, c_puct,
                            dirichlet_alpha, noise_eps, vl_k, n_vl,
                            request_queue, response_queue, replay_queue,
                            shutdown_event, response_timeout):
    set_cpu_threads(1)
    seed_worker(wid)
    board = Board(width=bw, height=bh, n_in_row=n_in_row)
    game = Game(board)
    client = RemotePolicyValueClient(wid, bw, bh, request_queue, response_queue,
                                      response_timeout=response_timeout)
    mcts_player = MCTSPlayer(
        policy_value_function=client.policy_value_fn,
        policy_value_batch_function=client.policy_value_batch_fn,
        c_puct=c_puct, n_playout=n_playout, is_selfplay=1,
        dirichlet_alpha=dirichlet_alpha, noise_eps=noise_eps,
        vl_k=vl_k, n_vl=n_vl,
    )
    while not shutdown_event.is_set():
        winner, play_data = game.start_self_play(mcts_player, temp=1.0,
                                                 temperature_moves=8)  # see Task 7
        play_data = list(play_data)
        augmented = get_equi_data(play_data, bw, bh)
        try:
            replay_queue.put(augmented, timeout=10.0)
        except queue.Full:
            # Trainer is behind; drop this game rather than block.
            # In practice with a properly-sized queue, shouldn't happen.
            pass
```

**Validation.**
- Running 10 minutes of training with the new pipeline should produce ≥ 2× more
  self-play games than the synchronous baseline in the same wall time.
- GPU utilization (nvidia-smi `dmon`) should stay > 60% during steady state, vs the old
  pipeline's sawtooth pattern.
- No hangs on Ctrl-C: shutdown event must propagate. Test by killing trainer mid-run and
  confirming all child processes exit within 30 s.
- Replay buffer should never go empty during training (good steady-state). If it does,
  workers can't keep up — increase `num_workers` or decrease `n_playout`.

**Crash recovery.** A worker that crashes (e.g., due to a bug in MCTS) should not hang the
trainer. Wrap `persistent_worker_loop` in a try/except that pushes an error to a separate
error_queue, then exits. The trainer monitors `worker_proc.is_alive()` and respawns
crashed workers.

---

### Task 5 — GPU Evaluator Inference Optimizations

**Priority:** P1.

**Files touched:** `policy_value_net_pytorch.py`, `train_gpu_evaluator.py` (evaluator
initialization).

This task bundles three independent inference-side optimizations.

#### 5a. `channels_last` memory format + persistent FP16

Conv2d on Tensor Cores requires NHWC layout. Combined with FP16, it gives 1.2–1.5× on
modern GPUs even for small models.

```python
# In persistent_evaluator_loop init:
net = PolicyValueNet(bw, bh, model_file=weights_path, use_gpu=use_gpu, use_amp=False)
net.policy_value_net = net.policy_value_net.to(memory_format=torch.channels_last)
if use_gpu:
    net.policy_value_net = net.policy_value_net.half()
net.policy_value_net.eval()

# In policy_value (or a new policy_value_inference method):
def policy_value_inference(self, state_batch):
    self.policy_value_net.eval()
    with torch.no_grad():
        x = torch.from_numpy(state_batch).pin_memory().to(self.device, non_blocking=True)
        if self.use_gpu:
            x = x.half()
        x = x.to(memory_format=torch.channels_last)
        log_p, v = self.policy_value_net(x)
    return torch.exp(log_p.float()).cpu().numpy(), v.float().cpu().numpy()
```

Important: only the *evaluator process* uses FP16 inference. The *trainer* keeps FP32
weights with AMP autocast inside `train_step`. This is consistent with the codebase's
existing AMP pattern.

When weights are reloaded in the evaluator, re-apply `.half()` and `channels_last`:

```python
state = torch.load(weights_path, map_location=net.device)
net.policy_value_net.load_state_dict(state)              # loads as fp32
net.policy_value_net = net.policy_value_net.half().to(memory_format=torch.channels_last)
net.policy_value_net.eval()
```

#### 5b. CUDA Graphs for fixed-batch inference

CUDA Graphs capture a static sequence of kernels and replay it with new inputs, eliminating
Python and kernel-launch overhead. For a 10×128 ResNet on small input, launch overhead is a
significant fraction of forward time.

```python
class CudaGraphInferenceWrapper:
    def __init__(self, net, batch_size, in_shape, device, dtype=torch.float16):
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.static_in = torch.zeros((batch_size, *in_shape),
                                     device=device, dtype=dtype) \
                              .to(memory_format=torch.channels_last)
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = net(self.static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_log_p, self.static_v = net(self.static_in)

    def run(self, batch_np):
        n = batch_np.shape[0]
        if n > self.batch_size:
            raise ValueError(f"batch {n} > captured size {self.batch_size}")
        # Pad to fixed batch
        host = torch.from_numpy(batch_np).pin_memory()
        if self.dtype == torch.float16:
            host = host.half()
        self.static_in[:n].copy_(host.to(self.device, non_blocking=True),
                                 non_blocking=True)
        if n < self.batch_size:
            self.static_in[n:].zero_()
        self.graph.replay()
        torch.cuda.synchronize()
        priors = torch.exp(self.static_log_p[:n].float()).cpu().numpy()
        values = self.static_v[:n].float().cpu().numpy()
        return priors, values
```

In the evaluator, capture the graph at startup with `batch_size = eval_batch_size` (e.g.,
256). When fewer than `eval_batch_size` requests are pending, pad to full size. Padding
overhead is small (zeros copy fast), and replay is much cheaper than eager forward.

⚠ **Weight reload + CUDA Graphs.** When weights change, the captured graph still holds
references to the old tensors via its parameters. CUDA Graphs in PyTorch *do* see updated
parameter values automatically because they share the same memory addresses, BUT this
relies on `load_state_dict` being in-place. Confirm by:

```python
old_id = net.policy_value_net.stem[0].weight.data_ptr()
net.policy_value_net.load_state_dict(new_state)
new_id = net.policy_value_net.stem[0].weight.data_ptr()
assert old_id == new_id, "load_state_dict reallocated; CUDA Graph will be stale"
```

If the assert fails, recapture the graph after every reload (relatively cheap — ~200 ms).

#### 5c. Larger eval_batch_size for Gomoku 15×15

The codebase's default `eval_batch_size=128` was chosen with Go-sized inputs in mind. For
Gomoku 15×15, input tensor size is `4 × 15 × 15 × 4 bytes = 3.6 KB` per state, vs Go's
`17 × 19 × 19 × 4 bytes = 24.5 KB`. With 7× smaller states, GPU memory permits much larger
batches.

Recommended: `eval_batch_size = 256` initially. With Task 1 (`vl_k=4`) and 16 workers, the
inflight pool is ~64 — nowhere near 256, so the timeout will fire and batches will be
smaller. But CUDA Graphs run a fixed-size graph anyway, so the cost of "256 captured but
only 60 pending" is just the zero-pad. Increase to 512 only if measurements show the GPU
underutilized at 256 with high inflight load.

**Validation.**
- Microbenchmark `policy_value_inference(batch)` for batch = {1, 8, 32, 64, 128, 256}
  before and after. Expect 1.5–3× speedup at batch ≥ 32 with FP16+channels_last+CUDA
  Graphs combined.
- Strength check: 50-game match between FP32-trained model evaluated FP16 vs FP32. For
  AlphaZero-scale networks the gap is typically below noise (< 2 Elo). If evaluator uses
  FP16 but training is FP32, this gap is benign as long as it's sub-noise.

---

### Task 6 — Markov State Caching in `Board`

**Priority:** P2.

**Files touched:** `game.py`.

**Problem.** `Board.current_state()` reconstructs all 4 feature planes from scratch on
every call. It is called once per ply during self-play (to log the state for training)
and once per leaf during MCTS (`copy_fast` followed by `current_state()`). For Gomoku,
where the state is fully Markov and `do_move` only flips a small region, this is wasteful.

**Design.** Maintain feature planes incrementally:

```python
class Board:
    def init_board(self, ...):
        ...
        # planes[0]: current player's stones
        # planes[1]: opponent's stones
        # planes[2]: last move marker
        # planes[3]: color-to-play indicator (constant within a ply)
        self._planes = np.zeros((4, self.height, self.width), dtype=np.float32)
        self._last_move_loc = None
        # planes[3] is filled based on current_player at state read time

    def do_move(self, move):
        h, w = self.move_to_location(move)   # or whatever the existing API is
        self._planes[0, h, w] = 1.0           # add current player's stone

        # Update last-move plane
        if self._last_move_loc is not None:
            lh, lw = self._last_move_loc
            self._planes[2, lh, lw] = 0.0
        self._planes[2, h, w] = 1.0
        self._last_move_loc = (h, w)

        # Player switch: swap planes 0 and 1
        self._planes[[0, 1]] = self._planes[[1, 0]]

        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[1] if self.current_player == self.players[0] \
                              else self.players[0]
        self.last_move = move

    def current_state(self):
        # planes 0–2 are kept up to date by do_move.
        # plane 3 needs a fresh fill since it depends on current_player.
        if self.current_player == self.players[0]:
            self._planes[3].fill(1.0)
        else:
            self._planes[3].fill(0.0)
        return self._planes  # caller must NOT mutate; or return a copy if they will
```

**Caveat — `copy_fast` and `_planes`.** When MCTS calls `copy_fast`, it must clone
`_planes` because `do_move` will mutate it in-place:

```python
def copy_fast(self):
    new = Board.__new__(Board)
    new.width = self.width
    new.height = self.height
    new.n_in_row = self.n_in_row
    new.players = self.players
    new.states = dict(self.states)
    new.availables = list(self.availables)
    new.current_player = self.current_player
    new.last_move = self.last_move
    new._planes = self._planes.copy()                 # 4 × 15 × 15 × 4 bytes ≈ 3.6 KB
    new._last_move_loc = self._last_move_loc
    return new
```

A 3.6 KB numpy copy is dramatically faster than reconstructing 4 planes from scratch on
every `current_state` call from a Python dict. Net win expected: 3–5× faster MCTS leaf
preparation.

**Validation.**
- Equivalence test: random 100 boards, compute `current_state()` with old code and new
  code; arrays must be identical.
- Mutation safety: after `b.do_move(m)`, `b._planes` must reflect new state (test by
  manual assertion on a known position).

---

### Task 7 — Temperature Schedule and Dirichlet α Tuning

**Priority:** P0 (cheapest change, affects data quality).

**Files touched:** `game.py` (`Game.start_self_play`), `mcts_alphaZero.py` (`MCTSPlayer`).

**Paper reference:** Self-play. "For the first 30 moves of each game, the temperature is
set to τ=1; this selects moves proportionally to their visit count in MCTS, and ensures a
diverse set of positions are encountered. For the remainder of the game, an infinitesimal
temperature is used, τ→0."

**Problem.** `MCTSPlayer.get_action` is called with a fixed `temp` (default 1e-3 in eval,
sometimes 1.0 in self-play depending on call site). The current `Game.start_self_play`
passes a single `temp` for the whole game. With τ→0 throughout, opening positions in the
replay buffer are repetitive (the network always plays its current best opening), starving
the network of exposure to diverse openings.

**Design.** Add a `temperature_moves` parameter to `start_self_play`. For move index
< `temperature_moves`, use τ=1; afterwards, τ=1e-3.

```python
# game.py
def start_self_play(self, player, is_shown=0, temperature_moves=8,
                    temp_high=1.0, temp_low=1e-3):
    self.board.init_board()
    p1, p2 = self.board.players
    states, mcts_probs, current_players = [], [], []
    move_idx = 0
    while True:
        cur_temp = temp_high if move_idx < temperature_moves else temp_low
        move, move_probs = player.get_action(self.board, temp=cur_temp, return_prob=1)
        states.append(self.board.current_state())
        mcts_probs.append(move_probs)
        current_players.append(self.board.current_player)
        self.board.do_move(move)
        move_idx += 1
        if is_shown:
            self.graphic(self.board, p1, p2)
        end, winner = self.board.game_end()
        if end:
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            player.reset_player()
            return winner, zip(states, mcts_probs, winners_z)
```

**Note.** The `current_state()` returned from `Board` after Task 6 returns a reference to
the internal `_planes`. Ensure `start_self_play` snapshots it (e.g.,
`states.append(self.board.current_state().copy())`) — otherwise all stored states will be
identical (the latest one).

**Recommended values for Gomoku 15×15:**
- `temperature_moves = 8` (≈20% of an average game; paper uses ~12% but Gomoku is shorter
  so a slightly higher fraction maintains opening diversity).
- Dirichlet α: paper formula `α ≈ 10/avg_legal_moves`. For Gomoku 15×15:
  - Root (start of game): 224 legal → α ≈ 0.045.
  - Use **`dirichlet_alpha = 0.05`** as the default. Current default is 0.03 (Go value),
    which under-explores for Gomoku.
- `noise_eps = 0.25` (unchanged from paper).

These three values (`temperature_moves`, `dirichlet_alpha`, `noise_eps`) should be
exposed as command-line arguments for sweeping.

**Validation.**
- After Task 7 only, compare entropy of move probabilities `mcts_probs` from the first 8
  plies of self-play games before and after. Should increase materially.
- Check unique opening sequences in the replay buffer over 100 games. Before: a handful
  of dominant openings. After: dozens of distinct sequences in the first 8 plies.

---

### Task 8 — Win-in-1 Leaf Shortcut (Optional, Purity Tradeoff)

**Priority:** P2 (optional).

**Files touched:** `mcts_alphaZero.py`, `game.py` (need `Board.is_winning_move(action)`).

**Paper purity note.** The paper claims AlphaGo Zero uses no domain knowledge beyond rules.
A "win-in-1" detector falls in a gray zone:
- *Pro-shortcut*: detecting a 5-in-a-row from the rule set is using rule knowledge, not
  pattern heuristics. Same kind as the paper's terminal check.
- *Con*: the network is supposed to *learn* tactical recognition; shortcutting may dampen
  the gradient signal that teaches it to value tactics correctly.

For research-faithful runs, skip this task. For wall-clock-optimized production runs, it's
worth 10–20% NN-eval reduction in late training when the network plays tactically.

**Design.** When expanding a leaf, before calling the network, check if any legal move
ends the game. If yes, set its prior to 1.0, others to 0.0, and back up `+1` immediately.

```python
def _check_win_in_one(self, sim_state):
    """Return action index that wins immediately, or None. O(legal × 4) directions."""
    cur = sim_state.get_current_player()
    for a in sim_state.availables:
        # Need an O(1) "would_win_here" check; if Board exposes is_winning_move(a), use it.
        # Otherwise, simulate + check + undo (requires Task 6's incremental planes to be
        # cheap to revert, OR a dedicated is_winning_move on Board).
        if sim_state.is_winning_move(a, cur):
            return a
    return None

# In the search loop, after detecting a non-terminal leaf:
win_action = self._check_win_in_one(sim_state)
if win_action is not None:
    # Synthesize a one-hot prior and a winning value
    leaf.expand([(win_action, 1.0)])
    self._backup_and_revert(path, -1.0)   # leaf player wins → negate for parent
    continue
# else: proceed to NN batch
```

**Required**: a fast `Board.is_winning_move(action, player)` that doesn't allocate. A clean
implementation walks 4 directions (horizontal, vertical, two diagonals) from `action` and
counts consecutive `player` stones (including the hypothetical one at `action`). For
`n_in_row=5`, this is at most 16 cell reads.

```python
# game.py
def is_winning_move(self, move, player):
    h, w = self.move_to_location(move)
    n = self.n_in_row
    # 4 directions
    for dh, dw in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        count = 1
        # forward
        for k in range(1, n):
            hh, ww = h + k * dh, w + k * dw
            if 0 <= hh < self.height and 0 <= ww < self.width \
                    and self.states.get(self.location_to_move((hh, ww))) == player:
                count += 1
            else:
                break
        # backward
        for k in range(1, n):
            hh, ww = h - k * dh, w - k * dw
            if 0 <= hh < self.height and 0 <= ww < self.width \
                    and self.states.get(self.location_to_move((hh, ww))) == player:
                count += 1
            else:
                break
        if count >= n:
            return True
    return False
```

**Validation.** Self-play with shortcut on/off; compare playing strength via 200-game
match. Shortcut should be non-regressing or marginally better. If strength regresses, the
sign of the backup is wrong or the value `+1` should be from a different perspective.

---

### Task 9 — Resignation Threshold

**Priority:** P2.

**Files touched:** `mcts_alphaZero.py` (`MCTSPlayer` exposes root Q), `game.py`
(`Game.start_self_play`), `train_gpu_evaluator.py` (auto-tune logic).

**Paper reference:** Self-play. "the resignation threshold v_resign is selected
automatically to keep the fraction of false positives ... below 5%. To measure false
positives, we disable resignation in 10% of self-play games and play until termination."

**Problem.** Every game is currently played to terminal. For Gomoku this means the loser
plays out 5–15 obviously-lost moves at the end. Less savings than in Go (where lost games
can drag on for 50+ moves), but still worthwhile.

**Design.**

1. `MCTSPlayer.get_action` exposes the root Q value after search:

   ```python
   self.last_root_value = self.mcts._root.Q()  # in -1..+1
   ```

2. `Game.start_self_play` accepts a `v_resign` argument and a `disable_resign` flag (set
   to True for 10% of self-play games). When the player to move has root Q < `v_resign`
   for two consecutive moves and the game has progressed past some minimum (e.g., 10
   plies), resign that side.

3. The trainer maintains an estimate of false-positive rate:
   - For each `disable_resign=True` game, record (player_who_would_have_resigned,
     min_root_q_for_that_player, actual_winner). False positive = the would-have-resigned
     side actually won the played-out game.
   - Adjust `v_resign` to keep FP rate ≤ 5% over a rolling window of, say, 100
     no-resign games.

   Adjustment rule (simple, robust):
   ```
   if fp_rate > 0.05:  v_resign -= 0.05  (more conservative — resign less often)
   if fp_rate < 0.02:  v_resign += 0.05  (more aggressive — resign more often)
   clamp v_resign to [-0.95, -0.5]
   ```

   Initial `v_resign = -0.85`.

**Resignation tuple labeling.** When a side resigns, the game is scored as a win for the
opponent. All training tuples from the resigning side carry `z = -1`; the opponent's
tuples carry `z = +1`. This is the same as if the resigning side had played to a real loss.

**Validation.**
- Average game length should drop from ~40 to ~30–32 plies.
- Total wall-clock for a fixed number of self-play games: 10–20% reduction.
- FP rate should hover around 3–5% in steady state. Log this rate.
- Strength: 200-game match between resigned-trained model and play-to-end-trained model
  must show resigned model ≥ play-to-end model. If it's worse, FP rate is too high or
  the resigning side's tuples are mislabeled.

---

### Task 10 — Shared-Memory State Transfer (Refinement of Task 4)

**Priority:** P3.

**Files touched:** `train_gpu_evaluator.py` (queue setup, `RemotePolicyValueClient`,
`gpu_evaluator_loop`).

**Problem.** Currently, every NN request goes through `mp.Queue.put(state_array)`, which
pickles the numpy array. With high request rates (e.g., 16 workers × 800 playouts × ~30
plies / few seconds = ~10⁵ requests/sec), pickle cost is non-trivial.

**Design.** Use `multiprocessing.shared_memory.SharedMemory` (Python 3.8+). Allocate two
ring buffers at startup:

- `shm_in`: float32 array of shape `(N_SLOTS, in_channels, H, W)`.
- `shm_out`: float32 array of shape `(N_SLOTS, board_size + 1)` (priors + value).

A free-list of slot indices (a ctx.Queue of ints) gates allocation. Workers acquire a
slot, write state into `shm_in[slot]`, send `(slot, worker_id, request_id)` over the
request queue, wait for response with same `slot`, read `shm_out[slot]`, release slot.

```python
# Setup (in main, before spawning):
N_SLOTS = max(1024, num_workers * vl_k * 4)
shm_in = SharedMemory(create=True, size=N_SLOTS * in_channels * H * W * 4)
shm_out = SharedMemory(create=True, size=N_SLOTS * (H * W + 1) * 4)
slot_pool_queue = ctx.Queue()
for i in range(N_SLOTS):
    slot_pool_queue.put(i)

# Pass shm names to workers + evaluator; they reattach via SharedMemory(name=...).

# Worker:
slot = slot_pool_queue.get()
in_view = np.ndarray((N_SLOTS, in_channels, H, W), dtype=np.float32, buffer=shm_in.buf)
out_view = np.ndarray((N_SLOTS, H*W + 1), dtype=np.float32, buffer=shm_out.buf)
in_view[slot] = state
self.request_queue.put((slot, self.worker_id, rid))
# wait response
self.response_queue.get(timeout=...)  # the response now just carries (slot, rid)
priors = out_view[slot, :H*W].copy()
value = float(out_view[slot, H*W])
slot_pool_queue.put(slot)
```

⚠ **Cleanup.** Shared memory survives process exit on Linux unless explicitly unlinked.
The trainer must `shm_in.close(); shm_in.unlink()` in a `finally` block.

⚠ **Race condition.** A worker must not release its slot until *after* it has read the
output, and must not write a new state into a slot until it has been re-acquired. The
free-list queue handles this naturally as long as `release == queue.put` happens after
read.

**When to do this task.** Only after Task 4 is in place and profiling confirms queue
serialization is a bottleneck (`mp.queues.put` showing up in cProfile or bench tool). With
< 50% GPU utilization at steady state and small batches, this task could move the needle
2–4×. With ≥ 70% GPU util, it won't matter much.

---

### Task 11 — Adaptive Batch Timeout (Refinement of Task 4)

**Priority:** P3.

**Files touched:** `train_gpu_evaluator.py` (`persistent_evaluator_loop`).

**Problem.** A fixed `eval_timeout_ms` (default 8) is a blunt instrument. Under low load
(few inflight requests), the timeout fires before a useful batch accumulates. Under high
load, batches fill before timeout — timeout doesn't matter. The bad case is the transition
zone where average batch size is, say, 20% of `eval_batch_size`.

**Design.** Adapt timeout based on recent average batch fill ratio:

```python
class AdaptiveTimeout:
    def __init__(self, init_ms=8, min_ms=1, max_ms=30):
        self.timeout_ms = init_ms
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.recent_fills = deque(maxlen=100)  # fill ratio per batch

    def update(self, fill_ratio):
        self.recent_fills.append(fill_ratio)
        if len(self.recent_fills) < 20:
            return
        avg = sum(self.recent_fills) / len(self.recent_fills)
        if avg < 0.4:                # batches too small — wait longer
            self.timeout_ms = min(self.timeout_ms * 1.3, self.max_ms)
        elif avg > 0.85:             # batches usually full — can wait less
            self.timeout_ms = max(self.timeout_ms * 0.85, self.min_ms)

# In evaluator loop:
adaptive = AdaptiveTimeout()
# ...
fill = len(pending) / float(eval_batch_size)
adaptive.update(fill)
timeout_sec = adaptive.timeout_ms / 1000.0
```

**Validation.** Log `(timeout_ms, fill_ratio)` periodically. At steady state, expect
`fill_ratio` to converge toward 0.6–0.8 (a healthy zone) and `timeout_ms` to reach a
stable point.

---

## 4. Configuration Recommendations

Concrete starting hyperparameters for Gomoku 15×15 after all tasks are landed:

| Parameter | Old default | New default | Notes |
|---|---|---|---|
| `num_workers` | 10 | 16 | More inflight requests → larger GPU batches |
| `n_playout` (self-play) | 800 | 800 | Unchanged |
| `eval_n_playout` | 1600 | 1600 | Unchanged |
| `vl_k` | n/a | 4 | New: leaf-parallel virtual loss group size |
| `n_vl` | n/a | 1.0 | New: virtual loss magnitude. Do not start at 3 |
| `max_oversample` | n/a | 3 | New: terminal overshoot factor |
| `c_puct` | 3.0 | 3.0 | Unchanged. Re-tune later if strength stalls |
| `dirichlet_alpha` | 0.03 | 0.05 | Adapted: `≈10/avg_legal_moves` for Gomoku 15×15 |
| `noise_eps` | 0.25 | 0.25 | Unchanged |
| `temperature_moves` | n/a | 8 | New: τ=1 for first 8 plies, then τ→0 |
| `eval_batch_size` | 128 | 256 | Larger batch fits Gomoku state well |
| `eval_timeout_ms` | 8 | 8 (init for adaptive) | After Task 11 |
| `batch_size` (train) | 512 | 512 | Unchanged. Increase to 1024 if GPU memory allows |
| `buffer_size` | 500_000 | 1_500_000 | Each Gomoku game = ~280 augmented positions; need more games |
| `recent_sample_window` | 200_000 | 500_000 | Wider window covers ~1800 distinct games |
| `weight_push_every` | n/a | 4 (updates) | New: how often to flush weights to evaluator |
| `eval_every` | 50 (game batches) | 100 (updates) | Reinterpreted in async pipeline |
| `v_resign` | n/a | -0.85 (initial) | Auto-tuned thereafter |
| `disable_resign_prob` | n/a | 0.10 | For FP rate measurement |

These are starting points. Sweep with the benchmarking protocol below before committing.

---

## 5. Benchmarking Protocol

Two kinds of benchmarks: **throughput** (does it run faster?) and **strength** (does it
play better?). Both must pass for any task to be considered done.

### 5.1 Throughput benchmark

Run for 30 minutes from a fixed initial checkpoint. Report:

- Self-play games completed
- Total NN eval requests served
- Average GPU evaluator batch size
- Average wall-clock per game
- GPU utilization (sample `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1`)

Each task should not regress any of these vs. the immediately preceding state. Tasks 1+2+3
should improve games/30-min by ≥ 2× and average batch by ≥ 4× vs baseline. Task 4 should
improve games/30-min by another 1.5–2× on top.

### 5.2 Strength benchmark

After each task is implemented and integrated, run a 200-game match between the new build
and the previous build, both starting from the *same* initial checkpoint and trained for
the *same* wall-clock budget (e.g., 4 hours). Use 1600 simulations per move for the match,
both sides, alternating start_player every 2 games. Compute Elo difference with
`celo = 1/400`.

Acceptance criterion: new ≥ old at p < 0.05 (binomial test against 50%) — i.e., new wins
≥ 113 / 200 games. If below, suspect a correctness bug.

### 5.3 Sanity tests after each task

- **Tasks 1–3**: visit-count equivalence test on toy game (see §3, Task 1 validation).
- **Task 4**: graceful Ctrl-C; replay queue never empty for > 30 s in steady state.
- **Task 5**: FP16 evaluator ≤ FP32 by < 2 Elo on identical weights.
- **Task 6**: `current_state()` byte-identical before/after.
- **Task 7**: opening-position diversity in buffer increases.
- **Task 8**: 200-game match shows shortcut model ≥ no-shortcut model.
- **Task 9**: average game length decreases; FP rate ≈ 3–5% in steady state.

---

## 6. Pitfalls Specific to Gomoku

### 6.1 Virtual loss too aggressive

In Go, paper uses `n_vl = 3`. **In Gomoku, this is wrong.** Gomoku tactics are forced
sequences (open three, four-threat, double-three) where there is often *one* move to
play. With `n_vl = 3`, after one collector finds the right move, the next two collectors
will be pushed to other moves entirely, wasting their NN eval and weakening the search's
focus on the only winning line. Start at `n_vl = 1`. If you want to experiment higher,
sweep up to 2 max — verify with strength matches.

### 6.2 Terminal density breaking batches

Gomoku has high terminal density in the search tree (5-row threats reach terminal in 2–4
plies). A naive "pick K leaves, batch them" without oversampling will produce many batches
of size 1 or 2 because the rest were terminal. The `max_oversample` mechanism in Task 1
solves this. *Do not skip oversampling.*

### 6.3 Dirichlet noise interacting with deep tree reuse

When `MCTSPlayer.update_with_move(move)` reuses the subtree, Dirichlet noise was applied
to the *old root*, but the new root is what was previously a child. The new root's priors
no longer have noise. The current code handles this correctly (noise is added in
`get_action` after the tree update). When implementing virtual loss with W/N storage,
ensure the noise is added to `_priors` (not `_W`) and that subsequent visits accumulate
into `_W`/`_N` cleanly. *Test by inspecting root priors before/after `update_with_move` +
`get_action`*.

### 6.4 `current_state` reference vs copy

After Task 6, `Board.current_state()` returns a reference to internal `_planes`. The
caller in `start_self_play` *appends* this reference to a list. Without a `.copy()`, all
list entries point to the same array — by end of game, all of them show the final state.
**This bug will silently produce garbage training data**. Test fix: after one game,
assert `np.any(states[0] != states[-1])`.

### 6.5 Replay buffer staleness

A replay buffer of 1.5M positions sounds huge, but in Gomoku that's only ~5400 games
(after 8× augmentation). If the network is improving rapidly, the policy that generated
the oldest games in the buffer is meaningfully different from current — leading to
distribution shift in training. Mitigation:

- Keep `recent_sample_window` ≤ 1/3 of `buffer_size`.
- Track the median "age in updates" of sampled positions. If > 50, increase
  `weight_push_every` (push more often) or shrink the window.

### 6.6 First-player bias

Gomoku without any rule (no swap2, no opening restriction) gives a substantial
first-player advantage on 15×15. If self-play games are scored as
`1 if winner == current_player else -1` and no opening rule is applied, the first player
wins ~60–70% of self-play games at convergence, biasing the value head. Two options:

- **Status quo**: live with it; the network learns "moving first is worth +0.4" and
  compensates. This is what the codebase does today.
- **Add swap2 or pro-style opening rule** at the game level. Out of scope for this
  optimization document but worth noting if value-head MSE refuses to converge.

### 6.7 Worker silently dies and the pipeline stalls

In Task 4's persistent-worker design, if one worker crashes due to a bug (e.g., an MCTS
error after specific board state), it stops producing games but doesn't take down the
trainer. The replay queue gets thinner, training continues but on stale data. Mitigation:
trainer monitors `worker_proc.is_alive()` every minute. If a worker is dead, log + respawn.

### 6.8 CUDA Graphs + AMP scaler

If you accidentally apply AMP autocast inside the captured region, scaler state gets
captured too — gives wrong-looking outputs after a few replays. CUDA Graphs in the
*evaluator* should be capturing pure FP16 forward (no autocast, no scaler). AMP belongs
in the *trainer*, not the evaluator. Keep them strictly separate.

### 6.9 Not all of `mcts_pure` evaluation should be removed

Task 4's design suggests replacing "evaluate vs MCTS_pure" with "evaluate best vs
current". Keep `mcts_pure` as a periodic sanity check (e.g., every 1000 updates). If the
network ever loses to pure MCTS at, say, 5000 playouts, something has gone catastrophically
wrong — best-vs-current can't catch this kind of regression because both networks degrade
together.

---

## 7. Summary — Recommended Execution Order

1. **Task 7** (Temperature schedule + Dirichlet α=0.05). 1 hour. Improves data quality,
   gives baseline.
2. **Tasks 1 + 2 + 3** as a single PR (virtual-loss MCTS + `Board.copy_fast` + numpy
   children arrays). 2–3 days. Largest single throughput win.
3. **Task 4** (persistent async pipeline). 1–2 days. Largest pipeline-level win.
4. **Task 5** (FP16 + channels_last + CUDA Graphs + larger batch). 0.5–1 day.
5. **Task 6** (Markov state caching). 0.5 day.
6. **Task 9** (Resignation threshold). 0.5 day.
7. **Task 8** (Win-in-1 shortcut, OPTIONAL). 0.5 day. Skip if pursuing paper purity.
8. **Task 10** (Shared memory state transfer). 1 day. Only if profiling shows queue
   serialization is a bottleneck.
9. **Task 11** (Adaptive timeout). 0.25 day. Quality-of-life refinement.

After all P0 tasks (7, 1+2+3, 4): expect ~5–10× wall-clock speedup and meaningfully
better training data quality. After all P1 tasks added (5): another 1.5–2×. After P2
tasks (6, 9): incremental gains, ~10–20% each. P3 tasks (10, 11) are diminishing-returns.

The fastest path to a working production-grade pipeline is therefore the sequence
**7 → 1+2+3 → 4 → 5**, after which the trainer should look very close to the paper's
described architecture, properly adapted for Gomoku.

---

## Appendix A — Files Modified Summary

| File | Tasks touching it |
|---|---|
| `mcts_alphaZero.py` | 1, 3, 7, 8, 9 |
| `game.py` | 2, 6, 7, 8, 9 |
| `policy_value_net_pytorch.py` | 5 |
| `train_gpu_evaluator.py` | 1, 4, 5, 9, 10, 11 |

## Appendix B — New Command-Line Arguments to Add

```
--vl-k                 (default 4)   leaf-parallel virtual-loss group size
--n-vl                 (default 1.0) virtual loss magnitude
--max-oversample       (default 3)   terminal overshoot factor in MCTS
--temperature-moves    (default 8)   plies of τ=1 before switching to τ→0
--weight-push-every    (default 4)   updates between weight pushes to evaluator
--v-resign             (default -0.85) initial resignation threshold
--disable-resign-prob  (default 0.10)  fraction of self-play with no resignation
--use-cuda-graphs      (default True) enable CUDA Graph capture in evaluator
--inference-fp16       (default True) FP16 + channels_last in evaluator
```

## Appendix C — Smoke Test Script Recommended

After each task, run a 5-minute smoke test before launching real training:

```bash
python train_gpu_evaluator.py \
  --num-workers 4 --n-playout 200 \
  --batch-size 128 --game-batch-num 5 \
  --check-freq 5 --eval-games 0 \
  --vl-k 4 --n-vl 1.0 --temperature-moves 8 \
  --dirichlet-alpha 0.05 --eval-batch-size 64
```

Expected: completes 5 batches in < 5 minutes, no errors, no zombie processes
(`pgrep -f selfplay-worker` returns empty after exit), `current_policy.model` is updated.

---

*End of document.*
