"""
One-shot script to rewrite rit_star.py to match Algorithm 1 from main_v3.tex.

Changes:
  1. Add self._x_samples persistent sample pool in __init__
  2. Rewrite plan() to match Alg 1 flow: prune -> update I_R -> sample -> pool -> r -> Q_E -> process -> CARM -> stats, with smoothing at end
  3. Rewrite plan_stepwise() to match the same flow
  4. Replace _extend_tree() with _process_edge_queue() + _rewire_neighbours()
  5. Modify _prune() to prune the sample pool
"""

import sys

filepath = '/home/muhayy/TRO-old/rit_star/rit_star.py'
with open(filepath) as f:
    lines = f.readlines()

# ── locate method boundaries (0-indexed line numbers) ──────────────
markers = {}
for i, line in enumerate(lines):
    stripped = line.strip()
    if 'self.c_best = np.inf' in line and 'markers' not in line:
        markers['c_best_init'] = i
    if stripped.startswith('def plan(self) -> Tuple'):
        markers['plan_start'] = i
    if stripped.startswith('def plan_stepwise(self):'):
        markers['stepwise_start'] = i
    if stripped.startswith('def get_stats(self) -> list:'):
        markers['stats_start'] = i
    if stripped.startswith('def _extend_tree(self, samples'):
        markers['extend_start'] = i
    if stripped.startswith('def _maybe_rebuild_carm_cache(self'):
        markers['rebuild_start'] = i
    if stripped.startswith('def _prune(self):'):
        markers['prune_start'] = i
    if stripped.startswith('def _update_informed_set(self):'):
        markers['informed_start'] = i

for key in ['c_best_init', 'plan_start', 'stepwise_start', 'stats_start',
            'extend_start', 'rebuild_start', 'prune_start', 'informed_start']:
    if key not in markers:
        print(f"ERROR: could not find marker '{key}'")
        sys.exit(1)
    print(f"  {key}: line {markers[key]+1}")

# ── new code blocks ────────────────────────────────────────────────

INIT_INSERT = """\
        self._x_samples: List[np.ndarray] = []  # persistent sample pool (Alg 1)
"""

NEW_PLAN = '''\
    def plan(self) -> Tuple[List[np.ndarray], float]:
        """Run the RIT* planning loop (Algorithm 1).

        Returns
        -------
        path : list of (d,) arrays from start to goal, or empty if no
               solution found.
        cost : float -- final best cost (inf if unsolved).
        """
        self._t0 = time.time()

        for it in range(self.max_iterations):
            # Phase 1: Prune, update I_R, sample (Alg 1, lines 3-7)
            if self.c_best < np.inf:
                self._prune()
                self._update_informed_set()

            new_samples = self._sample_batch(it)

            # Add collision-free samples to persistent pool
            for s in new_samples:
                is_goal = np.allclose(s, self.x_goal, atol=1e-8)
                if is_goal and self.goal_node is not None:
                    continue  # goal already in tree; rewire handles updates
                if self.collision_free(s):
                    self._x_samples.append(s.copy())
                elif self._adaptive_mode:
                    self._carm.add_collision_point(s)

            # Compute connection radius over |V| + |X_samples|
            n_total = len(self.vertices) + len(self._x_samples)
            r = self._compute_connection_radius(n_total)

            # Phase 2: Build & process edge queue (Alg 1, lines 8-14)
            self._process_edge_queue(r)

            # Phase 3: CARM feedback (Alg 1, lines 15-17)
            if self._adaptive_mode:
                self._maybe_rebuild_carm_cache(it)

            elapsed = time.time() - self._t0
            self._record_stats(it, elapsed)

        # Post-process: MultiStrategySmooth (Alg 1, line 20)
        path = self._extract_path()
        if path and len(path) > 1:
            path = self._shortcut_path(
                path, n_attempts=max(300, 100 * self.dim))
            exact = sum(riemannian_edge_cost(path[i], path[i + 1], self.metric)
                        for i in range(len(path) - 1))
            self.c_best = exact
            if self._stats:
                self._stats[-1][\'c_best\'] = exact
        return path, self.c_best

'''

NEW_STEPWISE = '''\
    def plan_stepwise(self):
        """Generator that yields tree state after each iteration.

        Yields
        ------
        dict with keys: iteration, vertices (list of Node), edges
        (list of (parent_x, child_x) tuples), path (list of arrays or []),
        c_best (float).
        """
        self._t0 = time.time()

        for it in range(self.max_iterations):
            # Phase 1: Prune, update I_R, sample (Alg 1, lines 3-7)
            if self.c_best < np.inf:
                self._prune()
                self._update_informed_set()
                self._update_stall_counter()

            new_samples = self._sample_batch(it)

            for s in new_samples:
                is_goal = np.allclose(s, self.x_goal, atol=1e-8)
                if is_goal and self.goal_node is not None:
                    continue
                if self.collision_free(s):
                    self._x_samples.append(s.copy())
                elif self._adaptive_mode:
                    self._carm.add_collision_point(s)

            n_total = len(self.vertices) + len(self._x_samples)
            r = self._compute_connection_radius(n_total)

            # Phase 2: Build & process edge queue (Alg 1, lines 8-14)
            self._process_edge_queue(r)

            # Phase 3: CARM feedback
            if self._adaptive_mode:
                self._maybe_rebuild_carm_cache(it)

            elapsed = time.time() - self._t0
            self._record_stats(it, elapsed)

            edges = []
            for v in self.vertices:
                if v.parent is not None:
                    edges.append((v.parent.x.copy(), v.x.copy()))
            path = self._extract_path()
            yield {
                \'iteration\': it,
                \'vertices\': [v.x.copy() for v in self.vertices],
                \'edges\': edges,
                \'path\': path,
                \'c_best\': self.c_best,
            }

        # Post-process: MultiStrategySmooth (Alg 1, line 20)
        path = self._extract_path()
        if path and len(path) > 1:
            path = self._shortcut_path(
                path, n_attempts=max(300, 100 * self.dim))
            exact = sum(riemannian_edge_cost(path[i], path[i + 1], self.metric)
                        for i in range(len(path) - 1))
            self.c_best = exact
            if self._stats:
                self._stats[-1][\'c_best\'] = exact

'''

NEW_PROCESS_EDGE_QUEUE = '''\
    def _process_edge_queue(self, r: float):
        """Build and process edge priority queue (Algorithm 1, lines 8-14).

        Builds Q_E = {(v, x) : v in V, x in X_samples, d_R(v,x) <= r}
        sorted by f(e) = g(v) + c_hat_R(v,x) + h_hat_R(x).

        Uses three-level cascading edge evaluation:
          L1 (midpoint): queue ordering and initial filter
          L2 (Simpson): secondary filter
          L3 (10-pt Gauss-Legendre) + collision: final accept/reject
        """
        mc = self._mc
        if not self._x_samples:
            return

        # Build KD-tree over unconnected samples
        sample_arr = np.array(self._x_samples)
        if self._use_weighted_kd:
            sample_kd = KDTree(sample_arr * self._sqrt_w)
        else:
            sample_kd = KDTree(sample_arr)

        # Precompute heuristics for all samples
        h_samples = np.array([mc.heuristic(s, self.x_goal)
                              for s in self._x_samples])

        # Build edge queue: expand all tree vertices
        edge_queue = []
        _cnt = 0
        connected = set()
        c_best = self.c_best

        for v in self.vertices:
            # Skip vertices that cannot improve c_best
            if c_best < np.inf and v.cost >= c_best:
                continue
            if self._use_weighted_kd:
                q = v.x * self._sqrt_w
            else:
                q = v.x
            idxs = sample_kd.query_ball_point(q, r)
            if not idxs:
                _, idx = sample_kd.query(q)
                idxs = [int(idx)]
            for si in idxs:
                ec_l1 = mc.edge_cost_l1(v.x, self._x_samples[si])
                f_e = v.cost + ec_l1 + h_samples[si]
                if c_best < np.inf and f_e >= c_best:
                    continue
                heapq.heappush(edge_queue, (f_e, _cnt, id(v), si))
                _cnt += 1

        # Map vertex id -> Node for queue processing
        vert_map = {id(v): v for v in self.vertices}

        # Process edges in priority order
        while edge_queue:
            f_e, _, vid, si = heapq.heappop(edge_queue)

            if f_e >= self.c_best:
                break

            if si in connected:
                continue

            v = vert_map.get(vid)
            if v is None:
                continue

            s = self._x_samples[si]
            h_s = h_samples[si]

            # L2: Simpson\\\'s rule filter
            ec_l2 = mc.edge_cost_l2(v.x, s)
            cost_l2 = v.cost + ec_l2
            if cost_l2 + h_s >= self.c_best:
                continue

            # L3: exact Gauss-Legendre cost
            ec_l3 = mc.edge_cost_exact(v.x, s)
            cost_l3 = v.cost + ec_l3
            if cost_l3 + h_s >= self.c_best:
                continue

            # Collision check
            if self._adaptive_mode:
                is_free, coll_pt = check_edge_collision_with_feedback(
                    v.x, s, self.collision_free)
                if not is_free:
                    if coll_pt is not None:
                        self._carm.add_collision_point(coll_pt)
                    continue
            else:
                if not check_edge_collision(v.x, s, self.collision_free):
                    continue

            # Edge accepted
            connected.add(si)
            is_goal = np.allclose(s, self.x_goal, atol=1e-8)

            if is_goal and self.goal_node is not None:
                # Update goal if this path is cheaper
                if cost_l3 < self.goal_node.cost:
                    if self.goal_node.parent is not None:
                        old_p = self.goal_node.parent
                        if self.goal_node in old_p.children:
                            old_p.children.remove(self.goal_node)
                    self.goal_node.parent = v
                    self.goal_node.cost = cost_l3
                    self.goal_node.f_value = cost_l3
                    v.children.append(self.goal_node)
                    self.c_best = cost_l3
                continue

            new_node = Node(s.copy(), cost=cost_l3)
            new_node.parent = v
            new_node.heuristic = 0.0 if is_goal else h_s
            new_node.f_value = cost_l3 + new_node.heuristic
            v.children.append(new_node)
            self.vertices.append(new_node)
            vert_map[id(new_node)] = new_node

            if is_goal:
                self.goal_node = new_node
                self.c_best = cost_l3

            # Rewire neighbours (Alg 1, line 14)
            self._rewire_neighbours(new_node, r)

            # Expand new vertex: add its outgoing edges to the queue
            if self._use_weighted_kd:
                q_new = new_node.x * self._sqrt_w
            else:
                q_new = new_node.x
            new_idxs = sample_kd.query_ball_point(q_new, r)
            for nsi in new_idxs:
                if nsi in connected:
                    continue
                ec_l1 = mc.edge_cost_l1(new_node.x, self._x_samples[nsi])
                f_e = new_node.cost + ec_l1 + h_samples[nsi]
                if f_e < self.c_best:
                    heapq.heappush(edge_queue, (f_e, _cnt, id(new_node), nsi))
                    _cnt += 1

            # Refresh c_best after rewiring
            if self.goal_node is not None:
                self.c_best = self.goal_node.cost

        # Remove connected samples from pool
        if connected:
            self._x_samples = [
                s for i, s in enumerate(self._x_samples)
                if i not in connected
            ]

    def _rewire_neighbours(self, new_node: Node, r: float):
        """Rewire existing tree vertices through new_node (Alg 1, line 14).

        For each tree vertex u within radius r of new_node, if
        g(new_node) + c_R(new_node, u) < g(u), rewire u through
        new_node.
        """
        mc = self._mc
        c_best = self.c_best

        kd = self._build_kd_tree()
        if self._use_weighted_kd:
            q = new_node.x * self._sqrt_w
        else:
            q = new_node.x
        idxs = kd.query_ball_point(q, r)

        for idx in idxs:
            if idx >= len(self.vertices):
                continue
            v = self.vertices[idx]
            if v is new_node or v is self.start_node or v is new_node.parent:
                continue
            if c_best < np.inf and v.f_value > c_best:
                continue

            # L1 fast filter
            ec_l1 = mc.edge_cost_l1(new_node.x, v.x)
            if new_node.cost + ec_l1 >= v.cost:
                continue

            # L2 Simpson
            ec_l2 = mc.edge_cost_l2(new_node.x, v.x)
            new_cost = new_node.cost + ec_l2
            if new_cost >= v.cost:
                continue
            if c_best < np.inf and new_cost + v.heuristic >= c_best:
                continue

            # Collision check
            if self._adaptive_mode:
                is_free, coll_pt = check_edge_collision_with_feedback(
                    new_node.x, v.x, self.collision_free)
                if not is_free:
                    if coll_pt is not None:
                        self._carm.add_collision_point(coll_pt)
                    continue
            else:
                if not check_edge_collision(
                        new_node.x, v.x, self.collision_free):
                    continue

            # Rewire
            if v.parent is not None and v in v.parent.children:
                v.parent.children.remove(v)
            v.parent = new_node
            v.cost = new_cost
            v.f_value = new_cost + v.heuristic
            new_node.children.append(v)
            self._propagate_cost(v)

'''

NEW_PRUNE = '''\
    def _prune(self):
        """Remove vertices and samples that cannot improve c_best.

        Implements the pruning step of Algorithm 1 -- vertices and
        unconnected samples that cannot possibly lie on a path cheaper
        than c_best are discarded.
        """
        if self.c_best == np.inf:
            return
        threshold = self.prune_thresh * self.c_best
        kept = []
        for v in self.vertices:
            if v is self.start_node:
                kept.append(v)
                continue
            if v is self.goal_node:
                kept.append(v)
                continue
            if v.f_value <= threshold:
                kept.append(v)
            else:
                # Detach from parent
                if v.parent is not None and v in v.parent.children:
                    v.parent.children.remove(v)
        self.vertices = kept
        # Invalidate KD-tree cache after pruning
        self._kd_index.clear()
        self._kd_n_verts = 0

        # Prune sample pool: remove samples outside informed set
        mc = self._mc
        c = self.c_best
        self._x_samples = [
            s for s in self._x_samples
            if mc.heuristic(self.x_start, s) + mc.heuristic(s, self.x_goal) < c
        ]
        # Cap pool size to avoid unbounded growth
        max_pool = 5 * self.batch_size
        if len(self._x_samples) > max_pool:
            h_vals = np.array([
                mc.heuristic(self.x_start, s) + mc.heuristic(s, self.x_goal)
                for s in self._x_samples
            ])
            keep_idx = np.argsort(h_vals)[:max_pool]
            self._x_samples = [self._x_samples[int(i)] for i in keep_idx]

'''

# ── assemble the new file ──────────────────────────────────────────
out = []

# Part 1: Everything up to and including self.c_best = np.inf (+ insert _x_samples)
out.extend(lines[:markers['c_best_init'] + 1])
out.append(INIT_INSERT)

# Part 2: From after c_best line to just before plan()
out.extend(lines[markers['c_best_init'] + 1 : markers['plan_start']])

# Part 3: New plan()
out.append(NEW_PLAN)

# Part 4: New plan_stepwise() (replaces old plan_stepwise)
out.append(NEW_STEPWISE)

# Part 5: get_stats() through _build_kd_tree (up to _extend_tree) -- unchanged
out.extend(lines[markers['stats_start'] : markers['extend_start']])

# Part 6: New _process_edge_queue() + _rewire_neighbours() (replaces _extend_tree)
out.append(NEW_PROCESS_EDGE_QUEUE)

# Part 7: _maybe_rebuild_carm_cache through _propagate_cost -- unchanged
out.extend(lines[markers['rebuild_start'] : markers['prune_start']])

# Part 8: New _prune()
out.append(NEW_PRUNE)

# Part 9: _update_informed_set and everything after -- unchanged
out.extend(lines[markers['informed_start']:])

result = ''.join(out)

# Sanity checks
assert 'def plan(self)' in result
assert 'def plan_stepwise(self)' in result
assert '_process_edge_queue' in result
assert '_rewire_neighbours' in result
assert '_x_samples' in result
assert '_shortcut_path' in result
assert 'def _prune(self)' in result
assert 'def _maybe_rebuild_carm_cache' in result
assert 'def get_stats' in result
assert 'def _sample_batch' in result
# Must NOT contain the old _extend_tree
assert 'def _extend_tree' not in result, "_extend_tree should be removed!"
# Must contain McSS
assert 'MultiStrategySmooth' in result or '_shortcut_path' in result

with open(filepath, 'w') as f:
    f.write(result)

n_lines = result.count('\n')
print(f"\nDone! Wrote {n_lines} lines to {filepath}")
