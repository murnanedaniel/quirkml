"""CC & walkthrough copied from ML-Tracking-Review lrt_paper/track_building/ccwalk.py (acorn cc_and_walk port,
score_guided_walk, guidance_mode 2). Only the graph algorithm is kept; no helix steps. See build_tracks.py."""

import numpy as np

def build_adjacency(src, dst, scores, keep):
    succ, pred = {}, {}
    for e in np.nonzero(keep)[0]:
        u, v = int(src[e]), int(dst[e])
        s = float(scores[e])
        succ.setdefault(u, {})[v] = s
        succ.setdefault(v, {})
        pred.setdefault(v, {})[u] = s
        pred.setdefault(u, {})
    return succ, pred


def topo_order(succ, pred):
    nodes = sorted(succ.keys())
    indeg = {v: len(pred[v]) for v in nodes if len(pred[v]) > 0}
    zero = [v for v in nodes if len(pred[v]) == 0]
    order = []
    while zero:
        gen, zero = zero, []
        for node in gen:
            for child in succ[node]:
                indeg[child] -= 1
                if indeg[child] == 0:
                    zero.append(child)
                    del indeg[child]
        order.extend(gen)
    if indeg:
        raise RuntimeError("cycle in the oriented graph")
    return order


def simple_path_components(succ, pred, order):
    seen = set()
    cc_tracks, removed = [], set()
    for start in order:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            n = stack.pop()
            comp.append(n)
            for m in succ[n]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
            for m in pred[n]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        if all(len(succ[n]) <= 1 and len(pred[n]) <= 1 for n in comp):
            if len(comp) > 2:
                cc_tracks.append(comp)
            removed.update(comp)
    return cc_tracks, removed


def find_next_hits(succ, current, used, th_min, th_add):
    neighbors = list(set(succ[current].keys()).difference(used))
    if len(neighbors) < 1:
        return None
    sc = succ[current]
    neighbors_scores = [sc[i] for i in neighbors]
    if max(neighbors_scores) <= th_min:
        return None
    sorted_idx = list(reversed(np.argsort(neighbors_scores)))
    next_hits = [neighbors[sorted_idx[0]]]
    for ii in range(1, len(sorted_idx)):
        idx = sorted_idx[ii]
        if neighbors_scores[idx] > th_add:
            next_hits.append(neighbors[idx])
        else:
            break
    return next_hits


# Optional cap on the number of roads grown from one start (None = ACORN behaviour, unlimited). Where the scores
# of many edges exceed th_add the number of roads grows exponentially with the road length; above the cap every
# road continues only with its best-scoring next hit (one V2 event, 80365 with the M3 Simon fine-tune, needed it).
MAX_ROADS = None
CAPPED = {"starts": 0}


def build_roads(succ, pred_sorted, start, used, th_min, th_add):
    alternative = False
    capped = False
    next_hits = find_next_hits(succ, start, used, th_min, th_add)
    if next_hits is None:
        return [(start, None)], alternative
    path = []
    for hit in next_hits:
        bestprob, thebest = -1, None
        for p in pred_sorted[hit]:
            if succ[p][hit] > bestprob:
                bestprob, thebest = succ[p][hit], p
        if thebest == start:
            path.append((start, hit))
        else:
            alternative = True
            path.append((thebest, hit))
    while not all(pp[-1] is None for pp in path):
        new_path = []
        for pp in path:
            if pp[-1] is None:
                new_path.append(pp)
                continue
            nh = find_next_hits(succ, pp[-1], used.union(pp), th_min, th_add)
            if nh is None:
                new_path.append(pp + (None,))
            else:
                for hit in (nh[:1] if capped else nh):
                    new_path.append(pp + (hit,))
        if MAX_ROADS is not None and not capped and len(new_path) > MAX_ROADS:
            capped = True
            CAPPED["starts"] += 1
            # rebuild this step with the best next hit only, then keep one branch per road from here on
            new_path = []
            for pp in path:
                if pp[-1] is None:
                    new_path.append(pp)
                    continue
                nh = find_next_hits(succ, pp[-1], used.union(pp), th_min, th_add)
                new_path.append(pp + ((None,) if nh is None else (nh[0],)))
        path = new_path
    return path, alternative


def _pick_road_local(succ, a_road, b_road):
    for i in range(0, min(len(a_road), len(b_road)) - 2):
        sa = succ[a_road[i]][a_road[i + 1]]
        sb = succ[b_road[i]][b_road[i + 1]]
        if abs(sa - sb) > 1e-5:
            return a_road if sa > sb else b_road
    return a_road


def choose_road_local(succ, road):
    res, length = road[0], len(road[0])
    for i in range(1, len(road)):
        if len(road[i]) > length + 1:
            res, length = road[i], len(road[i])
        elif abs(len(road[i]) - length) <= 1:
            res = _pick_road_local(succ, res, road[i])
            length = len(res)
    return res


def walk(succ, pred, order, removed, th_min, th_add, emit_extra_min=None, short_min=None):
    """Score-guided walk. Returns (claimed, extra, short):

    claimed : the road chosen at each start (claims its nodes), ACORN behaviour
    extra   : (emit-all-roads) every OTHER valid road of that start with >= emit_extra_min
              nodes that is not a subset of the claimed road; never claims nodes
    short   : roads built but too short to emit (short_min <= nodes < emit_extra_min),
              including claimed roads below the length rule; input to padding
    """
    pred_sorted = {v: sorted(pred[v].keys()) for v in pred}
    used = set()
    claimed, extra, short = [], [], []
    seen_extra = set()
    for node in order:
        if node in removed:
            continue
        alternative = True
        while alternative:
            if node in used:
                alternative = False
                continue
            road, alternative = build_roads(succ, pred_sorted, node, used, th_min, th_add)
            a_road = choose_road_local(succ, road) if len(road) > 1 else road[0]
            if len(a_road) < 3:
                used.add(node)
                claimed.append([node])
                continue
            nodes = list(dict.fromkeys(a_road[:-1]))
            claimed.append(nodes)
            used.update(nodes)
            if emit_extra_min is None:
                continue
            claimed_set = frozenset(nodes)
            for rd in road:
                if rd is a_road:
                    continue
                rd_nodes = [n for n in rd if n is not None]
                key = frozenset(rd_nodes)
                if key in seen_extra or key <= claimed_set:
                    continue
                seen_extra.add(key)
                if len(rd_nodes) >= emit_extra_min:
                    extra.append(rd_nodes)
                elif short_min is not None and len(rd_nodes) >= short_min:
                    short.append(rd_nodes)
    return claimed, extra, short


def flatten_partition(track_lists):
    """ACORN add_track_labels: the first track that contains a node keeps it."""
    seen = set()
    out = []
    for t in track_lists:
        tt = [n for n in t if n not in seen]
        seen.update(tt)
        if tt:
            out.append(tt)
    return out


