#!/usr/bin/env python3
"""Push-optimal Sokoban solver (sokoban-solver-01 bounty).

Input : a directory of instances, each <id>/level.txt in XSB
        (# wall, space floor, . goal, $ box, * box on goal, @ player, + player on goal).
Output: for each instance, <out>/<id>/solution.txt holding the player's moves in
        LURD (u/d/l/r; an upper-case letter marks a push -- case is not checked).

Objective: push every box onto a goal in as few pushes as possible. The checker
awards 1000 points iff the push count is <= P* (the push-optimal count) and
500*P*/pushes otherwise, so the whole game is hitting the push optimum.

Method
------
Levels are generated as independent rooms hanging off a one-wide hallway; a box
pushed out of its room can never be pushed back, so a level's push-optimal count
is the sum of its rooms' optima. We:

  1. decompose the level into rooms (cutting the dead hallway squares out of the
     floor), falling back to whole-level search if it is not a set of rooms;
  2. solve each room to push-optimality with A* over *push* states. A state is
     (frozenset of box squares, canonical player region). Edge cost is one push.
     The heuristic is an admissible minimum-cost bipartite matching between boxes
     and goals over reverse-push distances -- a far tighter lower bound than the
     plain "sum of nearest-goal distances", so the true optimum is proven within
     a far smaller state budget than a plain BFS needs;
  3. prune deadlocks a box can never escape: dead squares (no push sequence to
     any goal, ignoring other boxes) and frozen 2x2 blocks of walls/boxes with a
     box off its goal;
  4. stitch the per-room push sequences into one move string, walking the player
     between pushes with a shortest-path BFS.

Pure and deterministic: fixed node budget, no clock, no randomness. Two runs on
one input write identical bytes.
"""

import heapq
import os
import sys
from collections import deque

DIRS = ((0, -1, "u", "U"), (0, 1, "d", "D"), (-1, 0, "l", "L"), (1, 0, "r", "R"))

# Max A* expansions per room / per whole-level fallback. Work, never the clock.
SEARCH_STATES = 2_000_000


class Level:
    __slots__ = ("w", "h", "wall", "goals", "boxes", "player", "size")

    def __init__(self, w, h, wall, goals, boxes, player):
        self.w = w
        self.h = h
        self.wall = wall          # bytearray, 1 == wall
        self.goals = goals        # sorted list[int]
        self.boxes = boxes        # sorted list[int]
        self.player = player      # int
        self.size = w * h


def parse_level(text):
    rows = [r for r in text.split("\n") if len(r) > 0]
    h = len(rows)
    w = max((len(r) for r in rows), default=0)
    size = w * h
    wall = bytearray(b"\x01") * size
    goals = []
    boxes = []
    player = -1
    for y, row in enumerate(rows):
        base = y * w
        for x, ch in enumerate(row):
            i = base + x
            if ch == "#":
                continue
            wall[i] = 0
            if ch in ".*+":
                goals.append(i)
            if ch in "$*":
                boxes.append(i)
            if ch in "@+":
                player = i
    # The outermost frame is always wall: nothing leaves the board.
    for y in range(h):
        wall[y * w] = 1
        wall[y * w + (w - 1)] = 1
    for x in range(w):
        wall[x] = 1
        wall[(h - 1) * w + x] = 1
    goals.sort()
    boxes.sort()
    return Level(w, h, wall, goals, boxes, player)


def live_squares(level):
    """Squares from which a box alone can be pushed onto some goal.

    A box at goal g could have been pushed from y = g - d if both y and the
    square behind it (y - d) are open floor; walk that relation backwards.
    """
    w, wall, goals = level.w, level.wall, level.goals
    size = level.size
    live = bytearray(size)
    queue = deque()
    for g in goals:
        live[g] = 1
        queue.append(g)
    while queue:
        x = queue.popleft()
        for dx, dy, _, _ in DIRS:
            d = dy * w + dx
            y = x - d
            if 0 <= y < size and not wall[y] and not wall[y - d] and not live[y]:
                live[y] = 1
                queue.append(y)
    return live


def _reach(wall, occupied, low, size, dist, stamp, seen, queue):
    """Flood the player's region from `low` over open, unoccupied squares.

    Fills dist[] with step distance (only meaningful for cells with seen==stamp),
    marks seen[] with stamp, and returns the lowest square index in the region.
    """
    head = 0
    queue[0] = low
    seen[low] = stamp
    dist[low] = 0
    lowest = low
    n = 1
    w = _reach.w
    while head < n:
        p = queue[head]
        head += 1
        if p < lowest:
            lowest = p
        nd = dist[p] + 1
        # up, down, left, right -- guard left/right against row wrap via wall frame
        q = p - w
        if not wall[q] and not occupied[q] and seen[q] != stamp:
            seen[q] = stamp
            dist[q] = nd
            queue[n] = q
            n += 1
        q = p + w
        if not wall[q] and not occupied[q] and seen[q] != stamp:
            seen[q] = stamp
            dist[q] = nd
            queue[n] = q
            n += 1
        q = p - 1
        if not wall[q] and not occupied[q] and seen[q] != stamp:
            seen[q] = stamp
            dist[q] = nd
            queue[n] = q
            n += 1
        q = p + 1
        if not wall[q] and not occupied[q] and seen[q] != stamp:
            seen[q] = stamp
            dist[q] = nd
            queue[n] = q
            n += 1
    return lowest


# width is needed inside _reach without passing it everywhere
_reach.w = 0


def _frozen_block(wall, occupied, goal, t, w):
    """True if the box now at t sits in a frozen 2x2 block with a box off its goal."""
    for a in (t, t - 1, t - w, t - w - 1):
        b = a + 1
        c = a + w
        d = c + 1
        if (wall[a] or occupied[a]) and (wall[b] or occupied[b]) \
           and (wall[c] or occupied[c]) and (wall[d] or occupied[d]):
            if (occupied[a] and not goal[a]) or (occupied[b] and not goal[b]) \
               or (occupied[c] and not goal[c]) or (occupied[d] and not goal[d]):
                return True
    return False


def solve_push_optimal(level, max_states=SEARCH_STATES):
    """Push-optimal solution of one (sub)level via A* over push states.

    Returns (pushes, moves) with moves in LURD, or None if the search gives up
    or the level is unsolvable.
    """
    w, wall, goals = level.w, level.wall, level.goals
    size = level.size
    _reach.w = w
    goal = bytearray(size)
    for g in goals:
        goal[g] = 1
    live = live_squares(level)

    start_boxes = tuple(level.boxes)
    # A box already on a dead square => unsolvable.
    for b in start_boxes:
        if not live[b]:
            return None

    occupied = bytearray(size)
    # Two separate mark arrays sharing one stamp counter, exactly as the reference
    # does: `seen` floods the player region during expansion, `mark` floods the
    # canonical region used in a successor's key. Using one array for both corrupts
    # reachability for sibling pushes at the same node (a key-flood stamps cells the
    # expansion flood still needs), which makes the search miss most successors.
    seen = [0] * size
    mark = [0] * size
    dist = [0] * size
    queue = [0] * size
    stamp = [0]

    def region_low(boxes, frm):
        for b in boxes:
            occupied[b] = 1
        stamp[0] += 1
        low = _reach(wall, occupied, frm, size, dist, stamp[0], mark, queue)
        for b in boxes:
            occupied[b] = 0
        return low

    # --- admissible heuristic: min-cost bipartite matching boxes<->goals ---
    # Precompute per-goal pull distances to every square (a box can travel from
    # s to g iff g can be pulled back to s). pull[g][s] = pushes, or -1.
    goal_index = {g: gi for gi, g in enumerate(goals)}
    m = len(goals)
    INF = 1 << 30
    pull = []  # pull[gi][s]
    for g in goals:
        dg = [INF] * size
        dg[g] = 0
        dq = deque([g])
        while dq:
            x = dq.popleft()
            for dx, dy, _, _ in DIRS:
                d = dy * w + dx
                y = x - d
                if 0 <= y < size and not wall[y] and not wall[y - d] and dg[y] == INF:
                    dg[y] = dg[x] + 1
                    dq.append(y)
        pull.append(dg)

    def heuristic(boxes):
        n = len(boxes)
        if n == 1:
            dg = pull[0]
            v = dg[boxes[0]]
            return v if v < INF else INF
        # cost[i][j] = pushes to move box i to goal j (INF if impossible)
        cost = [[pull[j][b] for j in range(m)] for b in boxes]
        # Hungarian algorithm, O(n^2 * m), n <= m.
        u = [0] * (n + 1)
        v = [0] * (m + 1)
        p = [0] * (m + 1)
        way = [0] * (m + 1)
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [INF] * (m + 1)
            used = [False] * (m + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = INF
                j1 = 0
                row = cost[i0 - 1]
                for j in range(1, m + 1):
                    if not used[j]:
                        cur = row[j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                # If the cheapest augmenting edge is INF, matching is impossible.
                if delta >= INF:
                    return INF
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        ans = -v[0]
        return ans if ans >= 0 else INF

    # --- A* over push states ---
    # state: (boxes_tuple_sorted, canonical_region_low)
    boxes0 = tuple(sorted(start_boxes))
    low0 = region_low(boxes0, level.player)
    h0 = heuristic(boxes0)
    if h0 >= INF:
        return None
    if all(goal[b] for b in boxes0):
        return (0, "")

    # node arrays
    s_boxes = [boxes0]
    s_low = [low0]
    s_g = [0]
    s_parent = [-1]
    s_pushbox = [-1]   # index within parent's boxes of the box pushed
    s_pushto = [-1]    # destination square of the pushed box
    best = {(boxes0, low0): 0}
    counter = 0
    heap = [(h0, 0, counter, 0)]  # (f, g, tie, node)
    expanded = 0
    found = -1

    while heap and expanded < max_states:
        f, g, _, node = heapq.heappop(heap)
        if best.get((s_boxes[node], s_low[node])) != g:
            continue
        boxes = s_boxes[node]
        if all(goal[b] for b in boxes):
            found = node
            break
        expanded += 1

        # player region for this state
        for b in boxes:
            occupied[b] = 1
        stamp[0] += 1
        _reach(wall, occupied, s_low[node], size, dist, stamp[0], seen, queue)
        here = stamp[0]

        n = len(boxes)
        for bi in range(n):
            b = boxes[bi]
            for dx, dy, _, _ in DIRS:
                d = dy * w + dx
                stand = b - d      # square the player must stand on
                t = b + d          # box destination
                if seen[stand] != here:
                    continue
                if wall[t] or occupied[t] or not live[t]:
                    continue
                # apply push
                occupied[b] = 0
                occupied[t] = 1
                frozen = _frozen_block(wall, occupied, goal, t, w)
                if frozen:
                    occupied[t] = 0
                    occupied[b] = 1
                    continue
                nb = list(boxes)
                nb[bi] = t
                nb.sort()
                nbt = tuple(nb)
                # canonical player region: player ends at b, flood with new boxes
                stamp[0] += 1
                low = _reach(wall, occupied, b, size, dist, stamp[0], mark, queue)
                occupied[t] = 0
                occupied[b] = 1

                ng = g + 1
                key = (nbt, low)
                prev = best.get(key)
                if prev is not None and prev <= ng:
                    continue
                h = heuristic(nbt)
                if h >= INF:
                    continue
                best[key] = ng
                counter += 1
                nn = len(s_boxes)
                s_boxes.append(nbt)
                s_low.append(low)
                s_g.append(ng)
                s_parent.append(node)
                s_pushbox.append(bi)
                s_pushto.append(t)
                heapq.heappush(heap, (ng + h, ng, counter, nn))

        for b in boxes:
            occupied[b] = 0

    if found < 0:
        return None

    # --- reconstruct push path ---
    path = []
    s = found
    while s > 0:
        path.append(s)
        s = s_parent[s]
    path.reverse()

    # --- emit moves: walk to each push's stand square, then push ---
    moves = []
    player = level.player
    boxes = list(boxes0)
    occ = bytearray(size)
    for s in path:
        parent = s_parent[s]
        bi = s_pushbox[s]
        t = s_pushto[s]
        # box that moved: in parent's boxes at index bi -> from ; to = t
        pboxes = s_boxes[parent]
        frm = pboxes[bi]
        d = t - frm
        letter = None
        for dx, dy, low_, up_ in DIRS:
            if dy * w + dx == d:
                letter = up_
                break
        stand = frm - d
        for b in boxes:
            occ[b] = 1
        moves.append(_walk_to(wall, occ, w, size, player, stand))
        for b in boxes:
            occ[b] = 0
        moves.append(letter)
        player = frm
        # update boxes
        boxes[bi] = t
        boxes.sort()
    return (len(path), "".join(moves))


def _walk_to(wall, occupied, w, size, frm, to):
    """Shortest walk frm->to around walls/boxes, in lower-case LURD."""
    if frm == to:
        return ""
    prev = [-1] * size
    how = [""] * size
    prev[frm] = frm
    q = deque([frm])
    while q:
        p = q.popleft()
        if p == to:
            break
        for dx, dy, letter, _ in DIRS:
            d = dy * w + dx
            nxt = p + d
            if wall[nxt] or occupied[nxt] or prev[nxt] != -1:
                continue
            prev[nxt] = p
            how[nxt] = letter
            q.append(nxt)
    if prev[to] == -1:
        raise RuntimeError("walk_to: push square out of reach")
    out = []
    p = to
    while p != frm:
        out.append(how[p])
        p = prev[p]
    out.reverse()
    return "".join(out)


def rooms_of(level):
    """Decompose into rooms, or None if the level is not a clean set of rooms."""
    w, wall, goals, boxes, player = level.w, level.wall, level.goals, level.boxes, level.player
    size = level.size
    live = live_squares(level)
    row = player // w
    cut = bytearray(size)
    for i in range(row * w, (row + 1) * w):
        if not wall[i] and not live[i]:
            cut[i] = 1
    for b in boxes:
        if cut[b]:
            return None
    part = [-1] * size
    rooms = []
    for s in range(size):
        if wall[s] or cut[s] or part[s] >= 0:
            continue
        squares = [s]
        part[s] = len(rooms)
        idx = 0
        while idx < len(squares):
            cur = squares[idx]
            idx += 1
            for dx, dy, _, _ in DIRS:
                d = dy * w + dx
                q = cur + d
                if not wall[q] and not cut[q] and part[q] < 0:
                    part[q] = len(rooms)
                    squares.append(q)
        rooms.append({"squares": squares, "boxes": [], "goals": [], "done": False})
    for b in boxes:
        rooms[part[b]]["boxes"].append(b)
    for g in goals:
        if part[g] >= 0:
            rooms[part[g]]["goals"].append(g)
    if any(part[g] < 0 for g in goals):
        return None
    if any(len(r["boxes"]) != len(r["goals"]) for r in rooms):
        return None
    rooms = [r for r in rooms if r["boxes"]]
    return rooms or None


def _end_of(w, frm, moves):
    p = frm
    for ch in moves:
        c = ch.lower()
        p += -w if c == "u" else w if c == "d" else -1 if c == "l" else 1
    return p


def solve(level, max_states=SEARCH_STATES):
    """Solve a whole level to push-optimality, room by room when it decomposes."""
    rooms = rooms_of(level)
    if rooms is None:
        return solve_push_optimal(level, max_states)
    w, size = level.w, level.size
    player = level.player
    total_pushes = 0
    all_moves = []
    for room in rooms:
        wall = level.wall[:]  # copy
        for other in rooms:
            if other is not room:
                cells = other["goals"] if other["done"] else other["boxes"]
                for b in cells:
                    wall[b] = 1
        sub = Level(
            w,
            level.h,
            wall,
            sorted(room["goals"]),
            sorted(room["boxes"]),
            player,
        )
        res = solve_push_optimal(sub, max_states)
        if res is None:
            # Room search gave up: fall back to whole-level search once.
            return solve_push_optimal(level, max_states)
        pushes, moves = res
        total_pushes += pushes
        all_moves.append(moves)
        player = _end_of(w, player, moves)
        room["done"] = True
    return (total_pushes, "".join(all_moves))


def main(argv):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR")
    args = argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--input" and i + 1 < len(args):
            input_dir = args[i + 1]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        else:
            i += 1
    if not input_dir:
        input_dir = "/input"
    if not output_dir:
        output_dir = "/output"

    ids = sorted(d for d in os.listdir(input_dir)
                 if os.path.isfile(os.path.join(input_dir, d, "level.txt")))
    for inst in ids:
        with open(os.path.join(input_dir, inst, "level.txt"), "r") as fh:
            text = fh.read()
        moves = ""
        try:
            level = parse_level(text)
            res = solve(level)
            if res is not None:
                moves = res[1]
        except Exception as exc:  # never crash the whole run
            sys.stderr.write("solver: %s failed: %s\n" % (inst, exc))
            moves = ""
        outdir = os.path.join(output_dir, inst)
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "solution.txt"), "w") as fh:
            fh.write(moves + "\n")


if __name__ == "__main__":
    main(sys.argv)