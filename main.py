import heapq
import math
import random
import sys
import time
import tkinter as tk

BASE_CELL = 45
GRID = (10, 10)  # change GRID to test other board sizes
START = (0, 0)
STATIC_ITEMS = {(2, 3), (7, 1), (4, 6)}
DYNAMIC_PLAN = [(2500, None), (5000, None), (8000, None)]
STEP_MS = 180
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def mst_cost(nodes):
    pts = list(nodes)
    if len(pts) < 2:
        return 0
    used = {pts.pop()}
    total = 0
    while pts:
        dist, idx = min((min(manhattan(p, u) for u in used), i) for i, p in enumerate(pts))
        total += dist
        used.add(pts.pop(idx))
    return total


def make_dynamic_cells(cols, rows, static, start):
    rng = random.Random((cols, rows))
    candidates = [
        (x, y)
        for x in range(cols)
        for y in range(rows)
        if (x, y) not in static and (x, y) != start
    ]
    rng.shuffle(candidates)
    return candidates


def build_heuristics(start):
    def zero(state):
        return 0

    def nearest(state):
        pos, left = state
        if not left:
            return manhattan(pos, start)
        return min(manhattan(pos, g) for g in left)

    def nearest_return(state):
        pos, left = state
        if not left:
            return manhattan(pos, start)
        to_goal = min(manhattan(pos, g) for g in left)
        back_home = min(manhattan(start, g) for g in left)
        return to_goal + back_home

    def mst(state):
        pos, left = state
        if not left:
            return manhattan(pos, start)
        tree = mst_cost(left | {start})
        connect = min(manhattan(pos, g) for g in left)
        return connect + tree

    return {"zero": zero, "nearest": nearest, "return": nearest_return, "mst": mst}


def search_plan(pos, objects, cols, rows, start, mode, heuristic_fn):
    start_state = (pos, frozenset(objects - {pos}))
    frontier = []
    g0 = 0
    h0 = heuristic_fn(start_state) if mode == "A*" else 0
    heapq.heappush(frontier, (g0 + h0, g0, start_state, []))
    best = {start_state: 0}
    expanded = 0
    t0 = time.perf_counter()
    while frontier:
        f, g, state, path = heapq.heappop(frontier)
        if g > best[state]:
            continue
        expanded += 1
        pos, remaining = state
        if not remaining and pos == start:
            return path, expanded, time.perf_counter() - t0
        for dx, dy in MOVES:
            nxt = (pos[0] + dx, pos[1] + dy)
            if not (0 <= nxt[0] < cols and 0 <= nxt[1] < rows):
                continue
            new_state = (nxt, remaining - {nxt})
            ng = g + 1
            if ng < best.get(new_state, float("inf")):
                best[new_state] = ng
                hn = heuristic_fn(new_state) if mode == "A*" else 0
                heapq.heappush(frontier, (ng + hn, ng, new_state, path + [nxt]))
    return [], expanded, time.perf_counter() - t0


def simulate_run(grid, mode, heuristic_name):
    cols, rows = grid
    start = START
    static = {pos for pos in STATIC_ITEMS if 0 <= pos[0] < cols and 0 <= pos[1] < rows and pos != start}
    objects = set(static)
    heuristics = build_heuristics(start)
    heuristic_fn = heuristics[heuristic_name if mode == "A*" else "zero"]
    dynamic_cells = make_dynamic_cells(cols, rows, static, start)
    dyn_idx = 0
    robot = start
    total_expanded = 0
    total_time = 0.0
    path = []
    step = 0
    events = [(max(1, math.ceil(delay / STEP_MS)), pos) for delay, pos in DYNAMIC_PLAN]
    event_idx = 0

    def next_dynamic():
        nonlocal dyn_idx
        while dyn_idx < len(dynamic_cells):
            candidate = dynamic_cells[dyn_idx]
            dyn_idx += 1
            if candidate not in objects and candidate != robot:
                return candidate
        return None

    while True:
        if not path:
            path, expanded, elapsed = search_plan(robot, objects, cols, rows, start, mode, heuristic_fn)
            total_expanded += expanded
            total_time += elapsed
            if not path:
                break
        robot = path.pop(0)
        if robot in objects:
            objects.remove(robot)
        step += 1
        while event_idx < len(events) and step >= events[event_idx][0]:
            _, preset = events[event_idx]
            event_idx += 1
            pos = preset
            if pos is not None and not (0 <= pos[0] < cols and 0 <= pos[1] < rows):
                pos = None
            if pos is None or pos in objects or pos == robot:
                pos = next_dynamic()
            if pos is not None:
                objects.add(pos)
                path = []
        if not objects and robot == start and not path:
            break
    return total_expanded, total_time


def run_experiments():
    combos = [("UCS", "zero"), ("A*", "nearest"), ("A*", "return"), ("A*", "mst")]
    grids = [(10, 10), (20, 20), (50, 50)]
    rows = []
    for grid in grids:
        for mode, heuristic in combos:
            expanded, elapsed = simulate_run(grid, mode, heuristic)
            rows.append((f"{grid[0]}x{grid[1]}", mode, heuristic, expanded, elapsed))
    print("Grid\tAlgorithm\tHeuristic\tExpanded\tTime(s)")
    for grid, mode, heuristic, expanded, elapsed in rows:
        print(f"{grid}\t{mode}\t{heuristic}\t{expanded}\t{elapsed:.4f}")

class GridWorld:
    def __init__(self):
        self.start = START
        self.step_ms = STEP_MS
        self.static = set(STATIC_ITEMS)
        self.heuristics = build_heuristics(self.start)
        self.root = tk.Tk()
        self.root.title("AI Robot Collector")
        self.cell = BASE_CELL
        self.canvas = tk.Canvas(self.root, width=GRID[0] * self.cell, height=GRID[1] * self.cell, bg="white")
        self.canvas.grid(row=1, column=0, padx=10, pady=10)
        panel = tk.Frame(self.root)
        panel.grid(row=1, column=1, sticky="ns")
        tk.Label(panel, text="Algorithm").pack(pady=2)
        self.mode_var = tk.StringVar(value="A*")
        tk.OptionMenu(panel, self.mode_var, "UCS", "A*").pack(fill="x")
        tk.Label(panel, text="Heuristic").pack(pady=2)
        self.heur_var = tk.StringVar(value="nearest")
        tk.OptionMenu(panel, self.heur_var, *self.heuristics.keys()).pack(fill="x")
        tk.Button(panel, text="Start", command=self.start_run).pack(fill="x", pady=6)
        tk.Button(panel, text="Reset", command=self.reset).pack(fill="x")
        self.status = tk.StringVar()
        tk.Label(panel, textvariable=self.status, wraplength=160, justify="left").pack(pady=10)
        self.robot = self.start
        self.objects = set()
        self.path = []
        self.after_ids = []
        self.move_id = None
        self.running = False
        self.static_in_bounds = set()
        self.dynamic_cells = []
        self.dynamic_idx = 0
        self.last_plan_stats = {"expanded": 0, "time": 0.0}
        self.total_stats = {"expanded": 0, "time": 0.0}
        self.grid_var = tk.StringVar(value=f"{GRID[0]}x{GRID[1]}")
        top = tk.Frame(self.root)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        tk.Label(top, text="Grid Size").pack(side="left")
        grid_options = []
        for opt in (self.grid_var.get(), "10x10", "20x20", "40x40", "50x50"):
            if opt not in grid_options:
                grid_options.append(opt)
        grid_menu = tk.OptionMenu(top, self.grid_var, *grid_options)
        grid_menu.pack(side="left", padx=5)
        self.grid_var.trace_add("write", self.on_grid_change)
        self.change_grid(self.grid_var.get())

    def on_grid_change(self, *_):
        self.change_grid(self.grid_var.get())

    def compute_cell_size(self, cols, rows):
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        avail_w = max(240, screen_w - 260)  # leave room for control panel
        avail_h = max(240, screen_h - 220)
        size = min(avail_w // cols, avail_h // rows, BASE_CELL * 2)
        return max(8, size)

    def change_grid(self, value):
        cols, rows = map(int, value.split("x"))
        self.cell = self.compute_cell_size(cols, rows)
        self.canvas.config(width=cols * self.cell, height=rows * self.cell)
        self.cols, self.rows = cols, rows
        self.reset()

    def box(self, cell, pad=0):
        x, y = cell
        size = self.cell
        return x * size + pad, y * size + pad, (x + 1) * size - pad, (y + 1) * size - pad

    def center(self, cell):
        x, y = cell
        size = self.cell
        return x * size + size // 2, y * size + size // 2

    def reset(self):
        self.running = False
        if self.move_id:
            self.root.after_cancel(self.move_id)
            self.move_id = None
        for aid in self.after_ids:
            self.root.after_cancel(aid)
        self.after_ids.clear()
        self.robot = self.start
        self.static_in_bounds = {
            pos for pos in self.static if 0 <= pos[0] < self.cols and 0 <= pos[1] < self.rows and pos != self.start
        }
        self.objects = set(self.static_in_bounds)
        self.path = []
        self.dynamic_cells = self.build_dynamic_cells()
        self.dynamic_idx = 0
        self.last_plan_stats = {"expanded": 0, "time": 0.0}
        self.total_stats = {"expanded": 0, "time": 0.0}
        self.status.set("Waiting for start.")
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        for x in range(self.cols):
            for y in range(self.rows):
                self.canvas.create_rectangle(*self.box((x, y)), outline="#bbb")
                cx, cy = self.center((x, y))
                self.canvas.create_text(
                    cx, cy, text=f"{x},{y}", fill="#999", font=("Helvetica", max(8, self.cell // 3))
                )
        self.canvas.create_rectangle(*self.box(self.start, 3), outline="#4caf50", width=2)
        for obj in self.objects:
            self.canvas.create_oval(*self.box(obj, 10), fill="#ff9800", outline="")
        route = [self.robot] + self.path
        if len(route) > 1:
            coords = [c for cell in route for c in self.center(cell)]
            self.canvas.create_line(*coords, fill="#00bcd4", width=3, dash=(6, 2))
        self.canvas.create_oval(*self.box(self.robot, 6), fill="#2196f3", outline="")
        stats = self.last_plan_stats
        totals = self.total_stats
        label = f"Remaining: {len(self.objects)}"
        if stats["expanded"]:
            label += f" | Expanded: {stats['expanded']} | {stats['time']*1000:.1f} ms"
        if totals["expanded"]:
            label += f"\nTotal Expanded: {totals['expanded']} | Total Time: {totals['time']:.3f}s"
        self.canvas.create_text(5, 5, anchor="nw", text=label)

    def start_run(self):
        self.reset()
        self.running = True
        self.status.set("Collecting objects...")
        self.schedule_dynamic()
        self.move_step()

    def schedule_dynamic(self):
        for delay, pos in DYNAMIC_PLAN:
            aid = self.root.after(delay, lambda p=pos: self.add_dynamic(p))
            self.after_ids.append(aid)

    def build_dynamic_cells(self):
        return make_dynamic_cells(self.cols, self.rows, self.static_in_bounds, self.start)

    def add_dynamic(self, pos=None):
        if not self.running:
            return
        if pos is None:
            pos = self.next_dynamic_cell()
        if pos is None or pos in self.objects or pos == self.robot:
            pos = self.next_dynamic_cell()
        if pos is None:
            return
        self.objects.add(pos)
        self.status.set(f"Dynamic object spawned at {pos}")
        self.path = []
        self.draw()

    def next_dynamic_cell(self):
        while self.dynamic_idx < len(self.dynamic_cells):
            candidate = self.dynamic_cells[self.dynamic_idx]
            self.dynamic_idx += 1
            if candidate not in self.objects and candidate != self.robot:
                return candidate
        return None

    def move_step(self):
        if not self.running:
            return
        if not self.path:
            self.path = self.plan_path()
            if not self.path:
                if not self.objects and self.robot == self.start:
                    self.running = False
                    self.status.set("Mission complete.")
                else:
                    self.status.set("No plan available.")
                return
            self.draw()
            self.move_id = self.root.after(self.step_ms, self.move_step)
            return
        nxt = self.path.pop(0)
        self.robot = nxt
        if nxt in self.objects:
            self.objects.remove(nxt)
        self.draw()
        if not self.objects and self.robot == self.start and not self.path:
            self.running = False
            self.status.set("Mission complete.")
            return
        self.move_id = self.root.after(self.step_ms, self.move_step)

    def plan_path(self):
        mode = self.mode_var.get()
        heuristic_fn = self.heuristics[self.heur_var.get()] if mode == "A*" else self.heuristics["zero"]
        path, expanded, elapsed = search_plan(
            self.robot, self.objects, self.cols, self.rows, self.start, mode, heuristic_fn
        )
        self.last_plan_stats = {"expanded": expanded, "time": elapsed}
        self.total_stats["expanded"] += expanded
        self.total_stats["time"] += elapsed
        return path

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if "--experiments" in sys.argv:
        run_experiments()
    else:
        GridWorld().run()
