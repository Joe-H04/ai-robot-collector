import heapq
import math
import random
import sys
import time
import tkinter as tk
import os

try:
    from PIL import Image, ImageTk, ImageOps  # type: ignore
except Exception:
    Image = None
    ImageTk = None
    ImageOps = None

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


def deterministic_presets(cols, rows, start, static=(), count=3):
    def in_bounds(p):
        return 0 <= p[0] < cols and 0 <= p[1] < rows
    # Base desired spots derived from grid fractions, consistent per grid size
    desired = [
        (max(0, cols - 2), rows // 2),
        (cols // 2, max(0, rows - 2)),
        (max(0, cols - 2), max(0, rows - 2)),
        (cols // 3, rows // 3),
        (2 * cols // 3, 2 * rows // 3),
        (cols // 3, 2 * rows // 3),
        (2 * cols // 3, rows // 3),
    ]
    static_set = set(static)
    picks = []
    seen = set()
    for p in desired:
        if len(picks) >= count:
            break
        px = (min(cols - 1, max(0, p[0])), min(rows - 1, max(0, p[1])))
        if px in seen or not in_bounds(px):
            continue
        if px == start or px in static_set:
            continue
        seen.add(px)
        picks.append(px)
    # If not enough, fill with a deterministic scan from center
    if len(picks) < count:
        cx, cy = cols // 2, rows // 2
        for y in range(rows):
            for x in range(cols):
                px = (x, y)
                # prioritize by manhattan distance to center
        
        # Build ring-sorted order
        order = sorted(((abs(x - cx) + abs(y - cy), (x, y)) for x in range(cols) for y in range(rows)))
        for _, px in order:
            if len(picks) >= count:
                break
            if px in seen or px == start or px in static_set:
                continue
            seen.add(px)
            picks.append(px)
    return picks[:count]


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


def search_plan(pos, objects, cols, rows, start, mode, heuristic_fn, obstacles=None):
    if obstacles is None:
        obstacles = set()
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
            if nxt in obstacles:
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
    preset_positions = deterministic_presets(cols, rows, start, static, count=len(DYNAMIC_PLAN))
    events = []
    for idx, (delay, _) in enumerate(DYNAMIC_PLAN):
        pos = preset_positions[idx] if idx < len(preset_positions) else None
        events.append((max(1, math.ceil(delay / STEP_MS)), pos))
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
            path, expanded, elapsed = search_plan(
                robot, objects, cols, rows, start, mode, heuristic_fn, obstacles=set()
            )
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
        # Start in fullscreen; fallback to maximized if unavailable
        try:
            self.root.attributes("-fullscreen", True)
        except Exception:
            try:
                self.root.state("zoomed")
            except Exception:
                pass
        # Allow exiting fullscreen and quitting via keyboard
        self.root.bind("<Escape>", lambda e: self.exit_fullscreen())
        self.root.bind("<Control-q>", lambda e: self.root.destroy())
        self.cell = BASE_CELL
        self.canvas = tk.Canvas(
            self.root,
            width=GRID[0] * self.cell,
            height=GRID[1] * self.cell,
            bg="#0b1e0b",
            highlightthickness=0,
        )
        self.canvas.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        # Recompute sizing when the canvas itself resizes
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        # Click to add a collectible point; right-click to toggle obstacle
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        panel = tk.Frame(self.root)
        panel.grid(row=1, column=1, sticky="ns")
        self.panel = panel
        tk.Label(panel, text="Algorithm").pack(pady=2)
        self.mode_var = tk.StringVar(value="A*")
        tk.OptionMenu(panel, self.mode_var, "UCS", "A*").pack(fill="x")
        tk.Label(panel, text="Heuristic").pack(pady=2)
        self.heur_var = tk.StringVar(value="nearest")
        tk.OptionMenu(panel, self.heur_var, *self.heuristics.keys()).pack(fill="x")
        # Speed control (UI only)
        tk.Label(panel, text="Speed (ms/step)").pack(pady=(6, 2))
        self.speed_var = tk.IntVar(value=self.step_ms)
        tk.Scale(
            panel,
            from_=5,
            to=400,
            resolution=5,
            orient="horizontal",
            variable=self.speed_var,
            command=lambda v: self.on_speed_change(v),
        ).pack(fill="x")
        tk.Button(panel, text="Start", command=self.start_run).pack(fill="x", pady=6)
        tk.Button(panel, text="Reset", command=self.reset).pack(fill="x")
        tk.Button(panel, text="Quit", command=self.root.destroy).pack(fill="x", pady=(6, 0))
        # Sidebar status and stats
        self.status = tk.StringVar()
        tk.Label(panel, textvariable=self.status, wraplength=160, justify="left").pack(pady=(10, 4))
        tk.Label(panel, text="Stats").pack(anchor="w")
        self.stats_var = tk.StringVar(value="")
        tk.Label(panel, textvariable=self.stats_var, wraplength=160, justify="left").pack(pady=(0, 10), anchor="w")
        # Load theme images (F-35 for robot, radar for obstacles, airbase for targets)
        self._src_robot = None
        self._src_obstacle = None
        self._src_target = None
        base_dir = os.path.dirname(__file__)
        robot_path = os.path.join(base_dir, "b2.png")
        radar_path = os.path.join(base_dir, "radar.png")
        target_path = os.path.join(base_dir, "airbase.png")
        if Image is not None and ImageTk is not None:
            try:
                self._src_robot = Image.open(robot_path)
            except Exception:
                self._src_robot = None
            try:
                self._src_obstacle = Image.open(radar_path)
            except Exception:
                self._src_obstacle = None
            try:
                self._src_target = Image.open(target_path)
            except Exception:
                self._src_target = None
        else:
            try:
                self._src_robot = tk.PhotoImage(file=robot_path, master=self.root)
            except Exception:
                self._src_robot = None
            try:
                self._src_obstacle = tk.PhotoImage(file=radar_path, master=self.root)
            except Exception:
                self._src_obstacle = None
            try:
                self._src_target = tk.PhotoImage(file=target_path, master=self.root)
            except Exception:
                self._src_target = None
        self.robot_img = None
        self.obstacle_img = None
        self.target_img = None
        # Default heading: facing left (matches base image)
        self.heading = (-1, 0)
        self.robot = self.start
        self.objects = set()
        self.path = []
        self.after_ids = []
        self.move_id = None
        self.running = False
        self.static_in_bounds = set()
        self.dynamic_cells = []
        self.obstacles = set()
        self.step_counter = 0
        self.dynamic_events = []  # (step_threshold, preset_pos)
        self.dynamic_idx = 0
        self.last_plan_stats = {"expanded": 0, "time": 0.0}
        self.total_stats = {"expanded": 0, "time": 0.0}
        self.grid_var = tk.StringVar(value=f"{GRID[0]}x{GRID[1]}")
        top = tk.Frame(self.root)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        self.top = top
        tk.Label(top, text="Grid Size").pack(side="left")
        grid_options = []
        for opt in (self.grid_var.get(), "10x10", "20x20", "40x40", "50x50"):
            if opt not in grid_options:
                grid_options.append(opt)
        grid_menu = tk.OptionMenu(top, self.grid_var, *grid_options)
        grid_menu.pack(side="left", padx=5)
        self.grid_var.trace_add("write", self.on_grid_change)
        # Allow canvas column/row to expand
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        # Recompute cell size on window resize
        self.root.bind("<Configure>", self.on_root_resize)
        self.change_grid(self.grid_var.get())

    def on_grid_change(self, *_):
        self.change_grid(self.grid_var.get())

    def compute_cell_size(self, cols, rows):
        self.root.update_idletasks()
        # Prefer actual canvas size for precise fill
        c_w = self.canvas.winfo_width() if hasattr(self, "canvas") else 0
        c_h = self.canvas.winfo_height() if hasattr(self, "canvas") else 0
        if c_w >= 50 and c_h >= 50:
            return max(5, int(min(c_w // cols, c_h // rows)))
        # Fallback: approximate available space from window/panel/top
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height()
        if win_w < 100 or win_h < 100:
            win_w = self.root.winfo_screenwidth()
            win_h = self.root.winfo_screenheight()
        panel_w = getattr(self, "panel", None).winfo_width() if hasattr(self, "panel") else 240
        if not panel_w:
            panel_w = getattr(self, "panel", None).winfo_reqwidth() if hasattr(self, "panel") else 240
        top_h = getattr(self, "top", None).winfo_height() if hasattr(self, "top") else 40
        if not top_h:
            top_h = getattr(self, "top", None).winfo_reqheight() if hasattr(self, "top") else 40
        avail_w = max(200, win_w - panel_w)
        avail_h = max(200, win_h - top_h)
        return max(5, int(min(avail_w // cols, avail_h // rows)))

    def change_grid(self, value):
        cols, rows = map(int, value.split("x"))
        self.cell = self.compute_cell_size(cols, rows)
        self.cols, self.rows = cols, rows
        # Rescale images to the new cell size
        self._update_scaled_images()
        self.reset()

    def on_root_resize(self, event):
        # Recompute cell size to fill available space, and redraw if changed
        if not hasattr(self, "cols") or not hasattr(self, "rows"):
            return
        new_size = self.compute_cell_size(self.cols, self.rows)
        if new_size != self.cell:
            self.cell = new_size
            self._update_scaled_images()
            self.draw()

    def on_canvas_resize(self, event):
        if not hasattr(self, "cols") or not hasattr(self, "rows"):
            return
        # Use the actual canvas size to set cell size so we fill perfectly
        c_w = max(1, int(self.canvas.winfo_width()))
        c_h = max(1, int(self.canvas.winfo_height()))
        new_size = max(5, int(min(c_w // self.cols, c_h // self.rows)))
        if new_size != self.cell:
            self.cell = new_size
            self._update_scaled_images()
            self.draw()

    def exit_fullscreen(self):
        # Exit fullscreen so window controls are visible again
        try:
            self.root.attributes("-fullscreen", False)
        except Exception:
            pass

    def on_speed_change(self, value):
        try:
            self.step_ms = max(1, int(float(value)))
        except Exception:
            return
        # Apply change immediately by rescheduling the next move
        if self.running and self.move_id is not None:
            try:
                self.root.after_cancel(self.move_id)
            except Exception:
                pass
            self.move_id = self.root.after(self.step_ms, self.move_step)

    def box(self, cell, pad=0):
        x, y = cell
        size = self.cell
        return x * size + pad, y * size + pad, (x + 1) * size - pad, (y + 1) * size - pad

    def center(self, cell):
        x, y = cell
        size = self.cell
        return x * size + size // 2, y * size + size // 2

    def _scale_image(self, src, target):
        if src is None:
            return None
        target_px = max(8, int(target * 0.9))
        # PIL path
        if Image is not None and ImageTk is not None and hasattr(src, "size"):
            try:
                w, h = src.size
                if w == 0 or h == 0:
                    return None
                scale = min(target_px / w, target_px / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                resized = src.resize((new_w, new_h), Image.LANCZOS)
                return ImageTk.PhotoImage(resized)
            except Exception:
                return None
        # Tk PhotoImage path
        try:
            w = src.width(); h = src.height()
            if w == 0 or h == 0:
                return None
            factor = max(1, int(max(w / target_px, h / target_px)))
            return src.subsample(factor, factor) if factor > 1 else src
        except Exception:
            return None

    def _update_scaled_images(self):
        # Build directional variants for robot if PIL is available
        self.robot_imgs_by_dir = {}
        if Image is not None and ImageTk is not None and isinstance(self._src_robot, Image.Image):
            try:
                # Make oriented variants from original, then resize each
                left_img = self._src_robot.copy()
                right_img = ImageOps.mirror(left_img) if ImageOps is not None else left_img.transpose(Image.FLIP_LEFT_RIGHT)
                up_img = left_img.rotate(-90, expand=True)
                down_img = left_img.rotate(90, expand=True)
                self.robot_imgs_by_dir["left"] = self._scale_image(left_img, self.cell)
                self.robot_imgs_by_dir["right"] = self._scale_image(right_img, self.cell)
                self.robot_imgs_by_dir["up"] = self._scale_image(up_img, self.cell)
                self.robot_imgs_by_dir["down"] = self._scale_image(down_img, self.cell)
            except Exception:
                # Fallback to single image
                self.robot_img = self._scale_image(getattr(self, "_src_robot", None), self.cell)
                self.robot_imgs_by_dir = {}
        else:
            # Without PIL, we cannot rotate; just scale once
            self.robot_img = self._scale_image(getattr(self, "_src_robot", None), self.cell)
            self.robot_imgs_by_dir = {}
        self.obstacle_img = self._scale_image(getattr(self, "_src_obstacle", None), self.cell)
        self.target_img = self._scale_image(getattr(self, "_src_target", None), self.cell)

    def _dir_key(self, dx, dy):
        if dx == 1 and dy == 0:
            return "right"
        if dx == -1 and dy == 0:
            return "left"
        if dx == 0 and dy == -1:
            return "up"
        if dx == 0 and dy == 1:
            return "down"
        # Default to last known or left
        return "left"

    def reset(self):
        self.running = False
        if self.move_id:
            self.root.after_cancel(self.move_id)
            self.move_id = None
        for aid in self.after_ids:
            self.root.after_cancel(aid)
        self.after_ids.clear()
        self.robot = self.start
        # Clear obstacles on reset
        if hasattr(self, "obstacles"):
            self.obstacles.clear()
        # Recalculate cell size to fit current window and resize canvas/images
        if hasattr(self, "cols") and hasattr(self, "rows"):
            new_size = self.compute_cell_size(self.cols, self.rows)
            if new_size != self.cell:
                self.cell = new_size
                self.canvas.config(width=self.cols * self.cell, height=self.rows * self.cell)
                self._update_scaled_images()
        self.static_in_bounds = {
            pos for pos in self.static if 0 <= pos[0] < self.cols and 0 <= pos[1] < self.rows and pos != self.start
        }
        self.objects = set(self.static_in_bounds)
        self.path = []
        self.step_counter = 0
        self.dynamic_events = []
        self.dynamic_cells = self.build_dynamic_cells()
        self.dynamic_idx = 0
        self.last_plan_stats = {"expanded": 0, "time": 0.0}
        self.total_stats = {"expanded": 0, "time": 0.0}
        self.status.set("Waiting for start.")
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        grid_color = "#145c14"
        for x in range(self.cols):
            for y in range(self.rows):
                self.canvas.create_rectangle(*self.box((x, y)), outline=grid_color)
        self.canvas.create_rectangle(*self.box(self.start, 3), outline="#2ecc71", width=2)
        # Obstacles as radar images (fallback to green squares)
        for obs in getattr(self, "obstacles", set()):
            if self.obstacle_img is not None:
                cx, cy = self.center(obs)
                self.canvas.create_image(cx, cy, image=self.obstacle_img)
            else:
                self.canvas.create_rectangle(*self.box(obs, 2), fill="#2e7d32", outline="#1b5e20")
        # Targets as airbase images (fallback to orange dots)
        for obj in self.objects:
            if self.target_img is not None:
                cx, cy = self.center(obj)
                self.canvas.create_image(cx, cy, image=self.target_img)
            else:
                self.canvas.create_oval(*self.box(obj, 10), fill="#ff9800", outline="")
        route = [self.robot] + self.path
        if len(route) > 1:
            coords = [c for cell in route for c in self.center(cell)]
            self.canvas.create_line(*coords, fill="#39ff14", width=3, dash=(6, 2))
        # Choose oriented robot image if available
        rx, ry = self.center(self.robot)
        oriented = None
        if getattr(self, "robot_imgs_by_dir", None):
            if self.path:
                dx = self.path[0][0] - self.robot[0]
                dy = self.path[0][1] - self.robot[1]
                dir_key = self._dir_key(dx, dy)
            else:
                dir_key = self._dir_key(self.heading[0], self.heading[1])
            oriented = self.robot_imgs_by_dir.get(dir_key)
        if oriented is not None:
            self.canvas.create_image(rx, ry, image=oriented)
        elif self.robot_img is not None:
            self.canvas.create_image(rx, ry, image=self.robot_img)
        else:
            self.canvas.create_oval(*self.box(self.robot, 6), fill="#2196f3", outline="")
        # Update sidebar stats instead of drawing on the grid
        stats = self.last_plan_stats
        totals = self.total_stats
        label = f"Remaining: {len(self.objects)}"
        if stats["expanded"]:
            label += f" | Expanded: {stats['expanded']} | {stats['time']*1000:.1f} ms"
        if totals["expanded"]:
            label += f"\nTotal Expanded: {totals['expanded']} | Total Time: {totals['time']:.3f}s"
        self.stats_var.set(label)

    def start_run(self):
        self.reset()
        self.running = True
        self.status.set("Collecting objects...")
        self.schedule_dynamic()
        self.move_step()

    def schedule_dynamic(self):
        # Build step-based deterministic spawn schedule so fast runs don't miss spawns
        presets = deterministic_presets(
            self.cols, self.rows, self.start, self.static_in_bounds, count=len(DYNAMIC_PLAN)
        )
        events = []
        for idx, (delay_ms, _pos) in enumerate(DYNAMIC_PLAN):
            # Convert absolute spawn delay to a fixed step threshold based on the baseline STEP_MS.
            steps = max(1, int(math.ceil(delay_ms / float(max(1, STEP_MS)))))
            preset = presets[idx] if idx < len(presets) else None
            events.append((steps, preset))
        self.dynamic_events = sorted(events, key=lambda e: e[0])

    def build_dynamic_cells(self):
        # Exclude obstacles from potential dynamic spawn locations
        return make_dynamic_cells(self.cols, self.rows, self.static_in_bounds | self.obstacles, self.start)

    def add_dynamic(self, pos=None):
        if not self.running:
            return
        # Try preset position; if blocked, pick nearest free cell to the preset.
        if pos is None:
            pos = self.next_dynamic_cell()
        if pos is not None:
            pos = self.nearest_free_cell(pos)
        if pos is None:
            # Fallback to next candidate list if everything around preset is blocked
            pos = self.next_dynamic_cell()
        if pos is None:
            return
        self.objects.add(pos)
        self.status.set(f"Dynamic object spawned at {pos}")
        self.path = []
        self.draw()

    def spawn_due_events(self):
        # Spawn all events whose threshold in steps has been reached
        while self.dynamic_events and self.step_counter >= self.dynamic_events[0][0]:
            _, preset = self.dynamic_events.pop(0)
            self.add_dynamic(preset)

    def nearest_free_cell(self, target):
        # Find nearest free cell to target, avoiding start, robot, obstacles, and existing objects
        if target is None:
            return None
        tx, ty = target
        tx = min(max(0, tx), self.cols - 1)
        ty = min(max(0, ty), self.rows - 1)
        blocked = set(self.objects)
        blocked.update(getattr(self, 'obstacles', set()))
        blocked.add(self.start)
        blocked.add(self.robot)
        if (tx, ty) not in blocked:
            return (tx, ty)
        # Spiral search by increasing manhattan distance
        max_r = max(self.cols, self.rows)
        for r in range(1, max_r + 2):
            for dx in range(-r, r + 1):
                dy = r - abs(dx)
                for sy in (-1, 1) if dy != 0 else (1,):
                    nx, ny = tx + dx, ty + sy * dy
                    if 0 <= nx < self.cols and 0 <= ny < self.rows and (nx, ny) not in blocked:
                        return (nx, ny)
        return None

    def on_canvas_click(self, event):
        # Convert pixel coords to grid cell
        try:
            col = int(event.x // self.cell)
            row = int(event.y // self.cell)
        except Exception:
            return
        if not (0 <= col < getattr(self, 'cols', GRID[0]) and 0 <= row < getattr(self, 'rows', GRID[1])):
            return
        pos = (col, row)
        # Avoid placing on start or duplicate object
        if pos == self.start or pos in self.objects or pos in self.obstacles:
            return
        self.objects.add(pos)
        self.status.set(f"Point added at {pos}")
        # Force replan if currently running
        self.path = []
        self.draw()

    def on_canvas_right_click(self, event):
        # Toggle obstacle at clicked cell (right-click)
        try:
            col = int(event.x // self.cell)
            row = int(event.y // self.cell)
        except Exception:
            return
        if not (0 <= col < getattr(self, 'cols', GRID[0]) and 0 <= row < getattr(self, 'rows', GRID[1])):
            return
        pos = (col, row)
        if pos == self.start or pos == self.robot:
            return
        if pos in self.objects:
            return
        if pos in self.obstacles:
            self.obstacles.remove(pos)
            self.status.set(f"Obstacle removed at {pos}")
        else:
            self.obstacles.add(pos)
            self.status.set(f"Obstacle added at {pos}")
        # Clear current plan so replanning accounts for new obstacles
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
        # Advance tick and spawn any due dynamic events
        self.step_counter += 1
        self.spawn_due_events()
        if not self.path:
            self.path = self.plan_path()
            if not self.path:
                # If no path now, keep ticking if more events are pending; otherwise finish when safe
                if self.dynamic_events:
                    self.move_id = self.root.after(self.step_ms, self.move_step)
                    return
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
        # Update heading based on movement delta
        try:
            dx = nxt[0] - self.robot[0]
            dy = nxt[1] - self.robot[1]
            self.heading = (dx, dy)
        except Exception:
            pass
        self.robot = nxt
        if nxt in self.objects:
            self.objects.remove(nxt)
        self.draw()
        # Keep ticking if there are future events pending
        if self.dynamic_events:
            self.move_id = self.root.after(self.step_ms, self.move_step)
            return
        if not self.objects and self.robot == self.start and not self.path:
            self.running = False
            self.status.set("Mission complete.")
            return
        self.move_id = self.root.after(self.step_ms, self.move_step)

    def plan_path(self):
        mode = self.mode_var.get()
        heuristic_fn = self.heuristics[self.heur_var.get()] if mode == "A*" else self.heuristics["zero"]
        path, expanded, elapsed = search_plan(
            self.robot, self.objects, self.cols, self.rows, self.start, mode, heuristic_fn, obstacles=self.obstacles
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
