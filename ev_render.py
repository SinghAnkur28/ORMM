# ev_render.py
from typing import List, Tuple

import matplotlib.pyplot as plt


class EVRenderer:
    """
    Matplotlib renderer for EVRoutingEnv.
    - Depot: blue square
    - Customers: red (unserved) / green (served) circles
    - Stations: orange triangles
    - Vehicles: colored X with SoC labels
    - Trails: faint polylines showing each vehicle's path
    - HUD: time, distance, revenue, served/total
    """

    def __init__(self, env, figsize: Tuple[float, float] = (6, 6),
                 show_labels: bool = True, show_trails: bool = True):
        self.env = env
        self.show_labels = show_labels
        self.show_trails = show_trails

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.paths: List[List[Tuple[float, float]]] = [[] for _ in range(self.env.maxV)]

        # simple color palette for vehicles
        self.palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf"
        ]

        self._init_canvas()

    def reset_paths(self):
        self.paths = [[] for _ in range(self.env.maxV)]

    def draw(self, block: bool = False, title: str = "EV Routing"):
        env = self.env
        ax = self.ax
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

        # Depot
        dx, dy = env.depot_xy
        ax.scatter([dx], [dy], s=120, marker="s", label="Depot")

        # Customers (red=unserved, green=served)
        cx, cy, ccolors = [], [], []
        for j in range(env.maxC):
            if j >= len(env.cust_xy):
                continue
            x, y = env.cust_xy[j]
            demand = env.cust_demand[j]
            if demand <= 0.0:
                continue
            served = env.c_served[j]
            cx.append(x); cy.append(y)
            ccolors.append("tab:green" if served else "tab:red")
        if cx:
            ax.scatter(cx, cy, s=60, marker="o", c=ccolors, label="Customers")

        # Stations
        sx, sy = [], []
        for k in range(env.maxS):
            if k >= len(env.st_xy):
                continue
            x, y = env.st_xy[k]
            sx.append(x); sy.append(y)
        if sx:
            ax.scatter(sx, sy, s=80, marker="^", label="Stations", color="tab:orange")

        # Trails (colored per vehicle)
        if self.show_trails:
            for i in range(env.maxV):
                node_idx = env.v_loc[i] if i < len(env.v_loc) else 0
                x, y = env.node_coords[node_idx]
                self.paths[i].append((x, y))
                pts = self.paths[i]
                if len(pts) >= 2:
                    xs, ys = zip(*pts)
                    color = self.palette[i % len(self.palette)]
                    ax.plot(xs, ys, linewidth=2, alpha=0.6, color=color)

        # Vehicles (positions + SoC labels)
        for i in range(env.maxV):
            node_idx = env.v_loc[i] if i < len(env.v_loc) else 0
            x, y = env.node_coords[node_idx]
            soc = env.v_soc[i] if i < len(env.v_soc) else 0.0
            color = self.palette[i % len(self.palette)]
            ax.scatter([x], [y], s=90, marker="X", color=color,
                       label="Vehicles" if i == 0 else None)
            if self.show_labels:
                ax.text(x + 0.8, y + 0.8, f"V{i}(SoC={soc:.1f}kWh)", fontsize=8, color=color)

        # HUD: time, distance, revenue, served/total
        t = getattr(env, "t", 0.0)
        dist = getattr(env, "total_distance", 0.0)
        rev = getattr(env, "total_revenue", 0.0)
        served = sum(1 for j in range(env.maxC)
                     if env.cust_demand[j] > 0 and env.c_served[j])
        total = sum(1 for j in range(env.maxC)
                    if env.cust_demand[j] > 0)
        hud = f"t={t:.1f} | dist={dist:.1f} | rev={rev:.2f} | served={served}/{total}"
        ax.text(0.02, 0.98, hud, transform=ax.transAxes, va="top", ha="left")

        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.pause(0.001)
        if block:
            plt.show()

    def _init_canvas(self):
        self.ax.set_title("EV Routing")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.2)
        plt.tight_layout()
