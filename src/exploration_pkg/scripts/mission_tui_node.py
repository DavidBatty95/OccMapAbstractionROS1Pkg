#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mission_tui_node.py

Radiation-aware Mission TUI (display-only, non-intrusive)

Upgrades in this version (no feature removals):
- Uses /sim_rad_sensor as std_msgs/Float32 (µSv/s) -> converts to Sv/s
- Tracks cumulative distance + cumulative dose (Sv) with stable integration
- Smooths motion realistically using a critically-damped “virtual robot” observer
  (removes staircase/jagged distance even when pose updates are quantised/low-rate)
- Stable ncurses layout:
  - Header
  - Status
  - Metrics
  - Trends (sparklines)
  - INFO Logs (10 lines)
  - System diagnostics (CPU/RAM)
- Press 'p' to save a thesis-ready Matplotlib figure (distance + cumulative dose)
- Plot save path shown in INFO logs, saved to: ~/radtest_ws/mission_plots

ROS Noetic + Python3 + curses
"""

import rospy
import curses
import time
import math
import json
import psutil
from collections import deque
import os
import tf2_ros

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from std_msgs.msg import String, Float32
from rosgraph_msgs.msg import Log


# ================= helpers ================= #

def safe_addstr(win, y, x, s, attr=0):
    """Write safely into curses windows without crashing on resize."""
    try:
        h, w = win.getmaxyx()
        if 0 <= y < h and 0 <= x < w:
            win.addstr(y, x, str(s)[: max(0, w - x - 1)], attr)
    except Exception:
        pass


def sparkline(values, width):
    if not values or width <= 0:
        return " " * width
    chars = " ▁▂▃▄▅▆▇█"
    vals = list(values)[-width:]
    vmax = max(vals) if max(vals) > 1e-12 else 1.0
    return "".join(
        chars[min(int(v / vmax * (len(chars) - 1)), len(chars) - 1)]
        for v in vals
    ).ljust(width)


def draw_box(win, title):
    win.attron(curses.A_DIM)
    win.box()
    win.attroff(curses.A_DIM)
    safe_addstr(win, 0, 2, f" {title} ", curses.A_BOLD)


def format_dose(dose_sv):
    if dose_sv < 1e-6:
        return f"{dose_sv * 1e9:.2f} nSv"
    if dose_sv < 1e-3:
        return f"{dose_sv * 1e6:.2f} µSv"
    if dose_sv < 1.0:
        return f"{dose_sv * 1e3:.2f} mSv"
    return f"{dose_sv:.3f} Sv"


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# ================= TUI ================= #

class MissionTUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        curses.curs_set(0)
        stdscr.nodelay(True)

        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_YELLOW, -1)
        curses.init_pair(5, curses.COLOR_RED, -1)

        # ---------------- state ----------------
        self.mission_state = "UNKNOWN"
        self.nav_state = "IDLE"
        self.radiation_aware = False

        # ---------------- metrics ----------------
        self.total_distance_m = 0.0         # meters
        self.total_dose_sv = 0.0            # Sieverts
        self.current_dose_rate_sv_s = 0.0   # Sv/s

        # ---------------- time bases ----------------
        self.t0 = time.time()
        self.last_draw_time = time.time()
        self.last_dose_time = time.time()

        # ---------------- plotting history ----------------
        # Keep these for the figure on 'p'
        self.time_hist = []
        self.dist_hist = []
        self.dose_hist_plot = []

        # ---------------- sparklines ----------------
        self.spark_len = int(rospy.get_param("~spark_len", 160))
        self.graph_dt = float(rospy.get_param("~graph_dt", 0.5))
        self.last_graph_sample = time.time()

        self.distance_hist = deque(maxlen=self.spark_len)
        self.dose_hist = deque(maxlen=self.spark_len)

        # ---------------- logs ----------------
        self.logs = deque(maxlen=int(rospy.get_param("~log_lines", 10)))

        # ---------------- motion smoothing ----------------
        # We build a "virtual robot" that smoothly follows the measured robot pose.
        # This destroys the jagged staircase artefacts from quantised/low-rate poses.
        self.meas_xy = None             # latest measured x,y
        self.meas_time = None

        self.vx = 0.0
        self.vy = 0.0
        self.x = None                   # virtual position
        self.y = None

        # Dynamics knobs (tuned to look like a real base in RViz)
        # Higher wn => snappier tracking; lower => floatier.
        self.wn = float(rospy.get_param("~motion_wn", 3.0))          # rad/s
        self.max_speed = float(rospy.get_param("~max_speed", 1.5))   # m/s
        self.max_accel = float(rospy.get_param("~max_accel", 2.5))   # m/s^2

        # Outlier rejection (teleports / bad packets)
        self.max_jump_m = float(rospy.get_param("~max_jump_m", 2.0))

        # Integration rate for the virtual robot (independent of draw_hz)
        self.motion_hz = float(rospy.get_param("~motion_hz", 60.0))
        self.last_motion_update = time.time()

        # ROS init + subs
        # IMPORTANT: init_node MUST happen before creating TF listeners/buffers.
        rospy.init_node("mission_tui", anonymous=False)

        # TF fallback for robot pose (map -> base_link)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.last_tf_xy = None

        rospy.Subscriber("/mission_state", String, self._on_mission)
        rospy.Subscriber("/exploration_policy/debug_state", String, self._on_debug)
        rospy.Subscriber("/sim_rad_sensor", Float32, self._on_radiation)
        rospy.Subscriber("/rosout", Log, self._on_log)

        self.draw_hz = float(rospy.get_param("~draw_hz", 6.0))

        self._init_windows()

        # First message for plot path clarity
        self.plot_dir = os.path.expanduser("~/radtest_ws/src/exploration_pkg/mission_plots")
        self.logs.appendleft(f"[MissionTUI] Plot dir: {self.plot_dir} (press 'p' to save)")

    # ---------- TF helpers ---------- #

    def _read_tf_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_link", rospy.Time(0), rospy.Duration(0.01)
            )
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

    # ---------- callbacks ---------- #

    def _on_mission(self, msg):
        try:
            self.mission_state = (msg.data or "").strip() or "UNKNOWN"
        except Exception:
            self.mission_state = "UNKNOWN"

    def _on_debug(self, msg):
        try:
            d = json.loads(msg.data)
            self.nav_state = "NAVIGATING" if d.get("active_goal") else "IDLE"
            self.radiation_aware = bool(d.get("use_radiation", False))

            xy = d.get("robot_xy")
            if xy and len(xy) >= 2:
                x = float(xy[0])
                y = float(xy[1])
                now = time.time()

                # Initialize virtual state on first measurement
                if self.meas_xy is None:
                    self.meas_xy = (x, y)
                    self.meas_time = now
                    self.x, self.y = x, y
                    self.vx, self.vy = 0.0, 0.0
                    return

                # Reject huge jumps (teleports / broken measurement)
                if math.hypot(x - self.meas_xy[0], y - self.meas_xy[1]) > self.max_jump_m:
                    self.logs.appendleft(f"[MissionTUI] Ignored pose jump > {self.max_jump_m:.1f} m")
                    self.meas_xy = (x, y)
                    self.meas_time = now
                    # Re-anchor smoothly rather than snapping distance
                    self.x, self.y = x, y
                    self.vx, self.vy = 0.0, 0.0
                    return

                self.meas_xy = (x, y)
                self.meas_time = now
        except Exception:
            pass

    def _on_radiation(self, msg):
        # msg.data expected: µSv/s -> convert to Sv/s
        try:
            self.current_dose_rate_sv_s = max(0.0, float(msg.data)) * 1e-6
        except Exception:
            self.current_dose_rate_sv_s = 0.0

    def _on_log(self, msg):
        try:
            if msg.level == Log.INFO and msg.msg:
                self.logs.appendleft(str(msg.msg).strip())
        except Exception:
            pass

    # ---------- physics / integration ---------- #

    def _update_dose(self):
        """Integrate dose using wall-clock time base."""
        now = time.time()
        dt = now - self.last_dose_time
        if dt > 0.0:
            self.total_dose_sv += self.current_dose_rate_sv_s * dt
        self.last_dose_time = now

    def _update_virtual_motion(self):
        """
        Critically-damped 2nd order tracker:
        a = wn^2*(x_meas - x) - 2*wn*v
        with acceleration and speed limits.

        Distance is integrated from the *virtual* robot motion, not the raw steps.
        """

        # --- TF fallback if debug_state robot_xy is missing ---
        tf_xy = self._read_tf_pose()
        if tf_xy is not None:
            self.meas_xy = tf_xy

        # No pose at all yet
        if self.meas_xy is None:
            return

        # First ever pose → initialise virtual robot here
        if self.x is None or self.y is None:
            self.x, self.y = self.meas_xy
            self.vx, self.vy = 0.0, 0.0
            self.last_motion_update = time.time()
            return

        now = time.time()
        # Step at a stable rate (motion_hz), but tolerate delays
        dt = now - self.last_motion_update
        if dt <= 0.0:
            return

        # Substep if draw stalls, so dynamics stays stable
        max_dt = 1.0 / self.motion_hz
        n = int(max(1, math.ceil(dt / max_dt)))
        sub_dt = dt / n

        # critical damping
        wn = self.wn
        c = 2.0 * wn

        for _ in range(n):
            mx, my = self.meas_xy

            ex = mx - self.x
            ey = my - self.y

            ax = (wn * wn) * ex - c * self.vx
            ay = (wn * wn) * ey - c * self.vy

            # acceleration limiting
            a_mag = math.hypot(ax, ay)
            if a_mag > self.max_accel:
                s = self.max_accel / max(a_mag, 1e-9)
                ax *= s
                ay *= s

            # integrate velocity
            self.vx += ax * sub_dt
            self.vy += ay * sub_dt

            # speed limiting
            v_mag = math.hypot(self.vx, self.vy)
            if v_mag > self.max_speed:
                s = self.max_speed / max(v_mag, 1e-9)
                self.vx *= s
                self.vy *= s

            # integrate position
            prev_x, prev_y = self.x, self.y
            self.x += self.vx * sub_dt
            self.y += self.vy * sub_dt

            # distance from virtual motion
            self.total_distance_m += math.hypot(self.x - prev_x, self.y - prev_y)

        self.last_motion_update = now

    def _sample_histories(self):
        """Downsample time series and sparklines at graph_dt to keep them stable."""
        now = time.time()
        if now - self.last_graph_sample < self.graph_dt:
            return

        t = now - self.t0
        self.time_hist.append(t)
        self.dist_hist.append(self.total_distance_m)
        self.dose_hist_plot.append(self.total_dose_sv)

        self.distance_hist.append(self.total_distance_m)
        self.dose_hist.append(self.total_dose_sv)

        self.last_graph_sample = now

    # ---------- plotting ---------- #

    def save_plot(self):
        if len(self.time_hist) < 10:
            self.logs.appendleft("[MissionTUI] Not enough samples to plot yet.")
            return

        os.makedirs(self.plot_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(self.plot_dir, f"mission_{ts}.png")

        # Thesis-ready: clean, high DPI, consistent sizing
        plt.figure(figsize=(7.5, 4.5))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        # -------------------------------
        # Distance (blue, left axis)
        # -------------------------------
        dist_line, = ax1.plot(
            self.time_hist,
            self.dist_hist,
            linewidth=2.2,
            color="tab:blue",
            label="Cumulative distance (m)"
        )
        ax1.set_ylabel("Distance (m)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # -------------------------------
        # Dose (red, right axis)
        # -------------------------------
        dose_line, = ax2.plot(
            self.time_hist,
            [d * 1e6 for d in self.dose_hist_plot],  # Sv → µSv
            linewidth=2.2,
            color="tab:red",
            label="Cumulative absorbed dosage (µSv)"
        )
        ax2.set_ylabel("Cumulative dose (µSv)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        # -------------------------------
        # Axes, grid, legend
        # -------------------------------
        ax1.set_xlabel("Time (s)")
        ax1.grid(True, alpha=0.3)

        ax1.legend(
            [dist_line, dose_line],
            ["Distance (m)", "Cumulative dose (µSv)"],
            loc="upper left",
            frameon=True
        )

        plt.title("")
        plt.tight_layout()
        plt.savefig(fname, dpi=600)
        plt.close()

        self.logs.appendleft(f"[MissionTUI] Plot saved: {fname}")

    # ---------- windows ---------- #

    def _init_windows(self):
        H, W = self.stdscr.getmaxyx()

        # Minimum sensible layout
        self.min_H = 34
        self.min_W = 100

        self.header = self.stdscr.subwin(1, W, 0, 0)
        self.status = self.stdscr.subwin(7, 38, 3, 2)
        self.metrics = self.stdscr.subwin(7, 38, 11, 2)

        graph_h = 15
        self.graphs = self.stdscr.subwin(graph_h, max(10, W - 44), 3, 42)

        # Logs + system below graphs (keeps your "info box" from disappearing)
        logs_y = 3 + graph_h + 1
        self.logs_win = self.stdscr.subwin(10, max(10, W - 4), logs_y, 2)
        self.sys_win = self.stdscr.subwin(5, max(10, W - 4), logs_y + 11, 2)

    def _maybe_relayout(self):
        """Handle terminal resizes gracefully."""
        H, W = self.stdscr.getmaxyx()
        # If resized, re-init windows to avoid overlap/glitches
        if getattr(self, "_last_HW", None) != (H, W):
            self._last_HW = (H, W)
            try:
                curses.resizeterm(H, W)
            except Exception:
                pass
            self._init_windows()

    # ---------- draw ---------- #

    def draw(self):
        self._maybe_relayout()
        H, W = self.stdscr.getmaxyx()

        if H < self.min_H or W < self.min_W:
            self.stdscr.erase()
            safe_addstr(self.stdscr, 0, 0, f"Terminal too small (min {self.min_W}x{self.min_H}). Current: {W}x{H}")
            safe_addstr(self.stdscr, 2, 0, "Resize the window, or reduce TUI elements.")
            self.stdscr.refresh()
            return

        # Update “physics”
        self._update_virtual_motion()
        self._update_dose()
        self._sample_histories()

        # Header
        self.header.erase()
        safe_addstr(
            self.header, 0, 0,
            " RADIATION-AWARE EXPLORATION — MISSION TELEMETRY ".center(W),
            curses.color_pair(1) | curses.A_BOLD
        )
        self.header.noutrefresh()

        # Status
        self.status.erase()
        draw_box(self.status, "Status")
        ra_txt = "ON" if self.radiation_aware else "OFF"
        ra_col = curses.color_pair(2 if self.radiation_aware else 4) | curses.A_BOLD
        safe_addstr(self.status, 2, 2, f"Mission  : {self.mission_state}", curses.A_BOLD)
        safe_addstr(self.status, 3, 2, f"Nav      : {self.nav_state}")
        safe_addstr(self.status, 4, 2, f"Radiation: {ra_txt}", ra_col)
        safe_addstr(self.status, 5, 2, "Keys     : [p] plot   [q] quit", curses.A_DIM)
        self.status.noutrefresh()

        # Metrics
        self.metrics.erase()
        draw_box(self.metrics, "Metrics")
        safe_addstr(self.metrics, 2, 2, f"Distance : {self.total_distance_m:8.2f} m", curses.color_pair(3))
        safe_addstr(self.metrics, 3, 2, f"Dose     : {format_dose(self.total_dose_sv)}", curses.color_pair(2))
        safe_addstr(self.metrics, 5, 2, f"Dose rate: {self.current_dose_rate_sv_s * 1e6:6.2f} µSv/s", curses.color_pair(2))
        self.metrics.noutrefresh()

        # Trends
        self.graphs.erase()
        draw_box(self.graphs, "Trends")
        iw = self.graphs.getmaxyx()[1] - 4

        safe_addstr(self.graphs, 2, 2, "Distance", curses.color_pair(3) | curses.A_BOLD)
        safe_addstr(self.graphs, 3, 2, sparkline(self.distance_hist, iw), curses.color_pair(3))

        safe_addstr(self.graphs, 7, 2, "Dose", curses.color_pair(2) | curses.A_BOLD)
        safe_addstr(self.graphs, 8, 2, sparkline(self.dose_hist, iw), curses.color_pair(2))
        self.graphs.noutrefresh()

        # Logs
        self.logs_win.erase()
        draw_box(self.logs_win, "INFO Logs")
        for i, l in enumerate(self.logs):
            safe_addstr(self.logs_win, 1 + i, 2, l, curses.A_DIM)
        self.logs_win.noutrefresh()

        # System diagnostics
        self.sys_win.erase()
        draw_box(self.sys_win, "System Diagnostics")
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        safe_addstr(self.sys_win, 2, 2, f"CPU usage : {cpu:5.1f} %")
        safe_addstr(self.sys_win, 3, 2, f"RAM usage : {mem:5.1f} %")
        self.sys_win.noutrefresh()

        curses.doupdate()

    # ---------- main loop ---------- #

    def run(self):
        rate = rospy.Rate(self.draw_hz)
        while not rospy.is_shutdown():
            k = self.stdscr.getch()
            if k == ord('q'):
                break
            if k == ord('p'):
                self.save_plot()

            self.draw()
            rate.sleep()


def main(stdscr):
    MissionTUI(stdscr).run()


if __name__ == "__main__":
    curses.wrapper(main)