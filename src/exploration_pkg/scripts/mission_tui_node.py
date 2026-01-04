#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mission_tui_node.py

Radiation-aware Mission TUI (display-only, non-intrusive)

Features:
- Mission / navigation state
- Radiation awareness indicator (status box)
- Distance + cumulative dose tracking
- Live dose rate (µSv/s)
- Time-driven dose integration
- Motion-driven distance integration
- Stable double-height trend graphs
- INFO log feed (10 lines)
- CPU + RAM diagnostics box
- Flicker-free ncurses rendering

ROS Noetic + Python3 + curses
"""

import rospy
import curses
import time
import math
import json
import psutil
from collections import deque

from std_msgs.msg import String
from rosgraph_msgs.msg import Log
from gazebo_radiation_plugins.msg import Simulated_Radiation_Msg


# ================= helpers ================= #

def safe_addstr(win, y, x, s, attr=0):
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
    vmax = max(vals)
    if vmax <= 1e-12:
        return (" " * len(vals)).ljust(width)
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

        # -------- mission state --------
        self.mission_state = "UNKNOWN"
        self.nav_state = "IDLE"
        self.radiation_aware = False

        # -------- metrics --------
        self.total_distance = 0.0
        self.total_dose_sv = 0.0
        self.current_dose_rate = 0.0

        self.last_xy = None
        self.last_dose_time = time.time()

        # -------- graphs --------
        self.hist_len = int(rospy.get_param("~history_len", 160))
        self.graph_dt = float(rospy.get_param("~graph_dt", 0.5))
        self.last_graph_sample = time.time()

        self.distance_hist = deque(maxlen=self.hist_len)
        self.dose_hist = deque(maxlen=self.hist_len)

        # -------- logs --------
        self.logs = deque(maxlen=10)

        rospy.init_node("mission_tui", anonymous=False)

        rospy.Subscriber("/mission_state", String, self._on_mission)
        rospy.Subscriber("/exploration_policy/debug_state", String, self._on_debug)
        rospy.Subscriber("/radiation_sensor_plugin/sensor_0",
                         Simulated_Radiation_Msg, self._on_radiation)
        rospy.Subscriber("/rosout", Log, self._on_log)

        self.draw_hz = float(rospy.get_param("~draw_hz", 6.0))
        self.last_draw = 0.0

        self._init_windows()

    # ---------- window layout ---------- #

    def _init_windows(self):
        H, W = self.stdscr.getmaxyx()

        self.header = self.stdscr.subwin(1, W, 0, 0)

        self.status = self.stdscr.subwin(7, 38, 3, 2)
        self.metrics = self.stdscr.subwin(7, 38, 11, 2)

        graph_h = 15
        self.graphs = self.stdscr.subwin(graph_h, W - 44, 3, 42)

        self.logs_win = self.stdscr.subwin(10, W - 4, 3 + graph_h + 1, 2)
        self.sys_win = self.stdscr.subwin(5, W - 4, 3 + graph_h + 12, 2)

    # ---------- callbacks ---------- #

    def _on_mission(self, msg):
        self.mission_state = msg.data.strip()

    def _on_debug(self, msg):
        try:
            d = json.loads(msg.data)
            self.nav_state = "NAVIGATING" if d.get("active_goal") else "IDLE"
            self.radiation_aware = bool(d.get("use_radiation", False))
            xy = d.get("robot_xy")
            if xy:
                self._update_distance(xy[0], xy[1])
        except Exception:
            pass

    def _on_radiation(self, msg):
        self.current_dose_rate = max(0.0, msg.value) * 1e-6

    def _on_log(self, msg):
        if msg.level == Log.INFO and msg.msg:
            self.logs.appendleft(msg.msg.strip())

    # ---------- physics ---------- #

    def _update_distance(self, x, y):
        if self.last_xy:
            self.total_distance += math.hypot(x - self.last_xy[0],
                                               y - self.last_xy[1])
        self.last_xy = (x, y)

    def _update_dose(self):
        now = time.time()
        dt = now - self.last_dose_time
        if dt > 0:
            self.total_dose_sv += self.current_dose_rate * dt
        self.last_dose_time = now

    def _update_graphs(self):
        now = time.time()
        if now - self.last_graph_sample >= self.graph_dt:
            self.distance_hist.append(self.total_distance)
            self.dose_hist.append(self.total_dose_sv)
            self.last_graph_sample = now

    # ---------- drawing ---------- #

    def draw(self):
        self._update_dose()
        self._update_graphs()

        H, W = self.stdscr.getmaxyx()
        if H < 34 or W < 100:
            self.stdscr.erase()
            safe_addstr(self.stdscr, 0, 0, "Terminal too small (min 100x34)")
            self.stdscr.refresh()
            return

        # ----- header -----
        self.header.erase()
        safe_addstr(
            self.header, 0, 0,
            " RADIATION-AWARE EXPLORATION — MISSION TELEMETRY ".center(W),
            curses.color_pair(1) | curses.A_BOLD
        )
        self.header.noutrefresh()

        # ----- status -----
        self.status.erase()
        draw_box(self.status, "Status")
        ra = "ENABLED" if self.radiation_aware else "DISABLED"
        ra_col = curses.color_pair(2 if self.radiation_aware else 4) | curses.A_BOLD
        safe_addstr(self.status, 2, 2, f"Mission   : {self.mission_state}", curses.A_BOLD)
        safe_addstr(self.status, 3, 2, f"Nav       : {self.nav_state}")
        safe_addstr(self.status, 4, 2, f"Radiation : {ra}", ra_col)
        self.status.noutrefresh()

        # ----- metrics -----
        self.metrics.erase()
        draw_box(self.metrics, "Metrics")
        safe_addstr(self.metrics, 2, 2,
                    f"Distance : {self.total_distance:8.2f} m",
                    curses.color_pair(3))
        safe_addstr(self.metrics, 3, 2,
                    f"Dose     : {format_dose(self.total_dose_sv)}",
                    curses.color_pair(2))
        safe_addstr(self.metrics, 5, 2,
                    f"Dose rate: {self.current_dose_rate * 1e6:6.2f} µSv/s",
                    curses.color_pair(2))
        self.metrics.noutrefresh()

        # ----- graphs -----
        self.graphs.erase()
        draw_box(self.graphs, "Trends")
        iw = self.graphs.getmaxyx()[1] - 4
        safe_addstr(self.graphs, 2, 2, "Distance", curses.color_pair(3) | curses.A_BOLD)
        safe_addstr(self.graphs, 3, 2, sparkline(self.distance_hist, iw),
                    curses.color_pair(3))
        safe_addstr(self.graphs, 7, 2, "Dose", curses.color_pair(2) | curses.A_BOLD)
        safe_addstr(self.graphs, 8, 2, sparkline(self.dose_hist, iw),
                    curses.color_pair(2))
        self.graphs.noutrefresh()

        # ----- logs -----
        self.logs_win.erase()
        draw_box(self.logs_win, "Latest INFO Logs")
        for i, l in enumerate(self.logs):
            safe_addstr(self.logs_win, 1 + i, 2, l, curses.A_DIM)
        self.logs_win.noutrefresh()

        # ----- system diagnostics -----
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
            if self.stdscr.getch() == ord('q'):
                break
            now = time.time()
            if now - self.last_draw >= 1.0 / self.draw_hz:
                self.last_draw = now
                self.draw()
            rate.sleep()


def main(stdscr):
    MissionTUI(stdscr).run()


if __name__ == "__main__":
    curses.wrapper(main)