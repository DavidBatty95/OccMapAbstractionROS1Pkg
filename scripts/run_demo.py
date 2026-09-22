#!/usr/bin/env python3
"""Run, record and serve a bounded ROS experiment without a host ROS install."""
import argparse
import csv
from datetime import datetime, timezone
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import threading
import time
import xmlrpc.client

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


class Recording:
    def __init__(self, viewport):
        import rospy
        from nav_msgs.msg import OccupancyGrid, Path as RosPath
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray
        self.lock = threading.RLock()
        self.viewport = viewport
        self.map = self.rad = None
        self.markers = []
        self.metrics = {}
        self.state = {}
        self.mission = 'STARTING'
        self.trail = []
        self.messages = {'map': 0, 'radiation': 0, 'graph': 0, 'metrics': 0}
        rospy.Subscriber('/map', OccupancyGrid, self.on_map, queue_size=1)
        rospy.Subscriber('/radiation_costmap', OccupancyGrid, self.on_rad, queue_size=1)
        rospy.Subscriber('/exploration_markers', MarkerArray, self.on_markers, queue_size=1)
        rospy.Subscriber('/demo/metrics', String, self.on_metrics, queue_size=1)
        rospy.Subscriber('/exploration_policy/debug_state', String, self.on_state, queue_size=1)
        rospy.Subscriber('/mission_state', String, self.on_mission, queue_size=1)

    def on_map(self, msg):
        with self.lock:
            self.map = msg
            self.messages['map'] += 1

    def on_rad(self, msg):
        with self.lock:
            self.rad = msg
            self.messages['radiation'] += 1

    def on_markers(self, msg):
        with self.lock:
            self.markers = msg.markers
            if any(m.ns == 'graph' and m.points for m in msg.markers):
                self.messages['graph'] += 1

    def on_metrics(self, msg):
        with self.lock:
            self.metrics = json.loads(msg.data)
            self.trail.append((self.metrics['x'], self.metrics['y']))
            self.messages['metrics'] += 1

    def on_state(self, msg):
        with self.lock:
            self.state = json.loads(msg.data)

    def on_mission(self, msg):
        self.mission = msg.data

    def render(self, title):
        with self.lock:
            canvas = Image.new('RGB', (1100, 670), '#0c1424')
            draw = ImageDraw.Draw(canvas)
            draw.text((28, 20), 'OCCMAP / RADIATION-AWARE EXPLORATION', font=font(25), fill='#e7edf6')
            draw.text((28, 60), title + '  |  2D ROS reference simulation', font=font(16), fill='#94a9c7')
            draw.text((28, 103), 'OBSERVED MAP + ABSTRACTION GRAPH', font=font(16), fill='#52ded0')
            draw.text((571, 103), 'ONLINE RADIATION ESTIMATE + TRAJECTORY', font=font(16), fill='#ffa95e')
            if self.map is None:
                draw.text((30, 200), 'Starting ROS nodes and loading the map...', fill='white', font=font(20))
                return canvas
            msg = self.map
            h, w = msg.info.height, msg.info.width
            occupancy = np.array(msg.data).reshape(h, w)
            # Fixed visual-only viewport removes empty map padding. Ground-truth
            # bounds affect camera framing only and are never sent to the policy.
            r0, r1, c0, c1 = self.viewport
            vh, vw = r1-r0, c1-c0
            scale = min(510 / vw, 420 / vh)
            size = (max(1, round(vw * scale)), max(1, round(vh * scale)))
            inset = (510-size[0]) // 2
            top = 139 + (420-size[1]) // 2
            rgb = np.empty((h, w, 3), dtype=np.uint8)
            rgb[:] = (25, 38, 57)
            rgb[occupancy == 0] = (214, 224, 231)
            rgb[occupancy == 100] = (78, 99, 122)
            radiation = np.zeros((h, w))
            if self.rad is not None and len(self.rad.data) == h * w:
                radiation = np.maximum(0, np.array(self.rad.data).reshape(h, w)) / 100
            heat = rgb.copy()
            mask = (occupancy == 0) & (radiation > 0)
            hot = np.stack([255 * np.ones((h, w)), 205 - 160 * radiation, 90 - 60 * radiation], axis=-1)
            heat[mask] = hot[mask].astype(np.uint8)
            for offset, data in ((28, rgb), (571, heat)):
                panel = Image.fromarray(np.flipud(data[r0:r1, c0:c1])).resize(size, Image.Resampling.NEAREST)
                canvas.paste(panel, (offset + inset, top))

            def point(x, y, offset):
                gx = (x - msg.info.origin.position.x) / msg.info.resolution
                gy = (y - msg.info.origin.position.y) / msg.info.resolution
                return offset + inset + (gx-c0) * size[0] / vw, top + (r1-gy) * size[1] / vh

            for marker in self.markers:
                if marker.ns == 'graph':
                    for a, b in zip(marker.points[::2], marker.points[1::2]):
                        draw.line([point(a.x, a.y, 28), point(b.x, b.y, 28)], fill='#27949c', width=1)
                elif marker.ns == 'frontier_candidates':
                    for p in marker.points:
                        x, y = point(p.x, p.y, 28)
                        draw.ellipse((x-2, y-2, x+2, y+2), fill='#ab48d9')
                elif marker.ns == 'corridor' and len(marker.points) > 1:
                    draw.line([point(p.x, p.y, 28) for p in marker.points], fill='#ee8b32', width=3)
            if len(self.trail) > 1:
                for offset in (28, 571):
                    draw.line([point(x, y, offset) for x, y in self.trail], fill='#007baf', width=3)
                    x, y = point(*self.trail[-1], offset)
                    draw.ellipse((x-5, y-5, x+5, y+5), fill='#4bf0ce', outline='#08203a', width=2)
            m = self.metrics
            text = ('t = %.1f s     observed free = %.1f%%     path = %.2f m' %
                    (m.get('simulation_time_s', 0), 100 * m.get('observed_free_fraction', 0), m.get('distance_m', 0)))
            draw.text((28, 580), text, fill='#e7edf6', font=font(20))
            draw.text((28, 615), 'Dose integral: %.1f relative-rate seconds  |  %s' %
                      (m.get('dose_relative_rate_seconds', 0), self.mission), fill='#94a9c7', font=font(17))
            return canvas


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>OccMap research demo</title><style>body{background:#0c1424;color:#dce7f5;font:17px system-ui;max-width:1100px;margin:30px auto;padding:16px}a{color:#52ded0}img{width:100%;border-radius:12px}pre{white-space:pre-wrap}nav{margin:20px 0}</style>
<h1>OccMap research demo</h1><p>Live ROS observations, abstraction graph and online radiation estimate.</p>
<img id="preview" src="live.png" alt="Live map and radiation estimate"><nav><a href="summary.json">Run summary</a> · <a href="manifest.json">Provenance</a> · <a href="metrics.csv">Metrics CSV</a> · <a href="demo.gif">Recorded animation</a> · <a href="ros.log">ROS log</a></nav>
<p>2D holonomic reference simulation. Radiation is in relative units. This run is not a Gazebo or hardware validation.</p>
<pre id="status">Starting…</pre><script>setInterval(async()=>{document.getElementById('preview').src='live.png?t='+Date.now();try{let r=await fetch('summary.json?t='+Date.now());if(r.ok)document.getElementById('status').textContent=JSON.stringify(await r.json(),null,2)}catch(e){}},1000)</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', default='reactor_room_sim', choices=['reactor_room_sim', 'barrel_store_sim', 'base_env_sim', 'hut_interior', 'hut_expanded'])
    parser.add_argument('--duration', type=float, default=120, help='Wall-clock recording seconds after ROS startup')
    parser.add_argument('--factor', type=int, default=5, help='Conservative map downsampling factor')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-radiation', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--serve', action='store_true', help='Serve live/results page on port 8080; stay running afterward')
    parser.add_argument('--check', action='store_true', help='Require movement, graph, map and (if enabled) radiation estimates')
    args = parser.parse_args()
    if not 5 <= args.duration <= 3600 or args.factor < 1:
        parser.error('duration must be 5..3600 seconds and factor must be positive')
    mode = 'unaware' if args.no_radiation else 'aware'
    name = '%s-%s-s%d-%s' % (args.map, mode, args.seed, datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output = args.output.resolve() / name
    output.mkdir(parents=True)
    (output / 'index.html').write_text(HTML)
    hashes = {}
    for directory in ('src/exploration_pkg', 'scripts', 'docker'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'schema_version': 1, 'run': name, 'map': args.map, 'factor': args.factor,
                'radiation_aware': not args.no_radiation, 'seed': args.seed, 'duration_wall_s': args.duration,
                'runtime': platform.platform(), 'python': platform.python_version(),
                'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'torch', 'gpytorch', 'Pillow')},
                'files_sha256': hashes, 'model': '2D holonomic reference; unknown map; point radiation sensing',
                'dose_units': 'relative_dose_rate * simulated seconds',
                'determinism': 'Seeds fixed; ROS scheduling is asynchronous, so whole runs are not bitwise deterministic.'}
    atomic_json(output / 'manifest.json', manifest)
    server = None
    if args.serve:
        server = ThreadingHTTPServer(('0.0.0.0', 8080), partial(SimpleHTTPRequestHandler, directory=str(output)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print('Live demo: http://localhost:8080', flush=True)
    print('Artifacts:', output, flush=True)
    processes, frames, records = [], [], []
    log = (output / 'ros.log').open('w')
    summary = {'status': 'starting', 'run': name}
    atomic_json(output / 'summary.json', summary)
    record = None
    interrupted = threading.Event()
    def stop(_signum, _frame):
        interrupted.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    error = None
    try:
        processes.append(subprocess.Popen(['roscore'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + 30
        while True:
            try:
                xmlrpc.client.ServerProxy('http://localhost:11311').getUri('/demo_recorder')
                break
            except OSError:
                if time.monotonic() > deadline or processes[0].poll() is not None:
                    raise RuntimeError('ROS master did not start; see ros.log')
                time.sleep(0.2)
        import rospy
        rospy.init_node('demo_recorder', disable_signals=True)
        from occmap.grid import load_map
        visual_map = load_map(ROOT / 'src/exploration_pkg/maps' / args.map, args.factor)
        known = np.argwhere(visual_map.occupancy >= 0)
        lo, hi = known.min(axis=0), known.max(axis=0) + 1
        height, width = visual_map.occupancy.shape
        viewport = (max(0, lo[0]-5), min(height, hi[0]+5), max(0, lo[1]-5), min(width, hi[1]+5))
        record = Recording(viewport)
        processes.append(subprocess.Popen(['roslaunch', 'exploration_pkg', 'demo.launch',
            'map:=' + args.map, 'factor:=' + str(args.factor), 'seed:=' + str(args.seed),
            'use_radiation:=' + str(not args.no_radiation).lower()], stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        started = time.monotonic()
        # At most 180 frames even for hour-long runs.
        next_frame = started
        while time.monotonic() - started < args.duration and not interrupted.is_set():
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('A ROS process exited unexpectedly; see ros.log')
            image = record.render(args.map + ' / ' + mode)
            image.save(output / 'live.tmp', format='PNG')
            (output / 'live.tmp').replace(output / 'live.png')
            if time.monotonic() >= next_frame:
                frames.append(image)
                next_frame = time.monotonic() + max(1.0, args.duration / 180)
            with record.lock:
                if record.metrics:
                    records.append(dict(record.metrics))
                summary = {'status': 'running', 'run': name, 'mission_state': record.mission,
                           'metrics': dict(record.metrics), 'messages': dict(record.messages), 'policy': dict(record.state)}
            atomic_json(output / 'summary.json', summary)
            time.sleep(0.5)
    except Exception as exc:
        error = str(exc)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        log.close()
    if records:
        with (output / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if frames:
        frames[0].save(output / 'demo.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
    metrics = summary.get('metrics', {})
    messages = summary.get('messages', {})
    smoke = (metrics.get('distance_m', 0) > 1 and messages.get('graph', 0) > 0 and
             messages.get('map', 0) > 0 and (args.no_radiation or (
                 record is not None and record.rad is not None and max(record.rad.data) > 0)))
    log_text = (output / 'ros.log').read_text(errors='replace')
    callback_error = 'Traceback (most recent call last)' in log_text or 'Error in Timer callback' in log_text
    if callback_error:
        error = error or 'Python exception in ROS log'
    summary.update(status='failed' if error else ('interrupted' if interrupted.is_set() else 'recorded'),
                   error=error, smoke_pass=bool(smoke and not error),
                   acceptance_scope='Pipeline smoke test, not full exploration or dose-reduction validation',
                   animation={'frames': len(frames), 'playback_frame_ms': 100})
    atomic_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if server and not error and not interrupted.is_set():
        print('Recording complete. Results remain at http://localhost:8080; Ctrl+C to stop.', flush=True)
        while not interrupted.wait(0.5):
            pass
    if server:
        server.shutdown()
    return 1 if error or (args.check and not smoke) else 0


if __name__ == '__main__':
    raise SystemExit(main())
