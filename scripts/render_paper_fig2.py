#!/usr/bin/env python3
"""Reproducible, newly computed Fig. 2 illustrations (not historical replay).

Uses the bundled reactor map and repository LOS/clearance utilities. No ROS
runtime required. Writes figure exports plus full numerical provenance.
"""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
from scipy.ndimage import label as connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/exploration_pkg/python'))
from occmap.grid import clearance, line_cells, load_map, segment_free


def build_graph(m, blocked, field, start, goal, penalty):
    """Clearance-ranked coverage sampling, with a soft radiation penalty."""
    clear = clearance(blocked) * m.resolution
    ys, xs = np.where(~blocked)
    stride = max(1, round(0.25 / m.resolution))
    keep = (xs % stride == 0) & (ys % stride == 0)
    candidates = np.column_stack((xs[keep], ys[keep]))
    score = clear[candidates[:, 1], candidates[:, 0]] - penalty * field[candidates[:, 1], candidates[:, 0]]
    nodes = [m.world_to_grid(*start), m.world_to_grid(*goal)]
    for p in candidates[np.argsort(-score, kind='stable')]:
        if np.min(np.linalg.norm(np.asarray(nodes) - p, axis=1)) * m.resolution >= 0.8:
            nodes.append(tuple(p))
    nodes = np.asarray(nodes)
    world = np.array([m.grid_to_world(*p) for p in nodes])
    edges, adj = [], [[] for _ in nodes]
    for a, b in sorted(cKDTree(world).query_pairs(3.0)):
        if not (segment_free(blocked, nodes[a], nodes[b]) and
                segment_free(blocked, nodes[b], nodes[a])):
            continue
        cells = np.array(list(line_cells(nodes[a], nodes[b])))
        length = float(np.linalg.norm(world[a] - world[b]))
        values = field[cells[:, 1], cells[:, 0]]
        dose = float(np.trapz(values, dx=length / (len(values) - 1)))
        edges.append((a, b, length, dose))
        adj[a].append((b, length, dose))
        adj[b].append((a, length, dose))
    return nodes, world, edges, adj


def route(graph, weight):
    nodes, world, edges, adj = graph
    queue, costs, parents = [(0., 0)], {0: 0.}, {}
    while queue:
        _, a = heapq.heappop(queue)
        if a == 1:
            path = [a]
            while path[-1] != 0:
                path.append(parents[path[-1]])
            return path[::-1]
        for b, length, dose in adj[a]:
            cost = costs[a] + length + weight * dose
            if cost < costs.get(b, float('inf')):
                costs[b], parents[b] = cost, a
                heapq.heappush(queue, (cost + np.linalg.norm(world[b] - world[1]), b))
    raise RuntimeError('Disconnected endpoints')


def metrics(m, blocked, field, graph, path):
    nodes, world, edges, _ = graph
    length = dose = 0.
    peak = 0.
    for a, b in zip(path[:-1], path[1:]):
        assert segment_free(blocked, nodes[a], nodes[b]), 'Route collision'
        cells = np.array(list(line_cells(nodes[a], nodes[b])))
        values = field[cells[:, 1], cells[:, 0]]
        d = float(np.linalg.norm(world[a] - world[b]))
        length += d
        dose += float(np.trapz(values, dx=d / (len(values) - 1)))
        peak = max(peak, float(values.max()))
    return {'length_m': length, 'integrated_relative_intensity_m': dose,
            'peak_relative_intensity': peak, 'nodes': len(nodes), 'edges': len(edges),
            'path_world_xy': world[path].tolist(), 'collision_checked': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/results/fig2-redesign')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    directory = ROOT / 'src/exploration_pkg/maps/reactor_room_sim'
    m = load_map(directory, factor=5)
    blocked = clearance(m.occupancy != 0) * m.resolution <= 0.12 + m.resolution / np.sqrt(2)
    yy, xx = np.indices(m.occupancy.shape)
    wx = m.origin[0] + (xx + 0.5) * m.resolution
    wy = m.origin[1] + (yy + 0.5) * m.resolution
    extent = [m.origin[0], m.origin[0] + m.occupancy.shape[1] * m.resolution,
              m.origin[1], m.origin[1] + m.occupancy.shape[0] * m.resolution]
    start, goal = (7., -7.), (0., 8.)
    right_sources = [(5., 1., 3.), (5., 5., 3.), (7., 3., 3.)]
    scenarios = [('Avoidance', right_sources),
                 ('Mitigation', right_sources + [(-5., 1., 3.), (-5., 5., 3.), (-8., 3., 3.)])]
    hazard_threshold = 0.35
    records, panels = {}, []
    for title, sources in scenarios:
        # Regularised inverse square; shared physical/numerical scale in both panels.
        field = sum(strength / ((wx - sx) ** 2 + (wy - sy) ** 2 + 1.)
                    for sx, sy, strength in sources)
        for p in (start, goal):
            x, y = m.world_to_grid(*p)
            assert not blocked[y, x], f'Invalid endpoint {p}'
        base = build_graph(m, blocked, field, start, goal, penalty=0.)
        aware = build_graph(m, blocked, field, start, goal, penalty=2.)
        bp, ap = route(base, 0.), route(aware, 18.)
        # Eight-connected flood fill is deliberately permissive (even diagonal
        # corner cutting). Disconnection here also excludes conservative routes.
        components, _ = connected_components(~blocked & (field < hazard_threshold),
                                             structure=np.ones((3, 3), dtype=int))
        sx, sy = m.world_to_grid(*start)
        gx, gy = m.world_to_grid(*goal)
        safe_connected = bool(components[sy, sx] != 0 and
                              components[sy, sx] == components[gy, gx])
        records[title] = {'sources_xy_strength': sources,
                          'hazard_threshold': hazard_threshold,
                          'below_threshold_start_goal_connected': safe_connected,
                          'endpoint_intensities': [float(field[sy, sx]), float(field[gy, gx])],
                          'distance_only': metrics(m, blocked, field, base, bp),
                          'radiation_aware': metrics(m, blocked, field, aware, ap)}
        record = records[title]
        assert record['radiation_aware']['integrated_relative_intensity_m'] < record['distance_only']['integrated_relative_intensity_m']
        if title == 'Avoidance':
            assert safe_connected and record['radiation_aware']['peak_relative_intensity'] < hazard_threshold
        else:
            assert not safe_connected and max(record['endpoint_intensities']) < hazard_threshold
        panels.append((title, sources, field, base, aware, bp, ap))
        np.savez_compressed(args.output / (title.lower() + '_data.npz'),
                            occupancy=m.occupancy, field=field, blocked=blocked,
                            baseline_nodes=base[1], aware_nodes=aware[1],
                            baseline_edges=np.asarray(base[2]), aware_edges=np.asarray(aware[2]),
                            baseline_path=bp, aware_path=ap)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    cmap = LinearSegmentedColormap.from_list('radiation', ['#ffffff', '#fff0cb', '#fbd384', '#ee994b', '#cf542e', '#8c2323'])
    norm = Normalize(0, 4)
    blue, ink = '#0072B2', '#323b43'
    valid = np.argwhere(m.occupancy == 0)
    lo, hi = valid.min(axis=0), valid.max(axis=0)
    limits = [m.origin[0] + lo[1]*m.resolution - .4, m.origin[0] + hi[1]*m.resolution + .4,
              m.origin[1] + lo[0]*m.resolution - .4, m.origin[1] + hi[0]*m.resolution + .4]
    for show_graph, column in ((False, False), (True, False), (False, True)):
        fig = plt.figure(figsize=(3.5, 2.8) if column else (7.16, 4.65), facecolor='white')
        axes = [fig.add_axes([.025 + i*.49, .205, .455, .69]) for i in range(2)]
        for i, (ax, panel) in enumerate(zip(axes, panels)):
            title, sources, field, base, aware, bp, ap = panel
            bg = np.ones((*m.occupancy.shape, 3))
            bg[m.occupancy == -1] = [0.92, .93, .94]
            bg[m.occupancy == 100] = [.24, .27, .30]
            ax.imshow(bg, origin='lower', extent=extent, interpolation='nearest')
            heat = np.ma.masked_where(m.occupancy != 0, field)
            ax.imshow(heat, origin='lower', extent=extent, cmap=cmap, norm=norm, interpolation='nearest')
            ax.contour(wx, wy, heat, levels=[hazard_threshold], colors=['#a66a35'],
                       linewidths=.55 if column else .75, linestyles=':')
            if show_graph:
                segments = [aware[1][[a, b]] for a, b, _, _ in aware[2]]
                ax.add_collection(LineCollection(segments, colors='#687d89', linewidths=.18, alpha=.11))
                ax.scatter(*aware[1].T, s=2, color='#687d89', alpha=.3, linewidths=0)
            for graph, path, colour, style, width in [(base, bp, ink, (0, (3, 2)), .8 if column else 1.4), (aware, ap, blue, '-', 1.2 if column else 2.)]:
                points = graph[1][path]
                ax.plot(*points.T, color=colour, linestyle=style, lw=width,
                        solid_capstyle='round', dash_capstyle='round',
                        path_effects=[pe.Stroke(linewidth=width+(.7 if column else 1.4), foreground='white'), pe.Normal()])
            ax.scatter(*np.asarray(sources)[:, :2].T, marker='X', s=14 if column else 30, c='#8c2323', edgecolors='white', linewidths=.5, zorder=8)
            for xy, marker, label in [(aware[1][0], 'o', 'S'), (aware[1][1], '*', 'G')]:
                ax.scatter(*xy, s=(24 if marker == '*' else 12) if column else (48 if marker == '*' else 24), marker=marker, c=ink, edgecolors='white', linewidths=.6, zorder=9)
                ax.annotate(label, xy, xytext=(4, 2) if column else (6, 2), textcoords='offset points', fontsize=6 if column else 8, weight='bold',
                            path_effects=[pe.withStroke(linewidth=2, foreground='white')], zorder=10)
            x, y = limits[0]+3.5, limits[2]+.8
            ax.plot([x, x+2], [y, y], color=ink, lw=1.5)
            ax.text(x+1, y+.25, '2 m', ha='center', fontsize=5.5 if column else 7)
            ax.set(xlim=limits[:2], ylim=limits[2:], aspect='equal')
            ax.axis('off')
            ax.set_title(f'({chr(97+i)})  {title}', loc='left', pad=7, fontsize=7 if column else 10, weight='semibold')
        handles = [Line2D([], [], color=ink, ls='--', lw=1.4, label='Distance-only'),
                   Line2D([], [], color=blue, lw=2, label='Radiation-aware'),
                   Line2D([], [], marker='X', color='#8c2323', ls='', label='Sources', markersize=5),
                   Line2D([], [], color='#a66a35', ls=':', lw=.75, label=r'$I = 0.35$ boundary')]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .115), ncol=2 if column else 4, frameon=False, handlelength=1.8 if column else 2.5, columnspacing=1. if column else 2., fontsize=5.8 if column else 8)
        cax = fig.add_axes([.29, .075, .42, .019])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', ticks=[0, 1, 2, 3, 4])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5.5 if column else 7, length=2, pad=2)
        cb.set_label('Radiation intensity (relative units)', fontsize=6 if column else 8, labelpad=3)
        stem = 'fig2_column' if column else ('fig2_graph' if show_graph else 'fig2_clean')
        for fmt in ('pdf', 'svg', 'png'):
            fig.savefig(args.output / f'{stem}.{fmt}', dpi=600, facecolor='white', bbox_inches='tight', pad_inches=.04)
        plt.close(fig)

    manifest = {'description': 'New illustrative runs, not a replay of the original Figure 2 or experimental evidence for its reported aggregate results.',
                'repository': 'https://github.com/DavidBatty95/OccMapAbstractionROS1Pkg',
                'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'map': str(directory.relative_to(ROOT)), 'map_sha256': hashlib.sha256((directory/'reactor_room_sim.pgm').read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'resolution_m': m.resolution, 'origin': m.origin,
                'field_model': 'I(x,y)=sum_i K_i/(distance_to_source_i_squared+1); no attenuation; synthetic relative units',
                'hazard_definition': 'I >= 0.35 relative units; illustrative threshold, not a physical safety limit. Eight-connected flood fill checks start-goal connectivity after excluding hazard cells.',
                'graph': {'requested_candidate_stride_m': .25, 'effective_candidate_stride_m': max(1, round(.25/m.resolution))*m.resolution, 'minimum_separation_m': .8, 'connection_radius_m': 3., 'aware_score': 'clearance_m - 2 * intensity', 'path_objective': 'length_m + 18 * integral(I ds)'},
                'baseline': 'Distance-only search on a geometry-ranked sparse graph (not the paper full-grid baseline)',
                'scenarios': records}
    (args.output/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for title, record in records.items():
        base = record['distance_only']['integrated_relative_intensity_m']
        aware = record['radiation_aware']['integrated_relative_intensity_m']
        print(f'{title}: collision checks passed; illustrative integrated exposure {base:.3f} -> {aware:.3f}')
    print(f'Exports: {args.output}')


if __name__ == '__main__':
    main()
