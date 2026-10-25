# Figure 2 — publication redesign

These are **new illustrative planning runs**, using this repository's
`reactor_room_sim` occupancy map and `occmap.grid` map loading, clearance and
collision-checking utilities. They are not a reconstruction of the old screenshot
trajectories or part of the paper's reported evaluation dataset.

## Files

* **fig2_column.pdf** — compact two-panel version for a single IEEE column.
* **fig2_clean.pdf** — larger route-focused version; recommended for a wider figure.
* **fig2_graph.pdf** — same examples with a subdued sparse-graph overlay.
* Each design is also exported as an editable SVG and 600 dpi PNG.
* `avoidance_data.npz`, `mitigation_data.npz` retain occupancy, fields, graphs and routes.
* `provenance.json` records source hashes, numerical settings, routes and metrics.

The PDF/SVG preserve vector routes, graph edges, labels and legends; the occupancy
grid and scalar field remain raster layers. SVG text is editable. Colours and line
styles distinguish routes without depending solely on colour vision.

## Interpretation and proposed caption

**Illustrative radiation-aware planning on the repository reactor-room map.**
The dashed charcoal route minimises distance on a geometry-ranked sparse graph;
the solid blue route incorporates radiation into node ranking and path cost.
(a) Avoidance: three sources are clustered on the right, allowing the
radiation-aware route to bypass the high-intensity region on the left.
(b) Mitigation: three additional sources on the left create an unavoidable
high-intensity crossing; the planner chooses a lower-exposure route.
Dotted contours denote the illustrative threshold I = 0.35 relative units.
S and G denote the shared start and goal. Both panels use the same
relative-intensity scale. Fields and routes are newly generated illustrative
examples, rather than the original experimental trials.

Important for manuscript integration: the dashed baseline is **sparse-graph
distance-only**, not the full-grid baseline used in the paper's quantitative
evaluation. There is no Move Base run here. The new graph builder is a standalone
clearance-ranked illustration of the paper's scoring rule, not the ROS online
exploration policy. Avoidance means bypassing the hotspot, not zero exposure:
the regularised inverse-square field has nonzero tails everywhere.

The distinction is checked numerically on the 0.10 m planning grid. For avoidance,
every raster-sampled point on the blue route has intensity below 0.35. For
mitigation, both endpoints are below 0.35, but removing all cells with intensity
at or above 0.35 disconnects them even with permissive eight-connected flood
fill. Thus an above-threshold crossing is unavoidable in this discretised map,
not just an artefact of sparse graph sampling. The threshold defines the
illustration and is not a physical radiation safety limit.

The computed line-integral exposure proxies (relative intensity times metres)
are 19.260 versus 2.278 for avoidance and 20.577 versus 10.654 for mitigation
(distance-only versus radiation-aware). These selected examples illustrate the
behaviours and do not establish general performance guarantees.

## Reproduce

From the repository root, using Python with NumPy, SciPy, Matplotlib, Pillow and
PyYAML installed:

```bash
python3 scripts/render_paper_fig2.py
```

The script uses a 0.10 m conservatively pooled map, 0.12 m robot radius plus
half-cell diagonal margin, 0.8 m node separation and 3 m LOS connection radius.
Both directions of every undirected LOS edge are checked. Every plotted route
segment is checked again before export. The synthetic field is
`I = sum_i 3 / (r_i² + 1)` in relative units, with no wall attenuation. Panel (a)
has sources at (5, 1), (5, 5), (7, 3) m; panel (b) adds sources at (-5, 1),
(-5, 5), (-8, 3) m. The start and goal stay fixed. Source coordinates,
graph settings and numerical line-integral metrics are recorded in the manifest.

The examples are deliberately chosen to illustrate the two behaviours; their
exposure reductions must not replace the aggregate statistics in Figure 1.

## LaTeX

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{fig2_column.pdf}
  \caption{Illustrative radiation-aware planning on the reactor-room map.
  Dashed charcoal: distance-only sparse-graph route; solid blue:
  radiation-aware route. (a) Three sources on the right allow the
  high-intensity region to be bypassed on the left. (b) Additional sources
  on the left make an above-threshold crossing unavoidable; the planner
  selects a lower-exposure route. Dotted contours denote the illustrative
  threshold $I=0.35$ in relative units. S and G denote start and goal.
  Both panels share the relative-intensity scale. These are newly
  generated illustrative examples.}
  \label{fig:radiation-behaviours}
\end{figure}
```
