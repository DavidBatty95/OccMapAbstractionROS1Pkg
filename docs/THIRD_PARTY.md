# Attribution and third-party materials

David Batty's project code and new documentation/tooling use the root
[BSD-3-Clause licence](../LICENSE). The repository's existing map/field assets are
retained from the original research workspace; their source filenames, metadata
and contents are preserved. No new third-party worlds or robot meshes are
redistributed as part of this refresh.

The optional sources in [`legacy.repos`](../legacy.repos) remain separate works:

- University of Manchester's **gazebo_radiation_plugin** and
  **simple_radiation_layer**: retain upstream notices and cite the upstream work
  when using those components.
- Unitree Robotics' **unitree_ros**: retain the upstream package and asset notices.

The root licence does not relicense those external repositories. Consult each
upstream revision for its own terms, including any component-specific asset terms.

The container installs ROS, Ubuntu packages, NumPy, SciPy, PyTorch, GPyTorch,
scikit-learn, Pillow and other dependencies from their upstream package sources.
Their licences continue to apply. They are not copied into this Git repository.

The SVGs are editable project schematics. The GIFs and PNGs are rendered from ROS
observations collected by `scripts/run_demo.py`; each published recording has a
manifest with source/map and artifact hashes. Neither the SVGs nor the idealised
2D simulation should be presented as a real robot or Gazebo recording.
