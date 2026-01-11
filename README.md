# ☢️ OccMapAbstractionROS1Pkg

A **ROS 1 Gazebo simulation workspace** for research into  
**radiation-aware autonomous exploration and navigation**.

This repository contains the full simulation, mapping, and planning stack used to develop and validate **occupancy-map abstraction**, **frontier-based exploration**, and **radiation-informed path planning** for hazardous nuclear environments.

---

## 🧭 Overview

This workspace provides a complete and reproducible experimental environment for studying how mobile robots can explore and navigate when both **geometry and radiation fields are initially unknown**.

It includes:

🧪 **Gazebo nuclear environments**  
Custom reactor-style worlds, radiation sources, and sensor plugins.

🤖 **Autonomous exploration stack**  
Occupancy-map abstraction, frontier detection, navigation coordination, and radiation-aware decision making.

📈 **Radiation field estimation**  
Online radiation prediction (e.g. Gaussian Process Regression) to inform safe path planning.

🗺 **Maps and datasets**  
Real-world and synthetic environments for benchmarking and reproducibility.

This repository represents the **primary simulation platform** used to evaluate the methods described in the associated research and PhD thesis.

---

## 📁 Repository Structure

OccMapAbstractionROS1Pkg/
├── src/ # All ROS packages
├── launch/ # Simulation & experiment launch files
├── worlds/ # Gazebo nuclear environments
├── maps/ # Test and real-world maps
├── models/ # Gazebo models
└── README.md


---

# 🧠 What This Workspace Implements
☢️ **Radiation-Aware Simulation**

Custom Gazebo plugins provide radiation sources and sensors that publish realistic radiation measurements into ROS topics.

🗺 **Occupancy Map Abstraction**

Dense occupancy grids are converted into sparse node graphs suitable for planning in large environments.

🧭 **Frontier-Based Exploration**

The robot selects navigation targets at the boundary between known and unknown space to drive systematic exploration.

📊 **Radiation Field Estimation**

Online models estimate the radiation distribution in the environment, allowing the robot to reason about dose and risk.

🤖 **Navigation Coordination**

High-level logic chooses safe and efficient routes through hazardous environments.

---

# 🎯 Intended Use

This repository is designed for:

🔬 Research and experimentation

📄 Thesis and paper reproducibility

🧪 Simulation-based validation of radiation-aware autonomy

It is not intended to be a plug-and-play ROS navigation stack for production robots.

---

# 🔧 Development Status

This is an active research workspace.
Code, package structure, and launch files will continue to evolve as experiments and publications progress.

Planned improvements include:

Cleaner package modularisation

More documented launch pipelines

Example experiment configurations

---

# 👤 Author

Author: David Batty
Email: dwbatty@liverpool.ac.uk
University of Liverpool — School of Engineering
Research focus: Radiation-aware autonomous exploration in hazardous environments

---

# 📜 License

License will be finalised once the workspace stabilises.
For now, please contact the author regarding reuse or redistribution.
