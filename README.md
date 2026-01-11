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

