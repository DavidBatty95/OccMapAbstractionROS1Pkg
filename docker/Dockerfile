FROM ros:noetic-ros-base-focal@sha256:72b8bc59035dc0a5b8e07aae28c16caa84192971d72d207c72ed734fb1d5e97d
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-opencv python3-matplotlib python3-psutil python3-tqdm \
    fonts-dejavu-core ros-noetic-tf ros-noetic-tf2-geometry-msgs \
    ros-noetic-move-base-msgs && rm -rf /var/lib/apt/lists/*
COPY docker/requirements.txt /opt/occmap/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /opt/occmap/requirements.txt
WORKDIR /opt/occmap
COPY src/exploration_pkg /opt/occmap/src/exploration_pkg
RUN source /opt/ros/noetic/setup.bash && catkin_make install -DPYTHON_EXECUTABLE=/usr/bin/python3
COPY scripts /opt/occmap/scripts
COPY tests /opt/occmap/tests
COPY docker/entrypoint.sh /opt/occmap/entrypoint.sh
RUN chmod +x /opt/occmap/entrypoint.sh
EXPOSE 8080
ENTRYPOINT ["/opt/occmap/entrypoint.sh"]
CMD ["python3", "/opt/occmap/scripts/run_demo.py", "--serve", "--output", "/results"]
