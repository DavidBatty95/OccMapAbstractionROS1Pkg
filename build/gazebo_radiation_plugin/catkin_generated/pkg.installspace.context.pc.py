# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include".split(';') if "${prefix}/include" != "" else []
PROJECT_CATKIN_DEPENDS = "gazebo_dev;pluginlib;rosbash;message_runtime".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-lgazebo_radiation_plugins;-lradiation_source;-lradiation_source_plugin;-lradiation_sensor;-lradiation_sensor_plugin".split(';') if "-lgazebo_radiation_plugins;-lradiation_source;-lradiation_source_plugin;-lradiation_sensor;-lradiation_sensor_plugin" != "" else []
PROJECT_NAME = "gazebo_radiation_plugins"
PROJECT_SPACE_DIR = "/home/david/radtest_ws/install"
PROJECT_VERSION = "1.0.1"
