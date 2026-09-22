cmake_minimum_required(VERSION 3.0.2)
project(exploration_pkg)
find_package(catkin REQUIRED COMPONENTS
  actionlib geometry_msgs move_base_msgs nav_msgs rospy std_msgs tf tf2_ros
  tf2_geometry_msgs visualization_msgs)
catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS
  scripts/exploration_policy_node.py scripts/explore_sim_node.py
  scripts/navigation_corridor_node.py scripts/rad_GPR_node.py
  scripts/mission_tui_node.py scripts/precompute_rad.py scripts/research_demo_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch maps config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
