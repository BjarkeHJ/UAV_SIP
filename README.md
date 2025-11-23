# UAV_SIP
UAV SIP (Skeleton-based Inspection Planning) is a novel perception approach to navigate unmanned aerial vehicles in unknown environments for visual inspection purposes

## Prerequisites
- Ubuntu 22.05
- ROS2 Humble

## Installation
```
# Clone including submodules
git clone --recursive <your-workspace-repo-url> px4_sim_ws
cd px4_sim_ws

# If they forget --recursive:
# git submodule update --init --recursive

# Build
# source /opt/ros/humble/setup.bash
colcon build

# Source workspace
source install/setup.bash
```