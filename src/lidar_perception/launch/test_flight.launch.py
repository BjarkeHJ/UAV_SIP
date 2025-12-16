
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    test_flight_node = Node(
        package="lidar_perception",
        executable="test_flight",
        name="test_flight_node",
        output="screen",
        parameters=[]
    )
    
    return LaunchDescription([test_flight_node])