from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('surfel_mapping'),
            'config',
            'surfel_mapping.yaml'
        ]),
        description='Path to the configuration file'
    )
    
    pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value='/pointcloud',
        description='Input pointcloud topic'
    )
    
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/odom',
        description='Odometry topic'
    )

    # Surfel mapping node
    surfel_mapping_node = Node(
        package='surfel_mapping',
        executable='surfel_mapping_node',
        name='surfel_mapping_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
        remappings=[
            ('pointcloud', LaunchConfiguration('pointcloud_topic')),
            ('odom', LaunchConfiguration('odom_topic')),
        ]
    )

    return LaunchDescription([
        config_file_arg,
        pointcloud_topic_arg,
        odom_topic_arg,
        surfel_mapping_node,
    ])
