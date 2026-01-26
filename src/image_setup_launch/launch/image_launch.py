from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, FindPackageShare

def generate_launch_description():
    """Launch the image setup and depth camera nodes."""
    
    spear_vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('spear_vision'),
                'launch',
                'small_board_pose.launch.py',
            ])
        )
    )

    return LaunchDescription([
        # Depth camera node
        Node(
            package='depth_camera_pkg',
            executable='depth_camera_node',
            name='depth_camera',
            output='screen',
        ),
        # Image setup node
        Node(
            package='image',
            executable='image_setup_node',
            name='image_setup',
            output='screen',
        ),
        spear_vision_launch,
    ])