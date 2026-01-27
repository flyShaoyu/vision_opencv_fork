from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

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

        # R1机器人：二维码显示节点
        # Node(
        #     package='vision_opencv',
        #     executable='qr_detect_node',
        #     name='qr_display_r1',
        #     output='screen',
        #     parameters=[{'node_type': 'R1'}],
        #     emulate_tty=True,
        # ),
        
        # R2机器人：摄像头节点
        Node(
            package='vision_opencv',
            executable='camera_node',
            name='camera_r2',
            output='screen',
            parameters=[
                {'camera_index': 11},
                {'fps': 60},
                {'brightness': 10.0},
                {'contrast': 8.0},
                {'exposure': 300.0}
            ],
            emulate_tty=True,
        ),
        
        # R2机器人：二维码识别节点
        Node(
            package='vision_opencv',
            executable='qr_detect_node',
            name='qr_detect_r2',
            output='screen',
            parameters=[{'node_type': 'R2'}],
            emulate_tty=True,
        ),
        spear_vision_launch,
    ])