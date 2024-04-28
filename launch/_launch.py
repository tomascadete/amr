import launch
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='amr_sim',
            executable='perception',
            name='perception'),
        launch_ros.actions.Node(
            package='amr_sim',
            executable='tracker',
            name='tracker'),
        launch_ros.actions.Node(
            package='amr_sim',
            executable='planner',
            name='planner'),
        launch_ros.actions.Node(
            package='amr_sim',
            executable='control',
            name='control'),
        launch_ros.actions.Node(
            package='amr_sim',
            executable='crossing',
            name='crossing'),
        launch_ros.actions.Node(
            package='amr_sim',
            executable='traffic_light_detection',
            name='traffic_light_detection'),
    ])
