import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit


# Launch file for launching the first world training simulation environment
def generate_launch_description():
    ld = LaunchDescription()

    package_dir = get_package_share_directory("multi_robot_active_slam_simulations")
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")

    # World file path
    world = os.path.join(
        package_dir,
        "worlds",
        "main_directions.world",
    )

    # Model file path
    model_folder = "turtlebot3_burger"
    model_path = os.path.join(
        get_package_share_directory("multi_robot_active_slam_simulations"),
        "models",
        model_folder,
        "model.sdf",
    )

    # URDF file path (redundant F)
    urdf_file_name = "turtlebot3_burger.urdf"
    urdf_path = os.path.join(
        get_package_share_directory("multi_robot_active_slam_simulations"),
        "robot_descriptions",
        urdf_file_name,
    )
    # We have to parse urdf files for robot state publisher to read
    with open(urdf_path, "r") as infp:
        robot_desc = infp.read()

    # Handles gazebo_ros
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world}.items(),
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
        )
    )

    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)

    # Spawn robot instances in gazebo
    last_action = None

    poses = [[-2, 7], [0, 7], [2, 7]]

    for i in range(1, 4):
        name = "robot" + str(i)
        namespace = "/robot" + str(i)

        # Create state publisher node
        robot_state_publisher = Node(
            package="robot_state_publisher",
            namespace=namespace,
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"use_sim_time": True, "robot_description": robot_desc}],
        )

        spawner = Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=[
                "-entity",
                name,
                "-file",
                model_path,
                "-robot_namespace",
                namespace,
                "-x",
                str(poses[i - 1][0]),
                "-y",
                str(poses[i - 1][1]),
                "-z",
                "0.01",
            ],
            output="screen",
        )

        if last_action is None:
            # Call add_action directly for the first robot to facilitate chain instantiation via RegisterEventHandler
            ld.add_action(robot_state_publisher)
            ld.add_action(spawner)
        else:
            # Use RegisterEventHandler to ensure next robot creation happens only after the previous one is completed.
            # Simply calling ld.add_action for spawn_entity introduces issues due to parallel run.
            spawn_turtlebot3_event = RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=last_action,
                    on_exit=[spawner, robot_state_publisher],
                )
            )
            ld.add_action(spawn_turtlebot3_event)

        # Save last instance for next RegisterEventHandler
        last_action = spawner

    return ld
