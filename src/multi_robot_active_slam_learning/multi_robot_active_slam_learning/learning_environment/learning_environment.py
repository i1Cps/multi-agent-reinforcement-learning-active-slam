import rclpy
from rclpy.node import Node

import numpy as np
import scipy
import time
import math

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, Pose
from multi_robot_active_slam_interfaces.srv import StepEnv, ResetEnv, ResetGazeboEnv

from multi_robot_active_slam_learning.common.settings import (
    NUMBER_OF_ROBOTS,
    NUMBER_OF_SCANS,
)
from multi_robot_active_slam_learning.learning_environment.reward_function import (
    reward_function,
)


# This Node is reponsible for providing an interface for agents to take actions and recieve new states, rewards or both
# Contains direct communication with our physics simulator, gazebo.
class LearningEnvironment(Node):
    def __init__(self):
        super().__init__("learning_environment")

        self.num_robots = NUMBER_OF_ROBOTS

        # -------- Initialise subscribers, publishers, clients and services ------- #

        self.scan_subscribers = []
        self.covariance_matrix_subscribers = []
        self.odom_subscribers = []
        self.cmd_vel_publishers = []

        # Initialize all resources for each robot
        for i in range(1, self.num_robots + 1):
            namespace = f"/robot{i}"  # Subscribers

            self.scan_subscribers.append(
                self.create_subscription(
                    LaserScan,
                    f"{namespace}/scan",
                    lambda msg, robot_id=i - 1: self.scan_callback(msg, robot_id),
                    1,
                )
            )

            self.covariance_matrix_subscribers.append(
                self.create_subscription(
                    PoseWithCovarianceStamped,
                    f"{namespace}/pose",
                    lambda msg, robot_id=i - 1: self.covariance_matrix_callback(
                        msg, robot_id
                    ),
                    10,
                )
            )
            self.odom_subscribers.append(
                self.create_subscription(
                    Odometry,
                    f"{namespace}/odom",
                    lambda msg, robot_id=i - 1: self.odom_callback(msg, robot_id),
                    10,
                )
            )

            self.cmd_vel_publishers.append(
                self.create_publisher(Twist, f"{namespace}/cmd_vel", 10)
            )

        self.goal_position_reset_pose_subscriber = self.create_subscription(
            Pose,
            "/goal_position_reset_pose",
            self.goal_position_reset_pose_callback,
            10,
        )

        # ------------------ Clients ---------------------------- #
        self.initialise_gazebo_environment_client = self.create_client(
            Empty, "/initialise_gazebo_environment"
        )

        self.environment_success_client = self.create_client(
            Empty, "/environment_success"
        )
        self.environment_reset_client = self.create_client(
            ResetGazeboEnv, "/environment_reset"
        )

        # -------------------------------- Services --------------------------------- #
        self.environment_step = self.create_service(
            StepEnv, "/environment_step", self.environment_step_callback
        )
        self.reset_environment = self.create_service(
            ResetEnv, "/reset_environment_rl", self.reset_callback
        )

        # ------------------------------- ros style multi thread (lol) ---------------- #
        # Sub Node to await request within a call back. At this point im a principle ROS engineer
        self.sub_node = rclpy.create_node("sub_node")
        self.sub_client = self.sub_node.create_client(
            ResetGazeboEnv, "/environment_reset"
        )

        #################################################################
        #               CONSTANTS AND VARIABLES                         #
        #################################################################

        # Robot CONSTANTS
        self.MAX_LINEAR_SPEED = 0.22
        self.MAX_ANGULAR_SPEED = 2.0
        self.NUMBER_OF_SCANS = NUMBER_OF_SCANS
        self.MAX_SCAN_DISTANCE = 3.5

        # Robot Variables
        self.collided = np.full(self.num_robots, False, dtype=bool)
        self.found_goal = np.full(self.num_robots, False, dtype=bool)

        self.actual_poses = np.full((self.num_robots, 2), None, dtype=object)
        self.estimated_poses = np.full((self.num_robots, 2), None, dtype=object)

        self.scans = (
            np.ones((self.num_robots, self.NUMBER_OF_SCANS), dtype=np.float32) * 2
        )

        self.d_opts = np.full(self.num_robots, 0.01)
        self.linear_velocities = np.zeros(self.num_robots)
        self.angular_velocities = np.zeros(self.num_robots)

        # Environment CONSTANTS
        self.MAX_STEPS = 1000
        self.GOAL_DISTANCE = 0.7
        self.COLLISION_DISTANCE = 0.18

        # Environment Variables
        self.done = False
        self.truncated = False
        self.step_counter = 0
        self.goal_position = np.array([0.0, 0.0])
        self.distance_to_goal = np.full(self.num_robots, np.inf)

        # DEBUG
        self.reward_debug = False

        ################################################################
        #        Initialise Node
        ####################################################

        req = Empty.Request()
        while not self.initialise_gazebo_environment_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info("Waiting for initialise gazebo environment service")
        self.initialise_gazebo_environment_client.call_async(req)
        self.get_logger().info("Sucessfully initialised Learning Environment Node")

    # Callback function for LiDAR scan subscriber
    def scan_callback(self, msg, robot_id):
        scan = np.empty(len(msg.ranges), dtype=np.float32)
        # Resize scan data, data itself returns extra info about the scan, scan.ranges just gets.... the ranges
        for i in range(len(msg.ranges)):
            if msg.ranges[i] == float("Inf"):
                scan[i] = self.MAX_SCAN_DISTANCE
            elif np.isnan(msg.ranges[i]):
                scan[i] = 0
            else:
                scan[i] = msg.ranges[i]
        self.scans[robot_id] = scan

    def odom_callback(self, msg, robot_id):
        # Update robot-specific data
        self.actual_poses[robot_id][0] = msg.pose.pose.position.x
        self.actual_poses[robot_id][1] = msg.pose.pose.position.y
        self.distance_to_goal[robot_id] = np.sqrt(
            (self.goal_position[0] - self.actual_poses[robot_id][0]) ** 2
            + (self.goal_position[1] - self.actual_poses[robot_id][1]) ** 2
        )
        # print(f"Actual Robot: {self.actual_pose[0]} , {self.actual_pose[1]}")
        # print(f" goals pose: {self.goal_position[0]} , {self.goal_position[1]}")
        # print(f"Distance to goal: {self.distance_to_goal}")

    # Calculates yaw angle from quaternions, they deprecated Pose2D for some reason???? so this function is useless
    def calculate_yaw(self, q_ang):
        return math.atan2(
            2.0 * (q_ang.w * q_ang.z + q_ang.x * q_ang.y),
            1.0 - 2.0 * (q_ang.y**2 + q_ang.z**2),
        )

    # Callback function for covariance matrix subscriber
    def covariance_matrix_callback(self, msg, robot_id):
        # Get D-Optimality
        EIG_TH = 1e-6  # or any threshold value you choose
        data = msg.pose
        matrix = data.covariance
        covariance_matrix = np.array(matrix).reshape((6, 6))
        eigenvalues = scipy.linalg.eigvalsh(covariance_matrix)
        if np.iscomplex(eigenvalues.any()):
            print("Error: Complex Root")
        eigv = eigenvalues[eigenvalues > EIG_TH]
        n = np.size(covariance_matrix, 1)
        d_optimality = np.exp(np.sum(np.log(eigv)) / n)
        self.d_opts[robot_id] = d_optimality

        # Slam toolbox gets excited at the start of episodes sometimes
        # if self.step_counter < 20 and self.d_opts[robot_id] == 1:
        if self.step_counter < 20:
            self.d_opts[robot_id] = 0.01

        # Get Pose
        pose = data.pose
        self.estimated_poses[robot_id] = np.array([pose.position.x, pose.position.y])

    def goal_position_reset_pose_callback(self, data):
        self.goal_position[0] = data.position.x
        self.goal_position[1] = data.position.y
        print(f"new goal pose: [ {self.goal_position[0]} , {self.goal_position[1]} ]")

    def reset_callback(self, request, response):
        # Reset robot variables to prevent reset loop
        self.scans = (
            np.ones((self.num_robots, self.NUMBER_OF_SCANS), dtype=np.float32) * 2
        )
        self.actual_poses = np.full((self.num_robots, 2), None, dtype=object)
        self.estimated_poses = np.full((self.num_robots, 2), None, dtype=object)
        self.d_opts = np.full(self.num_robots, 0.01)
        self.linear_velocities = np.zeros(self.num_robots)
        self.angular_velocities = np.zeros(self.num_robots)
        self.step_counter = 0

        # Reset simulation
        req = ResetGazeboEnv.Request()
        while not self.environment_reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Environment reset service is not available, I'll wait"
            )

        future = self.sub_client.call_async(req)
        rclpy.spin_until_future_complete(self.sub_node, future)
        future_result: ResetGazeboEnv.Response = future.result()
        poses = future_result.poses

        # Process received poses
        for i in range(self.num_robots):
            self.estimated_poses[i, :] = [poses[i].position.x, poses[i].position.y]
            self.actual_poses[i, :] = [
                poses[i].position.x,
                poses[i].position.y,
            ]

        self.done = False
        self.truncated = False
        self.collided = np.full(self.num_robots, False, dtype=bool)

        for i in range(self.num_robots):
            scan_part = self.scans[i].astype(np.float32)
            pose_part = self.estimated_poses[i].astype(np.float32)
            d_opt_part = np.array([self.d_opts[i] * 10], dtype=np.float32)
            state_vector = np.concatenate((scan_part, pose_part, d_opt_part))
            response.multi_robot_states[i].state = state_vector

        time.sleep(2)
        # TODO: Use ros approved variation of time.sleep()
        # Pause execution

        return response

    # Calculates the rewards based on collisions, angular velocity and map certainty
    def get_rewards(
        self,
        found_goal: bool,
        collided: bool,
        angular_vel: float,
        linear_vel: float,
        d_opt: float | None,
    ) -> float:
        return reward_function(
            found_goal,
            collided,
            angular_vel,
            linear_vel,
            d_opt,
            self.MAX_LINEAR_SPEED,
            self.reward_debug,
        )

    def stop_robots(self):
        # Reset robots velocity
        for i in range(self.num_robots):
            self.cmd_vel_publishers[i].publish(Twist())

    # Check if the given scan shows a collision
    def has_collided(self, robot_idx):
        return bool(self.COLLISION_DISTANCE > np.min(self.scans[robot_idx]))

    def has_found_goal(self, robot_idx):
        return self.distance_to_goal[robot_idx] < self.GOAL_DISTANCE

    def handle_found_goal(self, found_goal):
        if not any(found_goal):
            return
        self.step_counter = 0
        self.done = False
        self.truncated = False
        environment_success_req = Empty.Request()
        while not self.environment_success_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Environment success service is not available, I'll wait"
            )
        self.environment_success_client.call_async(environment_success_req)

    # Populates the srv message response with obs,reward,truncated and, done
    def get_step_response(self, response):
        found_goal = []
        collided = []
        for i in range(self.num_robots):
            found_goal.append(self.has_found_goal(i) and self.step_counter > 130)
            collided.append(self.has_collided(i))
        self.truncated = self.step_counter > self.MAX_STEPS
        self.done = collided
        self.handle_found_goal(found_goal)
        for i in range(self.num_robots):
            scan_part = self.scans[i].astype(np.float32)
            pose_part = self.estimated_poses[i].astype(np.float32)
            d_opt_part = np.array([self.d_opts[i] * 10], dtype=np.float32)
            observation = np.concatenate((scan_part, pose_part, d_opt_part))
            response.multi_robot_states[i].state = observation
            response.multi_robot_rewards[i].reward = self.get_rewards(
                found_goal[i],
                collided[i],
                self.angular_velocities[i],
                self.linear_velocities[i],
                self.d_opts[i],
            )
            response.multi_robot_terminals[i].done = collided[i]
            response.multi_robot_terminals[i].truncated = self.truncated
        return response

    def environment_step_callback(self, request, response):
        self.step_counter += 1

        for i in range(self.num_robots):
            self.linear_velocities[i] = request.multi_robot_actions[i].actions[0]
            self.angular_velocities[i] = request.multi_robot_actions[i].actions[1]
            # if action_sum > 45:
            # reward += -50
            desired_vel_cmd = Twist()
            desired_vel_cmd.linear.x = self.linear_velocities[i].item()
            desired_vel_cmd.angular.z = self.angular_velocities[i].item()
            self.cmd_vel_publishers[i].publish(desired_vel_cmd)

        # Let simulation play out for a bit before observing
        time.sleep(0.1)

        # Return new state
        response = self.get_step_response(response)
        if self.done or self.truncated:
            self.stop_robots()
        return response


def main(args=None):
    rclpy.init(args=args)
    env = LearningEnvironment()
    rclpy.spin(env)
    env.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
