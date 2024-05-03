import rclpy
import torch
import numpy as np
from std_srvs.srv import Empty
from multi_robot_active_slam_interfaces.srv import StepEnv, ResetEnv
from multi_robot_active_slam_learning.common.settings import NUMBER_OF_ROBOTS


# Pause Gazebo Physics simulation
def pause_simulation(agent_self):
    while not agent_self.gazebo_pause.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info(
            "pause gazebo service not available, waiting again..."
        )
    future = agent_self.gazebo_pause.call_async(Empty.Request())
    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            return


# Unpause Gazebo Physics simulation
def unpause_simulation(agent_self):
    while not agent_self.gazebo_unpause.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info(
            "unpause gazebo service not available, waiting again..."
        )
    future = agent_self.gazebo_unpause.call_async(Empty.Request())
    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            return


# Function communicates with environment node, it request action and expects response
# containing observation, reward, and whether episode has finished or truncated
def step(agent_self, actions, discrete=False):
    req = StepEnv.Request()
    for i in range(NUMBER_OF_ROBOTS):
        agent_actions = np.array(actions[i], dtype=np.float32)
        req.multi_robot_actions[i].actions = agent_actions

    while not agent_self.environment_step_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info(
            "environment step service not available, waiting again..."
        )
    future = agent_self.environment_step_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                raw_obs = []
                rewards = []
                dones = []
                truncated = []

                for i in range(NUMBER_OF_ROBOTS):
                    raw_obs.append(res.multi_robot_states[i].state)
                    rewards.append(res.multi_robot_rewards[i].reward)
                    dones.append(res.multi_robot_terminals[i].done)
                    truncated.append(res.multi_robot_terminals[i].truncated)

                return raw_obs, rewards, dones, truncated
            else:
                agent_self.get_logger().error(
                    "Exception while calling service: {0}".format(future.exception())
                )
                print("Error getting step service response, this wont output anywhere")


# Make this spin til complete
def reset(agent_self):
    req = ResetEnv.Request()
    while not agent_self.reset_environment_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info(
            "reset environment service not available, waiting again..."
        )
    future = agent_self.reset_environment_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                states_list = [
                    np.array(state.state) for state in res.multi_robot_states
                ]
                return states_list
            else:
                agent_self.get_logger().error(
                    "Exception while calling service: {0}".format(future.exception())
                )
                print("Error resetting the env, if thats even possible ")
