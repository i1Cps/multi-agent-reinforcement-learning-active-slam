from multi_robot_active_slam_learning.common.settings import NUMBER_OF_ROBOTS
import rclpy
import numpy as np
from std_srvs.srv import Empty
from multi_robot_active_slam_interfaces.srv import StepEnv, ResetEnv
import matplotlib.pyplot as plt


def plot_training_data(
    steps_file,
    scores_file,
    goal_history_file,
    learning_plot_filename,
    goals_plot_filename,
    learning_title="Reinforcement Learning Algorithm Returns",
    goals_title="Average Goals per Episode",
):
    # Plot learning curve
    steps = np.load(steps_file)
    scores = np.load(scores_file)
    running_avg_scores = np.zeros(len(scores))
    for i in range(len(running_avg_scores)):
        running_avg_scores[i] = np.mean(scores[max(0, i - 100) : (i + 1)])

    f1 = plt.figure("Learning Curve")  # Start a new figure
    plt.plot(steps, running_avg_scores)
    plt.title(learning_title)
    plt.xlabel("Steps")
    plt.ylabel("Scores")
    plt.savefig(learning_plot_filename)

    # Plot goals history
    goal_history = np.load(goal_history_file)
    running_avg_goals = np.zeros(len(goal_history))
    for i in range(len(running_avg_goals)):
        running_avg_goals[i] = np.mean(goal_history[max(0, i - 100) : (i + 1)])

    f2 = plt.figure("Goals History")  # Start a new figure
    plt.plot(running_avg_goals)
    plt.title(goals_title)
    plt.xlabel("Episode")
    plt.ylabel("Average Goals")
    plt.savefig(goals_plot_filename)

    # Show all plots
    plt.show()


# Communicates to the Learning Environment Node that it should step the environment using these actions
# It then expects typical reinforcement learning items back
def step(agent_self, actions_array):
    req = StepEnv.Request()
    for robot in range(NUMBER_OF_ROBOTS):
        actions = np.array(
            (actions_array[robot][0], actions_array[robot][1]), dtype=np.float32
        )

        req.robot_requests[robot].actions = actions

    while not agent_self.step_environment_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info(
            "environment step service not available, waiting again..."
        )
    future = agent_self.step_environment_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()

                states = []
                rewards = []
                dones = []
                truncated = []

                collided = False
                goal_found = 0
                distance_to_goal = []

                for robot in range(NUMBER_OF_ROBOTS):
                    states.append(res.robot_responses[robot].observation)
                    rewards.append(res.robot_responses[robot].reward)
                    dones.append(res.robot_responses[robot].done)
                    truncated.append(res.robot_responses[robot].truncated)

                    collided = collided or res.robot_responses[robot].info.collided
                    goal_found = (
                        goal_found or res.robot_responses[robot].info.goal_found
                    )
                    distance_to_goal.append(
                        res.robot_responses[robot].info.distance_to_goal
                    )

                info = {
                    "collided": collided,
                    "goal_found": goal_found,
                    "distance_to_goal": distance_to_goal,
                }
                if info["goal_found"]:
                    print(info)
                    print(info["goal_found"])

                return states, rewards, dones, truncated, info
            else:
                agent_self.get_logger().error(
                    "Exception while calling service: {0}".format(future.exception())
                )
                print("Error getting step service response, this wont output anywhere")


# Communicates to the Learning Environment Node that it should reset everything
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
                observations = []
                for robot in range(NUMBER_OF_ROBOTS):
                    observations.append(res.robot_responses[robot].observation)
                return observations
            else:
                agent_self.get_logger().error(
                    "Exception while calling service: {0}".format(future.exception())
                )
                print("Error resetting the env, if thats even possible ")


# Communicates to the Learning Environment Node that it should skip a frame
def skip_frame(agent_self):
    req = Empty.Request()
    while not agent_self.skip_environment_frame_client.wait_for_service(
        timeout_sec=1.0
    ):
        agent_self.get_logger().info("Frame skip service not available, waiting ")
    future = agent_self.skip_environment_frame_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                return
            else:
                agent_self.get_logger().error(
                    "Exception while calling service: {0}".format(future.exception())
                )
                print("Error skipping the env, if thats even possible ")
