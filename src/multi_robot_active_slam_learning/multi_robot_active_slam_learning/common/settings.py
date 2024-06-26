import numpy as np


#########################################################
#                   ROBOT SETTINGS                      #
#########################################################

MAX_LINEAR_SPEED = 0.22
MAX_ANGULAR_SPEED = 2.0
MAX_SCAN_DISTANCE = 3.5
NUMBER_OF_SCANS = 90
COLLISION_DISTANCE = 0.18
ROBOT_NAME = "robot"

#########################################################
#                   ENVIRONMENT SETTINGS                #
#########################################################

# Scans + Pose_x + Pose_y
ROBOT_OBSERVATION_SPACE = NUMBER_OF_SCANS + 2
ROBOT_ACTION_SPACE = 2  # Linear and Angular Speed
MAX_CONTINUOUS_ACTIONS = np.array([MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED])
EPISODE_LENGTH_SEC = 60
EPISODE_STEPS = 1000
GOAL_PAD_RADIUS = 0.7
REWARD_DEBUG = True


#########################################################
#            REINFORCEMENT LEARNING SETTINGS            #
#########################################################

# GLOBAL SETTINGS
LOAD_MODEL = False
MODEL_PATH = "src/multi_robot_active_slam_learning/models"
NUMBER_OF_ROBOTS = 3
FRAME_BUFFER_DEPTH = 3
FRAME_BUFFER_SKIP = 4
RANDOM_STEPS = 25000
TRAINING_EPISODES = 2000
TRAINING_STEPS = 1_000_000

# MADDPG SETTINGS:
ACTOR_LEARNING_RATE_MADDPG = 0.00001
CRITIC_LEARNING_RATE_MADDPG = 0.00002
TAU = 0.005
GAMMA_MADDPG = 0.99
BATCH_SIZE_MADDPG = 512
ACTOR_MADDPG_FC1 = 500
ACTOR_MADDPG_FC2 = 512
CRITIC_MADDPG_FC1 = 512
CRITIC_MADDPG_FC2 = 512
MAX_MEMORY_SIZE = 1_000_000  # Adjust according to your system, I have 32GB RAM


# MAPPO SETTINGS:
ALPHA_MAPPO = 0.0001
BETA_MAPPO = 0.0003
GAMMA_MAPPO = 0.99
TRAJECTORY = 2048
NUM_MINI_BATCHES = 64
N_EPOCHS = 15
GAE_LAMBDA = 0.95
ENTROPHY_COEFFICIENT = 0.001
POLICY_CLIP = 0.2
ACTOR_MAPPO_FC1 = 512
ACTOR_MAPPO_FC2 = 512
CRITIC_MAPPO_FC1 = 512
CRITIC_MAPPO_FC2 = 512
