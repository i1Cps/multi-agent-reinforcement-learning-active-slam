import numpy as np

# THIS FILE IS VERY MESSY, SORRY FOR NOW

#########################################################
#                   ROBOT SETTINGS                      #
#########################################################

MAX_LINEAR_SPEED = 0.22
MAX_ANGULAR_SPEED = 2.0
NUMBER_OF_SCANS = 90


#########################################################
#                   ENVIRONMENT SETTINGS                #
#########################################################
NUMBER_OF_ROBOTS = 3
INITIAL_POSES = np.array([[-2.0, 7.0], [0, 7], [2, 7]])

MAX_CONTINUOUS_ACTIONS = np.array([MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED])


# Scans + Pose_x + Pose_y + D_Optimality
ROBOT_OBSERVATION_SPACE = (NUMBER_OF_SCANS + 3,)
ROBOT_ACTION_SPACE = (2,)

ENVIRONMENT_OBSERVATION_SPACE = (ROBOT_OBSERVATION_SPACE[0] * NUMBER_OF_ROBOTS,)

#########################################################
#            REINFORCEMENT LEARNING SETTINGS            #
#########################################################

# GLOBAL SETTINGS:

TRAINING_EPISODES = 2000
EPISODE_LENGTH_STEPS = 1000
EPISODE_LENGTH_SEC = 100
MAX_STEPS = 2_000_000
MAX_MEMORY_SIZE = 1_000_000  # Adjust according to your system, I have 32GB RAM
BATCH_SIZE = 64
FRAME_BUFFER_DEPTH = 3
FRAME_BUFFER_SKIP = 10

# MADDPG SETTINGS:

ALPHA_MADDPG = 0.001
BETA_MADDPG = 0.001
TAU_MADDPG = 0.005
ACTOR_MADDPG_FC1 = 400
ACTOR_MADDPG_FC2 = 512
CRITIC_MADDPG_FC1 = 512
CRITIC_MADDPG_FC2 = 512
GAMMA_MADDPG = 0.999
RANDOM_STEPS = 20000


# MAPPO SETTINGS:

ALPHA_MAPPO = 0.0005
BETA_MAPPO = 0.0005
TAU_MAPPO = 0.005
ACTOR_MAPPO_FC1 = 512
ACTOR_MAPPO_FC2 = 512
CRITIC_MAPPO_FC1 = 512
CRITIC_MAPPO_FC2 = 512
GAMMA_MAPPO = 0.99
MINI_BATCHES_MAPPO = 1
BATCH_SIZE_MAPPO = 584
N_EPOCHS = 15
GAE_LAMBDA = 0.95
ENTROPHY_COEFFICIENT = 0.01
POLICY_CLIP = 0.2
