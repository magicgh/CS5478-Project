import numpy as np
from env import PandaPickAndPlaceEnv
# from rrt_test.exp.rrt import rrt
from motion_planners.rrt import rrt
from motion_planners.utils import get_sample_function, get_distance, get_extend_function

env = PandaPickAndPlaceEnv(render_mode="rgb_array", renderer="OpenGL", control_type="joint")

observation, info = env.reset()

# Note: for Panda the last dimension is the fingers width

# Note: Parameters for RRT*
Q = np.array([(8, 4)])  # length of tree edges
r = 0.0001  # length of smallest edge to check for intersection with obstacles
max_samples = 65536  # max number of samples to take before timing out
rewire_count = 32768  # optional, number of nearby branches to rewire
prc = 0.2  # probability of checking for a connection to goal

joint_num = len(env.robot.get_revolute_joint_indices())
obstacles = env.task.get_obstacles()
get_sample = get_sample_function(joint_num)
get_extend = get_extend_function()

for _ in range(100):
    
    achieved_goal = env.robot.ee_position_to_target_arm_angles(observation["achieved_goal"])
    desired_goal = env.robot.ee_position_to_target_arm_angles(observation["desired_goal"])
    current_joint_angles = np.array([env.robot.get_joint_angle(i) for i in range(joint_num)])
    rrt(current_joint_angles, achieved_goal, get_distance, get_sample, get_extend)
    path = rrt(current_joint_angles, achieved_goal, 2000, 2, 0.6, env, obstacles, env.robot.get_body_id(), env.robot.get_revolute_joint_indices())
    
    if path is None:
        print("Goal not reached!")
        env.reset()
        continue
    
    for i, position in enumerate(path):
        observation, reward, terminated, truncated, info = env.step(position)

    current_joint_state = np.array([
                env.robot.get_joint_angle(i)
                for i in env.robot.get_revolute_joint_indices()
    ])
    
    if terminated or truncated:
        print("Episode terminated or truncated!")
        env.reset()
        
    elif get_distance(current_joint_state, achieved_goal) < 0.4:
        print("Goal reached!")
        env.reset()
        
    else:
        print("Goal not reached!")
        env.reset()

env.close()