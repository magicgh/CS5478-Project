from env import PandaPickAndPlaceEnv
from motion_planner.rrt.rrt_star import RRTStar
from motion_planner.rrt.rrt import RRT
from motion_planner.search_space.search_space import SearchSpace
import numpy as np
from panda_gym.utils import distance
from motion_planner.search_space.collision_utils import get_collision_fn
from motion_planner.exp.rrt import rrt

env = PandaPickAndPlaceEnv(render_mode="rgb_array", renderer="OpenGL", control_type="joint")

observation, info = env.reset()

X_dimensions = np.column_stack((env.action_space.low, env.action_space.high))[:-1]
# Note: for Panda the last dimension is the fingers width

# Note: Parameters for RRT*
Q = np.array([(8, 4)])  # length of tree edges
r = 0.0001  # length of smallest edge to check for intersection with obstacles
max_samples = 65536  # max number of samples to take before timing out
rewire_count = 32768  # optional, number of nearby branches to rewire
prc = 0.2  # probability of checking for a connection to goal
obstacles = env.task.get_obstacles() 

'''
collision_fn = get_collision_fn(env.robot.get_body_id(), env.robot.get_joint_indices(), obstacles)
'''
for _ in range(10000):
    
    achieved_goal = env.robot.ee_position_to_target_arm_angles(observation["achieved_goal"])
    desired_goal = env.robot.ee_position_to_target_arm_angles(observation["desired_goal"])
    current_joint_angles = np.array([env.robot.get_joint_angle(i) for i in range(7)])

    '''
    X = SearchSpace(X_dimensions, collision_fn)

    rrt = RRTStar(X, Q, current_joint_angles, achieved_goal, max_samples, r, prc, rewire_count)
    '''
    path = rrt(current_joint_angles, achieved_goal, 2000, 2, 0.6, env, obstacles, env.robot.get_body_id(), env.robot.get_joint_indices())
    print("path", path)
    print("current_joint_angles", current_joint_angles)
    print("achieved_goal", achieved_goal)
    env.robot.set_joint_angles(current_joint_angles)
    for i in range(len(path)-1):
        action = path[i+1] - path[i]
        print("action", action)
        action = np.append(action, 0)
        observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated or distance(current_joint_angles, achieved_goal) < 0.05:
        env.reset()
env.close()