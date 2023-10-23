from env import PandaPickAndPlaceEnv
from motion_planner.rrt.rrt_star import RRTStar
from motion_planner.rrt.rrt import RRT
from motion_planner.search_space.search_space import SearchSpace
import numpy as np
from panda_gym.utils import distance

env = PandaPickAndPlaceEnv(render_mode="rgb_array", renderer="OpenGL")

observation, info = env.reset()

# TODO: X_dimensions, and obs_length from the env
obs_length = 0.05
X_dimensions = np.array([(-0.15, 0.15), (-0.15, 0.15), (-0.1, 0.1)]) 

# Note: Parameters for RRT*
Q = np.array([(8, 4)])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal
fingers_width  = 0.003
is_pick = False
for _ in range(1000):
    obstacles_position = observation["observation"][7:].reshape(-1,3)
    achieved_goal = tuple(observation["achieved_goal"])
    desired_goal = tuple(observation["desired_goal"])
    current_position = tuple(observation["observation"][:3])
    Obstacles = np.array([
        [obstacle_position - obs_length, obstacle_position + obs_length]
        for obstacle_position in obstacles_position
    ]).reshape(-1, 6)

    if distance(np.array(current_position), np.array(achieved_goal)) < 0.03 and is_pick is False:
        action = np.array([0, 0, 0, fingers_width])
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        is_pick = True
        continue
    
    X = SearchSpace(X_dimensions,Obstacles)
    rrt = RRTStar(X, Q, current_position, desired_goal if is_pick else achieved_goal, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star()
    
    '''rrt = RRT(X, Q, current_position, desired_goal if is_pick else achieved_goal, max_samples, r, prc)
    path = rrt.rrt_search()'''
    

    action =  2*(np.array(path[1]) - np.array(path[0]))
    action = np.append(action, fingers_width)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break
env.close()