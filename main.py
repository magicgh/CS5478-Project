import numpy as np
from env import PandaPickAndPlaceEnv
# from rrt_test.exp.rrt import rrt
from motion_planners.rrt import rrt
from motion_planners.rrt_connect import rrt_connect
from motion_planners.rrt_star import rrt_star
from motion_planners.prm import prm
from motion_planners.utils import get_sample_function, get_distance, get_extend_function
from motion_planners.collision_utils import get_collision_fn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--planner", type=str, default="rrt", choices=["rrt", "rrt_connect", "rrt_star", "prm"], help="Planner to use")
parser.add_argument('--disp', action='store_true')
parser.add_argument('--obst_num', type=int, default=1)
args = parser.parse_args()

if args.disp:
    env = PandaPickAndPlaceEnv(render_mode="rgb_array", renderer="OpenGL", control_type="joint", obst_num=args.obst_num)
else:
    env = PandaPickAndPlaceEnv(render_mode="rgb_array", renderer="Tiny", control_type="joint", obst_num=args.obst_num)

observation, info = env.reset()

# Note: for Panda the last dimension is the fingers width

'''# Note: Parameters for RRT*
Q = np.array([(8, 4)])  # length of tree edges
r = 0.0001  # length of smallest edge to check for intersection with obstacles
max_samples = 65536  # max number of samples to take before timing out
rewire_count = 32768  # optional, number of nearby branches to rewire
prc = 0.2  # probability of checking for a connection to goal'''

joint_num = len(env.robot.get_revolute_joint_indices())
obstacles = env.task.get_obstacles()
get_sample = get_sample_function(joint_num)
get_extend = get_extend_function()
get_collision = get_collision_fn(env.robot.get_body_id(), env.robot.get_revolute_joint_indices(), obstacles, [env.task.get_table_id()])

cnt = 0
for _ in range(100):
    
    achieved_goal = env.robot.ee_position_to_target_arm_angles(observation["achieved_goal"])
    desired_goal = env.robot.ee_position_to_target_arm_angles(observation["desired_goal"])
    current_joint_angles = np.array([env.robot.get_joint_angle(i) for i in range(joint_num)])
    
    print(achieved_goal)
    if args.planner == "rrt":
        path = rrt(current_joint_angles, achieved_goal, get_distance, get_sample, get_extend, get_collision)
    elif args.planner == "rrt_connect":
        path = rrt_connect(current_joint_angles, achieved_goal, get_distance, get_sample, get_extend, get_collision)
    # NOTE rrt_star seems not work!!!
    elif args.planner == "rrt_star":
        path = rrt_star(current_joint_angles, achieved_goal, get_distance, get_sample, get_extend, get_collision, 1.0)
    elif args.planner == "prm":
        path = prm(current_joint_angles, achieved_goal, get_distance, get_sample, get_extend, get_collision)

    
    # path = rrt(current_joint_angles, achieved_goal, 2000, 2, 0.6, env, obstacles, env.robot.get_body_id(), env.robot.get_revolute_joint_indices())
    # assert 1==2

    print(path)
    print(achieved_goal)
    for i, position in enumerate(path):
        observation, reward, terminated, truncated, info = env.step(position)
    
    if terminated or truncated:
        print("Goal not reached!")
        env.reset()
        continue
    
    current_joint_angles = np.array([env.robot.get_joint_angle(i) for i in range(joint_num)])
    
    if get_distance(current_joint_angles, achieved_goal) < 0.06:
        print("Goal reached!")
        cnt += 1
        env.reset()
    else:
        print("Goal not reached!")
        env.reset()
        
env.close()
print("Planner: ", args.planner)
print("Obstacle number: ", args.obst_num)
print("Success rate: ", cnt / 100)