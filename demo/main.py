import sim
import pybullet as p
import random
import numpy as np
import argparse
from motion_planners.rrt_connect import rrt_connect
from motion_planners.rrt_star import rrt_star
from motion_planners.prm import prm
from motion_planners.rrt import rrt
from motion_planners.utils import get_sample_function, get_distance, get_extend_function, get_smooth_path
from motion_planners.collision_utils import get_collision_fn
from rrt import execute_path
from rrt import rrt as original_rrt

def test_robot_movement(num_trials, env):
    # Problem 1: Basic robot movement
    # Implement env.move_tool function in sim.py. More details in env.move_tool description
    passed = 0
    for i in range(num_trials):
        # Choose a reachable end-effector position and orientation
        random_position = env._workspace1_bounds[:, 0] + 0.15 + \
            np.random.random_sample((3)) * (env._workspace1_bounds[:, 1] - env._workspace1_bounds[:, 0] - 0.15)
        random_orientation = np.random.random_sample((3)) * np.pi / 4 - np.pi / 8
        random_orientation[1] += np.pi
        random_orientation = p.getQuaternionFromEuler(random_orientation)
        marker = sim.SphereMarker(position=random_position, radius=0.03, orientation=random_orientation)
        # Move tool
        env.move_tool(random_position, random_orientation)
        link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
        link_marker = sim.SphereMarker(link_state[0], radius=0.03, orientation=link_state[1], rgba_color=[0, 1, 0, 0.8])
        # Test position
        delta_pos = np.max(np.abs(np.array(link_state[0]) - random_position))
        delta_orn = np.max(np.abs(np.array(link_state[1]) - random_orientation))
        if  delta_pos <= 1e-3 and delta_orn <= 1e-3:
            passed += 1
        env.step_simulation(1000)
        # Return to robot's home configuration
        env.robot_go_home()
        del marker, link_marker
    print(f"[Robot Movement] {passed} / {num_trials} cases passed")

def test_grasping(num_trials, env):
    # Problem 2: Grasping
    passed = 0
    for _ in range(num_trials):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = env.get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)

        # Test for grasping success (this test is a necessary condition, not sufficient):
        object_z = p.getBasePositionAndOrientation(object_id)[0][2]
        if object_z >= 0.2:
            passed += 1
        env.reset_objects()
    print(f"[Grasping] {passed} / {num_trials} cases passed")

def test_rrt(args, num_trials, env):
    # Problem 3: RRT Implementation
    passed = 0
    for _ in range(num_trials):
        # grasp the object
        object_id = env._objects_body_ids[0]
        position, grasp_angle = env.get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            #print('grasp successful')
            # get a list of robot configuration in small step sizes
            get_sample = get_sample_function(len(env.get_joint_indices()))
            get_extend = get_extend_function()
            get_collision = get_collision_fn(env.get_robot_body_id(), env.get_joint_indices(), env.get_obstacles())
            if args.algo == 'rrt_star':
                path_conf = rrt_star(env.robot_home_joint_config, env.robot_goal_joint_config, get_distance, get_sample, get_extend, get_collision, 1.0, max_iterations=200)
            elif args.algo == 'rrt_connect':
                path_conf = rrt_connect(env.robot_home_joint_config, env.robot_goal_joint_config, get_distance, get_sample, get_extend, get_collision)
            elif args.algo == 'rrt':
                path_conf = rrt(env.robot_home_joint_config, env.robot_goal_joint_config, get_distance, get_sample, get_extend, get_collision)
            elif args.algo == 'prm':
                path_conf = prm(env.robot_home_joint_config, env.robot_goal_joint_config, get_distance, get_sample, get_extend, get_collision)
            elif args.algo == 'original_rrt':
                path_conf = original_rrt(env.robot_home_joint_config, env.robot_goal_joint_config, 10000, 2, 0.6, env)

            if path_conf is None:
                print(
                    "no collision-free path is found within the time budget. continuing ...")
            else:
                path_conf = get_smooth_path(path_conf, get_collision, get_extend)
                env.set_joint_positions(env.robot_home_joint_config)
                execute_path(path_conf, env)
            p.removeAllUserDebugItems()

        env.robot_go_home()

        # Test if the object was actually transferred to the second bin
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if object_pos[0] >= -0.8 and object_pos[0] <= -0.2 and\
            object_pos[1] >= -0.3 and object_pos[1] <= 0.3 and\
            object_pos[2] <= 0.2:
            passed += 1
        env.reset_objects()

    print(f"[RRT Object Transfer] {passed} / {num_trials} cases passed")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str,
                        help='part')
    parser.add_argument('--n', type=int, default=3,
                        help='number of trials')
    parser.add_argument('--disp', action='store_true')
    parser.add_argument('--algo', type=str, default='rrt_star', choices=['rrt_star', 'rrt_connect', 'prm', 'rrt', 'original_rrt'])
    args = parser.parse_args()

    random.seed(1)
    object_shapes = [
        "assets/objects/cube.urdf",
    ]
    env = sim.PyBulletSim(object_shapes = object_shapes, gui=args.disp)
    num_trials = args.n

    if args.part in ["2", "3", "all"]:
        env.load_gripper()
    if args.part in ["1", 'all']:
        test_robot_movement(num_trials, env)
    if args.part in ["2", 'all']:
        test_grasping(num_trials, env)
    if args.part in ["3", 'all']:
        test_rrt(args, num_trials, env)
    
