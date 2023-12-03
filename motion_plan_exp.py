import time
import math
import random
import os
import shutil

import pybullet as p

import pb_ompl
from my_planar_robot import MyPlanarRobot

import matplotlib.pyplot as plt
import seaborn as sns

pandaEndEffectorIndex = 11  # 8
pandaNumDofs = 7

ll = [-7] * pandaNumDofs
# upper limits for null space
# todo: set them to proper range
ul = [7] * pandaNumDofs
# joint ranges for null space
# todo: set them to proper range
jr = [7] * pandaNumDofs
# rest poses for null space
rp = [0.21, -0.03, -0.20, -1.99, -0.01, 1.98, 0.80]


class MotionPlanExp(object):
    def __init__(self, exp_num):
        p.setPhysicsEngineParameter(solverResidualThreshold=0)

        # Setup objects
        self.obstacles = []

        # Setup robot
        self.panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], [-0.707107, 0.0, 0.0, 0.707107], useFixedBase=True)
        self.robot = MyPlanarRobot(self.panda)
        
        # Results
        self.exp_num = exp_num
        self.results = {'Time': {'With Obstacles': {}, 'Without Obstacles': {}}, 'Distance': {'With Obstacles': {}, 'Without Obstacles': {}}}
        self.planners = ['PRM', 'RRTConnect', 'EST']

    def test(self):
        for planner in self.planners:
            self.test_core(planner, 'Without Obstacles')
        
        self.add_obstacles()
        
        for planner in self.planners:
            self.test_core(planner, 'With Obstacles')
        
        self.save_results()

    def test_core(self, planner, obstacle):
        self.results['Time'][obstacle][planner] = []
        self.results['Distance'][obstacle][planner] = []
        for i in range(self.exp_num):
            self.robot.reset()
            exp_time = -1
            while exp_time == -1:
                des_pos = self.generate_random_coordinate_and_angle()
                exp_time = self.get_path(des_pos, [math.pi / 2., 0., 0.], planner)
            end_effector_pos = p.getLinkState(self.panda, 11)[0]
            exp_dis = ((end_effector_pos[0] - des_pos[0])**2 + (end_effector_pos[1] - des_pos[1])**2 + (end_effector_pos[2] - des_pos[2])**2)**0.5
            self.results['Time'][obstacle][planner].append(exp_time)
            self.results['Distance'][obstacle][planner].append(exp_dis)

    def add_obstacles(self):
        self.obstacles.append(p.loadURDF('obstacles/block.urdf', basePosition=[0.3, 0.5, 0.3], useFixedBase=True))
        self.obstacles.append(p.loadURDF('obstacles/block.urdf', basePosition=[0.3, 0.5, -0.3], useFixedBase=True))
        self.obstacles.append(p.loadURDF("tray/traybox.urdf", [0, 0.2, -0.5], [0.5, -0.5, 0.5, 0.5]))
        self.obstacles.append(p.loadURDF("tray/traybox.urdf", [0.5, 0.2, 0], [-0.5, -0.5, -0.5, 0.5]))

    def generate_random_coordinate_and_angle(self):
        random.seed(time.time())
        x = random.uniform(-0.15, 0.15)
        y = 0.22
        z = random.uniform(-0.65, -0.35)
        
        return (x, y, z)
    
    def get_path(self, des_pos, des_orn, planner):
        quaternion_orn = p.getQuaternionFromEuler(des_orn)
        des_joint_state = p.calculateInverseKinematics(self.panda, 11, des_pos, quaternion_orn, ll, ul, jr, rp, maxNumIterations=100)

        pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        pb_ompl_interface.set_planner(planner)
        start_time = time.time()
        res, path = pb_ompl_interface.plan(des_joint_state)
        end_time = time.time()
        if res:
            pb_ompl_interface.execute(path, 0.0)
            return end_time - start_time
        else:
            print('--------------------get path error !!!--------------------')
            return -1

    def save_results(self):
        if os.path.exists('exp_results/motion_plan'):
            shutil.rmtree('exp_results/motion_plan')
        os.makedirs('exp_results/motion_plan')
        
        for data_type in ['Time', 'Distance']:
            for obstacle in ['Without Obstacles', 'With Obstacles']:
                for planner in self.planners:
                    self.save_text(planner + ' ' + obstacle + ' ' + data_type, self.results[data_type][obstacle][planner])
                self.save_plot('Motion Plan ' + data_type + ' Distribution of Algorithms ' + obstacle, self.planners, data_type, [self.results[data_type][obstacle][p] for p in self.planners])
    
    def save_text(self, title, value):
        mean = sum(value) / len(value)
        variance = sum((x - mean) ** 2 for x in value) / len(value)
        with open('exp_results/motion_plan/results.txt', 'a') as f:
            f.write(title + ':\n')
            f.write('\tmean: ' + str(mean) + '\n')
            f.write('\tvariance: ' + str(variance) + '\n')

    def save_plot(self, title, xticks, ylabel, value):
        plt.figure()
        sns.boxplot(data=value, orient='v')
        plt.title(title)
        plt.xticks([0, 1, 2], xticks)
        plt.ylabel(ylabel)
        plt.savefig('exp_results/motion_plan/' + title + '.png')
        plt.close()
