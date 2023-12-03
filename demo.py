import math
import random
import os
import yaml
import time

import pybullet as p
import pybullet_data as pd
import numpy as np

import pb_ompl
from my_planar_robot import MyPlanarRobot

from autolab_core import CameraIntrinsics, DepthImage, ColorImage, RgbdImage, BinaryImage
from gqcnn.grasping import CrossEntropyRobustGraspingPolicy, RgbdImageState

pandaEndEffectorIndex = 11  # 8
pandaNumDofs = 7

# lower limits for null space
ll = [-7] * pandaNumDofs
# upper limits for null space
# todo: set them to proper range
ul = [7] * pandaNumDofs
# joint ranges for null space
# todo: set them to proper range
jr = [7] * pandaNumDofs
# rest poses for null space
rp = [0.21, -0.03, -0.20, -1.99, -0.01, 1.98, 0.80]


class DemoSim(object):
    def __init__(self, exp_num, planner):
        p.setPhysicsEngineParameter(solverResidualThreshold=0)

        # Setup objects
        self.obstacles = []
        
        self.obstacles.append(p.loadURDF("tray/traybox.urdf", [0, 0.2, -0.5], [0.5, -0.5, 0.5, 0.5]))
        self.obstacles.append(p.loadURDF("tray/traybox.urdf", [0.5, 0.2, 0], [-0.5, -0.5, -0.5, 0.5]))
        
        # self.obstacles.append(p.loadURDF('obstacles/block.urdf', basePosition=[0.3, 0.5, 0.3], useFixedBase=True))
        self.obstacles.append(p.loadURDF('obstacles/block.urdf', basePosition=[0.3, 0.5, -0.3], useFixedBase=True))
        
        lego_pos = self.generate_random_coordinate()
        self.lego = p.loadURDF("lego/lego.urdf", lego_pos, [-0.5, -0.5, -0.5, 0.5], globalScaling=1.2)

        # Setup robot
        self.panda = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], [-0.707107, 0.0, 0.0, 0.707107], useFixedBase=True)
        self.robot = MyPlanarRobot(self.panda)
        self.finger_target = 0

        # Motion planning
        self.paths = list()

        # Setup state change
        self.state = 0
        self.cur_state = 0
        self.control_dt = 1. / 240.
        self.state_t = 0
        # 0：初始化、1：预测抓取位置并路径规划、2：张开机械手、3：移动到抓取处、4：闭合机械手、5：移动到放置处
        self.states = [0, 1, 2, 3, 4, 5, 2]
        self.state_duration = 1

        # Motion planning
        self.paths = list()
        self.planner = planner
        
        # Camera
        self.camera_intr = CameraIntrinsics.load('Grasping/primesense.intr')
        
        # Grasp
        with open('Grasping/gqcnn_pj.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.inpaint_rescale_factor = config["inpaint_rescale_factor"]
        self.policy_config = config["policy"]
        self.policy_config["metric"]["gqcnn_model"] = os.path.join('Grasping/models/GQCNN-2.1')
        
        self.graspPos = np.array([0, 0, 0, 0])
        
        # Result
        self.exp_num = exp_num
        self.cur_exp_num = 0

    def update_state(self):
        self.state_t += self.control_dt
        if self.state in [0, 2, 4]:
            if self.state_t >= self.state_duration:
                return self.next_state()
            else:
                return False
        else:
            return self.next_state()

    def next_state(self):
        self.cur_state += 1
        if self.cur_state >= len(self.states):
            self.cur_state = 0
            self.cur_exp_num += 1
            if self.cur_exp_num == self.exp_num:
                return True
            self.reset()
        self.state = self.states[self.cur_state]
        self.state_t = 0

        print("self.state=", self.state)
        return False

    def reset(self):
        p.removeBody(self.lego)
        lego_pos = self.generate_random_coordinate()
        self.lego = p.loadURDF("lego/lego.urdf", lego_pos, [-0.5, -0.5, -0.5, 0.5], globalScaling=1.2)
        
        self.robot.reset()
        self.finger_target = 0
        p.resetJointState(self.panda, 9, 0.0, targetVelocity=0)
        p.resetJointState(self.panda, 10, 0.0, targetVelocity=0)
        
        self.paths.clear()
        
        self.graspPos = np.array([0, 0, 0, 0])
        
    def generate_random_coordinate(self):
        random.seed(time.time())
        x = random.uniform(-0.15, 0.15)
        y = 0.3
        z = random.uniform(-0.65, -0.35)
        
        # print('random coordinate:' + str((x,y,z)))
        return (x, y, z)
    
    def step(self):
        if self.update_state():
            return True

        if self.state == 1:
            self.graspPos = self.get_grasp_pos()
            
            lego_pos = p.getBasePositionAndOrientation(self.lego)[0]
            p.removeBody(self.lego)
            
            p.resetJointState(self.panda, 9, 0.04, targetVelocity=0)
            p.resetJointState(self.panda, 10, 0.04, targetVelocity=0)
            self.get_path([self.graspPos[0], self.graspPos[1], self.graspPos[2]], [math.pi / 2., 0., 0.])
            self.get_path([0.6, 0.4, 0.0], [math.pi / 2., 0., 0.])
            
            self.robot.reset()
            p.resetJointState(self.panda, 9, 0.0, targetVelocity=0)
            p.resetJointState(self.panda, 10, 0.0, targetVelocity=0)
            
            self.lego = p.loadURDF("lego/lego.urdf", lego_pos, [-0.5, -0.5, -0.5, 0.5], globalScaling=1.2)
        elif self.state == 2:
            self.finger_target = 0.04
        elif self.state == 3:
            if len(self.paths) == 2:
                self.paths[0][0].execute(self.paths[0][1], self.control_dt, True)
        elif self.state == 4:
            self.finger_target = 0.0
        elif self.state == 5:
            if len(self.paths) == 2:
                self.paths[1][0].execute(self.paths[1][1], self.control_dt, True)
        
        if (self.state == 2 or self.state == 4) and self.state_t > 0.5:
            for i in [9, 10]:
                p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, self.finger_target, force=20)
        
        return False

    def get_path(self, des_pos, des_orn):
        quaternion_orn = p.getQuaternionFromEuler(des_orn)
        des_joint_state = p.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, des_pos,
                                                       quaternion_orn, ll, ul, jr, rp, maxNumIterations=100)
        
        pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        pb_ompl_interface.set_planner(self.planner)
        res, path = pb_ompl_interface.plan(des_joint_state)
        if res:
            pb_ompl_interface.execute(path, 0.0)
            # self.robot.set_state(des_joint_state)
            # print('des_pos:' + str(des_pos))
            # print('cur_pos:' + str(p.getLinkState(self.panda, 11)[0]))
            # print('des_state:' + str(des_joint_state))
            # print('cur_state:' + str(self.robot.get_cur_state()))
            self.paths.append((pb_ompl_interface, path))
        else:
            print('--------------------get path error !!!--------------------')

    def get_grasp_pos(self):
        width = 640
        height = 480

        fov = 49
        aspect = width / height
        near = 0.01
        far = 2.0
        
        cameraPos = [0, 0.8, -0.5]
        targetPos = [0, 0, -0.5]
        cameraUpPos = [1, 0, 0]

        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=cameraPos,
            cameraTargetPosition=targetPos,
            cameraUpVector=cameraUpPos,
            physicsClientId=0
        )
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        images = p.getCameraImage(width, height, viewMatrix, projection_matrix,
                                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)

        depth_data = np.reshape(images[3], (height, width, 1))
        depth_data = far * near / (far - (far - near) * depth_data)
        depth_im = DepthImage(depth_data, frame=self.camera_intr.frame)
        color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8),
                              frame=self.camera_intr.frame)
        
        seg_mask = np.reshape(images[4], (height, width, 1))
        seg_mask[seg_mask > 0] = 255
        seg_mask[seg_mask <= 0] = 0
        seg_mask = BinaryImage(seg_mask.astype(np.uint8))
        valid_px_mask = depth_im.invalid_pixel_mask().inverse()
        seg_mask = seg_mask.mask_binary(valid_px_mask)

        depth_im = depth_im.inpaint(rescale_factor=self.inpaint_rescale_factor)

        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        state = RgbdImageState(rgbd_im, self.camera_intr, segmask=seg_mask)
        
        policy = CrossEntropyRobustGraspingPolicy(self.policy_config)
        action = policy(state)
        
        pose = action.grasp.pose(grasp_approach_dir=np.array([0, -1, 0])).position
        angle = (math.pi - action.grasp.angle % math.pi) % math.pi
        if angle > math.pi / 2:
            angle = angle - math.pi
        return [cameraPos[0] - pose[1], 0.22, cameraPos[2] + pose[0]]


if __name__ == '__main__':
    createVideo = False
    fps = 240.
    timeStep = 1. / fps
    
    if createVideo:
        p.connect(p.GUI, options="--minGraphicsUpdateTimeMs=0 --mp4=\"pybullet_grasp.mp4\" --mp4fps=" + str(fps))
    else:
        p.connect(p.GUI)

    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=-45, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])
    p.setAdditionalSearchPath(pd.getDataPath())

    p.setTimeStep(timeStep)
    p.setGravity(0, -9.8, 0)

    demo_sim = DemoSim(2, 'PRM')
    while True:
        if demo_sim.step():
            break
        p.stepSimulation()
        if createVideo:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        if not createVideo:
            time.sleep(timeStep)
