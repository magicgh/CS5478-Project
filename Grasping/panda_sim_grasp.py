import sys
import time
import numpy as np
import math
import os
import yaml

from autolab_core import YamlConfig, CameraIntrinsics, DepthImage, ColorImage, RgbdImage, BinaryImage
from visualization import Visualizer2D as vis
from gqcnn.grasping import CrossEntropyRobustGraspingPolicy, RgbdImageState

from motion_planners.rrt_connect import rrt_connect
from motion_planners.rrt_star import rrt_star
from motion_planners.prm import prm
from motion_planners.rrt import rrt
from motion_planners.utils import get_sample_function, get_distance, get_extend_function
from motion_planners.collision_utils import get_collision_fn

useNullSpace = 1
ikSolver = 0
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
jointPositions = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions


class PandaSimAuto(object):
    def __init__(self, bullet_client, offset):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)

        # print("offset=",offset)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        self.obstacles = [
            # self.bullet_client.loadURDF('obstacles/block.urdf',
            #                             basePosition=[0.3, 0.3, 0.3],
            #                             useFixedBase=True
            #                             ),
            # self.bullet_client.loadURDF('obstacles/block.urdf',
            #                             basePosition=[0.3, 0.3, -0.3],
            #                             useFixedBase=True
            #                             ),
        ]
        self.legos = []

        self.obstacles.append(self.bullet_client.loadURDF("tray/traybox.urdf",
                                                          [0 + offset[0], 0 + offset[1], -0.6 + offset[2]],
                                                          [-0.5, -0.5, -0.5, 0.5],
                                                          flags=flags))
        self.obstacles.append(self.bullet_client.loadURDF("tray/traybox.urdf",
                                                          [0 + offset[0], 0 + offset[1], 0.6 + offset[2]],
                                                          [-0.5, -0.5, -0.5, 0.5],
                                                          flags=flags))

        self.legos.append(self.bullet_client.loadURDF("lego/lego.urdf",
                                                      np.array([0.1, 0.3, -0.6]) + self.offset,
                                                      flags=flags))
        # self.bullet_client.changeVisualShape(self.legos[0], -1, rgbaColor=[1, 0, 0, 1])  # 修改目标物体的颜色
        # self.legos.append(self.bullet_client.loadURDF("lego/lego.urdf",
        #                                               np.array([-0.1, 0.3, -0.5]) + self.offset,
        #                                               flags=flags))
        # self.legos.append(self.bullet_client.loadURDF("lego/lego.urdf",
        #                                               np.array([0.1, 0.3, -0.7]) + self.offset,
        #                                               flags=flags))
        #
        # self.sphereId = self.bullet_client.loadURDF("sphere_small.urdf",
        #                                             np.array([0, 0.3, -0.6]) + self.offset,
        #                                             flags=flags)
        # self.bullet_client.loadURDF("sphere_small.urdf",
        #                             np.array([0, 0.3, -0.5]) + self.offset,
        #                             flags=flags)
        # self.bullet_client.loadURDF("sphere_small.urdf",
        #                             np.array([0, 0.3, -0.7]) + self.offset,
        #                             flags=flags)

        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf",
                                                 np.array([0, 0, 0]) + self.offset,
                                                 [-0.707107, 0.0, 0.0, 0.707107],
                                                 useFixedBase=True, flags=flags)

        index = 0
        self.state = 0
        self.control_dt = 1. / 240.

        # create a constraint to keep the fingers centered
        c = self.bullet_client.createConstraint(self.panda,
                                                9,
                                                self.panda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            # print("info=",info)
            # jointName = info[1]
            joint_type = info[2]
            if joint_type == self.bullet_client.JOINT_PRISMATIC:
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
            if joint_type == self.bullet_client.JOINT_REVOLUTE:
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1

        self.diff_joint_state = np.zeros(9)
        self.graspPos = np.array([0, 0, 0, 0])
        self.finger_target = 0

        self.state_t = 0
        self.cur_state = 0
        # 0：初始状态、1：移动到空闲位置、2：获取深度图片计算最优抓取位置、3：移动到抓取处上方、4：张开机械手、5：移动到抓取处、6：闭合机械手、7：移动到放置处
        self.states = [0, 1, 2, 3, 4, 5, 6, 3, 1, 7, 4]
        self.state_durations = [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]

        current_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_path, 'gqcnn_pj.yaml'), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # config = YamlConfig(os.path.join(current_path, 'gqcnn_pj.yaml'))
        self.inpaint_rescale_factor = config["inpaint_rescale_factor"]
        self.policy_config = config["policy"]
        self.policy_config["metric"]["gqcnn_model"] = os.path.join(current_path, 'models/GQCNN-2.1')
        self.policy = CrossEntropyRobustGraspingPolicy(self.policy_config)

        self.camera_intr = CameraIntrinsics.load(os.path.join(current_path, 'primesense.intr'))

        self.get_sample = get_sample_function(9)
        self.get_extend = get_extend_function()
        # self.get_collision = get_collision_fn(self.panda, list(range(9)), self.obstacles)
        self.get_collision = get_collision_fn(self.panda, list(range(9)), self.obstacles)

        self.path = list()
        self.path_idx = 0

    def update_state(self):
        self.state_t += self.control_dt
        if self.state in [0, 2, 4, 6]:
            if self.state_t >= self.state_durations[self.state]:
                self.next_state()
        else:
            if self.path_idx >= len(self.path):
                self.next_state()

    def next_state(self):
        self.cur_state += 1
        if self.cur_state >= len(self.states):
            self.cur_state = 1
        self.state = self.states[self.cur_state]
        self.state_t = 0

        self.path = list()
        self.path_idx = 0

        print("self.state=", self.state)

    def step(self):
        self.bullet_client.submitProfileTiming("step")
        self.update_state()

        if self.state == 1:
            if len(self.path) == 0:
                self.path = self.get_path([0.6, 0.4, 0], [math.pi / 2., 0, 0.])
            next_joint_state = self.cal_next_joint_state(0.05)
        elif self.state == 2:
            self.graspPos = self.get_grasp_pos()
        elif self.state == 3:
            if len(self.path) == 0:
                self.path = self.get_path([self.graspPos[0], 0.4, self.graspPos[2]],
                                          [math.pi / 2., 0., 0.])
            next_joint_state = self.cal_next_joint_state(0.05)
        elif self.state == 4:
            self.finger_target = 0.04
        elif self.state == 5:
            if len(self.path) == 0:
                self.path = self.get_path([self.graspPos[0], self.graspPos[1], self.graspPos[2]],
                                          [math.pi / 2., self.graspPos[3], 0.])
            next_joint_state = self.cal_next_joint_state(0.05)
        elif self.state == 6:
            self.finger_target = 0.0
        elif self.state == 7:
            if len(self.path) == 0:
                self.path = self.get_path([0, 0.4, 0.6], [math.pi / 2., 0., 0.])
            next_joint_state = self.cal_next_joint_state(0.05)

        if self.state == 1 or self.state == 3 or self.state == 5 or self.state == 7:
            for i in range(9):
                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                         next_joint_state[i], force=5 * 240.)

            current_joint_state = np.array([self.bullet_client.getJointState(self.panda, i)[0] for i in range(9)])
            if np.sqrt(np.sum((current_joint_state - np.array(self.path[self.path_idx])) ** 2)) < 0.08:
                self.path_idx += 1
                self.diff_joint_state = np.zeros(9)

        if (self.state == 4 or self.state == 6) and self.state_t > 0.2:
            for i in [9, 10]:
                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                         self.finger_target, force=10)

    def get_path(self, des_pos, des_orn, algorithm='prm'):
        current_joint_state = [self.bullet_client.getJointState(self.panda, i)[0] for i in range(9)]
        quaternion_orn = self.bullet_client.getQuaternionFromEuler(des_orn)
        des_joint_state = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex,
                                                                        des_pos, quaternion_orn, ll, ul, jr, rp,
                                                                        maxNumIterations=20)
        if algorithm == 'rrt_star':
            return rrt_star(current_joint_state, des_joint_state, get_distance, self.get_sample, self.get_extend,
                            self.get_collision, 1.0, max_iterations=200)
        elif algorithm == 'rrt_connect':
            return rrt_connect(current_joint_state, des_joint_state, get_distance, self.get_sample, self.get_extend,
                               self.get_collision)
        elif algorithm == 'rrt':
            return rrt(current_joint_state, des_joint_state, get_distance, self.get_sample, self.get_extend,
                       self.get_collision)
        elif algorithm == 'prm':
            result = prm(current_joint_state, des_joint_state, get_distance, self.get_sample, self.get_extend,
                         self.get_collision)
            # result.append(des_joint_state)
            print(len(result))
            return result
        else:
            return list()

    def cal_next_joint_state(self, speed):
        current_joint_state = np.array([self.bullet_client.getJointState(self.panda, i)[0] for i in range(9)])
        if np.all(self.diff_joint_state == 0):
            self.diff_joint_state = self.path[self.path_idx] - current_joint_state
        return current_joint_state + self.diff_joint_state * speed

    def get_grasp_pos(self):
        width = 640  # 图像宽度
        height = 480  # 图像高度

        fov = 49  # 相机视角
        aspect = width / height  # 宽高比
        near = 0.01  # 最近拍摄距离
        far = 2.0  # 最远拍摄距离

        # cameraPos = self.prev_pos  # 相机位置
        cameraPos = [0, 0.6, -0.6]
        targetPos = [0, 0, -0.6]  # 目标位置，与相机位置之间的向量构成相机朝向
        cameraUpPos = [1, 0, 0]  # 相机顶端朝向

        viewMatrix = self.bullet_client.computeViewMatrix(
            cameraEyePosition=cameraPos,
            cameraTargetPosition=targetPos,
            cameraUpVector=cameraUpPos,
            physicsClientId=0
        )  # 计算视角矩阵
        projection_matrix = self.bullet_client.computeProjectionMatrixFOV(fov, aspect, near, far)  # 计算投影矩阵
        images = self.bullet_client.getCameraImage(width, height, viewMatrix, projection_matrix,
                                                   renderer=self.bullet_client.ER_BULLET_HARDWARE_OPENGL)

        depth_data = np.reshape(images[3], (height, width, 1))
        depth_data = far * near / (far - (far - near) * depth_data)
        # depth_data = np.load('depth_0.npy')
        depth_im = DepthImage(depth_data, frame=self.camera_intr.frame)
        color_im = ColorImage(np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8),
                              frame=self.camera_intr.frame)

        # seg_mask = depth_im.invalid_pixel_mask().inverse()
        seg_mask = np.reshape(images[4], (height, width, 1))
        seg_mask[seg_mask > 0] = 255
        seg_mask[seg_mask <= 0] = 0
        seg_mask = BinaryImage(seg_mask.astype(np.uint8))
        valid_px_mask = depth_im.invalid_pixel_mask().inverse()
        seg_mask = seg_mask.mask_binary(valid_px_mask)

        depth_im = depth_im.inpaint(rescale_factor=self.inpaint_rescale_factor)

        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        state = RgbdImageState(rgbd_im, self.camera_intr, segmask=seg_mask)
        action = self.policy(state)

        if self.policy_config["vis"]["final_grasp"]:
            vis.figure(size=(10, 10))
            vis.imshow(rgbd_im.depth,
                       vmin=self.policy_config["vis"]["vmin"],
                       vmax=self.policy_config["vis"]["vmax"])
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
                action.grasp.depth, action.q_value))
            vis.show()

        pose = action.grasp.pose(grasp_approach_dir=np.array([0, -1, 0])).position
        x_axis = action.grasp.pose().x_axis
        y_axis = action.grasp.pose().y_axis
        z_axis = action.grasp.pose().z_axis
        pose_world = pose[0] * x_axis + pose[1] * y_axis + pose[2] * z_axis
        angle = (math.pi - action.grasp.angle % math.pi) % math.pi
        if angle > math.pi / 2:
            angle = angle - math.pi
        # return [cameraPos[0] - pose[1], 0.0, cameraPos[2] - pose[0], angle]
        return [cameraPos[0] - pose[1], 0.0, cameraPos[2] + pose[0], angle]
        # return [-pose_world[0], pose_world[2], pose_world[1], angle]
        # return [cameraPos[0] + pose_world[0], cameraPos[1] + pose_world[1], cameraPos[2] + pose_world[2], angle]
