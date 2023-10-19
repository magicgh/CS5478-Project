import time
import numpy as np
import math

from autolab_core import YamlConfig, CameraIntrinsics, DepthImage, ColorImage, RgbdImage
from visualization import Visualizer2D as vis
from gqcnn.grasping import CrossEntropyRobustGraspingPolicy, RgbdImageState

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
        self.legos = []

        self.bullet_client.loadURDF("tray/traybox.urdf",
                                    [0 + offset[0], 0 + offset[1], -0.6 + offset[2]],
                                    [-0.5, -0.5, -0.5, 0.5],
                                    flags=flags)

        self.legos.append(self.bullet_client.loadURDF("lego/lego.urdf",
                                                      np.array([0.1, 0.3, -0.5]) + self.offset,
                                                      flags=flags))
        self.bullet_client.changeVisualShape(self.legos[0], -1, rgbaColor=[1, 0, 0, 1])  # 修改目标物体的颜色
        self.legos.append(self.bullet_client.loadURDF("lego/lego.urdf",
                                                      np.array([-0.1, 0.3, -0.5]) + self.offset,
                                                      flags=flags))
        self.legos.append(self.bullet_client.loadURDF("lego/lego.urdf",
                                                      np.array([0.1, 0.3, -0.7]) + self.offset,
                                                      flags=flags))

        self.sphereId = self.bullet_client.loadURDF("sphere_small.urdf",
                                                    np.array([0, 0.3, -0.6]) + self.offset,
                                                    flags=flags)
        self.bullet_client.loadURDF("sphere_small.urdf",
                                    np.array([0, 0.3, -0.5]) + self.offset,
                                    flags=flags)
        self.bullet_client.loadURDF("sphere_small.urdf",
                                    np.array([0, 0.3, -0.7]) + self.offset,
                                    flags=flags)

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
            jointType = info[2]
            if jointType == self.bullet_client.JOINT_PRISMATIC:
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
            if jointType == self.bullet_client.JOINT_REVOLUTE:
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1

        self.prev_pos = self.bullet_client.getLinkState(
            self.panda, pandaEndEffectorIndex, computeForwardKinematics=True)[0]
        self.prev_orn = self.bullet_client.getQuaternionFromEuler([math.pi / 2., 0, 0.])
        self.graspPos = np.array([0, 0, 0, 0])
        self.finger_target = 0

        self.state_t = 0
        self.cur_state = 0
        # 0：空闲、1：移动到托盘中心上方、2：获取深度图片计算最优抓取位置、3：移动到目标上方、4：张开机械手、5：移动到目标处、6：闭合机械手、3：移动到目标上方、7：移动到结束位置
        self.states = [0, 1, 2, 3, 4, 5, 6, 3, 7, 4]
        self.state_durations = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

        config = YamlConfig('gqcnn_pj.yaml')
        self.inpaint_rescale_factor = config["inpaint_rescale_factor"]
        self.policy_config = config["policy"]
        self.policy_config["metric"]["gqcnn_model"] = 'models/GQCNN-2.1'
        self.policy = CrossEntropyRobustGraspingPolicy(self.policy_config)

        self.camera_intr = CameraIntrinsics.load('primesense.intr')

    def update_state(self):
        self.state_t += self.control_dt
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
            self.state_t = 0
            self.state = self.states[self.cur_state]
            print("self.state=", self.state)

    def step(self):
        self.bullet_client.submitProfileTiming("step")
        self.update_state()

        pos = self.prev_pos
        orn = [math.pi / 2., 0, 0.]  # 需要根据预测结果调整第二个维度
        alpha = 0.01  # 移动速度，越接近1越慢
        if self.state == 1:
            pos = self.get_next_pos(0, 0.4, -0.6, alpha)
        elif self.state == 2:
            self.graspPos = self.get_grasp_pos()
        elif self.state == 3:
            pos = self.get_next_pos(self.graspPos[0], 0.4, self.graspPos[2], alpha)
            orn = self.get_next_orn(math.pi / 2., 0., 0., alpha)
        elif self.state == 4:  # 张开机械手
            self.finger_target = 0.04
        elif self.state == 5:
            # gripper_height = 0.034
            pos = self.get_next_pos(self.graspPos[0], self.graspPos[1] + 0.034, self.graspPos[2], alpha)
            orn = self.get_next_orn(math.pi / 2., self.graspPos[3], 0., alpha)
        elif self.state == 6:  # 闭合机械手
            self.finger_target = 0.01
        elif self.state == 7:
            pos = self.get_next_pos(0.6, 0.4, 0, alpha)

        if pos != self.prev_pos:
            self.prev_pos = pos
            self.prev_orn = orn
            quaternion_orn = self.bullet_client.getQuaternionFromEuler(orn)
            jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos,
                                                                       quaternion_orn, ll, ul, jr, rp,
                                                                       maxNumIterations=20)
            for i in range(pandaNumDofs):
                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                         jointPoses[i], force=5 * 240.)

        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_target, force=10)

    def get_grasp_pos(self):
        width = 640  # 图像宽度
        height = 480  # 图像高度

        fov = 49  # 相机视角
        aspect = width / height  # 宽高比
        near = 0.01  # 最近拍摄距离
        far = 1.0  # 最远拍摄距离

        cameraPos = self.prev_pos  # 相机位置
        targetPos = [self.prev_pos[0], 0, self.prev_pos[2]]  # 目标位置，与相机位置之间的向量构成相机朝向
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
        segmask = depth_im.invalid_pixel_mask().inverse()

        depth_im = depth_im.inpaint(rescale_factor=self.inpaint_rescale_factor)

        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        state = RgbdImageState(rgbd_im, self.camera_intr, segmask=segmask)
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

        pose = action.grasp.pose().position
        angle = (math.pi - action.grasp.angle % math.pi) % math.pi
        if angle > math.pi / 2:
            angle = angle - math.pi
        return [self.prev_pos[0] - pose[0], self.prev_pos[1] - pose[2], self.prev_pos[2] + pose[1], angle]

    def get_next_pos(self, x, y, z, alpha):
        pos = [0, 0, 0]
        pos[0] = self.prev_pos[0] * (1 - alpha) + alpha * x
        pos[1] = self.prev_pos[1] * (1 - alpha) + alpha * y
        pos[2] = self.prev_pos[2] * (1 - alpha) + alpha * z
        return pos

    def get_next_orn(self, x, y, z, alpha):
        orn = [0, 0, 0]
        orn[0] = self.prev_orn[0] * (1 - alpha) + alpha * x
        orn[1] = self.prev_orn[1] * (1 - alpha) + alpha * y
        orn[2] = self.prev_orn[2] * (1 - alpha) + alpha * z
        return orn
