import time
import numpy as np
import math

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
        self.finger_target = 0

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

        self.state_t = 0
        self.cur_state = 0
        # 0：空闲、1：移动到托盘中心上方、2：获取深度图片计算最优抓取位置、3：移动到目标上方、4：张开机械手、5：移动到目标处、6：闭合机械手、3：移动到目标上方、7：移动到结束位置
        self.states = [0, 1, 2, 3, 4, 5, 6, 3, 7]
        self.state_durations = [1, 1, 0, 1, 1, 1, 1, 1, 5]
        self.prev_pos = self.bullet_client.getLinkState(
            self.panda, pandaEndEffectorIndex, computeForwardKinematics=True)[0]
        self.graspPos = [0, 0, 0, 0]

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
        orn = self.bullet_client.getQuaternionFromEuler([math.pi / 2., 0, 0.])  # 需要根据预测结果调整第二个维度
        alpha = 0.9  # 移动速度，越接近1越慢
        if self.state == 1:
            pos = self.get_next_pos(0, 0.2, -0.6, 0.1)
            self.prev_pos = pos
        elif self.state == 2:
            self.graspPos = self.get_grasp_pos()
        elif self.state == 3:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.legos[0])  # 直接获取了某个物体的位置的渐进位置
            pos = [pos[0], alpha * self.prev_pos[1] + (1. - alpha) * 0.2, pos[2]]
            self.prev_pos = pos
        elif self.state == 4:  # 张开机械手
            self.finger_target = 0.04
        elif self.state == 5:
            pos, _ = self.bullet_client.getBasePositionAndOrientation(self.legos[0])  # 直接获取了某个物体的位置的渐进位置
            pos = [pos[0], alpha * self.prev_pos[1] + (1. - alpha) * 0.03, pos[2]]
            self.prev_pos = pos
        elif self.state == 6:  # 闭合机械手
            self.finger_target = 0.01
        elif self.state == 7:
            pos = self.get_next_pos(0, 0.2, -0.6, 0.1)
            self.prev_pos = pos

        jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll,
                                                                   ul, jr, rp, maxNumIterations=20)
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     jointPoses[i], force=5 * 240.)

        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_target, force=10)

    def get_grasp_pos(self):
        width = 1080  # 图像宽度
        height = 720  # 图像高度

        fov = 100  # 相机视角
        aspect = width / height  # 宽高比
        near = 0.01  # 最近拍摄距离
        far = 20  # 最远拍摄距离

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

        return [0, 0, 0, 0]

    def get_next_pos(self, x, y, z, alpha):
        pos = [0, 0, 0]
        pos[0] = self.prev_pos[0] * (1 - alpha) + alpha * x
        pos[1] = self.prev_pos[1] * (1 - alpha) + alpha * y
        pos[2] = self.prev_pos[2] * (1 - alpha) + alpha * z
        return pos
