import pybullet as p
import math

from pb_ompl import PbOMPLRobot


class MyPlanarRobot(PbOMPLRobot):
    def __init__(self, id) -> None:
        self.id = id
        self.num_dim = 7
        self.joint_idx = [0, 1, 2, 3, 4, 5, 6]
        self.reset()

        self.joint_bounds = []
        # self.joint_bounds.append([math.radians(-90), math.radians(90)])
        # self.joint_bounds.append([math.radians(-90), math.radians(90)])
        # self.joint_bounds.append([math.radians(-90), math.radians(90)])
        # self.joint_bounds.append([math.radians(-160), math.radians(-90)])
        # self.joint_bounds.append([math.radians(-90), math.radians(90)])
        # self.joint_bounds.append([math.radians(90), math.radians(180)])
        # self.joint_bounds.append([math.radians(90), math.radians(180)])
        self.joint_bounds.append([math.radians(-180), math.radians(180)])
        self.joint_bounds.append([math.radians(-180), math.radians(180)])
        self.joint_bounds.append([math.radians(-180), math.radians(180)])
        self.joint_bounds.append([math.radians(-180), math.radians(180)])
        self.joint_bounds.append([math.radians(-180), math.radians(180)])
        self.joint_bounds.append([math.radians(-180), math.radians(180)])
        self.joint_bounds.append([math.radians(-180), math.radians(180)])

        c = p.createConstraint(self.id, 9, self.id, 10, jointType=p.JOINT_GEAR,
                                                jointAxis=[1, 0, 0], parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        for i in range(7):
            p.changeDynamics(self.id, i, linearDamping=0, angularDamping=0)

    def get_joint_bounds(self):
        return self.joint_bounds

    def get_cur_state(self):
        return self.state

    def set_state(self, state):
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def reset(self):
        self._set_joint_positions(self.joint_idx, [0.21, -0.03, -0.20, -1.99, -0.01, 1.98, 0.80])
        self.state = [0.21, -0.03, -0.20, -1.99, -0.01, 1.98, 0.80]

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            p.resetJointState(self.id, joint, value, targetVelocity=0)
