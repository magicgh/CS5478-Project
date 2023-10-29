from typing import Optional

import numpy as np
from gymnasium import spaces

from gym_panda.robot import PyBulletRobot
from gym_panda.bullet import PyBullet
import time


class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(
            3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        # control (x, y z) if "ee", else, control the 7 joints
        n_action = 3 if self.control_type == "ee" else 7
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0,
                                  shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array(
                [87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )
        self.joint_epsilon = 1e-2
        self.revolute_joints_indices = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
        self.fingers_indices = np.array([9, 10], dtype=np.int32)
        self.neutral_joint_values = np.array(
            [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(
            self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(
            self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(
            self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(
            self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        if len(action) == 7:
            action = np.concatenate((action, [0.0]))
            
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(
                ee_displacement)
    
        elif self.control_type == "joint":
            target_arm_angles = action[:7]
    
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(
                arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        # Default control type is "joint", set position control
        target_angles = np.concatenate(
            (target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        
        self.control_joints(target_angles=target_angles, speed=0.03)

        timeout_t0 = time.time()
        
        while True:
            current_joint_state = [
                self.sim.get_joint_angle(self.body_name, i)
                for i in self.revolute_joints_indices
            ]
            if all([
                np.abs(
                    current_joint_state[i] - target_angles[i]) < self.joint_epsilon
                for i in range(len(self.revolute_joints_indices))
            ]):
                break
            if time.time()-timeout_t0 > 5:
                print(
                    "Timeout: robot is taking longer than 5s to reach the target joint state. Skipping...")
                self.reset()
                break

            self.step_simulation(1)

    def step_simulation(self, n_steps: int) -> None:
        """Step the simulation.

        Args:
            n_steps (int): Number of steps.
        """
        for _ in range(n_steps):
            self.sim.step_simulation()
    
    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * \
            0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([
                                                                                 1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def ee_position_to_target_arm_angles(self, ee_position: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector position.

        Args:
            ee_position (np.ndarray): End-effector position, as (x, y, z).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        # get the current position and the target position
        target_ee_position = ee_position.copy()

        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([
                                                                                 1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array(
            [self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate(
                (ee_position, ee_velocity, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity))
        return observation

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def set_joint_angles(self, angles: np.ndarray) -> None:
        return super().set_joint_angles(angles)
    
    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(
            self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(
            self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    def get_body_id(self):
        return self.sim.get_body_id_from_name(self.body_name)

    def get_revolute_joint_indices(self):
        return self.revolute_joints_indices