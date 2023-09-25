import gymnasium as gym
import panda_gym
from typing import Any, Dict
from panda_gym.envs.robots.panda import Panda

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance

from typing import Optional
from panda_gym.envs.core import RobotTaskEnv

class PickAndPlace(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
        obs_num: int = 3,
        obs_xy_range: float = 0.3,
        obs_z_range: float = 0.2,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.obs_num = obs_num
        self.obstacle_size = 0.04
        self.obs_range_low = np.array([-obs_xy_range / 2, -obs_xy_range / 2, 0])
        self.obs_range_high = np.array([obs_xy_range / 2, obs_xy_range / 2, obs_z_range])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        
        for i in range(self.obs_num):
            self.sim.create_box(
                body_name="obstacle_"+str(i),
                half_extents=np.ones(3) * self.object_size / 2,
                mass=0.0,
                ghost=True,
                position=np.array([0.0, 0.0, 0.05]),
                rgba_color=np.array([255, 255, 0, 0.3]),
            )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position
    
    def get_obstacle(self) -> list[np.ndarray]:
        obstacle_position_list = []
        for i in range(self.obs_num):
            obstacle_position = np.array(self.sim.get_base_position("obstacle_"+str(i)))
            obstacle_position_list.append(obstacle_position)
        return obstacle_position_list

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        obstacle_position_list = self._sample_obs()
        for i, sample_position in enumerate(obstacle_position_list):
            self.sim.set_base_pose("obstacle_"+str(i), sample_position, np.array([0.0, 0.0, 0.0, 1.0]))
            

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position
    
    def _sample_obs(self) -> list[np.ndarray]:
        """Randomize start position of obstacle."""
        obstacle_position_list = []
        for i in range(self.obs_num):
            obstacle_position = np.array([0.0, 0.0, self.obstacle_size / 2])
            noise = self.np_random.uniform(self.obs_range_low, self.obs_range_high)
            obstacle_position += noise
            obstacle_position_list.append(obstacle_position)
        return obstacle_position_list

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        obs_dist_list = []
        for dist in self.get_obstacle():
            obs_dist_list.append(distance(achieved_goal, dist))
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32) + np.array(np.min(obs_dist_list) > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32) + np.array(np.min(obs_dist_list), dtype=np.float32)
        
        

class PandaPickAndPlaceEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Rool of the camera. Defaults to 0.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlace(sim, reward_type=reward_type)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )
        
env = PandaPickAndPlaceEnv(render_mode="human")

observation, info = env.reset()

for _ in range(100000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
