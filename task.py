from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
import numpy as np
from typing import Any, Dict

class PickAndPlace(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
        obs_num: int = 5,
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
        self.obstacle_size = 0.05
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
                half_extents=np.ones(3) * self.obstacle_size / 2,
                mass=1.0,
                position=np.array([0.0, 0.0, self.obstacle_size/2]),
                rgba_color=np.array([0.9, 0.9, 0, 1.0]),
            )

    def get_obs(self) -> np.ndarray:
        # position of the object and obstacles
        obs_observation = np.array([self.sim.get_base_position("obstacle_"+str(i)) for i in range(self.obs_num)]).flatten()
        return np.array(obs_observation)

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
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)