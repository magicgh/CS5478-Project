import time
import math
import random
import os
import shutil
import yaml

import pybullet as p
import numpy as np
from autolab_core import CameraIntrinsics, DepthImage, ColorImage, RgbdImage, BinaryImage
from gqcnn.grasping import CrossEntropyRobustGraspingPolicy, RgbdImageState

import matplotlib.pyplot as plt
import seaborn as sns


class GraspExp(object):
    def __init__(self, exp_num):
        p.setPhysicsEngineParameter(solverResidualThreshold=0)

        p.loadURDF("tray/traybox.urdf", [0, 0, -0.6], [0.5, -0.5, 0.5, 0.5])
        
        # Camera
        self.camera_intr = CameraIntrinsics.load(os.path.join('Grasping/primesense.intr'))
        
        # GQCNN
        with open('Grasping/gqcnn_pj.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.inpaint_rescale_factor = config["inpaint_rescale_factor"]
        self.policy_config = config["policy"]
        self.policy_config["metric"]["gqcnn_model"] = os.path.join('Grasping/models/GQCNN-2.1')
        
        # Result
        self.exp_num = exp_num
        self.result = {'time': [], 'distance': []}

    def test(self):
        for i in range(self.exp_num):
            lego = self.add_lego()
            start_time = time.time()
            predict_pos = self.get_grasp_pos()
            end_time = time.time()
            true_pos = p.getBasePositionAndOrientation(lego)[0]
            distance = math.sqrt((predict_pos[0] - true_pos[0]) ** 2 + (predict_pos[2] - true_pos[2]) ** 2)
            if i:
                self.result['time'].append(end_time - start_time)
                self.result['distance'].append(distance)
            p.removeBody(lego)
        
        self.save_result()

    def add_lego(self):
        pos = self.generate_random_coordinate()
        return p.loadURDF("lego/lego.urdf", pos, [-0.5, -0.5, -0.5, 0.5], globalScaling=1.5)
        
    def generate_random_coordinate(self):
        random.seed(time.time())
        x = random.uniform(-0.15, 0.15)
        y = 0.02223
        z = random.uniform(-0.75, -0.45)
        
        return (x, y, z)

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

    def save_result(self):
        if os.path.exists('exp_results/grasp'):
            shutil.rmtree('exp_results/grasp')
        os.makedirs('exp_results/grasp')
        
        self.save_text('Grasp Time', self.result['time'])
        self.save_text('Grasp Distance', self.result['distance'])
        
        self.save_plot('Grasp Time Distribution', 'Time', self.result['time'])
        self.save_plot('Grasp Distance Distribution', 'Distance', self.result['distance'])

    def save_text(self, title, value):
        mean = sum(value) / len(value)
        variance = sum((x - mean) ** 2 for x in value) / len(value)
        with open('exp_results/grasp/results.txt', 'a') as f:
            f.write(title + ':\n')
            f.write('\tmean: ' + str(mean) + '\n')
            f.write('\tvariance: ' + str(variance) + '\n')

    def save_plot(self, title, label, value):
        plt.figure()
        sns.boxplot(data=value, orient='v')
        plt.title(title)
        plt.xlabel('GQCNN')
        plt.ylabel(label)
        plt.savefig('exp_results/grasp/' + title + '.png')
        plt.close()
