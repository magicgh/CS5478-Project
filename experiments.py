import time

import pybullet as p
import pybullet_data as pd

from motion_plan_exp import MotionPlanExp
from grasp_exp import GraspExp
from motion_plan_and_grasp_exp import MotionPlanAndGraspExp

fps = 240.
timeStep = 1. / fps

p.connect(p.DIRECT)

p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0.35, 0.13, 0])
p.setAdditionalSearchPath(pd.getDataPath())

p.setTimeStep(timeStep)
p.setGravity(0, -9.8, 0)

# motion_plan_exp = MotionPlanExp(20)
# motion_plan_exp.test()

# grasp_exp = GraspExp(20)
# grasp_exp.test()

motion_plan_and_grasp_exp = MotionPlanAndGraspExp(20)
while True:
    if motion_plan_and_grasp_exp.step():
        break
    p.stepSimulation()
    time.sleep(0.0)
