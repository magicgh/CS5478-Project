import pybullet as p
import numpy as np
import math


MAX_ITERS = 10000
delta_q = 2


def semiRandomSample(steer_goal_p, q_goal):
    # with steer_goal_p probability return q_goal
    # with 1 - steer_goal_p return a uniform sample
    uniform_sample = np.array([np.random.uniform(-np.pi/2, np.pi/2),
                            np.random.uniform(-np.pi/2, np.pi/2),
                            np.random.uniform(-np.pi/2, np.pi/2),
                            np.random.uniform(-np.pi/2, np.pi/2),
                            np.random.uniform(-np.pi/2, np.pi/2),
                            np.random.uniform(-np.pi/2, np.pi/2),
                            np.random.uniform(-np.pi/2, np.pi/2)
                            ],)
                        
    res = np.array([0, 1])
    output = np.random.choice(res, p = [steer_goal_p, 1-steer_goal_p])
    if output == 0:
        return q_goal
    return uniform_sample
    
def L1(v1, v2): # L1 computation used for nearest() func and distance computations in general
    res = 0
    for i in range(7):
        res += abs(v1[i] - v2[i])
    return res
    
def L2(v1, v2):
    res = 0
    for i in range(7):
        res += (abs(v1[i] - v2[i]))**2
    return math.sqrt(res)
    
def nearest(vertices, q_rand):
    # uses L2 distance to determine the nearest vertex in the current graph to q_rand

    min_dist = L2(vertices[0], q_rand)
    nearest_vertex = vertices[0]
    for v in vertices:
        curr_dist = L2(v, q_rand)
        if curr_dist < min_dist:
            min_dist = curr_dist
            nearest_vertex = v
            
    return nearest_vertex
    
def steer(q_nearest, q_rand, delta_q): # calculation of steer using HW3 Concept Review notes

    coefficient = delta_q / L2(q_rand, q_nearest)
    q_new = q_nearest + (q_rand - q_nearest) * coefficient
    
    return q_new

def set_joint_positions(values, robot_joint_indices, robot_body_id):
        assert len(robot_joint_indices) == len(values)
        for joint, value in zip(robot_joint_indices, values):
            p.resetJointState(robot_body_id, joint, value)

def check_collision(q, obstacles, robot_body_id, robot_joint_indices):
    set_joint_positions(q, robot_joint_indices, robot_body_id)
    for obstacle_id in obstacles:
        closest_points = p.getClosestPoints(
            robot_body_id, obstacle_id, 0.18)
        if closest_points is not None and len(closest_points) != 0:
            # print("check_collision: ", obstacle_id)
            return True
    return False

def obstacleFree(env, q_nearest, q_new, obstacles, robot_body_id, robot_joint_indices):
    res = check_collision(q_new, obstacles, robot_body_id, robot_joint_indices)
    # uncomment the line below if we need to move the robot back to q_nearest before proceeding:
    #env.set_joint_positions(q_nearest)
    # print("obstacleFree: ", not res)
    return not res # res is true if there is a collision and vice versa if there isn't one, so we return the negation of res from obstacleFree


def createGraph(vertices, edges):
    graph = dict()
    for v in vertices:
        tup = tuple(v)
        if tup in edges.keys():
            graph[tup] = edges[tup]
        else:
            neighbors = list()
            for key, val in edges.items():
                for x in val:
                    if tup == tuple(x):
                        neighbors.append(np.asarray(key))
                        continue
            graph[tup] = neighbors
    return graph # keys are tuple versions of vertices and values are lists of vertices

def bfsShortestPath(graph, start, end):
    seen = []
    queue = [[start]]

    while queue:
        path = queue.pop(0)
        node = tuple(path[-1])
        if node not in seen:

            for neighbor in graph[node]:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
                if tuple(neighbor) == tuple(end):
                    return new_path

            seen.append(node)

    return None
    

def calculatePath(vertices, edges, start, end):
    graph = createGraph(vertices, edges)
    return bfsShortestPath(graph, start, end)
    

def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env, obstacles, robot_body_id, robot_joint_indices):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :returns path: list of configurations (joint angles) if found a path within MAX_ITERS, else None
    """
    # ========= TODO: Problem 3 ========
    # Implement RRT code here. This function should return a list of joint configurations
    # that the robot should take in order to reach q_goal starting from q_init
    
    # Use visualize_path() to visualize the edges in the exploration tree for part (b)
    
    vertices = [q_init]
    edges = dict()
    
    for i in range(1,MAX_ITERS):
        q_rand = semiRandomSample(steer_goal_p, q_goal)
        q_nearest = nearest(vertices, q_rand)
        q_new = steer(q_nearest, q_rand, delta_q)
        if obstacleFree(env, q_nearest, q_new, obstacles, robot_body_id, robot_joint_indices):
            vertices.append(q_new)
            tup = tuple(q_nearest)
            if tup not in edges.keys():
                edges[tup] = [q_new]
            else:
                edges[tup].append(q_new)
            dist = L2(q_new, q_goal)
            if dist < delta_q:
                vertices.append(q_goal)
                tup = tuple(q_new)
                if tup not in edges.keys():
                    edges[tup] = [q_goal]
                else:
                    edges[tup].append(q_goal)
                #print('edges: ')
                path = calculatePath(vertices, edges, q_init, q_goal)
                return path
        
    # ==================================
    return None