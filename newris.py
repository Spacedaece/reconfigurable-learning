import numpy as np
import math
import scipy
import gym
from gym import spaces
from collections import deque
import random as rand # maybe?

class RISenv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):  
        super(RISenv, self).__init__()
        self.receiver = np.array([5, 5, 1]) 
        self.transmitter = np.array([3, 3, 2])
        self.reference = np.array([0, 0, 0])
        self.observation_space = spaces.Box(low=0, high=1, shape=(100,17))
        self.reward_scale = 10**(1) # Scaling factor for reward
        self.ep_length = 0
        self.ep_reward = 0
        self.total_reward = 0
        self.gamma_value = 1
        self.M = 10 #m rows
        self.N = 10 #n columns
        self.dx = 1 #width of unit cell
        self.dy = 1 #length of unit cell
        self.j = 5 #??
        self.λ = 3.8 #wavelength
        self.Γ = 2 #gamma n,m
        self.pt = 20 #power transmitted
        self.action_space = spaces.Box(low=0.01, high=2*math.pi, shape=(self.M * self.N,), dtype=np.float32)

        self.reset()  # Initialize the state



    def calculate_angle1(self, unit_cell):
        vector = self.transmitter - unit_cell #🤔check
        azimuth = np.arctan2(vector[1], vector[0])
        distance_xy = np.linalg.norm(vector[:2])
        elevation = np.arctan2(vector[2], distance_xy)
        distance = np.linalg.norm(vector)
        return azimuth, elevation, distance

    def calculate_angle2(self, unit_cell):
        vector = self.receiver - unit_cell
        azimuth = np.arctan2(vector[1], vector[0])
        distance_xy = np.linalg.norm(vector[:2])
        elevation = np.arctan2(vector[2], distance_xy)
        distance = np.linalg.norm(vector)
        return azimuth, elevation, distance


    def Ftx(self, theta, phi):
        modeltx = "paper1"
        if modeltx == "paper1":
            F = (math.cos(theta + phi)**3)
        elif modeltx == "simple":
            F = 1
        return F

    def Fu(self, theta, phi):
        F = math.cos(theta + phi)**3
        return F

    def Frx(self, theta, phi):
        modelrx = "paper1"
        if modelrx == "paper1":
            F = (math.cos(theta + phi)**3)
        elif modelrx == "simple":
            F = 1
        return F

    def F_combined(self, unit_cell):
        F_tx = self.Ftx(self.calculate_angle1(unit_cell)[1], self.calculate_angle1(unit_cell)[0])
        F_t = self.Fu(self.calculate_angle1(unit_cell)[1], self.calculate_angle1(unit_cell)[0])
        F_r = self.Fu(self.calculate_angle2(unit_cell)[1], self.calculate_angle2(unit_cell)[0])
        F_rx = self.Frx(self.calculate_angle2(unit_cell)[1], self.calculate_angle2(unit_cell)[0])
        return F_tx * F_t * F_r * F_rx
    
    def gain_calculation(self):
        self.gt = ((4*math.pi)/scipy.integrate.nquad(self.Ftx, [[0, math.pi],[0, 2*math.pi]])[0]) 
        self.g = ((4*math.pi)/scipy.integrate.nquad(self.Fu, [[0, math.pi],[0, 2*math.pi]])[0])
        self.gr = ((4*math.pi)/scipy.integrate.nquad(self.Frx, [[0, math.pi],[0, 2*math.pi]])[0])
        return self.gt, self.g, self.gr
    
    def step(self, action): #maybe create another whole function for everything in step?
        self.observation = np.zeros((self.M * self.N, 17))
        for x in range(self.M):
            for y in range(self.N):
                unit_cell = np.array([x, y, 0])
                self.gain_calculation()
                self.action = action[x * self.N + y]  # Get the corresponding action for this unit cell
                if np.isnan(action[x*self.N +y]):
                    self.gamma_value = math.pi
                else:
                    self.gamma_value = abs(self.Γ) * (math.e**(self.j * action))
                M = 2 - 2 * self.M
                N = 2 - 2 * self.N
                sigma = 0
                for i in range(int(self.M)):
                    for j in range(int(self.N)):
                        angle1_distance = self.calculate_angle1(unit_cell)[2]
                        angle2_distance = self.calculate_angle2(unit_cell)[2] #representing distances for each
                        sqrt_combined = math.sqrt(self.F_combined(unit_cell))
                        if angle1_distance * angle2_distance != 0:
                            sigma += (sqrt_combined * self.gamma_value / (angle1_distance * angle2_distance)) * math.e ** (
                                (-self.j * 2 * math.pi * ((angle1_distance + angle2_distance)) / self.λ))
                power_received = (
                    self.pt
                    * (
                        (self.gt * self.gr * self.g * self.dx * self.dy * (self.λ ** 2))
                        / (64 * (math.pi**3))
                    )
                ) * (sigma)
                self.reward = self.reward_scale * power_received # check
                self.ep_length += 1
                self.ep_reward += self.reward
                self.done = False
                info = {}        
                self.total_reward += self.reward
                print([(x, y), unit_cell, self.action, self.reward])
                #-------------------------------------

                # No need to concatenate, just append the observation to the list
                unit_cell = np.array([x, y, 0])
                azimuth1, elevation1, distance1 = self.calculate_angle1(unit_cell)
                azimuth2, elevation2, distance2 = self.calculate_angle2(unit_cell)
                observation = [x, y, azimuth1, elevation1, distance1, azimuth2, elevation2, distance2, self.M, self.N, *self.transmitter, *self.receiver, self.F_combined(unit_cell)]
                self.observation[x * self.N + y] = observation
    # Return the observations as a list of arrays
        return self.observation, self.reward, self.done, info



    def reset(self):
        self.observation = np.zeros((self.M * self.N, 17))  # Initialize a NumPy array to store the observations
        for x in range(self.M):
            for y in range(self.N):
                unit_cell = np.array([x, y, 0])
                azimuth1, elevation1, distance1 = self.calculate_angle1(unit_cell)
                azimuth2, elevation2, distance2 = self.calculate_angle2(unit_cell)
                observation = [x, y, azimuth1, elevation1, distance1, azimuth2, elevation2, distance2, self.M, self.N, *self.transmitter, *self.receiver, self.F_combined(unit_cell)]
                self.observation[x * self.N + y] = observation
        return self.observation

    def render(self, mode='human'):
        pass

# ----------------------------------------------------------------------------------------------------------------------------------
from stable_baselines3 import PPO, A2C
import os
from newris import RISenv
import time
from stable_baselines3.common.callbacks import BaseCallback

class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(CustomTensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Access the environment's episode length and cumulative reward
        ep_length = self.training_env.envs[0].ep_length
        ep_reward = self.training_env.envs[0].ep_reward

        # Log the episode length and reward to TensorBoard
        self.logger.record('rollout/ep_len_mean', ep_length)
        self.logger.record('rollout/ep_rew_mean', ep_reward)

        return True

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir): #saves model directory
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = RISenv()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 100
iters = 0

callback = CustomTensorboardCallback()

while True:
    iters += 1
    obs = env.reset()
    actions = []

    for _ in range(TIMESTEPS):
        action, _ = model.predict(obs)
        actions.append(action)
        obs, _, _, _ = env.step(action)
    actions = np.array(actions)

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO", callback=callback)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")


