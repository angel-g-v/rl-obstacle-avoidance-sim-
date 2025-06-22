import gym
from gym import spaces
import numpy as np
import cv2

class CameraEnv(gym.Env):
    def __init__(self):
        super(CameraEnv, self).__init__()

        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)  # 0 = left, 1 = straight, 2 = right

        self.steps = 0
        self.agent_x = 32
        self.obstacles = []
        self.reset()

    def reset(self):
        self.steps = 0
        self.agent_x = np.random.randint(24, 40)  # start near center with variation
        self.obstacles = [np.random.randint(10, 54) for _ in range(np.random.randint(3, 7))]
        return self._get_obs()

    def step(self, action):
        self.steps += 1

        if action == 0:
            self.agent_x -= 2
        elif action == 2:
            self.agent_x += 2
        self.agent_x = np.clip(self.agent_x, 0, 63)

        # Crash logic: near obstacle
        crash = any(abs(self.agent_x - x) < 4 for x in self.obstacles)
        done = crash or self.steps >= 200

        if crash:
            reward = -1.0
        else:
            near = any(abs(self.agent_x - x) < 8 for x in self.obstacles)
            if near and action in [0, 2]:
                reward = 1.2  # smart turning
            else:
                reward = 1.0

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(img, (self.agent_x, 60), 2, 255, -1)
        for x in self.obstacles:
            cv2.rectangle(img, (x - 2, 20), (x + 2, 30), 150, -1)
        return img.reshape(64, 64, 1)

    def render(self, mode="human"):
        img = self._get_obs().squeeze()
        cv2.imshow("Sim View", img)
        cv2.waitKey(1)
