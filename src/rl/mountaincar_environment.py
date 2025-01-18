import numpy as np
from environment import Environment

class MountainCarEnvironment(Environment):
    def __init__(self, env):
        super().__init__(env)
        
        self.position_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=20)
        self.velocity_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=20)

    def get_num_states(self):
        return len(self.position_space) * len(self.velocity_space)

    def get_num_actions(self):
        return self.env.action_space.n

    def get_state_id(self, state):
        position, velocity = state
        
        position_idx = np.digitize(position, self.position_space) - 1
        velocity_idx = np.digitize(velocity, self.velocity_space) - 1
        
        position_idx = np.clip(position_idx, 0, len(self.position_space) - 1)
        velocity_idx = np.clip(velocity_idx, 0, len(self.velocity_space) - 1)
        
        state_id = position_idx * len(self.velocity_space) + velocity_idx
        return state_id

    def get_random_action(self):
        return self.env.action_space.sample()

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        return state, reward, done, info
