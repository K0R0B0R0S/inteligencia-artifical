from timeit import default_timer as timer
import pickle
import numpy as np
from environment import Environment

from taxi_feature_extractor import TaxiFeatureExtractor
from blackjack_feature_extractor import BlackjackFeatureExtractor
from frozenlake_feature_extractor import FrozenLakeFeatureExtractor
from cliffwalking_feature_extractor import CliffWalkingFeatureExtractor

feature_extractors_dict = {
  "Blackjack-v1": BlackjackFeatureExtractor,
  "Taxi-v3": TaxiFeatureExtractor,
  "CliffWalking-v0": CliffWalkingFeatureExtractor,
  "FrozenLake-v1": FrozenLakeFeatureExtractor
}

class QLearningAgentLinear:

  def __init__(self, 
               gym_env: Environment, 
               epsilon_decay_rate, 
               learning_rate, 
               gamma):
    self.env = gym_env
    env_name = self.env.get_id()
    print(gym_env)
    print(gym_env.env)

    print("Calling fex constructor...")
    print(feature_extractors_dict[env_name])
    self.fex = feature_extractors_dict[env_name](gym_env.env)

    self.w = np.random.rand(self.fex.get_num_features() + 1)
    
    self.steps = 0

    self.epsilon = .5
    self.max_epsilon = 0.5
    self.min_epsilon = 0.1
    self.epsilon_decay_rate = epsilon_decay_rate
    self.learning_rate = learning_rate
    self.gamma = gamma 
    self.epsilon_history = []

  def choose_action(self, state, is_in_exploration_mode = True):
    exploration_tradeoff = np.random.uniform(0, 1)
    if is_in_exploration_mode and exploration_tradeoff < self.epsilon:
      # exploration
      action = self.env.get_random_action()
    else:
      action = self.policy(state)
    return action

  def policy(self, state):
    # exploitation (taking the biggest Q value for this state)
    return self.__get_action_and_value(state)[0]

  def get_value(self, state):
    return self.__get_action_and_value(state)[1]

  def get_qvalues(self, state):
    q_values = {}
    for action in range(self.env.action_space.n):
      q_values[action]  = self.get_qvalue(state, action)
    return q_values

  def get_features(self, state, action):
    feature_vector = self.fex.get_features(state, action)
    feature_vector = feature_vector.reshape(1, -1)
    feature_vector = feature_vector.flatten()    
    steps_feature = np.array([np.log10(self.steps+1)])
    feature_vector = np.concatenate([feature_vector, steps_feature])
    return feature_vector

  def get_qvalue(self, state, action):
    features = self.get_features(state, action)
    return np.dot(self.w, features)

  def __get_action_and_value(self, state):
    max_qvalue = float("-inf")
    best_action = 0
    for action in range(self.env.get_num_actions()):
      q_value  = self.get_qvalue(state, action)
      if q_value > max_qvalue:
        max_qvalue = q_value
        best_action = action
    return [best_action, max_qvalue]

  def update(self, state, action, reward, next_state):
    next_state_value = self.get_value(next_state)
    if self.fex.is_terminal_state(next_state):
      next_state_value = 0
    difference = (reward + (self.gamma * next_state_value)) - self.get_qvalue(state, action)
    if difference < -100:
       difference = -100
    if difference > 100:
       difference = 100
    features = self.get_features(state, action)
    new_w = self.w + self.learning_rate * difference * features
    self.w = new_w

  def train(self, num_episodes: int):

    successful_episodes = 0

    rewards_per_episode = []
    penalties_per_episode = []
    cumulative_successful_episodes = []

    start_time = timer()  # Record the start time

    for episode in range(num_episodes):
      terminated = False
      truncated = False

      state, _ = self.env.reset()

      total_rewards = 0
      self.steps = 0

      total_penalties = 0

      while not (terminated or truncated):
        self.steps += 1
        action = self.choose_action(state)
        new_state, reward, terminated, truncated, _ = self.env.step(action)

        if reward == -10:
            total_penalties += 1

        self.update(state, action, reward, new_state)
        total_rewards += reward

        assert not np.isnan(self.get_weights()).any()
        assert not (any(self.get_weights() > 1e6)), f"Weigths explosion: {self.get_weights()}"

        if (terminated or truncated):
          # Reduce epsilon to decrease the exploration over time
          self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
            np.exp(-self.epsilon_decay_rate * episode)
          self.epsilon_history.append(self.epsilon)
          if terminated:
            # assert reward == +20
            # assert new_state in [0, 85, 410, 475]
            successful_episodes += 1
        
        state = new_state

      rewards_per_episode.append(total_rewards)
      penalties_per_episode.append(total_penalties)
      cumulative_successful_episodes.append(successful_episodes)

      if episode % 50 == 0:
        end_time = timer()  # Record the end time
        execution_time = end_time - start_time
        print("Episode# %d/%d (%d successful)" % (episode, num_episodes, successful_episodes))
        print(f"\tElapsed time (from first episode): {execution_time:.2f}s")
        print("\tTotal rewards %d" % total_rewards)
        print("\tTotal steps: %d" % self.steps)
        print("\tCurrent epsilon: %.4f" % self.epsilon)
        print("\tTotal penalties: %d" % total_penalties)
        print("\tw:", self.w)
        # print("\tCurrent weights: %s" % self.get_weights())
        print()

    return penalties_per_episode, rewards_per_episode, cumulative_successful_episodes

  def get_weights(self):
    return self.w

  def save(self, filename):
    # open a file, where you ant to store the data
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(self, file)

    # close the file
    file.close()

  @staticmethod
  def load_agent(filename):
    # open a file, where you stored the pickled data
    file = open(filename, 'rb')

    # dump information to that file
    agent = pickle.load(file)

    return agent
