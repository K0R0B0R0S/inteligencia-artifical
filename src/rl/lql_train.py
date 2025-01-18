import argparse
import gymnasium as gym
from lql import QLearningAgentLinear
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from gymnasium.wrappers import TimeLimit

from taxi_environment import TaxiEnvironment
from blackjack_environment import BlackjackEnvironment
from cliffwalking_environment import CliffWalkingEnvironment
from frozenlake_environment import FrozenLakeEnvironment

environment_dict = {
    "Blackjack-v1": BlackjackEnvironment,
    "Taxi-v3": TaxiEnvironment,
    "CliffWalking-v0": CliffWalkingEnvironment,
    "FrozenLake-v1": FrozenLakeEnvironment
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum number of steps per training episode")
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--epsilon_decay_rate", type=float, default=0.0001, help="Decay rate for the exploration rate")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9, help="Gamma")
    parser.add_argument("--is_slippery", action="store_true", help="Set environment to be slippery")
    args = parser.parse_args()

    num_episodes = args.num_episodes
    max_steps = args.max_steps
    env_name = args.env_name
    epsilon_decay_rate = args.epsilon_decay_rate
    learning_rate = args.learning_rate
    gamma = args.gamma

    if env_name == "FrozenLake-v1":
        env = gym.make(env_name, is_slippery=args.is_slippery).env
    else:
        env = gym.make(env_name).env

    env = TimeLimit(env, max_episode_steps=args.max_steps)
    env = environment_dict[env_name](env)

    agent = QLearningAgentLinear(env, learning_rate=learning_rate, epsilon_decay_rate=epsilon_decay_rate, gamma=gamma)
    penalties_per_episode, rewards_per_episode, cumulative_successful_episodes = agent.train(num_episodes)
    agent.save(args.env_name + "-lql-agent.pkl")

    plt.subplot(2, 2, 1)
    plt.plot(savgol_filter(penalties_per_episode, 111, 2))
    plt.title(f"Penalties ({args.env_name})")

    plt.subplot(2, 2, 2)
    plt.plot(savgol_filter(rewards_per_episode, 111, 2))
    plt.title(f"Rewards ({args.env_name})")

    plt.subplot(2, 2, 3)
    plt.plot(cumulative_successful_episodes)
    plt.title(f"Successful episodes ({args.env_name})")

    plt.subplot(2, 2, 4)
    plt.plot(agent.epsilon_history)
    plt.title(f"Epsilon ({args.env_name})")

    plt.savefig(args.env_name + "-lql-results.png")
    plt.close()
