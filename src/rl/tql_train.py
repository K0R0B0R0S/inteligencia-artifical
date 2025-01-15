from timeit import default_timer as timer
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tql import QLearningAgentTabular

from taxi_environment import TaxiEnvironment
from blackjack_environment import BlackjackEnvironment

environment_dict = {
    "Blackjack-v1": BlackjackEnvironment,
    "Taxi-v3": TaxiEnvironment
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=6000, help="Number of episodes")
    parser.add_argument("--decay_rate", type=float, default=0.0001, help="Decay rate")
    parser.add_argument("--learning_rate", type=float, default=0.7, help="Learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.618, help="Discount factor (gamma)")
    args = parser.parse_args()

    num_episodes = args.num_episodes
    env_name = args.env_name
    decay_rate = args.decay_rate
    learning_rate = args.learning_rate
    gamma = args.gamma

    env = gym.make(env_name).env

    env = environment_dict[env_name](env)

    agent = QLearningAgentTabular(
        env=env,
        decay_rate=decay_rate,
        learning_rate=learning_rate,
        gamma=gamma
    )
    rewards = agent.train(num_episodes)

    agent.save(args.env_name + "-tql-agent.pkl")

    plt.plot(savgol_filter(rewards, 1001, 2))
    plt.title(f"Curva de aprendizado suavizada ({args.env_name})")
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa total')
    plt.savefig(args.env_name + "-tql-learning_curve.png")
    plt.close()

    plt.plot(agent.epsilons_)
    plt.title(f"Decaimento do valor de $\epsilon$ ({args.env_name})")
    plt.xlabel('Episódio')
    plt.ylabel('$\epsilon$')
    plt.savefig(args.env_name + "-tql-epsilons.png")
    plt.close()
