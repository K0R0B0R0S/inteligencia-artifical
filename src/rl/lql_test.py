import argparse
import gymnasium as gym
from lql import QLearningAgentLinear
from blackjack_feature_extractor import Actions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--is_slippery", action="store_true", help="Set environment to be slippery")
    args = parser.parse_args()
    assert args.num_episodes > 0

    agent = QLearningAgentLinear.load_agent(args.env_name + "-lql-agent.pkl")

    if args.env_name == "FrozenLake-v1":
        env_test = gym.make(args.env_name, is_slippery=args.is_slippery, render_mode="human" if args.render else None).env
    else:
        env_test = gym.make(args.env_name, render_mode="human" if args.render else None).env

    total_actions, total_rewards = 0, 0

    for episode in range(args.num_episodes):
        state, _ = env_test.reset()
        num_actions = 0
        reward = 0
        
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.policy(state)
            state, reward, terminated, truncated, info = env_test.step(action)
            num_actions += 1

        total_rewards += reward
        total_actions += num_actions
        
    print("***Results***********************")
    print(f"Average episode length: {total_actions / args.num_episodes}")
    print(f"Average rewards: {total_rewards / args.num_episodes}")
    print("**********************************")

    env_test.close()