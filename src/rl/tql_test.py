import argparse
import gymnasium as gym
from tql import QLearningAgentTabular

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Taxi-v3", help="Environment name")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes")
    args = parser.parse_args()
    assert args.num_episodes > 0

    agent = QLearningAgentTabular.load_agent(args.env_name + "-tql-agent.pkl")
    env_test = gym.make(args.env_name, render_mode="human")

    total_actions, total_rewards = 0, 0

    for episode in range(args.num_episodes):
        observation, info = env_test.reset()
        state = observation
        num_actions = 0
        reward = 0
        
        terminated = False
        truncated = False

        while not (terminated or truncated):
            env_test.render() 
            action = agent.choose_action(state, is_in_exploration_mode=False)
            state, reward, terminated, truncated, info = env_test.step(action)
            num_actions += 1
            total_rewards += reward

        total_actions += num_actions

    print("***Results***********************")
    print(f"Average episode length: {total_actions / args.num_episodes}")
    print(f"Average rewards: {total_rewards / args.num_episodes}")
    print("**********************************")

    env_test.close()