import gymnasium as gym
import minigrid
import torch
import numpy as np
from tqdm import tqdm

from agent.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.preprocess import preprocess_state


# Environment

env = gym.make(
    "MiniGrid-Empty-8x8-v0"
)

n_actions = env.action_space.n

agent = DQNAgent(n_actions)

buffer = ReplayBuffer(
    capacity=10000
)


# Hyperparameters

episodes = 200

batch_size = 32

target_update = 10

rewards_history = []

best_reward = -999999

# Training Loop


for episode in tqdm(range(episodes)):

    obs, info = env.reset()

    state = preprocess_state(obs)

    done = False

    total_reward = 0

    while not done:

        action = agent.select_action(
            state
        )

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        next_state = preprocess_state(
            next_obs
        )

        buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        total_reward += reward


        # Learn


        if len(buffer) > 1000:

            batch = buffer.sample(
                batch_size
            )

            loss = agent.train_step(
                batch
            )

    rewards_history.append(
        total_reward
    )

    # Save rewards after every episode
    np.save(
        "rewards.npy",
        np.array(rewards_history)
    )


    # Epsilon Decay


    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= (
            agent.epsilon_decay
        )

    # Update Target Net


    if episode % target_update == 0:

        agent.update_target_network()


    # Save Best Model


    if total_reward > best_reward:

        best_reward = total_reward

        torch.save(
            agent.model.state_dict(),
            "best_model.pth"
        )

    print(
        f"Episode {episode} "
        f"Reward {total_reward:.2f} "
        f"Epsilon {agent.epsilon:.3f}"
    )


# Save Reward History


np.save(
    "rewards.npy",
    np.array(rewards_history)
)

print("Training Finished")