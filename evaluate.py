import gymnasium as gym
import minigrid
import torch
import numpy as np

from models.dqn import DQN
from utils.preprocess import preprocess_state

ENV_NAME = "MiniGrid-Empty-8x8-v0"

env = gym.make(ENV_NAME)

n_actions = env.action_space.n

model = DQN(n_actions)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location="cpu"
    )
)

print(
    "Model Loaded Successfully!"
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: {total_params}"
)

model.eval()

num_episodes = 20

rewards = []

successes = 0

for episode in range(num_episodes):

    obs, info = env.reset()

    state = preprocess_state(obs)

    done = False

    total_reward = 0

    steps = 0

    while not done:
        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        with torch.no_grad():
            action = model(
                state_tensor
            ).argmax().item()

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_reward += reward

        steps += 1

        state = preprocess_state(next_obs)

    print(
        f"Reward: {total_reward}"
    )

    print(
        f"Steps: {steps}"
    )

avg_reward = np.mean(rewards)

success_rate = (successes / num_episodes) * 100

print("\n========== RESULTS ==========")
print(f"Average Reward : {avg_reward:.3f}")
print(f"Success Rate   : {success_rate:.2f}%")
print("=============================")