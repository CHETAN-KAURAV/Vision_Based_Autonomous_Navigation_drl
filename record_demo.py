import gymnasium as gym
import minigrid
import imageio
import torch

from models.dqn import DQN
from utils.preprocess import preprocess_state

env = gym.make(
    "MiniGrid-Empty-8x8-v0",
    render_mode="rgb_array"
)

n_actions = env.action_space.n

model = DQN(n_actions)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location="cpu"
    )
)

model.eval()

frames = []

obs, info = env.reset()

state = preprocess_state(obs)

done = False

while not done:

    frame = env.render()

    frames.append(frame)

    state_tensor = torch.FloatTensor(
        state
    ).unsqueeze(0)

    with torch.no_grad():

        action = model(
            state_tensor
        ).argmax().item()

    next_obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    state = preprocess_state(next_obs)

imageio.mimsave(
    "navigation_demo.gif",
    frames,
    fps=5
)

print("GIF Saved")