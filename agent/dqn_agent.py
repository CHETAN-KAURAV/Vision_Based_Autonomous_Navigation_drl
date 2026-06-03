import random
import numpy as np
import torch
import torch.nn as nn

from models.dqn import DQN

class DQNAgent:

    def __init__(self, n_actions):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.n_actions = n_actions

        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.model = DQN(n_actions).to(self.device)

        self.target_model = DQN(n_actions).to(self.device)

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4
        )

        self.loss_fn = nn.MSELoss()

    def select_action(self, state):

        if random.random() < self.epsilon:

            return random.randint(
                0,
                self.n_actions - 1
            )

        state = torch.FloatTensor(
            state
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            q_values = self.model(state)

        return q_values.argmax().item()

    def train_step(self, batch):

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(
            np.array(states)
        ).to(self.device)

        actions = torch.LongTensor(
            actions
        ).to(self.device)

        rewards = torch.FloatTensor(
            rewards
        ).to(self.device)

        next_states = torch.FloatTensor(
            np.array(next_states)
        ).to(self.device)

        dones = torch.FloatTensor(
            dones
        ).to(self.device)

        current_q = self.model(states)

        current_q = current_q.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze()

        with torch.no_grad():

            next_q = self.target_model(
                next_states
            ).max(1)[0]

            target_q = rewards + (
                1 - dones
            ) * self.gamma * next_q

        loss = self.loss_fn(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):

        self.target_model.load_state_dict(
            self.model.state_dict()
        )