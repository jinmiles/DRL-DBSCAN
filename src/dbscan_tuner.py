import numpy as np
from src.clustering import run_dbscan
from src.metrics import get_internal_metrics

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DBSCANDRLTuner:
    def __init__(
        self,
        X,
        eps_range=(0.1, 1.0),
        minpts_range=(3, 15),
        n_episodes=30,
        max_steps=15,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.95,
        lr=1e-3,
        device=None,
        verbose=True,
    ):
        self.X = X
        self.eps_range = eps_range
        self.minpts_range = minpts_range
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.verbose = verbose

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = 4  # [eps, minpts, silhouette, db]
        self.action_dim = 6  # [eps+, eps-, minpts+, minpts-, reset, no-op]
        self.qnet = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.memory = deque(maxlen=1000)
        self.batch_size = 32

        self.eps = np.mean(eps_range)
        self.minpts = int(np.mean(minpts_range))

    def get_state(self, eps, minpts, prev_metrics):
        return np.array(
            [
                eps,
                minpts,
                prev_metrics.get("silhouette", 0),
                prev_metrics.get("davies_bouldin", 0),
            ],
            dtype=np.float32,
        )

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.qnet(state)
        return q_values.argmax().item()

    def step(self, action, eps, minpts):
        delta_eps = 0.05 * (self.eps_range[1] - self.eps_range[0])
        delta_minpts = 1
        if action == 0:  # eps+
            eps = min(eps + delta_eps, self.eps_range[1])
        elif action == 1:  # eps-
            eps = max(eps - delta_eps, self.eps_range[0])
        elif action == 2:  # minpts+
            minpts = min(minpts + delta_minpts, self.minpts_range[1])
        elif action == 3:  # minpts-
            minpts = max(minpts - delta_minpts, self.minpts_range[0])
        elif action == 4:  # reset
            eps = np.mean(self.eps_range)
            minpts = int(np.mean(self.minpts_range))
        return eps, minpts

    def compute_reward(self, prev_metrics, curr_metrics):
        # 실루엣 점수 증가/감소에 따라 보상 부여 (Davies-Bouldin은 음수 방향)
        prev_sil = prev_metrics.get("silhouette", 0)
        curr_sil = curr_metrics.get("silhouette", 0)
        prev_db = prev_metrics.get("davies_bouldin", 10)
        curr_db = curr_metrics.get("davies_bouldin", 10)
        reward = (curr_sil - prev_sil) * 10 + (prev_db - curr_db)
        return reward

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.qnet(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.qnet(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def tune(self):
        best_metrics = {"silhouette": -1}
        best_params = {"eps": self.eps, "minpts": self.minpts}
        for episode in range(self.n_episodes):
            eps = np.mean(self.eps_range)
            minpts = int(np.mean(self.minpts_range))
            prev_metrics = {"silhouette": 0, "davies_bouldin": 10}
            state = self.get_state(eps, minpts, prev_metrics)
            for step in range(self.max_steps):
                action = self.select_action(state)
                next_eps, next_minpts = self.step(action, eps, minpts)
                labels = run_dbscan(self.X, eps=next_eps, min_samples=next_minpts)
                curr_metrics = get_internal_metrics(self.X, labels)
                reward = self.compute_reward(prev_metrics, curr_metrics)
                next_state = self.get_state(next_eps, next_minpts, curr_metrics)
                done = step == self.max_steps - 1

                self.memory.append((state, action, reward, next_state, done))
                self.optimize()

                # 업데이트
                if curr_metrics.get("silhouette", -1) > best_metrics.get(
                    "silhouette", -1
                ):
                    best_metrics = curr_metrics
                    best_params = {"eps": next_eps, "minpts": next_minpts}

                state = next_state
                eps, minpts = next_eps, next_minpts
                prev_metrics = curr_metrics

                if self.verbose:
                    print(
                        f"Ep{episode+1:02d} Step{step+1:02d} eps={eps:.3f} minpts={minpts} sil={curr_metrics.get('silhouette', np.nan):.3f} reward={reward:.2f}"
                    )

            # epsilon decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

        if self.verbose:
            print(
                f"\nBest params: eps={best_params['eps']:.3f}, minpts={best_params['minpts']}"
            )
            print(f"Best silhouette: {best_metrics.get('silhouette', np.nan):.3f}")
        return best_params, best_metrics


if __name__ == "__main__":
    from src.datasets import get_moons

    X, y = get_moons(n_samples=300, noise=0.06)
    tuner = DBSCANDRLTuner(X, n_episodes=5, max_steps=10, verbose=True)
    best_params, best_metrics = tuner.tune()
    print("Best params:", best_params)
    print("Best metrics:", best_metrics)
