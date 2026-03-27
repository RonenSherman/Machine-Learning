import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import freeze_support
import os

TRAIN_FROM_SCRATCH = True

# WARNING WARNING WARNING
NUM_ENVS = 16  # DO NOT CHANGE!!! WARNING WARNING DO NOT CHANGE ABOVE 32 AT ALL COSTS
MODEL_PATH = "humanoid_ppo_best.pth"       # best model path
BEST_REWARD_FILE = "best_reward.txt"       # file to store best reward

CHECKPOINT_INTERVAL = 50  # save every N episodes

def make_env():
    def _init():
        return gym.make("HumanoidStandup-v5")
    return _init

if __name__ == "__main__":
    freeze_support()  # fixes spawn_main / bootstrapping errors on Windows

    env = AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])

    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    # PPO Network (continuous actions)
    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()

            self.shared = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU()
            )

            self.actor_mean = nn.Linear(256, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))
            self.critic = nn.Linear(256, 1)

        def forward(self, x):
            x = self.shared(x)
            mean = self.actor_mean(x)
            std = torch.exp(self.log_std)
            value = self.critic(x)
            return mean, std, value


    model = ActorCritic()

    # Load previous best model if exists
    best_reward = float('-inf')
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("Loaded best model from previous run.")

    if os.path.exists(BEST_REWARD_FILE):
        with open(BEST_REWARD_FILE, "r") as f:
            best_reward = float(f.read())
            print(f"Previous best reward: {best_reward:.1f}")

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    gamma = 0.99
    clip_eps = 0.2
    epochs = 10
    batch_size = 2048


    def choose_action(obs):
        obs = torch.tensor(obs, dtype=torch.float32)

        mean, std, value = model(obs)
        dist = torch.distributions.Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.detach().numpy(), log_prob.detach(), value.detach()


    def compute_returns(rewards, dones):
        returns = []
        R = torch.zeros(NUM_ENVS)

        for r, d in zip(reversed(rewards), reversed(dones)):
            R = torch.tensor(r) + gamma * R * (1 - torch.tensor(d, dtype=torch.float32))
            returns.insert(0, R)

        return torch.stack(returns)


    # Training loop
    for episode in range(10000):

        obs, _ = env.reset()

        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        total_reward = 0

        for step in range(batch_size):

            action, log_prob, value = choose_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            states.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value.squeeze(-1))

            obs = next_obs
            total_reward += reward.sum()

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        returns = compute_returns(rewards, dones)
        advantages = returns - values

        # flatten
        states = states.reshape(-1, obs_dim)
        actions = actions.reshape(-1, act_dim)
        old_log_probs = old_log_probs.reshape(-1)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(epochs):

            mean, std, new_values = model(states)
            dist = torch.distributions.Normal(mean, std)

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - new_values.squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        avg_reward = total_reward / NUM_ENVS

        # Incremental checkpoint every N episodes
        if episode % CHECKPOINT_INTERVAL == 0:
            # Save incremental checkpoint
            torch.save(model.state_dict(), f"humanoid_ppo_ep{episode}.pth")

            # Save best model if improved
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(model.state_dict(), MODEL_PATH)
                with open(BEST_REWARD_FILE, "w") as f:
                    f.write(str(best_reward))
                print(f"Episode {episode} | New best reward {best_reward:.1f} | Model saved.")
            else:
                print(f"Episode {episode} | Reward {avg_reward:.1f} | Incremental checkpoint saved.")

        else:
            print(f"Episode {episode} | Reward {avg_reward:.1f}")
