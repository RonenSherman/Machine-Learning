import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

TRAIN_FROM_SCRATCH = True
MODEL_PATH = "humanoid_ppo.pth"

env = gym.make("HumanoidStandup-v5")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# PPO Network (continuous actions)
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        ) #

        # Actor outputs mean
        self.actor_mean = nn.Linear(256, act_dim)

        # Learnable log std
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)
        mean = self.actor_mean(x)
        std = torch.exp(self.log_std)
        value = self.critic(x)
        return mean, std, value


model = ActorCritic()

if not TRAIN_FROM_SCRATCH:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

optimizer = optim.Adam(model.parameters(), lr=3e-4)

gamma = 0.99
clip_eps = 0.2
epochs = 10
batch_size = 2048


def choose_action(obs):
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    mean, std, value = model(obs)
    dist = torch.distributions.Normal(mean, std)

    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)

    return action.squeeze(0).detach().numpy(), log_prob.detach(), value.detach()


def compute_returns(rewards, dones, values):
    returns = []
    R = 0

    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            R = 0
        R = r + gamma * R
        returns.insert(0, R)

    return torch.tensor(returns, dtype=torch.float32)


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

        done = terminated or truncated

        states.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)
        values.append(value)

        obs = next_obs
        total_reward += reward

        if done:
            obs, _ = env.reset()

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    old_log_probs = torch.stack(log_probs)
    values = torch.stack(values).squeeze()

    returns = compute_returns(rewards, dones, values)
    advantages = returns - values

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

    print(f"Episode {episode} | Reward {total_reward:.1f}")

    if episode % 50 == 0:
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved.")