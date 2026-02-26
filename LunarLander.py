import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

"""
Actor-Critic improves on REINFORCE because REINFORCE is a Monte Carlo method that only updates after an entire episode.
This makes learning slow and unstable since the total return can be very noisy.
When the agent crashes at the end of an episode, that outcome affects the learning signal for every action taken earlier,
which makes it difficult to know how specific actions impacted the state.
Actor-Critic solves this by introducing a critic that estimates the value of each state.
Instead of learning only from the final outcome, the actor learns whether an action was better or worse than expected in that state.
This reduces variance and allows the agent to learn more efficiently from each step, leading to faster and more stable training.
"""

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, 4)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


env = gym.make("LunarLander-v3", render_mode=None)

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # faster learning

gamma = 0.99


def choose_action(obs):
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, value = model(obs)
    logits = logits.squeeze(0)
    value = value.squeeze(0)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    return action.item(), dist.log_prob(action), dist.entropy(), value


# Training loop
for episode in range(2500):
    obs, _ = env.reset()
    done = False

    log_probs = []
    values = []
    rewards = []
    entropies = []

    while not done:
        action, log_prob, entropy, value = choose_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        entropies.append(entropy)

        obs = next_obs
        done = terminated or truncated

    # Step-wise TD(0) returns
    returns = []
    R = 0
    values = torch.stack(values).view(-1)
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, dtype=torch.float32)

    # Advantage
    advantages = returns - values
    if advantages.std() > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)

    # Actor and critic loss
    actor_loss = -(log_probs * advantages.detach()).mean()
    critic_loss = advantages.pow(2).mean()
    entropy_bonus = entropies.mean()

    # Increase entropy for exploration
    loss = actor_loss + 0.5 * critic_loss - 0.05 * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    print(f"Episode {episode} | Reward {sum(rewards):.1f}")