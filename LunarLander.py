import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

TRAIN_FROM_SCRATCH = False
MODEL_PATH = "lunarlander_a2c.pth"

"""
Actor-Critic improves on REINFORCE because REINFORCE is a Monte Carlo method that only updates after an entire episode.
This makes learning slow and unstable since the total return is very noisy.
When the agent crashes, that affects the learning signal for every action taken earlier,
which makes it difficult to know how specific actions impacted the state.
Actor-Critic solves this by introducing a critic that estimates the value of each state.
Instead of learning only from the final outcome, the actor learns whether an action was better or worse than expected in that state.
This reduces variance and allows the agent to learn more efficiently from each step,rather than being effected by the crashes at the end
"""

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.actor = nn.Linear(256, 4)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


render_mode = None if TRAIN_FROM_SCRATCH else 'human'
env = gym.make("LunarLander-v3", render_mode=render_mode)

model = ActorCritic()

if not TRAIN_FROM_SCRATCH:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

optimizer = None if not TRAIN_FROM_SCRATCH else optim.Adam(model.parameters(), lr=3e-4)

gamma = 0.99

# stronger exploration early
initial_entropy = 0.2
final_entropy = 0.01
decay_steps = 3000

reward_history = []
stop_exploring = False


def choose_action(obs, deterministic=False):

    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, value = model(obs)
    logits = logits.squeeze(0)
    value = value.squeeze(0)

    dist = torch.distributions.Categorical(logits=logits)

    if deterministic:
        action = dist.probs.argmax()
        log_prob = dist.log_prob(action)
        entropy = torch.tensor(0.0)

    else:
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

    return action.item(), log_prob, entropy, value


def compute_returns(rewards, last_value, done):

    returns = []
    R = 0 if done else last_value

    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    return torch.tensor(returns, dtype=torch.float32)


# Main loop trains if TRAIN_FROM_SCRATCH, else just demos trained model
for episode in range(5000):
    obs, _ = env.reset()

    done = False
    total_reward = 0

    log_probs = []
    values = []
    rewards = []
    entropies = []

    while not done:

        action, log_prob, entropy, value = choose_action(
            obs,
            deterministic=stop_exploring or not TRAIN_FROM_SCRATCH
        )

        next_obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if TRAIN_FROM_SCRATCH:

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

        obs = next_obs
        done = terminated or truncated

    if TRAIN_FROM_SCRATCH:

        with torch.no_grad():
            last_value = 0

        returns = compute_returns(rewards, last_value, done)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        advantages = returns - values

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        entropy_coef = max(final_entropy, initial_entropy * (1 - episode / decay_steps))
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_bonus = entropies.mean()

        loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        reward_history.append(total_reward)

        # decides when to stop learning and to make the sim visible again
        if len(reward_history) >= 100:
            avg_reward = sum(reward_history[-100:]) / 100

            if avg_reward > 100  and render_mode != 'human':
                render_mode = 'human'
                env = gym.make("LunarLander-v3", render_mode=render_mode)
                torch.save(model.state_dict(), MODEL_PATH)
                print("Model saved.")
                stop_exploring = True

    print(f"Episode {episode} | Reward {total_reward:.1f}")