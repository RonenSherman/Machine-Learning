import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

TRAIN_FROM_SCRATCH = True
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

render_mode = None
env = gym.make("LunarLander-v3", render_mode=render_mode)


model = ActorCritic()

if not TRAIN_FROM_SCRATCH:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

optimizer = None if not TRAIN_FROM_SCRATCH else optim.Adam(model.parameters(), lr=1e-4)

gamma = 0.99

#added this to make it explore less as it learns
initial_entropy = 0.01
final_entropy = 0.0
decay_steps = 5000

reward_history = []
stop_exploring = False


def choose_action(obs, deterministic=False):
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    # only use no_grad when running trained model
    if deterministic:
        with torch.no_grad():
            logits, value = model(obs)
    else:
        logits, value = model(obs)  # gradients enabled during training

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


# Training loop
for episode in range(5000 if TRAIN_FROM_SCRATCH else 1000):
    obs, _ = env.reset()
    done = False

    log_probs = [] # for policy gradient
    values = [] # for critic
    rewards = [] # rewards
    entropies = [] # exploration (think epsilon-greedy)
    next_values = [] # TD learning

    while not done:
        action, log_prob, entropy, value = choose_action(obs, deterministic=stop_exploring)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        with torch.no_grad():
            _, next_value = model(torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0))
            next_value = next_value.squeeze(0)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        entropies.append(entropy)
        next_values.append(next_value)

        obs = next_obs
        done = terminated or truncated

    values = torch.stack(values).view(-1)
    next_values = torch.stack(next_values).view(-1)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    targets = rewards + gamma * next_values # bootstraps estimates
    advantages = targets - values # advantage part of advantage actor critic, compares how much action actually helped

    if advantages.std() > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)

    entropy = max(final_entropy, initial_entropy * (1 - episode / decay_steps))

    actor_loss = -(log_probs * advantages.detach()).mean()
    critic_loss = advantages.pow(2).mean()
    entropy_bonus = entropies.mean()

    loss = actor_loss + 0.5 * critic_loss - entropy * entropy_bonus
    if optimizer is not None and not stop_exploring:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

    episode_reward = rewards.sum().item()
    reward_history.append(episode_reward)



    # decides when to stop learning and to make the sim visible again
    if len(reward_history) >= 100:
        avg_reward = sum(reward_history[-100:]) / 100
        if avg_reward > 200:
            render_mode = 'human'
            env = gym.make("LunarLander-v3", render_mode=render_mode)
            stop_exploring = True
            torch.save(model.state_dict(), MODEL_PATH)
            print("Model saved.")

    print(f"Episode {episode} | Reward {episode_reward:.1f}")