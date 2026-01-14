import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# Neural Network (Policy Network)
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # - Input: 4 values (CartPole state)
        # - Hidden layer: 32 neurons with ReLU activation
        # - Output: 2 values (probabilities for left/right actions)
        self.layers = nn.Sequential(
            nn.Linear(4, 32),   # State → hidden layer
            nn.ReLU(),          # Non-linear
            nn.Linear(32, 2)    # Hidden layer → output (action)
        )

    def forward(self, x):
        # Forward pass:
        # Converts state input into action probabilities
        # Softmax ensures probabilities sum to 1
        return torch.softmax(self.layers(x), dim=-1)


# Environment & Training Setup
# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Instantiate the policy network
Network = Network()

# Adam optimizer updates network weights using gradient descent
optimizer = optim.Adam(Network.parameters(), lr=1e-2)


# Action Selection Function
def choose_action(obs):
    # Convert observation (state) into a PyTorch tensor
    obs = torch.tensor(obs, dtype=torch.float32)

    # Get action probabilities from the policy network
    probs = Network(obs)

    # Create a categorical probability distribution over actions
    dist = torch.distributions.Categorical(probs)

    # Sample an action based on the probabilities
    action = dist.sample()

    # Return:
    # - action index (0 or 1)
    # - log probability of the chosen action (used for training)
    return action.item(), dist.log_prob(action)


# Training Loop (the reinforcement part part of the code)
for episode in range(200):
    # Reset environment at the start of each episode
    obs, info = env.reset()
    done = False

    # Store log probabilities and rewards for this episode
    log_probs = []
    rewards = []

    # Run one episode
    while not done:
        # Choose an action using the current policy
        action, log_prob = choose_action(obs)

        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Save data for learning later
        log_probs.append(log_prob)
        rewards.append(reward)

        # Episode ends if the pole falls or time limit is reached
        done = terminated or truncated

    # Compute Discounted Returns
    returns = []
    G = 0

    # Calculate discounted reward from the end of the episode backward
    for r in reversed(rewards):
        G = r + 0.99 * G  # γ = 0.99
        returns.insert(0, G)

    # Convert to tensor
    returns = torch.tensor(returns)

    # Normalize returns to stabilize training
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    # -------------------------------
    # Policy Gradient Loss
    # -------------------------------
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        # REINFORCE loss:
        # Encourage actions that led to higher returns
        loss -= log_prob * G

    # Backpropagation step
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update network parameters

    # Print episode performance
    print(f"Episode {episode}  |  Reward = {sum(rewards)}")

env.close()
