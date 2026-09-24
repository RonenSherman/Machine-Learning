import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env():
    return gym.make("HumanoidStandup-v5", render_mode="human")

demo_env = DummyVecEnv([make_env])

# Load normalization stats
demo_env = VecNormalize.load("ppo_HumanoidStandup-v5_vecnormalize.pkl", demo_env)
demo_env.training = False
demo_env.norm_reward = False

model = PPO.load("ppo_HumanoidStandup-v5")

obs = demo_env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = demo_env.step(action)