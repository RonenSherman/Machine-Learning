import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("HumanoidStandup-v5", n_envs=16)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("ppo_HumanoidStandup-v5")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_HumanoidStandup-v5")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")