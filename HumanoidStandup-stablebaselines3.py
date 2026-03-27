import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("HumanoidStandup-v5", n_envs=32)

model = PPO.load("ppo_HumanoidStandup-v5")
model.set_env(vec_env)

model.learn(total_timesteps=1000000)
#while model.learn:
#    model.save("ppo_HumanoidStandup-v5")
#    print("model saved")

model.save("ppo_HumanoidStandup-v5")
print("model saved")
del model


model = PPO.load("ppo_HumanoidStandup-v5")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")