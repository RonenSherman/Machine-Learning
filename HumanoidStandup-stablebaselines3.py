import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

N_ENVS = 32
ROLLOUT_SIZE = 2048 * N_ENVS  # 65536 steps per rollout

vec_env = make_vec_env("HumanoidStandup-v5", n_envs=N_ENVS)

# Normalize observations and rewards (helps stability for Humanoid)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

# Evaluation env
eval_env = make_vec_env("HumanoidStandup-v5", n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

# Load model
model = PPO.load("ppo_HumanoidStandup-v5", env=vec_env)

# Custom callback: overwrite latest model every rollout
class SaveLatestCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)  # overwrite latest
            print("latest model updated")
        return True

save_latest_callback = SaveLatestCallback(
    save_freq=ROLLOUT_SIZE,
    save_path="ppo_HumanoidStandup-v5"
)

# Evaluation callback: save best model only
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best/",
    log_path="./logs/",
    eval_freq=ROLLOUT_SIZE,
    deterministic=True,
    render=False
)

# Train the model
model.learn(
    total_timesteps=1_000_000,
    reset_num_timesteps=False,  # continue counting if resuming
    callback=[save_latest_callback, eval_callback]
)

# Final save of latest model
model.save("ppo_HumanoidStandup-v5")
print("model saved")

# Run trained model
del model
N_ENVS = 1
model = PPO.load("ppo_HumanoidStandup-v5", env=vec_env)

obs = vec_env.reset()
while True:
    action, _ = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
