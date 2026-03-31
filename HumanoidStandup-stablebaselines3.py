import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os

# Config
N_ENVS = 32
ROLLOUT_SIZE = 2048 * N_ENVS  # steps per rollout
TOTAL_TIMESTEPS = 1_000_000  # adjust as desired per run
MODEL_PATH = "ppo_HumanoidStandup-v5"
VECNORM_PATH = MODEL_PATH + "_vecnormalize.pkl"

# Vectorized Environment
=vec_env = make_vec_env("HumanoidStandup-v5", n_envs=N_ENVS)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

# Evaluation Environment
eval_env = make_vec_env("HumanoidStandup-v5", n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

# Load model if exists
if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH, env=vec_env)
    print("[INFO] Loaded existing model.")
else:
    model = PPO("MlpPolicy", vec_env, verbose=1)
    print("[INFO] Created new model.")

# Save Latest Callback
class SaveLatestCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            self.training_env.save(self.save_path + "_vecnormalize.pkl")
            print(f"[SAVE] Model & VecNormalize updated at step {self.num_timesteps}")
        return True

save_latest_callback = SaveLatestCallback(
    save_freq=ROLLOUT_SIZE,
    save_path=MODEL_PATH
)

# Evaluation Callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best/",
    log_path="./logs/",
    eval_freq=ROLLOUT_SIZE,
    deterministic=True,
    render=False,
)

# Train the model
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    reset_num_timesteps=False,
    callback=[save_latest_callback, eval_callback]
)

# Final save
model.save(MODEL_PATH)
vec_env.save(VECNORM_PATH)
print("[FINAL SAVE] Model & VecNormalize stats saved.")

# Demo Mode (Single Env)
print("\n[INFO] Starting demo...")
demo_env = gym.make("HumanoidStandup-v5", render_mode="human")

# Load normalization stats
demo_env = VecNormalize.load(VECNORM_PATH, demo_env)
demo_env.training = False
demo_env.norm_reward = False

# Load model
demo_model = PPO.load(MODEL_PATH)

obs = demo_env.reset()
while True:
    action, _ = demo_model.predict(obs, deterministic=True)
    obs, reward, done, info = demo_env.step(action)
    if done:
        obs = demo_env.reset()
