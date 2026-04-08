import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os


# Config

N_ENVS = 16
TOTAL_TIMESTEPS = 100_000_000
MODEL_PATH = "ppo_HumanoidStandup-v5"
VECNORM_PATH = MODEL_PATH + "_vecnormalize.pkl"


# Vectorized Training Env

vec_env = make_vec_env("HumanoidStandup-v5", n_envs=N_ENVS)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)


# Evaluation Env

eval_env = make_vec_env("HumanoidStandup-v5", n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
eval_env.training = False


# Load or Create Model

if os.path.exists(MODEL_PATH + ".zip"):
    if os.path.exists(VECNORM_PATH):
        vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        print("[INFO] Resuming training with VecNormalize stats")
    else:
        print("[WARNING] Missing VecNormalize stats!")

    model = PPO.load(MODEL_PATH, env=vec_env)

else:
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,

        # Optimized humanoid settings
        n_steps=2048,
        batch_size=32768,  # divides 2048*16 = 32768
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,

        policy_kwargs=policy_kwargs
    )

    print("[INFO] Created new model")


# Save Callback

class SaveLatestCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.model.save(self.save_path)
        self.training_env.save(self.save_path + "_vecnormalize.pkl")
        print(f"[SAVE] Step {self.num_timesteps}")

save_callback = SaveLatestCallback(MODEL_PATH)


# Evaluation Callback

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best/",
    log_path="./logs/",
    eval_freq=200_000,
    deterministic=True,
    render=False,
)


# Train
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    reset_num_timesteps=False,
    callback=[save_callback, eval_callback]
)


# Final Save

model.save(MODEL_PATH)
vec_env.save(VECNORM_PATH)
print("[FINAL SAVE COMPLETE]")


# test code

"""
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
"""
