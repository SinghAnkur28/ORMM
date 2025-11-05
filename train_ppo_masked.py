# train_ppo_masked.py
import os
import time
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from ev_routing_env import EVRoutingEnv, make_tiny_instance
from ev_masked_wrapper import EVMaskedDiscreteWrapper


# The ActionMasker expects a function(env) -> np.ndarray[bool]
def mask_fn(env: EVMaskedDiscreteWrapper):
    return env.action_mask()


def make_env(seed=1, discharge_bins=5):
    # base env
    inst = make_tiny_instance(seed)
    env = EVRoutingEnv(inst)
    # flatten actions + mask
    env = EVMaskedDiscreteWrapper(env, discharge_bins=discharge_bins)
    # plug into sb3-contrib masking
    env = ActionMasker(env, mask_fn)
    return env


def main():
    logdir = os.path.join("runs", f"ppo_masked_{int(time.time())}")
    os.makedirs(logdir, exist_ok=True)

    # ---- training envs ----
    train_env = make_vec_env(make_env, n_envs=4, seed=123, vec_env_cls=None)  # Subproc not required here
    # ---- eval env ----
    eval_env = make_env(seed=999)

    # ---- model ----
    # Observation is a flat Box; action is Discrete (masked). MlpPolicy is fine.
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=logdir,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        seed=123,
    )

    # ---- eval callback ----
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=logdir,
        log_path=logdir,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    # ---- train ----
    total_timesteps = 200_000  # bump up later
    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=True)

    # save
    save_path = os.path.join(logdir, "final_model.zip")
    model.save(save_path)
    print(f"Saved to {save_path}")

    # ---- quick eval run ----
    env = eval_env
    obs, info = env.reset()
    ep_ret = 0.0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += float(reward)
        if terminated or truncated:
            break
    print(f"Eval episode return: {ep_ret:.2f}")


if __name__ == "__main__":
    main()
