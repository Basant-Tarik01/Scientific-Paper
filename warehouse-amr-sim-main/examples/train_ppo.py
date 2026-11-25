#!/usr/bin/env python3
"""Train / evaluate the RL policies of A1 on the WarehouseNav-v0 environment (2D simulator).

    pip install "stable-baselines3>=2.0" "sb3-contrib>=2.0" torch      # CPU is fine for small runs

    # privileged teacher (sees the true health vector)
    python3 examples/train_ppo.py --privileged --timesteps 3000000 --n-envs 8 --out runs/teacher
    # domain-randomised PPO baseline (no health input, no memory)
    python3 examples/train_ppo.py --timesteps 3000000 --n-envs 8 --out runs/dr_ppo
    # end-to-end recurrent PPO baseline (LSTM, no teacher)
    python3 examples/train_ppo.py --recurrent --timesteps 3000000 --n-envs 8 --out runs/rec_ppo
    # evaluate any of them on held-out severities / combined faults
    python3 examples/train_ppo.py --eval runs/teacher/model.zip --privileged --scenario a1_eval_heldout --episodes 100

The student (recurrent encoder distilled from the teacher) is the method under study; it can be
trained with the same env: roll out the student, query the teacher with obs["health"] (use
privileged=True and drop the key for the student), and regress the teacher's actions (DAgger).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "amr_sim_core"))
import amr_sim_core  # noqa: E402,F401  (registers WarehouseNav-v0)


def make_env(scenario, privileged, seed):
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor

    def _f():
        env = gym.make("WarehouseNav-v0", scenario=scenario, privileged=privileged, seed=seed)
        return Monitor(env)
    return _f


def evaluate(model, scenario, privileged, episodes, recurrent, seed=12345):
    import gymnasium as gym
    env = gym.make("WarehouseNav-v0", scenario=scenario, privileged=privileged)
    results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        state, start = None, np.ones((1,), dtype=bool)
        done = False
        while not done:
            if recurrent:
                action, state = model.predict(obs, state=state, episode_start=start, deterministic=True)
                start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        m = info["episode_metrics"]
        m.update(faults="+".join(info["faults"]) or "none", density=info["density"])
        results.append(m)
    coll = np.mean([m["outcome"].startswith("collision") for m in results])
    succ = np.mean([m["success"] for m in results])
    nm = np.mean([m["near_miss_events"] for m in results])
    print(f"[{scenario}] episodes={episodes} success={succ:.2f} collision={coll:.2f} near-miss/ep={nm:.2f}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="a1_train")
    ap.add_argument("--privileged", action="store_true", help="teacher: observe the true health vector")
    ap.add_argument("--recurrent", action="store_true", help="RecurrentPPO (LSTM) from sb3-contrib")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--eval", default=None, help="path to a saved model.zip: evaluate instead of training")
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    if args.recurrent:
        from sb3_contrib import RecurrentPPO as Algo
        policy = "MultiInputLstmPolicy"
    else:
        from stable_baselines3 import PPO as Algo
        policy = "MultiInputPolicy"

    if args.eval:
        model = Algo.load(args.eval)
        res = evaluate(model, args.scenario, args.privileged, args.episodes, args.recurrent)
        out = os.path.splitext(args.eval)[0] + f"_eval_{args.scenario}.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"per-episode metrics: {out}")
        return

    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    VecEnv = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = VecEnv([make_env(args.scenario, args.privileged, args.seed + i) for i in range(args.n_envs)])
    os.makedirs(args.out, exist_ok=True)
    try:
        import tensorboard  # noqa: F401
        tb_dir = os.path.join(args.out, "tb")
    except ImportError:
        tb_dir = None
    model = Algo(policy, env, n_steps=512, batch_size=1024 if not args.recurrent else 256, learning_rate=3e-4,
                 gamma=0.99, gae_lambda=0.95, ent_coef=0.005, verbose=1, seed=args.seed,
                 tensorboard_log=tb_dir)
    model.learn(total_timesteps=args.timesteps,
                callback=CheckpointCallback(max(100_000 // args.n_envs, 1), os.path.join(args.out, "ckpt")))
    model.save(os.path.join(args.out, "model"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    print(f"saved {os.path.join(args.out, 'model.zip')}")
    env.close()
    evaluate(model, args.scenario, args.privileged, 10, args.recurrent)


if __name__ == "__main__":
    main()
