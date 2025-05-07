import time
import random
import argparse

import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from RL.ppo_continuous import PPOAgent, AttendPPOAgent
from RL.buffer import RolloutBuffer
from Env.multiRampEnv import MultiRampEnv


detector_df = pd.read_csv("./Env/MultiRamp/detectorPos.CSV")
ctrl_step, model_step = 30, 5
warmup_step, simulation_step, release_step = 600, 3600, 0
total_step = warmup_step + simulation_step + release_step

NET_DIR = "./Env/MultiRamp/multiRamp.net.xml"
DET_DIR = "./Env/MultiRamp/denseDetectors.add.xml"
CFG_DIR = "./Env/MultiRamp/multiRamp.sumocfg"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--filename", type=str, default="test")
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--randomize", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_env", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=250000)
    parser.add_argument("--save_freq", type=int, default=50)
    # parameters associate with attention layer
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--model_dim", type=int, default=16)
    parser.add_argument("--feed_forward_dim", type=int, default=256)
    parser.add_argument("--num_attend_layer", type=int, default=2)
    parser.add_argument("--num_head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    # parameters associate with reinforcement learning
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--k_epoch", type=int, default=4)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--max_grad_clip", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--train_step", type=int, default=64)
    parser.add_argument("--num_mini_batch", type=int, default=4)
    arguments = parser.parse_args()
    arguments.batch_size = int(arguments.train_step * arguments.num_env)
    arguments.mini_batch_size = int(arguments.batch_size // arguments.num_mini_batch)
    return arguments


def make_env(env_, seed, randomize, count=10000, ob_mean=None, ob_var=None, r_mean=None, r_var=None):
    from RL.wrappers import MySingleRecordEpisodeStatistics as RecordEpisodeStatistics, RecordNormalParam
    from RL.wrappers import MyNormalizeObservation as NormalizeObservation
    from RL.wrappers import MyNormalizeReward as NormalizeReward

    def thunk():
        env = gym.wrappers.ClipAction(env_)
        env = RecordEpisodeStatistics(env)
        env = NormalizeObservation(env, count=count, running_mean=ob_mean, running_var=ob_var)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10., 10., dtype=np.float32))
        # not subtract the mean (scaling the reward)
        env = NormalizeReward(env, count=count, running_mean=r_mean, running_var=r_var)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10., 10., dtype=np.float32))
        env = RecordNormalParam(env)
        # env.seed(seed)
        if not randomize:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        return env
    return thunk


def train_policy(env: gym.vector, agent: PPOAgent, writer: SummaryWriter, args):
    current_step = 0
    num_update = args.max_steps // args.batch_size
    if args.randomize:
        state, info = env.reset()
    else:
        state, info = env.reset(seed=[args.seed + i for i in range(args.num_env)])
    start_time = time.perf_counter()
    ob_mean, ob_var, r_mean, r_var = None, None, None, None
    for update_step in range(1, num_update + 1):
        lr = agent.lr_decay(current_step)
        writer.add_scalar("Global/lr", lr, current_step)
        for _ in range(args.train_step):
            with torch.no_grad():
                action, log_prob = agent.select_action(state)
            next_state, reward, terminated, _, info = env.step(action)
            current_step += args.num_env
            if "final_info" in info.keys():
                # some env terminated
                num_terminated = sum(info["_final_info"])
                tts, returns = np.zeros(args.num_env, dtype=np.float32), np.zeros(args.num_env, dtype=np.float32)
                length = 0
                # FIXME: currently Ignore the env early stop case here !!!
                for i, sub_info in enumerate(info["final_info"]):
                    tts[i] = sub_info["tts"]
                    returns[i] = sub_info["episode"]["r"].item()
                    length = sub_info["episode"]["l"].item()
                ob_mean, ob_var = info["final_info"][np.argmin(tts)]["ob_mean"], info["final_info"][np.argmin(tts)]["ob_var"]
                r_mean, r_var = info["final_info"][np.argmin(tts)]["r_mean"], info["final_info"][np.argmin(tts)]["r_var"]
                writer.add_scalar("Env/TTS", tts.mean(), current_step)
                writer.add_scalar("Env/episode_return", returns.mean(), current_step)
                writer.add_scalar("Env/episode_length", length, current_step)
                print("Step: ", current_step, "Return: ", returns.mean(), "Length: ", length)
            # get the real terminated/truncated observation, not the reset observation
            real_next_state = np.vstack(info["final_observation"]) if "final_observation" in info else next_state
            agent.rolloutBuffer.push(state, action, log_prob, reward, real_next_state, terminated)
            state = next_state
        actor_loss, critic_loss, entropy_loss = agent.train()

        writer.add_scalar("Global/step_per_second", current_step / (time.perf_counter() - start_time), current_step)
        writer.add_scalar("Loss/actor_loss", actor_loss, current_step)
        writer.add_scalar("Loss/critic_loss", critic_loss, current_step)
        writer.add_scalar("Loss/entropy_loss", entropy_loss, current_step)
        if update_step % args.save_freq == 0:
            agent.save("PPO/TrainData/{}".format(args.filename))
    env.close()
    return ob_mean, ob_var, r_mean, r_var


if __name__ == "__main__":
    args = parse_args()
    if not args.randomize:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    writer = SummaryWriter(log_dir="./Logs/{}".format(args.exp_name))
    writer.add_text("HyperParameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    if args.load_from is not None:
        # train from a pretrained model, load the corresponding mean and var
        from functools import partial
        ob_mean = np.loadtxt("{}_ob_mean.CSV".format(args.load_from))
        ob_var = np.loadtxt("{}_ob_var.CSV".format(args.load_from))
        r_mean = np.loadtxt("{}_r_mean.CSV".format(args.load_from))
        r_var = np.loadtxt("{}_r_var.CSV".format(args.load_from))
        make_env = partial(make_env, ob_mean=ob_mean, ob_var=ob_var, r_mean=r_mean, r_var=r_var)
    envs = gym.vector.AsyncVectorEnv(
        [make_env(
            MultiRampEnv(ctrl_step_length=ctrl_step, simulation_precision=1, total_step=total_step, warmup=warmup_step,
                         port=args.port + i, gui=False, terminate_when_unhealthy=False, network_filename=NET_DIR,
                         detector_frame=detector_df, detector_filename=DET_DIR, sumocfg_filename=CFG_DIR,
                         no_step_log=True, no_warnings=True),
            args.seed + i,
            args.randomize)
            for i in range(args.num_env)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    envs.call("set_aggregate_parameters", aggregate_length=model_step)

    state_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    num_mainline_cell, num_ramp_cell = envs.call("num_mainline_cell")[0], envs.call("num_ramp_cell")[0]
    feature_dim = state_dim // (num_mainline_cell + num_ramp_cell)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    buffer = RolloutBuffer(args.num_env, args.train_step, state_shape, action_shape, device)
    # ---------- ppo agent with attention mechanism ----------
    ppo_agent = AttendPPOAgent(
        num_mainline_cell, num_ramp_cell, feature_dim, action_dim, args.hidden_dim, args.model_dim,
        args.feed_forward_dim, args.num_attend_layer, args.num_head, args.dropout, buffer, device,
        args.max_steps, args.gamma, args.gae_lambda, args.k_epoch, args.lr, args.eps_clip, args.max_grad_clip,
        args.entropy_coef, args.batch_size, args.mini_batch_size
    )
    if args.load_from is not None:
        # train from a pretrained model, load the pretrained model
        ppo_agent.load(args.load_from)
    env_ob_mean, env_ob_var, env_r_mean, env_r_var = train_policy(envs, ppo_agent, writer, args)
    writer.close()
    if env_ob_mean is not None and env_ob_var is not None:
        print("observation_mean: ", env_ob_mean.tolist())
        # note that this is the variance, not the standard deviation
        print("observation_var: ", env_ob_var.tolist())
        np.savetxt("./TrainData/{}_ob_mean.CSV".format(args.filename), env_ob_mean, delimiter=",")
        np.savetxt("./TrainData/{}_ob_var.CSV".format(args.filename), env_ob_var, delimiter=",")
        np.savetxt("./TrainData/{}_r_mean.CSV".format(args.filename), np.array([env_r_mean]), delimiter=",")
        np.savetxt("./TrainData/{}_r_var.CSV".format(args.filename), np.array([env_r_var]), delimiter=",")
