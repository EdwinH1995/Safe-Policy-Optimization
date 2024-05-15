# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations

import os
import random
import sys
import time
from collections import deque

import numpy as np
try: 
    from isaacgym import gymutil
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, TensorDataset

from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.env import make_sa_mujoco_env, make_sa_isaac_env
from safepo.common.logger import EpochLogger
from safepo.common.model import ActorVCritic
from safepo.common.model import LyapunovFunction
from safepo.common.model import DeltaLyapunovCritic
from safepo.utils.config import single_agent_args, isaac_gym_map, parse_sim_params


default_cfg = {
    'hidden_sizes': [64, 64],
    'gamma': 0.99,
    'target_kl': 0.02,
    'batch_size': 64,
    'learning_iters': 40,
    'max_grad_norm': 40.0,
}

isaac_gym_specific_cfg = {
    'total_steps': 100000000,
    'steps_per_epoch': 32768,
    'hidden_sizes': [1024, 1024, 512],
    'gamma': 0.96,
    'target_kl': 0.016,
    'num_mini_batch': 4,
    'use_value_coefficient': True,
    'learning_iters': 8,
    'max_grad_norm': 1.0,
    'use_critic_norm': False,
}

def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f'{args.device}:{args.device_id}')


    if args.task not in isaac_gym_map.keys():
        env, obs_space, act_space = make_sa_mujoco_env(
            num_envs=args.num_envs, env_id=args.task, seed=args.seed
        )
        eval_env, _, _ = make_sa_mujoco_env(num_envs=1, env_id=args.task, seed=None)
        config = default_cfg

    else:
        sim_params = parse_sim_params(args, cfg_env, None)
        env = make_sa_isaac_env(args=args, cfg=cfg_env, sim_params=sim_params)
        eval_env = env
        obs_space = env.observation_space
        act_space = env.action_space
        args.num_envs = env.num_envs
        config = isaac_gym_specific_cfg

    # set training steps
    steps_per_epoch = config.get("steps_per_epoch", args.steps_per_epoch)
    total_steps = config.get("total_steps", args.total_steps)
    local_steps_per_epoch = steps_per_epoch // args.num_envs
    epochs = total_steps // steps_per_epoch
    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs,
        verbose=False,
    )
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=3e-4
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=3e-4
    )
    #Initializing Lyapunov module
    lyapunov_function = LyapunovFunction(obs_space.shape[0]).to(device)
    lyapunov_optimizer = torch.optim.Adam(lyapunov_function.parameters(), lr=3e-4)
    #Initializing DeltaCritic
    delta_lyapunov_critic = DeltaLyapunovCritic(obs_space.shape[0]).to(device)
    delta_lyapunov_optimizer = torch.optim.Adam(delta_lyapunov_critic.parameters(), lr=3e-4)
    # Create a list of num_instances DeltaCritics
    delta_lyapunov_critic_lag = DeltaLyapunovCritic(obs_space.shape[0]).to(device)
    delta_lyapunov_critic_lag_optimizer = torch.optim.Adam(delta_lyapunov_critic.parameters(), lr=3e-4)
    delta_lyapunov_critic_copy= DeltaLyapunovCritic(obs_space.shape[0]).to(device)
    delta_lyapunov_critic_copy_optimizer = torch.optim.Adam(delta_lyapunov_critic.parameters(), lr=3e-4)
    delta_lyapunov_critic_lag_copy= DeltaLyapunovCritic(obs_space.shape[0]).to(device)
    delta_lyapunov_critic_lag_copy_optimizer = torch.optim.Adam(delta_lyapunov_critic.parameters(), lr=3e-4)
    # Number of instances you want
    num_instances = 20
    # Loop to create and append each instance to the list
    #for _ in range(num_instances):
        #function_instance = DeltaLyapunovCritic(obs_space.shape[0]).to(device)
        #delta_lyapunov_critic_history.append(function_instance)
    #current_delta_lyapunov_critic_index=0
    #Initializing MSE Loss
    mse_loss = nn.MSELoss()
    # create the vectorized on-policy buffer
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=args.num_envs,
        gamma=config["gamma"],
    )
    #Set the lyapunov limits
    lyapunov_threshold = args.cost_limit  # or some predefined safety threshold
    lyapunov_initial_penalty_scale = 10  # Penalty scale for initial state violations
    lambda_lyapunov=0.98
    # set up the logger
    dict_args = vars(args)
    dict_args.update(config)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    rew_deque = deque(maxlen=50)
    cost_deque = deque(maxlen=50)
    len_deque = deque(maxlen=50)
    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    logger.save_config(dict_args)
    logger.setup_torch_saver(policy.actor)
    logger.log("Start with training.")
    obs,_= env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret, ep_cost, ep_len = (
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
    )
    # training loop
    is_start = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    terminate = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    lagrange_coefficient_lol=0.0
    lagrange_update=0.0
    current_step_int=0
    for epoch in range(epochs):
        #obs,_= env.reset()
        #obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        terminate_test=[]
        rollout_start_time = time.time()
        delta_lyapunov_critic_copy.load_state_dict(delta_lyapunov_critic.state_dict())
        delta_lyapunov_critic_lag_copy.load_state_dict(delta_lyapunov_critic_lag.state_dict())
        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                act, log_prob, value_r, value_c = policy.step(obs, deterministic=False)
            action = act.detach().squeeze() if args.task in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            next_obs, reward, cost, terminated, truncated, info = env.step(action)

            ep_ret += reward.cpu().numpy() if args.task in isaac_gym_map.keys() else reward
            ep_cost += cost.cpu().numpy() if args.task in isaac_gym_map.keys() else cost
            ep_len += 1
            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost, terminated, truncated)
            )
            if "final_observation" in info:
                info["final_observation"] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info["final_observation"]
                    ],
                )
                info["final_observation"] = torch.as_tensor(
                    info["final_observation"],
                    dtype=torch.float32,
                    device=device,
                )
            with torch.no_grad():
                lyapunov_current=lyapunov_function(obs)
                lyapunov_next=lyapunov_function(next_obs)
                delta_lyapunov=delta_lyapunov_critic(obs)
            if lyapunov_current.dim() == 0:  # It's a scalar tensor
                lyapunov_current = lyapunov_current.expand(args.num_envs)  # Expand to match the batch size
            if lyapunov_next.dim() == 0:  # It's a scalar tensor
                lyapunov_next = lyapunov_next.expand(args.num_envs)  # Expand to match the batch size
            if delta_lyapunov.dim() == 0:  # It's a scalar tensor
                delta_lyapunov = delta_lyapunov.expand(args.num_envs)  # Expand to match the batch size
            current_step = torch.full((args.num_envs,), current_step_int, dtype=torch.int, device=device)
            terminate=((terminated.clone()) > 0.) | ((truncated.clone()) > 0.)
            if terminated.item()>0. or truncated.item()>0.:
                terminate_test.append(steps)
            #print("terminated",terminated)
            #print("truncated",truncated)
            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
                next_obs=next_obs,
                lyapunov_current=lyapunov_current,
                lyapunov_next=lyapunov_next,
                is_start=is_start,
                terminate=terminate,
                delta_lyapunov=delta_lyapunov,
                current_step=current_step
            )
            current_step_int+=1
            if terminate.item():
                current_step_int=0
            is_start=terminate.clone()
            obs = next_obs
            epoch_end = steps >= local_steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obs[idx], deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    info["final_observation"][idx], deterministic=False
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        rew_deque.append(ep_ret[idx])
                        cost_deque.append(ep_cost[idx])
                        len_deque.append(ep_len[idx])
                        logger.store(
                            **{
                                "Metrics/EpRet": np.mean(rew_deque),
                                "Metrics/EpCost": np.mean(cost_deque),
                                "Metrics/EpLen": np.mean(len_deque),
                            }
                        )
                        ep_ret[idx] = 0.0
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0
                        logger.logged = False

                    buffer.finish_path(
                        last_value_r=last_value_r, last_value_c=last_value_c, idx=idx
                    )
        rollout_end_time = time.time()

        eval_start_time = time.time()

        eval_episodes = 1 if epoch < epochs - 1 else 10
        if args.use_eval:
            for _ in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device)
                eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
                while not eval_done:
                    with torch.no_grad():
                        act, log_prob, value_r, value_c = policy.step(eval_obs, deterministic=True)
                    next_obs, reward, cost, terminated, truncated, info = env.step(
                        act.detach().squeeze().cpu().numpy()
                    )
                    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                    eval_rew += reward
                    eval_cost += cost
                    eval_len += 1
                    eval_done = terminated[0] or truncated[0]
                    eval_obs = next_obs
                eval_rew_deque.append(eval_rew)
                eval_cost_deque.append(eval_cost)
                eval_len_deque.append(eval_len)
            logger.store(
                **{
                    "Metrics/EvalEpRet": np.mean(eval_rew),
                    "Metrics/EvalEpCost": np.mean(eval_cost),
                    "Metrics/EvalEpLen": np.mean(eval_len),
                }
            )

        eval_end_time = time.time()

        # update lagrange multiplier
        ep_costs = logger.get_stats("Metrics/EpCost")

        # update policy
        data = buffer.get()
        old_distribution = policy.actor(data["obs"])

        # comnpute advantage
        advantage = data["adv_r"]
        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["act"],
                data["log_prob"],
                data["target_value_r"],
                data["target_value_c"],
                advantage,
                data["cost"],
                data["next_obs"],
                data["lyapunov_current"],
                data["lyapunov_next"],
                data["is_start"],
                data["terminate"],
                data["delta_lyapunov"],
                data["current_step"]
            ),
            batch_size=config.get("batch_size", args.steps_per_epoch//config.get("num_mini_batch", 1)),
            shuffle=True,
        )
        update_counts = 0
        final_kl = torch.ones_like(old_distribution.loc)
        lambda_penalty = 10  # Coefficient for Lyapunov penalty
        total_loss_sum=0
        epsilon=0.01
        gamma = config['gamma']  # Discount factor from your config
        lagrange_coefficient_lol=lagrange_update
        mean_lyapunov_initial_penalty_sum=0.0
        start_states=0
        initial_target_value_c_mean=0.
        start_is_1000_multiple=True
        start_witness=0
        number_of_redressments=0
        for _ in range(config["learning_iters"]):
            for (
                obs_b,
                act_b,
                log_prob_b,
                target_value_r_b,
                target_value_c_b,
                adv_b,
                cost_b,
                next_obs_b,
                lyapunov_current_b,
                lyapunov_next_b,
                is_start_b,
                terminate_b,
                delta_lyapunov_b,
                current_step_b
            ) in dataloader:
                #zeroing gradients for the optimizer
                lyapunov_optimizer.zero_grad()
                while ((torch.relu(target_value_c_b-0.1-lyapunov_function(obs_b).detach()))>0.).any():
                    number_of_redressments+=1
                    lyapunov_current_prelim = lyapunov_function(obs_b)
                    loss_c_lyapunov=(torch.relu(target_value_c_b-lyapunov_current_prelim)).mean()*1000
                    """ for name, param in lyapunov_function.named_parameters():
                        print(f"{name} requires grad: {param.requires_grad}")
                    print("Condition:", ((torch.relu(target_value_c_b-0.1-lyapunov_current_prelim))>0.))
                    print("Does loss require gradients?:", loss_c_lyapunov.requires_grad)
                    print("lyapunov_current_prelim",lyapunov_current_prelim)
                    print("lyapunov_mse_loss", loss_c_lyapunov)
                    print("target_value_c_b",target_value_c_b) """
                    lyapunov_optimizer.zero_grad()
                    loss_c_lyapunov.backward()
                    lyapunov_optimizer.step()

                lyapunov_optimizer.zero_grad()
                #zeroing gradients for the optimizer
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                cost_critic_optimizer.zero_grad()
                loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b), target_value_c_b)
                lyapunov_optimizer.zero_grad()
                delta_lyapunov_optimizer.zero_grad()
                delta_lyapunov_critic_lag_optimizer.zero_grad()
                delta_lyapunov_critic_copy_optimizer.zero_grad()
                delta_lyapunov_critic_lag_copy_optimizer.zero_grad()
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost_critic.parameters():
                        loss_c += param.pow(2).sum() * 0.001
                actor_optimizer.zero_grad()
                #policy loss calculation
                distribution = policy.actor(obs_b)
                log_prob = distribution.log_prob(act_b).sum(dim=-1)
                ratio = torch.exp(log_prob - log_prob_b)
                ratio_cliped = torch.clamp(ratio, 0.85, 1.15)
                active_mask = ~terminate_b  # True for non-terminated states
                total_lyapunov_loss = torch.zeros(1, device=obs_b.device)
                #lyapunov_current_b=lyapunov_current_b.detach()
                #lyapunov_next_b=lyapunov_next_b.detach()

                lyapunov_current = lyapunov_function(obs_b[active_mask])
                lyapunov_next = lyapunov_function(next_obs_b[active_mask])

                """ if active_mask.any():
                    ratio_current = lyapunov_current / lyapunov_current_b
                    clipped_ratio_current = torch.clamp(ratio_current, 1-epsilon, 1+epsilon)
                    clipped_current = clipped_ratio_current * lyapunov_current_b
                    delta_lyapunov = cost_b +gamma*lyapunov_next-clipped_current
                    delta_lyapunov_masked=(delta_lyapunov*ratio_cliped)[active_mask]
                total_lyapunov_loss = torch.relu(delta_lyapunov_masked.mean()) """
                if active_mask.any():
                    # Calculate ratios and apply clipping
                    ratio_current = (lyapunov_current+0.1) / (lyapunov_current_b[active_mask]+0.1)
                    epsilon_tensor = torch.full_like(ratio_current, epsilon)
                    clipped_ratio_current = torch.min(ratio_current, 1+epsilon_tensor)
                    clipped_current = clipped_ratio_current * (lyapunov_current_b[active_mask]+0.1)-0.1
                    print("lyapunov_current:",lyapunov_current)
                    print("lyapunov_current_b:",lyapunov_current_b[active_mask])
                    print("ratio_current",ratio_current)
        
                    ratio_next = (lyapunov_next+0.1) / (lyapunov_next_b[active_mask]+0.1)
                    clipped_ratio_next = torch.max(ratio_next, 1-epsilon_tensor)
                    clipped_next = clipped_ratio_next * (lyapunov_next_b[active_mask]+0.1)-0.1
                    print("lyapunov_next:",lyapunov_next)
                    print("lyapunov_next_b:",lyapunov_next_b[active_mask])
                    print("ratio_next",ratio_current)
                    print("cost_b",cost_b[active_mask])
                    #Calculate clipping penalty
                    penalty_current = -torch.relu(ratio_current - 1 - epsilon) * (torch.abs(lyapunov_current_b[active_mask])+0.1)
                    penalty_next = torch.relu(1-epsilon-ratio_next) * (torch.abs(lyapunov_next_b[active_mask])+0.1)
                    #Calculate clipped delta loss
                    delta_lyapunov_clipped= (cost_b[active_mask] + gamma * torch.relu(clipped_next) - torch.relu(clipped_current))*((torch.pow(lambda_lyapunov, current_step_b[active_mask])))
                    #delta_lyapunov = (cost_b + gamma * lyapunov_next - lyapunov_current)*torch.pow(lambda_lyapunov, current_step_b) #+ penalty_current + penalty_next
                    print('delta_lyapunov:',delta_lyapunov)
                    delta_lyapunov_times_ratio_b=(cost_b[active_mask]+gamma*lyapunov_next_b[active_mask]-lyapunov_current_b[active_mask])*((torch.pow(lambda_lyapunov, current_step_b))[active_mask])
                    #delta_lyapunov_times_ratio=(delta_lyapunov* ratio_cliped)[active_mask]

                    delta_lyapunov_clipped_with_ratio_pi=torch.max(ratio[active_mask] * delta_lyapunov_clipped, ratio_cliped[active_mask] * delta_lyapunov_clipped)+2*penalty_next+2*penalty_current
                    print('delta_lyapunov_times_ratio_b:',delta_lyapunov_times_ratio_b)
                    #Calculate total Lyapunov loss
                    lyapunov_loss_b=(delta_lyapunov_times_ratio_b).mean()
                    lyapunov_loss= delta_lyapunov_clipped_with_ratio_pi.mean()
                    #lyapunov_loss_clipped=delta_lyapunov_clipped_with_ratio_pi.mean()
                    #total_lyapunov_loss = lyapunov_loss #+ penalty_current.mean() + penalty_next.mean()
                else:
                    total_lyapunov_loss=0.0
                #Dealing with projections?
                #delta_lyapunov = (cost_b + gamma * lyapunov_next - lyapunov_current)*torch.pow(lambda_lyapunov, current_step_b)
                #pi_proj=scipy(ratio,(cost_b + gamma * lyapunov_next - lyapunov_current)*torch.pow(lambda_lyapunov, current_step_b))


                loss_pi = -torch.min(ratio * adv_b, ratio_cliped * adv_b).mean()


                if is_start_b.any():
                    initial_lyapunov_divergence = torch.where(is_start_b[active_mask], lyapunov_current - lyapunov_threshold, torch.tensor(float('nan'), device=lyapunov_current.device))
                    initial_lyapunov_divergence_b = torch.where(is_start_b[active_mask], lyapunov_current_b[active_mask] - lyapunov_threshold, torch.tensor(float('nan'), device=lyapunov_current_b.device))
                    initial_target_value_c_mean += (target_value_c_b[is_start_b]).mean()
                    mean_initial_lyapunov_penalty=torch.relu(torch.nanmean(initial_lyapunov_divergence))
                    mean_lyapunov_initial_penalty_sum+=torch.nanmean(initial_lyapunov_divergence_b)
                    if ((current_step_b[is_start_b]) % 1000).any():
                        start_is_1000_multiple=False
                    start_states+=1
                else:
                    mean_initial_lyapunov_penalty=0.0
                #initial_lyapunov_divergence = torch.zeros_like(lyapunov_current)
                #initial_lyapunov_divergence_b = torch.zeros_like(lyapunov_current)
                #initial_lyapunov_divergence[is_start_b]=lyapunov_current[is_start_b] - lyapunov_threshold
                #initial_lyapunov_divergence_b[is_start_b]=lyapunov_current_b[is_start_b] - lyapunov_threshold
                #initial_lyapunov_penalty = torch.zeros_like(lyapunov_current)
                #initial_condition_violation = (lyapunov_current > lyapunov_threshold) & is_start_b
                #initial_lyapunov_penalty[initial_condition_violation] = lyapunov_initial_penalty_scale * (lyapunov_current[initial_condition_violation] - lyapunov_threshold)
                
                #Delta Lyapunov Critic Loss
                if active_mask.any():
                    with torch.no_grad():
                        delta_lyapunov_critic_lag_b=delta_lyapunov_critic_lag_copy(obs_b[active_mask])
                        delta_lyapunov_critic_b=delta_lyapunov_critic_copy(obs_b[active_mask])
                    delta_lyapunov_critic_error_mean=mse_loss((delta_lyapunov_critic(obs_b[active_mask])),delta_lyapunov_times_ratio_b)
                    delta_lyapunov_critic_lag_error_mean=mse_loss((delta_lyapunov_critic_lag_b+delta_lyapunov_critic_b),delta_lyapunov_critic_lag(obs_b[active_mask]))
                    #delta_lyapunov_critic_lag_error_mean=mse_loss(torch.relu(delta_lyapunov_critic_lag_b+delta_lyapunov_critic_b),delta_lyapunov_critic_lag(obs_b[active_mask]))
                    #delta_lyapunov_critic_error_mean=torch.abs(((delta_lyapunov_critic(obs_b))[active_mask]).mean()-lyapunov_loss_b)
                else:
                    delta_lyapunov_critic_error_mean=0.0

                #Compute lagrange coefficient
                with torch.no_grad():
                    lagrange_coefficient=delta_lyapunov_critic_lag_copy(obs_b[active_mask])+delta_lyapunov_critic_copy(obs_b[active_mask])
                #lagrange_coefficient=lyapunov_loss_b
                #with torch.no_grad():
                    #for i in range(min(num_instances,epoch-1)):
                        #lagrange_coefficient=lagrange_coefficient+((delta_lyapunov_critic_history[i])(obs_b[active_mask])).mean()
                        #print("delta_lyapuunov_critic_history",(delta_lyapunov_critic_history[i])(obs_b))
                        #print("delta_lyapunov_critic_history_mean",((delta_lyapunov_critic_history[i])(obs_b)).mean())



                """ lagrange_coefficient=torch.zeros_like(lyapunov_current, dtype=torch.float32, device=device)
                with torch.no_grad():
                    for i in range(min(num_instances,epoch-1)):
                        if not lagrange_coefficient.shape==(delta_lyapunov_critic_history[i])(obs_b[active_mask]).shape:
                            print("Shape mismatch detected!")
                            print("Shape of lagrange_coefficient:", lagrange_coefficient.shape)
                            print("Shape of obs_b:", obs_b.shape)
                            print("Shape of delta_output:", (delta_lyapunov_critic_history[i])(obs_b[active_mask]).shape)
                            print("lagrange:",lagrange_coefficient)
                            print("obs_b",obs_b)
                            print("(delta_lyapunov_critic_history[i])(obs_b)",(delta_lyapunov_critic_history[i])(obs_b[active_mask]))
                        lagrange_coefficient=lagrange_coefficient+((delta_lyapunov_critic_history[i])(obs_b[active_mask]))
                        print("delta_lyapuunov_critic_history",(delta_lyapunov_critic_history[i])(obs_b[active_mask]))
                        print("delta_lyapunov_critic_history_mean",((delta_lyapunov_critic_history[i])(obs_b[active_mask])).mean()) """
                lagrange_mask=(lagrange_coefficient>0) | (delta_lyapunov_critic_b>0)
                """ print("lagrange_coefficient:",lagrange_coefficient)
                print("lyapunov_loss",lyapunov_loss) """
                total_lyapunov_loss = 10*((torch.max(lagrange_coefficient[lagrange_mask],delta_lyapunov_critic_b[lagrange_mask])*delta_lyapunov_clipped_with_ratio_pi[lagrange_mask]).mean())
                #printing stuff


                total_loss_sum+=lyapunov_loss_b
                print("lagrange_coefficient:",lagrange_coefficient)
                print("lagrange_coefficient_mean:",lagrange_coefficient.mean())
                print("delta_lyapunov_critic_mean:",((delta_lyapunov_critic(obs_b[active_mask]))).mean())
                print("lyapunov_loss_b",lyapunov_loss_b)
                print("lyapunov_current.mean()",lyapunov_current_b.mean())
                print("lyapunov_next.mean()",lyapunov_next_b.mean())
                print("total_lyapunov_loss:",total_lyapunov_loss)
                print("is_start_b",is_start_b)
                print("loss pi:",loss_pi)
                print("total_lyapunov_sum:", total_loss_sum)
                print("Mean Initial Lyapunov Penalty:", mean_initial_lyapunov_penalty)
                print("Mean Initial Lyapunov Penalty Mean:", mean_lyapunov_initial_penalty_sum/(start_states+0.01))
                print("Mean initial target cost",initial_target_value_c_mean/(start_states+0.01))
                print("start_is_1000_m",start_is_1000_multiple)
                print("start_witness",start_witness)
                print("lagrange:",lagrange_coefficient_lol)
                print("number_of_redressments",number_of_redressments)
                if epoch>15:
                    total_loss = loss_pi + 2*loss_r + loss_c + total_lyapunov_loss + delta_lyapunov_critic_error_mean + mean_initial_lyapunov_penalty + delta_lyapunov_critic_lag_error_mean \
                        if config.get("use_value_coefficient", False) \
                        else loss_pi + loss_r + loss_c + total_lyapunov_loss + mean_initial_lyapunov_penalty + delta_lyapunov_critic_error_mean + delta_lyapunov_critic_lag_error_mean
                else:
                    total_loss = loss_pi + 2*loss_r + loss_c + total_lyapunov_loss + mean_initial_lyapunov_penalty + delta_lyapunov_critic_error_mean + delta_lyapunov_critic_lag_error_mean \
                        if config.get("use_value_coefficient", False) \
                        else loss_pi + loss_r + loss_c + total_lyapunov_loss + mean_initial_lyapunov_penalty + delta_lyapunov_critic_error_mean + delta_lyapunov_critic_lag_error_mean
                #total_loss/=(2+lagrange_coefficient)
                """ print("Before backward - checking gradients:")
                    for name, param in lyapunov_function.named_parameters():
                    if param.requires_grad:
                        print(f"{name} gradient: {param.grad}")
                   for name, param in policy.reward_critic.named_parameters():
                    if param.requires_grad:
                        print(f"{name} gradient: {param.grad}")
                for name, param in policy.cost_critic.named_parameters():
                    if param.requires_grad:
                        print(f"{name} gradient: {param.grad}")
                for name, param in policy.actor.named_parameters():
                    if param.requires_grad:
                        print(f"{name} gradient: {param.grad}")
                print("Before backward - checking requires gradients:")
                for name, param in lyapunov_function.named_parameters():
                    if param.requires_grad:
                        print(f"{name} will accumulate gradients: {param.requires_grad and param.grad is not None}")
                for name, param in policy.reward_critic.named_parameters():
                    if param.requires_grad:
                        print(f"{name} will accumulate gradients: {param.requires_grad and param.grad is not None}")
                for name, param in policy.cost_critic.named_parameters():
                    if param.requires_grad:
                        print(f"{name} will accumulate gradients: {param.requires_grad and param.grad is not None}")
                for name, param in policy.actor.named_parameters():
                    if param.requires_grad:
                        print(f"{name} will accumulate gradients: {param.requires_grad and param.grad is not None}") """
                total_loss.backward()

                """ print("After backward - checking gradients:")
                for name, param in lyapunov_function.named_parameters():
                    if param.requires_grad:
                        print(f"{name} gradient: {param.grad}")
                for name, param in policy.reward_critic.named_parameters():
                    if param.requires_grad:
                        print(f"{name} gradient: {param.grad}")
                for name, param in policy.cost_critic.named_parameters():
                    if param.requires_grad:
                        print(f"{name} gradient: {param.grad}")
                for name, param in policy.actor.named_parameters():
                    if param.requires_grad:
                        print(f"{name} gradient: {param.grad}")
                try:
                    total_loss.backward()  # This should raise an error if graph is already freed
                    print("Second backward call succeeded (unexpected under normal circumstances)")
                except RuntimeError as e:
                    print("Second backward call failed as expected: ", str(e)) """
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost_critic_optimizer.step()
                actor_optimizer.step()
                lyapunov_optimizer.step()
                delta_lyapunov_optimizer.step()
                delta_lyapunov_critic_lag_optimizer.step()
                logger.store(
                    **{
                        "Loss/Loss_reward_critic": loss_r.mean().item(),
                        "Loss/Loss_cost_critic": loss_c.mean().item(),
                        "Loss/Loss_actor": loss_pi.mean().item(),
                    }
                )

            new_distribution = policy.actor(data["obs"])
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            final_kl = kl
            update_counts += 1
            if kl > config["target_kl"]:
                break
        update_end_time = time.time()
        actor_scheduler.step()
        print("terminate_test:",terminate_test)
        if not logger.logged:
            # log data
            logger.log_tabular("Metrics/EpRet")
            logger.log_tabular("Metrics/EpCost")
            logger.log_tabular("Metrics/EpLen")
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost")
                logger.log_tabular("Metrics/EvalEpLen")
            logger.log_tabular("Train/Epoch", epoch + 1)
            logger.log_tabular("Train/TotalSteps", (epoch + 1) * args.steps_per_epoch)
            logger.log_tabular("Train/StopIter", update_counts)
            logger.log_tabular("Train/KL", final_kl)
            logger.log_tabular("Train/LR", actor_scheduler.get_last_lr()[0])
            logger.log_tabular("Loss/Loss_reward_critic")
            logger.log_tabular("Loss/Loss_cost_critic")
            logger.log_tabular("Loss/Loss_actor")
            logger.log_tabular("Time/Rollout", rollout_end_time - rollout_start_time)
            if args.use_eval:
                logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
            logger.log_tabular("Time/Update", update_end_time - eval_end_time)
            logger.log_tabular("Time/Total", update_end_time - rollout_start_time)
            logger.log_tabular("Value/RewardAdv", data["adv_r"].mean().item())
            logger.log_tabular("Value/CostAdv", data["adv_c"].mean().item())

            logger.dump_tabular()
            if (epoch+1) % 100 == 0 or epoch == 0:
                logger.torch_save(itr=epoch)
                if args.task not in isaac_gym_map.keys():
                    logger.save_state(
                        state_dict={
                            "Normalizer": env.obs_rms,
                        },
                        itr = epoch
                    )
    logger.close()


if __name__ == "__main__":
    args, cfg_env = single_agent_args()
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args, cfg_env)
    else:
        main(args, cfg_env)
