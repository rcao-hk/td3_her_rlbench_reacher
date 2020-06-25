import argparse
import gym
import rlbench.gym
import numpy as np
import time
import parl
from rlbench_reach_agent import RLBenchReachAgent
from rlbench_reach_model import RLBenchReachModel
from parl.utils import action_mapping, ReplayMemory, tensorboard
# from parl.utils import logger, tensorboard
import logging

MAX_EPISODES = 5000
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = int(1e6)
WARMUP_SIZE = 1e3
BATCH_SIZE = 256
ENV_SEED = 1
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise
MAX_STEPS_PER_EPISODES = 200


class LoggingInstance(object):
    def __init__(self, logfile):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler(logfile, mode='a')
        self.fh.setLevel(logging.DEBUG)  # 用于写到file的等级开关
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.logger.addHandler(self.fh)

    def logging_string(self, string_msg):
        self.logger.info(string_msg)

    def decorator(self):
        self.logger.removeHandler(self.fh)


def run_train_episode(env, agent, rpm):
    obs_list = []
    action_list = []
    reward_list = []
    terminal_info = []
    obs = env.reset()
    obs_list.append(obs)
    total_reward = 0
    steps = 0
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])
    episode_goal = np.expand_dims(obs[-3:], axis=0)
    while MAX_STEPS_PER_EPISODES-steps:
        steps += 1
        batch_obs = np.expand_dims(obs[8:15], axis=0)
        batch_obs_with_goal = np.concatenate((batch_obs, episode_goal), axis=1)
        if rpm.size() < WARMUP_SIZE:
            action = env.action_space.sample()
        else:
            action = agent.predict(batch_obs_with_goal.astype('float32'))
            action = np.squeeze(action)

            # Add exploration noise, and clip to [-max_action, max_action]
            # action += noise()
            action = np.random.normal(action, EXPL_NOISE * max_action)
            action = np.clip(action, min_action, max_action)

        next_obs, reward, done, info = env.step(action)
        obs_list.append(next_obs)
        action_list.append(action)
        reward_list.append(reward)
        terminal_info.append(done)

        obs = next_obs
        total_reward += reward
        # print(total_reward)
        if done:
            break

    for idx in range(steps):
        obs = obs_list[idx]
        next_obs = obs_list[idx + 1]
        obs_desired_goal = np.concatenate((obs[8:15], obs[-3:]))
        next_obs_desired_goal = np.concatenate((next_obs[8:15], next_obs[-3:]))
        action = action_list[idx]
        reward = reward_list[idx]
        done = terminal_info[idx]
        obs_achieved_goal = np.concatenate((obs[8:15], obs[22:25]))
        next_obs_achieved_goal = np.concatenate((next_obs[8:15], next_obs[22:25]))
        rpm.append(obs_desired_goal, action, reward, next_obs_desired_goal, done)
        rpm.append(obs_achieved_goal, action, 1, next_obs_achieved_goal, True)

    if rpm.size() > WARMUP_SIZE:
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
            BATCH_SIZE)
        agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                    batch_terminal)

    return total_reward


def run_evaluate_episode(env, agent, render):
    obs = env.reset()
    total_reward = 0
    episode_goal = np.expand_dims(obs[-3:], axis=0)
    steps = 0
    while MAX_STEPS_PER_EPISODES - steps:
        steps += 1
        batch_obs = np.expand_dims(obs[8:15], axis=0)
        batch_obs_with_goal = np.concatenate((batch_obs, episode_goal), axis=1)
        action = agent.predict(batch_obs_with_goal.astype('float32'))
        action = np.squeeze(action)
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)
        if render:
            env.render()
        # print(reward)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward


def main(args):
    env = gym.make(args.env)
    # env = gym.make(args.env, render_mode='human')
    env.reset()
    # env.seed(ENV_SEED)

    logger = LoggingInstance('RLBench/log/train.txt')
    obs_dim = 7
    goal_dim = 3

    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    model = RLBenchReachModel(act_dim, max_action)
    algorithm = parl.algorithms.TD3(
        model,
        max_action=max_action,
        gamma=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = RLBenchReachAgent(algorithm, obs_dim + goal_dim, act_dim)

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim + goal_dim, act_dim)

    agent.restore_critic('RLBench/train_model/critic_16000.ckpt')
    agent.restore_actor('RLBench/train_model/actor_16000.ckpt')

    test_flag = 0
    store_flag = 0
    total_episodes = 16000
    while total_episodes < args.train_total_episodes:
        train_reward = run_train_episode(env, agent, rpm)
        total_episodes += 1
        logger.logging_string('Episodes: {} Reward: {}'.format(total_episodes, train_reward))
        tensorboard.add_scalar('train/episode_reward', train_reward,
                               total_episodes)

        if total_episodes // args.test_every_episodes >= test_flag:
            while total_episodes // args.test_every_episodes >= test_flag:
                test_flag += 1
            evaluate_reward = run_evaluate_episode(env, agent, render=False)
            logger.logging_string('Episodes {}, Evaluate reward: {}'.format(
                total_episodes, evaluate_reward))

            tensorboard.add_scalar('eval/episode_reward', evaluate_reward,
                                   total_episodes)

        if total_episodes // args.store_every_episodes >= store_flag:
            while total_episodes // args.store_every_episodes >= store_flag:
                store_flag += 1
                agent.save_actor('RLBench/train_model/actor_' + str(total_episodes) + '.ckpt')
                agent.save_critic('RLBench/train_model/critic_' + str(total_episodes) + '.ckpt')

    logger.decorator()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', help='Fetch environment name', default='reach_target-state-v0')
    parser.add_argument(
        '--train_total_episodes',
        type=int,
        default=int(3e5),
        help='maximum training episodes')
    parser.add_argument(
        '--test_every_episodes',
        type=int,
        default=int(8e2),
        help='the step interval between two consecutive evaluations')
    parser.add_argument(
        '--store_every_episodes',
        type=int,
        default=int(4e3),
        help='the step interval for model store')

    args = parser.parse_args()
    main(args)



