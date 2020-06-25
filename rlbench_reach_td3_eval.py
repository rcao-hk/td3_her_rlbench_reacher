import argparse
import gym
import rlbench.gym
from rlbench_reach_agent import RLBenchReachAgent
from rlbench_reach_model import RLBenchReachModel
import logging
import parl
import numpy as np
from parl.utils import action_mapping
import time
import imageio
import os
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
MAX_STEPS_PER_EPISODES = 100


class ImageLogger(object):
    def __init__(self, path):
        self.path = path
        self.image_dict = []

    def __call__(self, _frame):
        self.image_dict.append(_frame)

    def save(self):
        imageio.mimsave(self.path, self.image_dict, 'GIF')


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


def run_evaluate_episode(env, agent, image_recoder):
    obs = env.reset()
    total_reward = 0
    episode_goal = np.expand_dims(obs[-3:], axis=0)
    steps = 0
    while MAX_STEPS_PER_EPISODES - steps:
        steps += 1
        image_recoder(env.render(mode='rgb_array'))
        batch_obs = np.expand_dims(obs[8:15], axis=0)
        batch_obs_with_goal = np.concatenate((batch_obs, episode_goal), axis=1)
        action = agent.predict(batch_obs_with_goal.astype('float32'))
        action = np.squeeze(action)
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)
        # time.sleep(0.1)
        # print(reward)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward


def main(args):
    env = gym.make(args.env, render_mode='rgb_array')

    # env = gym.make(args.env, render_mode='human')
    # env.seed(ENV_SEED)
    # env = Monitor(env, 'RLBench/records', force=True)

    env.reset()
    logger = LoggingInstance('RLBench/log/eval.txt')
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

    model_idx = 80000
    recode_path = 'RLBench/records/' + str(model_idx)
    if not os.path.exists(recode_path):
        os.makedirs(recode_path)

    agent.restore_critic('RLBench/train_model/critic_'+str(model_idx)+'.ckpt')
    agent.restore_actor('RLBench/train_model/actor_'+str(model_idx)+'.ckpt')

    for epics in range(1, 11):
        image_recoder = ImageLogger('RLBench/records/'+str(model_idx)+'/video_'+str(epics)+'.gif')
        evaluate_reward = run_evaluate_episode(env, agent, image_recoder)
        logger.logging_string('Episodes {}, Evaluate reward: {}'.format(
            epics, evaluate_reward))
        time.sleep(1)
        image_recoder.save()

    env.close()
    logger.decorator()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', help='RLBench environment name', default='reach_target-state-v0')
    parser.add_argument(
        '--eval_episodes',
        type=int,
        default=int(10),
        help='store episodes number')

    args = parser.parse_args()
    main(args)