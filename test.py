import torch
import gym
import gym3
from procgen import ProcgenGym3Env


def main():
    mod = torch.load('/home/zaynahjaved/phasic-policy-gradient/logs/ppg/model.jd') 
    env = ProcgenGym3Env(num=1, env_name='fruitbot', start_level=600, num_levels=400, distribution_mode='easy')
    step = 0
    rew_total = 0
    while True:
        rew, obs, first = env.observe()
        state_in = {'pi': obs, 'vf': 'x'}
        env.act(mod.act(obs,first,obs))
        rew_total += rew
        print(rew_total)
        step += 1




if __name__ == '__main__':
    main()
