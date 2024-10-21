from shooterenv import ShooterEnv
from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
import os
import time

timer = int(time.time())
MODEL_DIR = f'models/{timer}'
logdir = 'logs/'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = ShooterEnv()
env.reset()

episodes = 1000
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
for i in range(episodes):
    model.learn(total_timesteps=10000,reset_num_timesteps=False, tb_log_name=f"PPO-{timer}")
    model.save(f"{MODEL_DIR}/{episodes*i}")
