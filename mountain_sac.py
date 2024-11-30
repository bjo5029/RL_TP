import os, time
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecMonitor
from callback import SaveOnBestTrainingRewardCallback # 사용자 정의 콜백

import numpy as np
# np.bool8 = np.bool

def train(env_id, log_base_dir="logs", model_base_dir="models", model_name=None, total_timesteps=800000):   
    # 모델 저장 디렉토리가 존재하지 않을 경우 자동으로 생성
    os.makedirs(log_base_dir, exist_ok=True)
    os.makedirs(model_base_dir, exist_ok=True)

    # 환경 설정
    env = make_vec_env(env_id, n_envs=32)
    env = VecMonitor(env, log_base_dir)

    # 모델 이름 기본값 설정
    if model_name is None:    
        model_name = env_id + "_SAC"

    log_dir = os.path.join(log_base_dir, env_id)

    # PPO 모델 초기화
    model = SAC(
        policy="MlpPolicy",
        env=env,
        # -------------------------------------------------
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=512,
        ent_coef=0.1, # 0.1 -> 0.5
        train_freq=32,
        gradient_steps=32,
        gamma=0.9999,
        tau=0.01,
        learning_starts=0,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]),
        
        # n_steps=32,                 
        # n_epochs=10,                 
        # gamma=0.9999,                
        # gae_lambda=0.9,              
        # clip_range=0.1,
        # # clip_range_vf=None,
        # normalize_advantage=True,
        # vf_coef=0.19,
        # max_grad_norm=5,
        # use_sde=True,
        # # sde_sample_freq=-1,
        # # rollout_buffer_class=None,
        # # rollout_buffer_kwargs=None,
        # # target_kl=None,
        # # stats_window_size=100,
        # policy_kwargs=dict(          
        #     log_std_init=-3.29,
        #     ortho_init=False
        # ),
        # -------------------------------------------------
        tensorboard_log = log_dir,
        verbose=1,
        seed=None,
        device='auto',
    )            

    # 학습 콜백 설정
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_base_dir)   

    # 모델 학습
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=4,
        tb_log_name="SAC",
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # 모델 저장
    save_path = os.path.join(os.getcwd(), model_base_dir, model_name)
    model.save(save_path)

    # 환경 종료
    env.close()


def run(env_id, model_base_dir="models", model_name=None, n_episodes=5):
    # 환경 설정
    env = gym.make(env_id, render_mode='human')
    
    # 모델 로드
    if model_name is None:
        model_name = env_id + "_SAC"

    model_path = os.path.join(model_base_dir, model_name)
    model = SAC.load(model_path, env)

    # 학습된 모델 실행
    for episode in range(n_episodes):
        obs, info = env.reset()

        episode_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.01)
            episode_reward += reward
            if terminated or truncated:
                time.sleep(1.0)
                print("n_steps: {}".format(episode_reward))
                episode_reward = 0
                break
            
    # 환경 종료
    env.close()


if __name__ == "__main__":
    env_id = "MountainCarContinuous-v0"
    
    train(env_id)
    run(env_id)