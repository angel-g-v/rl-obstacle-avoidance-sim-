from camera_env import CameraEnv
from stable_baselines3 import PPO

env = CameraEnv()

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_logs"  # optional logging
)

# Custom loop with live rendering during training
TIMESTEPS = 10000
for i in range(30):  # 30 * 10k = 300k steps
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    
    # Render current behavior
    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        #env.render()
        if done:
            break

model.save("ppo_camera_model")
