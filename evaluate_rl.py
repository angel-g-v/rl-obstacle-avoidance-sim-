from camera_env import CameraEnv
from stable_baselines3 import PPO

env = CameraEnv()
model = PPO.load("ppo_camera_model")

obs = env.reset()
total_reward = 0

print("Starting evaluation (headless)...")
for step in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    
    print(f"Step {step+1:03d}: Action = {action}, Reward = {round(reward, 2)}")
    total_reward += reward

    if done:
        print("Episode ended early.")
        break

print("Total Episode Reward:", round(total_reward, 2))
