import cv2
from shooterenv import ShooterEnv
from stable_baselines3 import PPO


env = ShooterEnv()
model = PPO.load('models/1729416037/10000.zip') # Directory of model


episodes = 5

for episode in range(episodes):
    obs, _ = env.reset()
    done, truncated = False, False
    while not done and not truncated:
        action, _ = model.predict(obs)
        # print("action", random_action)
        obs, reward, done, truncated, info = env.step(action)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        cv2.imshow('Circle Shooter', env.frame)
        # print("reward", reward)

    print('I DIE')

cv2.destroyAllWindows()