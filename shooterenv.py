import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2 
import math
import random

width = 600
height = 800
MAX_ENEMY_UK = 10
MAX_FRAMES_TILL_DONE = 5000
MAX_FRIEND_KILL_COUNT = 2


class ShooterEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=np.array([-90, -1, -1, -1, -1]), high=np.array([90, width, height, width, height]),
                                            shape=(5,), dtype=np.int64)
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        
    def draw_game(self):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw player
        cv2.circle(frame, (int(self.player_x), int(self.player_y)), 
                  self.player_radius, (0, 255, 0), -1)
        
        # Draw gun
        angle_rad = math.radians(self.gun_angle)
        gun_length = 30
        gun_end_x = int(self.player_x - math.sin(angle_rad) * gun_length)
        gun_end_y = int(self.player_y - math.cos(angle_rad) * gun_length)
        cv2.line(frame, (int(self.player_x), int(self.player_y)), 
                (gun_end_x, gun_end_y), (255, 255, 255), 2)

        # Draw bullets
        for bullet in self.bullets:
            cv2.circle(frame, (int(bullet['x']), int(bullet['y'])), 3, (255, 255, 0), -1)

        # Draw enemies (red)
        for enemy in self.enemies:
            cv2.circle(frame, (int(enemy['x']), int(enemy['y'])), 
                      enemy['radius'], (0, 0, 255), -1)

        # Draw friends (blue)
        for friend in self.friends:
            cv2.circle(frame, (int(friend['x']), int(friend['y'])), 
                      friend['radius'], (255, 0, 0), -1)

        # Draw score
        cv2.putText(frame, f'Score: {self.score}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame

    def step(self, action):
        # move player
        if action == 0:  # Rotate gun left
            self.gun_angle = min(90, self.gun_angle + 5)
        elif action == 1:  # Rotate gun right
            self.gun_angle = max(-90, self.gun_angle - 5)
        elif action == 2:  # Shoot
            angle_rad = math.radians(self.gun_angle)
            self.bullets.append({
                'x': self.player_x,
                'y': self.player_y,
                'dx': -math.sin(angle_rad) * self.bullet_speed,
                'dy': -math.cos(angle_rad) * self.bullet_speed
            })
            self.reward = -2

        # spawn circle
        self.spawn_timer += 1
        if self.spawn_timer % 120 == 0:  # Spawn every 120 frames
            x = random.randint(30, self.width - 30)
            if random.random() < 0.7:  # 70% chance for enemy
                self.enemies.append({
                    'x': x,
                    'y': 30,
                    'radius': 20
                })
            else:  # 30% chance for friend
                self.friends.append({
                    'x': x,
                    'y': 30,
                    'radius': 15
                })    

        # update game state
        for bullet in self.bullets[:]:
            bullet['x'] += bullet['dx']
            bullet['y'] += bullet['dy']
            if bullet['y'] < 0 or bullet['y'] > self.height or \
               bullet['x'] < 0 or bullet['x'] > self.width:
                self.bullets.remove(bullet)

        # Update enemies and friends
        for enemy in self.enemies[:]:
            enemy['y'] += self.enemy_speed
            if enemy['y'] > self.height:
                self.enemies.remove(enemy)
                self.enemy_uk += 1
                self.score -= 5
                self.reward = -40

        for friend in self.friends[:]:
            friend['y'] += self.enemy_speed
            if friend['y'] > self.height:
                self.friends.remove(friend)
                self.reward = 20

        # Check collisions
        for bullet in self.bullets[:]:
            # Check enemy collisions
            for enemy in self.enemies[:]:
                dist = math.sqrt((bullet['x'] - enemy['x'])**2 + 
                               (bullet['y'] - enemy['y'])**2)
                if dist < enemy['radius']:
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    if enemy in self.enemies:
                        self.enemies.remove(enemy)
                        self.score += 10
                        self.reward = 160

            # Check friend collisions
            for friend in self.friends[:]:
                dist = math.sqrt((bullet['x'] - friend['x'])**2 + 
                               (bullet['y'] - friend['y'])**2)
                if dist < friend['radius']:
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    if friend in self.friends:
                        self.friends.remove(friend)
                        self.reward = -100
        
        self.frame = self.draw_game()
        
        if self.enemy_uk >= MAX_ENEMY_UK or self.friends_killed >= MAX_FRIEND_KILL_COUNT:
            self.done = True

        self.total_frame += 1
        if self.total_frame > MAX_FRAMES_TILL_DONE:
            self.truncated = True
        
        # reward tracking
        # if action == 3:
        #     self.reward = 1
        if self.truncated or self.done:
            self.reward = -150

        
        self.observation = np.array([self.gun_angle, 
                                    -1 if len(self.friends)==0 else self.friends[0]['x'], 
                                    -1 if len(self.friends)==0 else self.friends[0]['y'], 
                                    -1 if len(self.enemies)==0 else self.enemies[0]['x'], 
                                    -1 if len(self.enemies)==0 else self.enemies[0]['y']])

        info = {}
        return self.observation, self.reward, self.done, self.truncated, info

    def reset(self, seed=None, options=None):
        self.truncated = False
        self.done = False
        self.reward = 0
        self.enemy_uk = 0
        self.friends_killed = 0
        self.total_frame = 0
        self.width = width
        self.height = height
        self.player_x = width // 2
        self.player_y = height - 50 
        self.player_radius = 20
        self.gun_angle = random.randint(0, 90) * (-1 if random.random() > 0.5 else 1)  # degrees
        self.bullets = []
        self.enemies = [{
            'x': random.randint(30, self.width - 30),
            'y': 30,
            'radius': 20
        }]
        self.friends = [{
            'x': random.randint(30, self.width - 30),
            'y': 30,
            'radius': 15
        }]
        self.score = 0
        self.game_over = False
        self.bullet_speed = 20
        self.enemy_speed = 2
        self.spawn_timer = 0
        self.frame = self.draw_game()
        
        self.observation = np.array([self.gun_angle, 
                                    -1 if len(self.friends)==0 else self.friends[0]['x'], 
                                    -1 if len(self.friends)==0 else self.friends[0]['y'], 
                                    -1 if len(self.enemies)==0 else self.enemies[0]['x'], 
                                    -1 if len(self.enemies)==0 else self.enemies[0]['y']])
        info = {}

        return self.observation, info

    def render(self):
        ...

    def close(self):
        ...