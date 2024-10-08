import imageio
import cv2
import csv
import gym
import numpy as np
from tqdm import tqdm
import time
import uuid
import os
import random
import pygame
from pygame.locals import *

from random_agent import AtariPreprocessing

# Initate game

n_steps = 1_0000
n_seeds = 1

games = [
    # "SpaceInvadersNoFrameskip-v4",
    # 'BreakoutNoFrameskip-v4',
    # 'FreewayNoFrameskip-v4',
    # 'CarnivalNoFrameskip-v4',
    'PongNoFrameskip-v4'
]

log_video = False
output_name = 'random_policy_empa_like'


if __name__ == "__main__":

    for game in tqdm(games):
        for n_seed in range(n_seeds):
            env = gym.make(game, render_mode='human')
            env.reset()

            AP = AtariPreprocessing(
                env,
                terminal_on_life_loss=False,
                game_name=game.replace("NoFrameskip-v4", ""),
                tag=n_seed,
                log_video = log_video,
                output_name = output_name
            )
            lives = env.ale.lives()

            i = 0
            while i < n_steps:
                press_type = random.choice(["long", "short"])
                action = env.action_space.sample()
                print("action", action)

                if press_type == "long":
                    for _ in range(15):
                        observation, reward, game_over, info = AP.step(action)
                        i += 1
                        if game_over:
                            AP.reset()
                else:
                    for _ in range(4):
                        observation, reward, game_over, info = AP.step(action)
                        i += 1
                        if game_over:
                            AP.reset()
                    for _ in range(11):
                        observation, reward, game_over, info = AP.step(0)
                        i += 1
                        if game_over:
                            AP.reset()

    
# def key_converter(key_event):
#     """
#     'LEFT' for moving left, 'RIGHT' for moving right, and 'SPACE' for firing
#     else do nothing
#     """   
#     if key_event.type == KEYDOWN:
#         if key_event.key == K_LEFT:
#             return 3  
#         elif key_event.key == K_RIGHT:
#             return 2  
#         elif key_event.key == K_SPACE:
#             return 1  
#     return None  

# if __name__ == "__main__":

#     pygame.init()
#     clock = pygame.time.Clock()

#     for game in tqdm(games):
#         for n_seed in range(n_seeds):
#             env = gym.make(game, render_mode='human')
#             env.reset()

#             AP = AtariPreprocessing(
#                 env,
#                 terminal_on_life_loss=False,
#                 game_name=game.replace("NoFrameskip-v4", ""),
#                 tag=n_seed,
#                 log_video = log_video,
#                 output_name = output_name
#             )
#             lives = env.ale.lives()

#             action = 0
#             game_over = False
#             while True:
#                 for event in pygame.event.get():
#                     if event.type == QUIT:
#                         pygame.quit()
#                         quit()
#                     else:
#                         confirmed_action = key_converter(event)
#                         if confirmed_action is not None:
#                             action = confirmed_action

#                 observation, reward, game_over, info = AP.step(action) 

#                 if game_over:
#                     env.reset()
#                     AP.reset()

#                 clock.tick(600) 

#     pygame.quit()
