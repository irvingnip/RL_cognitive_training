import torch
import time
import sys
import csv
import os
import numpy as np
from VGDLEnv import VGDLEnv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import tkinter as tk

class Player(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Env = VGDLEnv(self.config["game_name"], 'all_games')
        self.Env.set_level(0)

        print('Initializing at seed = {}'.format(self.config["random_seed"]))

        self.ended = 0
        self.num_episodes = self.config["num_episodes"]

        self.ensure_dir('screens')
        self.ensure_dir('logs')

        # Initialize Tkinter for input handling and rendering
        self.root = tk.Tk()
        self.root.title("Game Window")

        self.action = 0

        # Bind keys to key events
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)

        # Initialize Matplotlib figure and canvas
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.Env.render(), animated=True)
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_screen(self):
        return self.Env.render()

    def update_rendering(self):
        self.fig.canvas.restore_region(self.background)
        self.im.set_array(self.Env.render())
        self.ax.draw_artist(self.im)
        self.canvas.draw()

    def on_key_press(self, event):
        """
        Handles key press events and sets the corresponding action.
        """
        if event.keysym == "Left":
            self.action = 2  # Move left
        elif event.keysym == "Right":
            self.action = 1  # Move right
        elif event.keysym == "Up":
            self.action = 3  # Move up
        elif event.keysym == "Down":
            self.action = 4  # Move down
        elif event.keysym == "space":
            self.action = 5  # Fire

    def on_key_release(self, event):
        """
        Resets action to no operation on key release.
        """
        self.action = 0  # No operation (noop)

    def level_step(self):
        if self.config["level_switch"] == 'sequential':
            if sum(self.recent_history) == int(self.config["criteria"].split('/')[0]): 
                if self.Env.lvl == len(self.Env.env_list) - 1:
                    return 1

                else:  
                    self.Env.lvl += 1
                    self.Env.set_level(self.Env.lvl)
                    print("Next Level!")
                    self.recent_history = [0] * int(self.config["criteria"].split('/')[1])
                    return 0

        elif self.config["level_switch"] == 'random':
            self.Env.lvl = np.random.choice(range(len(self.Env.env_list) - 1))
            self.Env.set_level(self.Env.lvl)
            return 0

        else:
            raise Exception('level switch not specified.')

    def play_game_with_human_input(self):
        print("Starting Game")
        print("-" * 25)

        self.steps = 0
        self.episode_steps = 0
        self.episode = 0
        self.episode_reward = 0

        with open('reward_histories/{}_reward_history_{}_trial{}.csv'.format(self.config['game_name'],
                                                                            self.config['level_switch'],
                                                                            self.config['trial_num']), "w") as file:
            writer = csv.writer(file)
            writer.writerow(["level", "steps", "ep_reward", "win", "game_name", "criteria"])

        with open('object_interaction_histories/{}_object_interaction_history_{}_trial{}.csv'.format(
                self.config['game_name'], self.config['level_switch'], self.config['trial_num']), "w") as file:
            interactionfilewriter = csv.writer(file)
            interactionfilewriter.writerow(
                ['agent_type', 'subject_ID', 'modelrun_ID', 'game_name', 'game_level', 'episode_number', 'event_name',
                'count'])
                
        self.recent_history = [0] * int(self.config["criteria"].split('/')[1])

        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.config["random_seed"])

        self.Env.reset()

        avatar_position_data = {'game_info': (self.Env.current_env._game.width, self.Env.current_env._game.height),
                                'episodes': [[(self.Env.current_env._game.sprite_groups['avatar'][0].rect.left,
                                               self.Env.current_env._game.sprite_groups['avatar'][0].rect.top,
                                               self.Env.current_env._game.time,
                                               self.Env.lvl)]]}

        event_dict = defaultdict(lambda: 0)

        while True:
            # Handle tkinter events for input handling and update
            self.root.update_idletasks()
            self.root.update()

            avatar_position_data['episodes'][-1].append((self.Env.current_env._game.sprite_groups['avatar'][0].rect.left,
                                                         self.Env.current_env._game.sprite_groups['avatar'][0].rect.top,
                                                         self.Env.current_env._game.time,
                                                         self.Env.lvl))

            timestep_events = set()

            for e in self.Env.current_env._game.effectListByClass:
                if e in [('changeResource', 'avatar', 'water'), ('changeResource', 'avatar', 'log')]:
                    pass
                else:
                    timestep_events.add(tuple(sorted((e[1], e[2]))))

            for e in timestep_events:
                event_dict[e] += 1

            self.steps += 1
            self.episode_steps += 1

            self.reward, self.ended, self.win = self.Env.step(self.action)

            self.update_rendering()

            self.episode_reward += self.reward
            self.reward = max(-1.0, min(self.reward, 1.0))

            # Handle the end of an episode
            if self.ended or self.episode_steps > self.config["timeout"]:
                if self.episode_steps > self.config["timeout"]:
                    print("Game Timed Out")

                self.episode += 1
                print(f"Level {self.Env.lvl}, episode reward at step {self.steps}: {self.episode_reward}")
                sys.stdout.flush()

                episode_results = [self.Env.lvl, self.steps, self.episode_reward, self.win, self.config["game_name"],
                                   int(self.config["criteria"].split('/')[0])]

                self.recent_history.insert(0, self.win)
                self.recent_history.pop()

                if self.level_step():
                    with open('reward_histories/{}_reward_history_{}_trial{}.csv'.format(self.config["game_name"],
                                                                                         self.config["level_switch"],
                                                                                         self.config["trial_num"]),
                              "a", newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(episode_results)
                    break

                # Reset the episode
                self.episode_reward = 0
                self.Env.reset()

                avatar_position_data['episodes'].append([(self.Env.current_env._game.sprite_groups['avatar'][0].rect.left,
                                                          self.Env.current_env._game.sprite_groups['avatar'][0].rect.top,
                                                          self.Env.current_env._game.time, self.Env.lvl)])

                event_dict = defaultdict(lambda: 0)
                self.episode_steps = 0

                with open('reward_histories/{}_reward_history_{}_trial{}.csv'.format(self.config["game_name"],
                                                                                     self.config["level_switch"],
                                                                                     self.config["trial_num"]),
                          "a", newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(episode_results)

        print("Game Finished")
        self.root.quit()  


config = {
    'game_name': 'aliens',
    'img_size': 64,
    'random_seed': 1,
    'num_episodes': 10,
    'level_switch': 'sequential',
    'criteria': '1/1',
    'timeout': 5000,
    'trial_num': 1,
}

player = Player(config)
player.play_game_with_human_input()
