import torch
import sys
import csv
import os
import base64
import io
import numpy as np
from PIL import Image
from VGDLEnv import VGDLEnv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import tkinter as tk
from openai import OpenAI
import time
import threading


class Player(object):
    def __init__(self, config):
        self.config = config
        self.api_key = config["api_key"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Env = VGDLEnv(self.config["game_name"], 'all_games')
        self.Env.set_level(0)
        self.client = OpenAI(api_key=self.api_key)

        print('Initializing at seed = {}'.format(self.config["random_seed"]))

        self.ended = 0
        
        self.num_episodes = self.config["num_episodes"]
        self.screen_history = []
        self.recent_history = [0] * int(self.config["criteria"].split('/')[1])

        self.game_state_history = []  # To store game states for assessment

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

        # Create an assessment file
        self.assessment_file = open('assessment_results.txt', 'w')
        self.assessment_file.write("Player Performance Assessments:\n")
        self.assessment_file.write("=" * 40 + "\n")

        # Initialize the GPT-4o thread to None
        self.gpt_thread = None
        self.assessment_result = None

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

    def get_screen_as_base64(self):
        screen = self.get_screen()
        screen = np.ascontiguousarray(screen, dtype=np.uint8)
        image = Image.fromarray(screen)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_assessment_per_steps(self, game_state):
        try:
            screen_base64 = game_state

            prompt = (
                "You are an expert in playing the Atari style game similar to SpaceInvaders based on visual features in an image. "
                "In this game, the blue block is your avatar, and you need to shoot enemies represented as yellow or light green blocks while avoid their bombs represented as red blocks."
                "For the given images, rate the player's performance, such as accuracy of shoots, movement and positioning Strategy, Ability to Avoid Enemy Fire. "
            )

            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screen_base64}"}}
            ]


            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=300,
            )

            # assessment = response.choices[0].message.content.strip()
            # return assessment

            self.assessment_result = response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating action: {e}")
            # return "Error"
            self.assessment_result = "Error"

    def on_key_press(self, event):
        """
        Handles key press events and sets the corresponding action.
        """
        if event.keysym == "Left":
            self.action = 2  
        elif event.keysym == "Right":
            self.action = 1  
        elif event.keysym == "Up":
            self.action = 3  
        elif event.keysym == "Down":
            self.action = 4  
        elif event.keysym == "space":
            self.action = 5

    def on_key_release(self, event):
        """
        Resets action to no operation on key release.
        """
        self.action = 0  

    def level_step(self):
        if self.config["level_switch"] == 'sequential':
            if sum(self.recent_history) == int(self.config["criteria"].split('/')[0]): 
                if self.Env.lvl == len(self.Env.env_list) - 1:
                    print("Learning Finished")
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

            # Capture the game state (as base64 image) and store it
            game_state_base64 = self.get_screen_as_base64()
            self.game_state_history.append(game_state_base64)

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

            self.reward, self.ended, self.win = self.Env.step(self.action)

            self.update_rendering()

            self.steps += 1
            self.episode_steps += 1
            self.episode_reward += self.reward

            # After every 5 steps, assess the performance using GPT-4o
            if len(self.game_state_history) >= 10:
                if self.gpt_thread is None or not self.gpt_thread.is_alive():
                    # Slice the first 10 states to send for assessment
                    states_to_assess = self.game_state_history[:10]
                    
                    # Start a new thread to handle the assessment
                    self.gpt_thread = threading.Thread(target=self.generate_assessment_per_steps, args=(states_to_assess,))
                    self.gpt_thread.start()
                    
                    # Remove the first 10 states from the history
                    del self.game_state_history[:10]
                else:
                    # To avoid overflow, keep the history manageable by limiting its size
                    # Here we keep the last 50 frames only; adjust as necessary
                    max_buffer_size = 50
                    if len(self.game_state_history) > max_buffer_size:
                        del self.game_state_history[:len(self.game_state_history) - max_buffer_size]


            # Check if assessment is complete and write to file
            if self.assessment_result:
                self.assessment_file.write(f"Assessment after {self.steps} steps:\n")
                self.assessment_file.write(self.assessment_result + "\n")
                self.assessment_file.write("=" * 40 + "\n")
                print(f"Assessment received from GPT-4o after {self.steps} steps: {self.assessment_result}")
                self.assessment_result = None  

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
    # 'game_name': 'bait',
    'game_name': 'aliens',
    'img_size': 64,
    'random_seed': 1,
    'num_episodes': 10,
    'level_switch': 'sequential',
    'criteria': '1/1',
    'max_steps': 1000,
    'timeout': 5000,
    'trial_num': 1,
    'api_key': 'sk-None-yrHBAgE1q3WBsVi3NB0MT3BlbkFJWGSZkgHGXfgT3sV2YTBF'
}

player = Player(config)
player.play_game_with_human_input()