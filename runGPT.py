import torch
from collections import defaultdict
import numpy as np
from PIL import Image
from VGDLEnv import VGDLEnv
import csv
import os
from pygame.locals import K_RIGHT, K_LEFT, K_UP, K_DOWN, K_SPACE
import matplotlib.pyplot as plt
import base64
import io
import sys
from openai import OpenAI
import threading
import time

class Player(object):
    def __init__(self, config):
        self.config = config
        self.api_key = config["api_key"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Env = VGDLEnv(self.config["game_name"], 'all_games')
        self.Env.set_level(0)
        self.client = OpenAI(api_key=self.api_key)

        print('Initializing at seed = {}'.format(self.config["random_seed"]))

        self.n_actions = len(self.Env.actions)
        self.ended = 0
        self.num_episodes = self.config["num_episodes"]
        self.screen_history = []
        self.recent_history = [0] * int(self.config["criteria"].split('/')[1])

        self.ensure_dir('screens')
        self.ensure_dir('logs')

    def ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_screen(self):
        return self.Env.render()

    def get_screen_as_base64(self):
        screen = self.get_screen()
        screen = np.ascontiguousarray(screen, dtype=np.uint8)
        image = Image.fromarray(screen)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_action_from_response(self, response_text):
        """
        Extract the action based on the keywords present in the GPT response.
        """
        action_keywords = ["left", "right", "up", "down", "fire"]
        response_text = response_text.lower()

        for action in action_keywords:
            if action in response_text:
                return action  

        return "noop"
    
    
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
        
    def generate_action_per_step(self, game_state, action_holder, action_lock):
        try:
            screen_base64 = game_state

            prompt = (
                "You are an expert in playing the Atari style game similar to SpaceInvaders based on visual features in an image. "
                "In this game, the blue block is your avatar, and you need to shoot enemies represented as yellow blocks while avoiding their bombs represented as red blocks."
                "For the given image, decide the next step you should take to shoot as many enemies as possible to win the game. "
                "Generate a pure step description such as 'move left', 'move right', 'move up', 'move down' or 'fire'.\n"
            )

            # prompt = (
            #     "You are an expert in playing the puzzle game named Bait based on visual features in an image. "
            #     "In this game, the blue block is your avatar, and you need to move brown blocks to clear the path to the red block where the key is located, and then take the key to the green block to win the game."
            #     "For the given image, decide the next step you should take to collect the key to win the game. "
            #     "Generate a pure step description such as 'move left', 'move right', 'move up', 'move down'.\n"
            # )

            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screen_base64}"}}
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": content}
                ],
                max_tokens=300,
            )

            response_text = response.choices[0].message.content.strip()
            extracted_action = self.extract_action_from_response(response_text)

            with action_lock:
                action_holder['action'] = extracted_action

        except Exception as e:
            print(f"Error generating action: {e}")
            with action_lock:
                action_holder['action'] = "Error"

    def initialize_rendering(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.Env.render(), animated=True)
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        plt.show()

    def update_rendering(self):
        self.fig.canvas.restore_region(self.background)
        self.im.set_array(self.Env.render())
        self.ax.draw_artist(self.im)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def play_game_with_gpt4(self):
        print("Starting Game")
        print("-" * 25)

        # Initialize time for FPS control
        target_fps = 30  
        time_per_frame = 1.0 / target_fps
        last_frame_time = time.time()

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

        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.config["random_seed"])

        self.Env.reset()
        avatar_position_data = {'game_info': (self.Env.current_env._game.width, self.Env.current_env._game.height),
                                'episodes': [[(self.Env.current_env._game.sprite_groups['avatar'][0].rect.left,
                                               self.Env.current_env._game.sprite_groups['avatar'][0].rect.top,
                                               self.Env.current_env._game.time,
                                               self.Env.lvl)]]}

        event_dict = defaultdict(lambda: 0)

        self.initialize_rendering()

        action_holder = {'action': None}
        action_lock = threading.Lock()  
        last_action = 0  
        gpt_thread = None

        while self.steps < self.config['max_steps']:

            # FPS Limiting
            current_time = time.time()
            if current_time - last_frame_time < time_per_frame:
                continue
            last_frame_time = current_time

            self.steps += 1
            self.episode_steps += 1

            # Start GPT action generation in a separate thread if one isn't already running
            if gpt_thread is None or not gpt_thread.is_alive():
                game_state = self.get_screen_as_base64()
                gpt_thread = threading.Thread(target=self.generate_action_per_step, args=(game_state, action_holder, action_lock))
                gpt_thread.start()

            with action_lock:
                if action_holder['action'] is None:
                    action = 0  
                else:
                    action = {
                        "left": 2,
                        "right": 1,
                        "up": 3,
                        "down": 4,
                        "fire": 5
                    }.get(action_holder['action'].lower(), 0)
                    action_holder['action'] = None  # Reset action after processing


            print(f"Action received from GPT-4o: {action}")

            # Step in the environment using the selected action
            self.reward, self.ended, self.win = self.Env.step(action)

            time.sleep(0.2)
            # Update rendering (Keep this on the main thread)
            self.update_rendering()

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

            self.episode_reward += self.reward
            self.reward = max(-1.0, min(self.reward, 1.0))

            if self.ended or self.episode_steps > self.config["timeout"]:
                if self.episode_steps > self.config["timeout"]:
                    print("Game Timed Out")

                with open('object_interaction_histories/{}_object_interaction_history_{}_trial{}.csv'.format(
                        self.config["game_name"], self.config["level_switch"], self.config["trial_num"]), "a", newline='') as file:
                    interactionfilewriter = csv.writer(file)
                    for event_name, count in event_dict.items():
                        row = ('GPT-4o', 'NA', 'NA', self.config["game_name"], self.Env.lvl, self.episode, event_name, count)
                        interactionfilewriter.writerow(row)

                self.episode += 1

                print("Level {}, episode reward at step {}: {}".format(self.Env.lvl, self.steps, self.episode_reward))
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

            if self.ended:
                print("Game Ended")
                break

        print("Game Finished")

config = {
    'game_name': 'aliens',
    # 'game_name': 'bait',
    'img_size': 64,
    'random_seed': 1,
    'num_episodes': 10,
    'level_switch': 'sequential',
    'criteria': '1/1',
    'max_steps': 10000,
    'timeout': 2000,
    'trial_num': 10,
    'api_key': 'sk-None-yrHBAgE1q3WBsVi3NB0MT3BlbkFJWGSZkgHGXfgT3sV2YTBF'
}

player = Player(config)
player.play_game_with_gpt4()