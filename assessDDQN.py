import torch
import sys
import csv
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from VGDLEnv import VGDLEnv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import tkinter as tk
import threading
from concurrent.futures import ThreadPoolExecutor
from rl_models import DQN

class Player(object):
    def __init__(self, config):
        self.config = config
        self.Env = VGDLEnv(self.config["game_name"], 'all_games')
        self.Env.set_level(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.game_size = np.shape(self.Env.render())
        self.input_channels = self.game_size[2]
        self.n_actions = len(self.Env.actions)

        print('Initializing at seed = {}'.format(self.config["random_seed"]))

        self.ended = 0
        
        self.num_episodes = self.config["num_episodes"]
        self.screen_history = []
        self.recent_history = [0] * int(self.config["criteria"].split('/')[1])
        self.game_state_history = []  

        self.resize = T.Compose([
            T.ToPILImage(),
            T.Pad((np.max(self.game_size[0:2]) - self.game_size[1],
                   np.max(self.game_size[0:2]) - self.game_size[0])),
            T.Resize((self.config['img_size'], self.config['img_size']), interpolation=Image.BICUBIC),
            T.ToTensor()
        ])

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
        self.assessment_file = open('assessment_results2.txt', 'w')
        self.assessment_file.write("Player Performance Assessments:\n")
        self.assessment_file.write("=" * 40 + "\n")

        self.executor = ThreadPoolExecutor(max_workers=1)  # Manage assessment threads

        self.load_ddqn_model()

        self.assessment_future = None  

        # Track total performance score based on DDQN assessments
        self.ddqn_total_score = 0
        self.ddqn_assessments_completed = 0 
        self.ddqn_assessment_score = 0

        # Initialize the DDQN assessment thread to None
        self.assessment_result = None
        self.game_state_buffer = []  
        self.assessment_result = "Awaiting first assessment..."

    def load_ddqn_model(self):
        """
        Loads the pre-trained DDQN model based on the game environment.
        """
        self.ddqn_model = DQN(self.input_channels, self.n_actions).to(self.device)

        model_path = 'model_weights/{}_trial{}_{}.pt'.format(self.config['game_name'], 
                                                             self.config['trial_num'], 
                                                             self.config['level_switch'])
        self.ddqn_model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.ddqn_model.eval()

    def ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_screen(self):
        """
        Capture the current screen, process it using the transformations defined, and convert it to a tensor.
        """
        screen = self.Env.render().transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return self.resize(screen).unsqueeze(0).to(self.device)
    
    def update_rendering(self):
        self.fig.canvas.restore_region(self.background)
        self.im.set_array(self.Env.render())
        self.ax.draw_artist(self.im)
        self.canvas.draw()

    def generate_assessment_per_steps_ddqn(self, game_states, game_scores, level_progress):
        """
        Uses the DDQN model along with game scores and level progress to assess the player's performance.
        This function runs in a separate thread.
        """
        try:
            # aggregate Q-values
            total_q_value_score = 0
            for state in game_states:
                with torch.no_grad():
                    q_values = self.ddqn_model(state)
                    total_q_value_score += q_values.max(1)[0].item()

            avg_q_value_score = total_q_value_score / len(game_states)

            # Incorporate game scores and level progress into the assessment
            max_game_score = 56
            avg_game_score = (sum(game_scores) / len(game_scores)) / max_game_score
            avg_level_progress = sum(level_progress) / len(level_progress)

            print('game_scores, avg_game_score:', game_scores, avg_game_score)
            # Create a composite score based on Q-values, game score, and level progression
            performance_score = (0.5 * avg_q_value_score) + (0.3 * avg_game_score) + (0.2 * avg_level_progress)

            # Classify performance based on the composite score
            if performance_score > 0.8:
                self.assessment_result = f"Your score is {performance_score}. Excellent performance based on model and game metrics."
            elif performance_score > 0.5:
                self.assessment_result = f"Your score is {performance_score}. Good performance. Keep improving!"
            else:
                self.assessment_result = f"Your score is {performance_score}. You need to improve your strategy."

        except Exception as e:
            print(f"Error generating assessment using DDQN: {e}")
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

        # Initialize containers for tracking game scores and level progress
        game_scores = []
        level_progress = []

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

            # Capture the game state (as a tensor) and store it
            game_state_tensor = self.get_screen()
            self.game_state_history.append(game_state_tensor)

            # Track the current score and level progression
            game_scores.append(self.Env.current_env._game.score)
            level_progress.append(self.Env.lvl)

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

            # Step in the environment using the selected action
            self.reward, self.ended, self.win = self.Env.step(self.action)

            self.update_rendering()

            self.steps += 1
            self.episode_steps += 1
            self.episode_reward += self.reward

            # After every 5 steps, run the DDQN assessment
            if len(self.game_state_history) == 5:
                threading.Thread(target=self.generate_assessment_per_steps_ddqn, 
                                args=(self.game_state_history, game_scores, level_progress)).start()
                # Reset the history for the next batch
                self.game_state_history = []
                game_scores = []
                level_progress = []

            # Check if assessment is complete and write to file
            if self.assessment_result:
                self.assessment_file.write(f"Assessment after {self.steps} steps:\n")
                self.assessment_file.write(self.assessment_result + "\n")

            # Calculate and log the overall performance dynamically, only if assessments have been completed
            if self.ddqn_assessments_completed > 0:
                avg_ddqn_score = self.ddqn_total_score / self.ddqn_assessments_completed
                self.assessment_file.write(f"Overall Performance Score (Average): {avg_ddqn_score:.2f}\n")
            else:
                self.assessment_file.write("No assessments completed yet.\n")

            self.assessment_file.write("=" * 40 + "\n")
            print(f"Assessment received from DDQN: {self.assessment_result}")
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
    'timeout': 5000,
    'trial_num': 1,
}

player = Player(config)
player.play_game_with_human_input()