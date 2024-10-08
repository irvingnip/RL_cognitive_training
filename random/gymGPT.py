import gym
import numpy as np
from tqdm import tqdm
from pygame.locals import *
import base64
import io
from PIL import Image
from openai import OpenAI
import threading
import time
import random
from random_agent import AtariPreprocessing

n_steps = 1_000
n_seeds = 1

games = [
    # "SpaceInvadersNoFrameskip-v4",
    # 'BreakoutNoFrameskip-v4',
    # 'FreewayNoFrameskip-v4',
    'CarnivalNoFrameskip-v4',
    # 'PongNoFrameskip-v4'
]

log_video = False
output_name = 'gpt_policy'


def get_screen_as_base64(env):
    # screen = np.empty((210, 160, 3), dtype=np.uint8)  
    screen = np.empty((214, 160, 3), dtype=np.uint8) 
    env.ale.getScreenRGB(screen)  # Fill the array with the RGB frame from the environment
    image = Image.fromarray(screen)  # Convert the array to an image
    buffered = io.BytesIO()  # Create an in-memory buffer
    image.save(buffered, format="JPEG")  # Save the image to the buffer in JPEG format
    return base64.b64encode(buffered.getvalue()).decode('utf-8')  

def generate_action_per_step(screen_base64, api_key):
    client = OpenAI(api_key=api_key)
    try:
        # prompt = (
        #     "You are an expert in playing Atari game SpaceInvaders based on visual features in an image. "
        #     "For the given image, decide the next step you should take to win the game. "
        #     "Generate a pure step description such as 'move left', 'move right', 'move up', 'move down', 'fire'.\n"
        # )
        # prompt = (
        #     "You are an expert in playing Atari game Freeway based on visual features in an image."
        #     "For the given image, decide the next step you should take to avoid cars while crossing the road to win points."
        #     "Generate a pure step description such as 'move up', 'move down'.\n"
        # )
        # prompt = (
        #     "You are an expert in playing Atari game Pong based on visual features in an image."
        #     "For the given image, decide the next step you should take to move the right paddle up or down and deflect the ball away."
        #     "Generate a pure step description such as 'move up', 'move down' or 'do nothing'.\n"
        # )
        prompt = (
            "You are an expert in playing Atari game Carnival based on visual features in an image."
            "Targets move horizontally across the screen and you must shoot them. You are in control of a gun that can be moved horizontally. The supply of ammunition is limited and chickens may steal some bullets from you if you don't hit them in time."
            "For the given image, decide the next step you should take to ."
            "Generate a pure step description such as 'move left', 'move right', or 'fire'.\n"
        )
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screen_base64}"}}
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=300,
        )

        if response:
            try:
                action = response.choices[0].message.content.strip()
                # debugging point
                print(f"Action received from GPT-4: {action}")
                return action
            except Exception as e:
                print(f"Error parsing action from response: {e}")
                return "Error"
        else:
            print(f"Error from GPT API: {response.status_code} - {response.text}")
            return "Error"
    except Exception as e:
        print(f"Error generating action: {e}")
        return "Error"

# Function for the GPT thread
def generate_action_thread(screen_base64, action_holder, api_key):
    action_holder['action'] = generate_action_per_step(screen_base64, api_key)

if __name__ == "__main__":
    # put your key here
    api_key = ""

    for game in tqdm(games):
        for n_seed in range(n_seeds):
            env = gym.make(game, render_mode='human')
            env.reset()

            AP = AtariPreprocessing(
                env,
                terminal_on_life_loss=False,
                game_name=game.replace("NoFrameskip-v4", ""),
                tag=n_seed,
                log_video=log_video,
                output_name=output_name
            )
            lives = env.ale.lives()

            # Default action to fallback when GPT is delayed
            last_gpt_action = 0 
            action_holder = {'action': None}
            gpt_thread = None
            i = 0

            while i < n_steps:
                try:
                    screen_base64 = get_screen_as_base64(env)
                except ValueError as e:
                    print(e)
                    break

                # start GPT action generation in a separate thread if one isn't already running
                if gpt_thread is None or not gpt_thread.is_alive():
                    gpt_thread = threading.Thread(target=generate_action_thread, args=(screen_base64, action_holder, api_key))
                    gpt_thread.start()

                if action_holder['action'] is not None:
                    gpt_action = action_holder['action']
                    action_holder['action'] = None  

                # Convert GPT action description to environment action
                # for games Freeway
                    # if "up" in gpt_action.lower():
                    #     action = 1
                    # elif "down" in gpt_action.lower():
                    #     action = 2

                    # for games pong
                    # if "up" in gpt_action.lower():
                    #     action = 2
                    # elif "down" in gpt_action.lower():
                    #     action = 3

                # for games SpaceInvaders and Carnival
                    if "fire" in gpt_action.lower():
                        action = 1
                    elif "right" in gpt_action.lower():
                        action = 2
                    elif "left" in gpt_action.lower():
                        action = 3
                    elif "left" in gpt_action.lower() and "fire" in gpt_action.lower():
                        action = 5
                    elif "right" in gpt_action.lower() and "fire" in gpt_action.lower():
                        action = 4

                    else:
                        action = 0  
                    # Store the last valid action
                    # last_gpt_action = action  
                else:
                    action = 0
                    # Store the last valid action
                    # action = last_gpt_action  

                observation, reward, game_over, info = AP.step(action)
                i += 1

                if game_over:
                    AP.reset()
                    break