Background
==
The purpose of this project is to apply reinforcement learning in assessing human performance in Atari games.  

To train model in Gym
==
Run the random_agent_empa_like.py in random folder  

To train DDQN model in VGDL
==
installation
--
```
pip install -r requirements
```
run DDQN
--
```
python runDDQN.py -game_name aliens
```
To train GPT to play Atari
==
```
python runGPT.py
```
or  
```
python gymGPT.py
```
To play games with human input
==
```
python runHuman.py
```
To assess human performance with DDQN
==
```
python assessDDQN.py
```
To assess human performance with GPT
==
```
python assessGPT.py
```
Code credits
==
The implementation of games environment including VGDL and Gym is based on [RC_RL](https://github.com/SergioArnaud/RC_RL).
