# td3_her_rlbench_reacher
A implementation for soving reach target task based on Twin Delayed DDPG(TD3) with Hindsight Experience Replay(HER) using PaddlePaddle.

![image](https://github.com/63445538/td3_her_rlbench_reacher/blob/master/RLBench/records/solved/video_7.gif)

# prerequisites

Python: Python 3.6+

[PaddlePaddle](https://github.com/PaddlePaddle/Paddle) : Deep learning framework

[PARL](https://github.com/PaddlePaddle/PARL) : Reinforcement learning toolbox based on PaddlePaddle

[gym](https://github.com/openai/gym) : Universal environment builder for RL tasks

[RLBench](https://github.com/stepjam/RLBench): RL tasks extension for robotics researches.

# Install
First, create a virtual environment by ```virtualenv```, in it, install PaddlePaddle, gym and PARL by

```pip install requirements.txt```

Then install RLBench via [RLBench](https://github.com/stepjam/RLBench). 

# Train

```python rlbench_reach_td3_train.py```

# Evaluate

```python rlbench_reach_td3_eval.py```

# Result
4 stages (initial model, after 40000-episode trained model, after 80000-episode trained model, fianl model) of training model is selected and the corresponded render result is restored as gif in the folder ```records```.
