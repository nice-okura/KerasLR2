import gym
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import *
from rl.memory import SequentialMemory
import numpy as np
from gym import spaces
from pprint import pprint as pp

# 指定の数（1~99）を複数枚のトランプの和で揃えるゲーム
class Addition(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX = 99
    n_actions = 13 # 1~13 の13パターン(トランプ)

    def __init__(self):
        self.action_space = spaces.Discrete(self.n_actions)
        self.action = 0
        self.cards_list = np.array([])
        self.cards_sum = 0
        self.goal = np.random.randint(0, self.MAX)
        self.reward = 0
        self.observation_space = spaces.Box(low=0, high=self.MAX+1, shape=(2,), dtype=np.int32)

    def reset(self):
        self.cards_sum = 0
        self.cards_list = np.array([])
        self.goal = np.random.randint(0, self.MAX)
        self.reward = 0

        return np.array([0, self.goal])

    def step(self, action):
        # print("cards_sum: ", self.cards_sum, "action: ", action)
        self.cards_sum += action
        done = False
        reward = 0

        if self.cards_sum == self.goal:
            # ゴール報酬は最大値の2倍（適当）
            reward = self.MAX*2
            done = True
        elif self.cards_sum > self.goal:
            # ゴール超えたら失格
            # print("Over the goal.")
            reward = -10
            done = True
        elif np.any(self.cards_list==action):
            # すでに出したカードを出したら失格
            # print("The same cards. action: ", action , "cards_list: ", self.cards_list)
            reward = -10
            done = True
        elif action == 0:
            # 0を出すとゲーム終了
            # print("Action 0. Game set.")
            reward = 0
            done = True
        else:
            # 目標数(goal)から近いほうが報酬が高い
            reward = action + (self.cards_sum/self.MAX) * 2

        if(action != 0):
            self.cards_list = np.append(self.cards_list, action)

        self.action = action
        self.reward = reward

        info = {}

        return np.array([self.cards_sum, self.goal]), reward, done, info

    def render(self, mode='human', close=False):
        if mode != 'human':
          raise NotImplementedError()
        print("CardSum: ",self.cards_sum, "Goal: ", self.goal)

# env = Addition()
# obs = env.reset()
# n_steps = 20
# for step in range(n_steps):
#     action = np.random.randint(1,13)
#     obs, reward, done, info = env.step(action)
#     print('obs=', obs, 'reward=', reward, 'done=', done)
#     env.render(mode='human')
#     if done:
#         print("Goal !!", "reward=", reward)
#         break

env = Addition()
obs = env.reset()
n_steps = 20
window_length = 1
input_shape =  (window_length,) + env.observation_space.shape
print("-Initial parameter-")
print(env.action_space) # input
print(env.observation_space) # output
print(env.reward_range) # rewards
print(env.action_space) # action
print(env.action_space.sample()) # action

nb_actions = env.action_space.n
c = input_data = Input(input_shape)
c = Flatten()(c)
c = Dense(128, activation='relu')(c)
c = Dense(128, activation='relu')(c)
c = Dense(128, activation='relu')(c)
c = Dense(128, activation='relu')(c)
c = Dense(nb_actions, activation='linear')(c)
model = Model(input_data, c)
print(model.summary())

# rl
memory = SequentialMemory(limit=50000, window_length=window_length)
policy = EpsGreedyQPolicy() #GreedyQPolicy()# SoftmaxPolicy()
# agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy)
# agent.compile(Adam())
# agent.fit(env, nb_steps=10000*60*6, visualize=False, verbose=1)
# agent.save_weights("weights.hdf5")

# predict
obs = env.reset()
n_steps = 20
for step in range(n_steps):
    obs = obs.reshape((1, 1, 2))
    action = model.predict(obs)
    # print("predict_action: ", action)
    action = np.argmax(action)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done, '\n')

    env.render(mode='human')

    if done:
        print("Goal !!", "reward=", reward)
        break
