"""
    algorithm: DDPG (Deep Deterministic Policy Gradient)
        "In this work we persent a model-free, off-policy, actor-critic algorithm using deep function
    approximators that can learn policies in high-dimensional, continuous action spaces."
        -- Quoted from the Paper' Continuous control with deep reinforcement learning'
        Of course, you can get more detail from the paper above.

        key points:
            1.DQN + Actor-Critic.
            2.soft update parameters.
            3.Four Neural Networks:
                1.Actor : s -> a
                2.Critic: (s,a) -> Q
                3.Target Actor: s'-> max a'
                4.Target Critic: (s',a') -> Q'

    environment: Pendulum-v0
    state:
        1.Angle:
            cos(angle) : [-1, 1]
            sin(angle) : [-1, 1]
        2.Angular velocity: [-8, 8]

    action:
        Motor control torque: [-2, 2]

    reward : - [angle**2 + .1*Angular velocity**2 + .001*(torque**2)]

    prerequisites:  tensorflow 2.2(tensorflow >= 2.0)

    author: Xinchen Han
    date: 2020/8/14

    Notice:

"""

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gym
from collections import deque
import random

"""Environment"""
env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = env.action_space.high


"""Random seed"""
env.seed(2)
np.random.seed(2)
tf.random.set_seed(2)

"""Set hyperparameters"""
gamma = 0.9
learning_rate = 1e-3
max_episodes = 3000
max_step = 200

render = True

class DDPG(object):
    def __init__(self):
        self.variance = 2.0
        self.ema = tf.train.ExponentialMovingAverage(decay = 1-1e-2)
        self.batch_size = 100
        self.replay_size = 3000
        self.replay_queue = deque(maxlen=self.replay_size)

        self.Actor_model = self.create_actor_model()
        self.Target_Actor_model = self.create_actor_model()
        self.copy_para(self.Actor_model, self.Target_Actor_model)

        self.Critic_model = self.create_critic_model()
        self.Target_Critic_mdoel = self.create_critic_model()
        self.copy_para(self.Critic_model, self.Target_Critic_mdoel)

        self.actor_optimizer = tf.optimizers.Adam(learning_rate)
        self.critic_optimizer = tf.optimizers.Adam(learning_rate * 2)


    def create_actor_model(self):
        input = keras.layers.Input(shape=state_dim)
        hidden1 = keras.layers.Dense(64, activation='relu')(input)
        hidden2 = keras.layers.Dense(32, activation='relu')(hidden1)
        hidden3 = keras.layers.Dense(action_dim, activation='tanh')(hidden2)
        output = keras.layers.Lambda(lambda x: x * action_range )(hidden3)
        model = keras.models.Model(inputs=[input], outputs=[output])
        return model


    def create_critic_model(self):
        state_input = keras.layers.Input(shape = state_dim)
        action_input = keras.layers.Input(shape = action_dim)
        input = keras.layers.concatenate([state_input, action_input])
        hidden1 = keras.layers.Dense(64, activation='relu')(input)
        hidden2 = keras.layers.Dense(32, activation='relu')(hidden1)
        output = keras.layers.Dense(1)(hidden2)
        model = keras.models.Model(inputs=[state_input, action_input], outputs=[output])
        return model


    def copy_para(self, from_model, to_model):
        for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
            j.assign(i)


    def choose_action(self, state):
        action = self.Actor_model(np.array([state], dtype=np.float32))[0]
        return np.clip(
            np.random.normal(action, self.variance), -action_range, action_range) # add noisy


    def ema_update(self):
        paras = self.Actor_model.trainable_weights + self.Critic_model.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.Target_Actor_model.trainable_weights + self.Target_Critic_mdoel.trainable_weights, paras):
            i.assign(self.ema.average(j))


    def fill_replay(self, state, action, state_, reward, done):
        self.replay_queue.append((state, action, state_, reward, done))


    def model_train(self):
        self.variance *= .9995

        replay_batch = random.sample(self.replay_queue, self.batch_size)
        state_batch = np.array([replay[0] for replay in replay_batch])
        action_batch = np.array([replay[1] for replay in replay_batch])
        next_state_batch = np.array([replay[2] for replay in replay_batch])
        reward_batch = np.array([replay[3] for replay in replay_batch])
        done_batch = np.array([replay[4] for replay in replay_batch])

        with tf.GradientTape() as tape:
            actions_ = self.Target_Actor_model(next_state_batch)
            q_ = self.Target_Critic_mdoel([next_state_batch, actions_])
            y = reward_batch + gamma * q_
            q = self.Critic_model([state_batch, action_batch])
            td_error = tf.losses.mean_squared_error(y, q)
        critic_grads = tape.gradient(td_error, self.Critic_model.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.Critic_model.trainable_weights))

        with tf.GradientTape() as tape:
            action = self.Actor_model(state_batch)
            q = self.Critic_model([state_batch, action])
            actor_loss = -tf.reduce_mean(q)
        actor_grads = tape.gradient(actor_loss, self.Actor_model.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.Actor_model.trainable_weights))
        self.ema_update()


if __name__ == "__main__":
    agent = DDPG()
    score_list = []
    for episode in range(max_episodes):
        state = env.reset()
        score = 0
        for step in range(max_step):
            if render:
                env.render()
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            agent.fill_replay(state, action, state_, reward, done)
            if len(agent.replay_queue) >= agent.replay_size:
                agent.model_train()

            score += reward
            state = state_

            if done:
                score_list.append(score)
                print('episode:', episode, 'score:', score, 'max_score:', np.max(score_list))
                break


    env.close()
    plt.plot(score_list, color='orange')
    plt.show()
