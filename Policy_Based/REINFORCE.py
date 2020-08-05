"""
    algorithm: REINFORCE
        REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility

        More details you can learn from the paper:
                https://link.springer.com/content/pdf/10.1007/BF00992696.pdf

        key points:
                Monte Carlo


    environment: CartPole-v0
    state:
        1.Cart Position:[-4.8,4.8],  2.Cart Velocity[-Inf,Inf],  3.Pole Angle[-24 deg, 24 deg]
        4.Pole Velocity [-Inf,Inf]
    action:
        0: left
        1: right
    reward: 1 for every step

    prerequisites:  tensorflow 2.2(tensorflow >= 2.0)
    notice：

    author: Xinchen Han
    date: 2020/8/5

"""

from tensorflow import keras
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
import numpy as np
import gym

"""Environment"""
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

"""Random seed"""
env.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

"""Set hyperparameters"""
gamma = 0.9
epsilon = 0.9
learning_rate = 1e-3
max_episodes = 500

render = True


class REINFORCE(object):

    def __init__(self):
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.actor = self.create_actor_model()
        self.optimizer = keras.optimizers.Adam(learning_rate)


    def create_actor_model(self):
        input = keras.layers.Input(shape=state_dim)
        hidden1 = keras.layers.Dense(64, activation='relu')(input)
        hidden2 = keras.layers.Dense(16, activation='relu')(hidden1)
        output = keras.layers.Dense(action_dim)(hidden2)
        model = keras.models.Model(inputs=[input], outputs=[output])
        return model


    def choose_action(self, state):
        probs = tf.nn.softmax(self.actor(np.array([state], np.float32))).numpy()
        return tl.rein.choice_action_by_probs(probs.ravel())


    def perceive(self, state, action, reward):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)


    def learn(self):
        discounted_reward_buffer_norm = self.discount_and_norm_rewards()

        with tf.GradientTape() as tape:
            _logits = self.actor(np.vstack(self.state_buffer))
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=_logits, labels=np.array(self.action_buffer)
            )
            loss = tf.reduce_mean(neg_log_prob * discounted_reward_buffer_norm)

        grad = tape.gradient(loss, self.actor.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.actor.trainable_weights))

        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []


    def discount_and_norm_rewards(self):
        discounted_reward_buffer = np.zeros_like(self.reward_buffer)
        running_add = 0
        for t in reversed(range(0, len(self.reward_buffer))):
            running_add = running_add * gamma + self.reward_buffer[t]
            discounted_reward_buffer[t] = running_add

        discounted_reward_buffer -= np.mean(discounted_reward_buffer)
        discounted_reward_buffer /= np.std(discounted_reward_buffer)
        return discounted_reward_buffer


if __name__ == "__main__":
    agent = REINFORCE()
    score_list = []
    for episode in range(max_episodes):
        state = env.reset()
        score = 0
        while True:
            action = agent.choose_action(state)
            if render:
                env.render()
            state_, reward, done, _ = env.step(action)
            agent.perceive(state, action, reward)
            state = state_
            score += reward
            if done:
                agent.learn()
                score_list.append(score)
                print('episode:', episode, 'score:', score, 'max_score:', np.max(score_list))
                break
        if np.mean(score_list[-10:]) > 180:
            break

    env.close()
    plt.plot(score_list, color='orange')
    plt.show()