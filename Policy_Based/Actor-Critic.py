"""
    algorithm: Actor - Critic
        In this actor-critic algorithm, the policy-based and value-based algorithm are combined.
Of course, the critic has many alternative forms, such as
        1. total reward of the trajectory (REINFORCE)
        2. reward following the action
        3. baseline version
        4. Q function
        5. advantage function
        6. TD - error : r + gamma * V(s') - V(s)
        Regarding the form of the critic function, a generalization is made in GAE, and you can see
more detials from the  : https://arxiv.org/pdf/1506.02438.pdf


        key points:
                Actor : policy network
                Critic : value network
            state -> Actor -> action -> env -> info -> critic -> TD-error -> train network


    environment: CartPole-v0
    state:
        1.Cart Position:[-4.8,4.8],  2.Cart Velocity[-Inf,Inf],  3.Pole Angle[-24 deg, 24 deg]
        4.Pole Velocity [-Inf,Inf]
    action:
        0: left
        1: right
    reward: 1 for every step

    prerequisites:  tensorflow 2.2(tensorflow >= 2.0)

    author: Xinchen Han
    date: 2020/8/11

    Notice:
        so far I found that my actions will eventually tend to 0,
        that is, no matter what the state, the final output action is 0.
        If possible, I would like to see you can help solve this problem, I am very grateful.

"""


from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gym

"""Environment"""
env = gym.make('CartPole-v1').unwrapped
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


"""Random seed"""
env.seed(3)
np.random.seed(3)
tf.random.set_seed(3)

"""Set hyperparameters"""
gamma = 0.9
learning_rate = 1e-4
max_episodes = 500

render = True

class Actor(object):
    def __init__(self):
        self.Actor_model = self.create_actor_model()
        self.optimizer = keras.optimizers.Adam(learning_rate)

    def create_actor_model(self):
        input = keras.layers.Input(shape=state_dim)
        hidden1 = keras.layers.Dense(64, activation='relu')(input)
        output = keras.layers.Dense(action_dim)(hidden1)
        model = keras.models.Model(inputs=[input], outputs=[output])
        return model

    def Actor_train_model(self, state, action, td_error):
        with tf.GradientTape() as tape:
             _logits = self.Actor_model(np.array([state]))
             cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = [action], logits=_logits)
             loss = tf.reduce_sum(tf.multiply(cross_entropy, td_error[0]))
        grad = tape.gradient(loss, self.Actor_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.Actor_model.trainable_weights))

    def choose_action(self, state):
        probs = tf.nn.softmax(self.Actor_model(np.array([state], np.float32))).numpy()
        # probs = np.array(tf.clip_by_value(probs, 0.3, 0.7))
        # print(probs)
        return np.random.choice(np.arange(action_dim), p = probs.ravel())

class Critic(object):
    def __init__(self):
        self.Critic_model = self.create_critic_model()
        self.optimizer = keras.optimizers.Adam(learning_rate * 10)

    def create_critic_model(self):
        input = keras.layers.Input(shape=state_dim)
        hidden1 = keras.layers.Dense(64, activation='relu')(input)
        # hidden2 = keras.layers.Dense(16, activation='relu')(hidden1)
        output = keras.layers.Dense(1)(hidden1)
        model = keras.models.Model(inputs=[input], outputs=[output])
        return model

    def Critic_train_model(self, state, reward, next_state):
        next_v = self.Critic_model(np.array([next_state]))
        with tf.GradientTape() as tape:
            v = self.Critic_model(np.array([state]))
            ## TD_error = r + gamma * V(newS) - V(S)
            td_error = reward + gamma * next_v - v
            loss = tf.square(td_error)

        grad = tape.gradient(loss, self.Critic_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.Critic_model.trainable_weights))
        return td_error


if __name__ == "__main__":

    actor = Actor()
    critic = Critic()
    score_list = []
    for episode in range(max_episodes):
        state = env.reset()
        score = 0
        while True:
            action = actor.choose_action(state)
            if render:
                env.render()
            state_, reward, done, _ = env.step(action)
            TD_error = critic.Critic_train_model(state, reward, state_)
            actor.Actor_train_model(state, action, TD_error)
            score += reward
            state = state_
            if done:
                score_list.append(score)
                print('episode:', episode, 'score:', score, 'max_score:', np.max(score_list))
                break
        if np.mean(score_list[-10:]) > 180:
            break

    env.close()
    plt.plot(score_list, color='orange')
    plt.show()


