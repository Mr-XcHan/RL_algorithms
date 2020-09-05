"""
    algorithm: TRPO (Asynchronous Advantage Actor-Critic)
        In policy gradient algorithms, the learning rate is important,a large step can
        collapse the policy performance, even with a small step can lead a large differences
        in policy.But the TRPO can guaranteed the policy monotonic improvement. Of course ,
        it has made several approximations to the theoretically-justified procedure.

        More detail you can learn from the paper 'Trust Region Policy Optimization'

    key points:
        1. Strong mathematical theory
        2. Guaranteed the policy monotonic improvement

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
    date: 2020/8/22

    Notice:
    Basic Framework is cited from:
    https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_TRPO.py

"""



import argparse
import copy
import os
import time
import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'Pendulum-v0'  # environment id
RANDOM_SEED = 2  # random seed
RENDER = False

ALG_NAME = 'TRPO'
TRAIN_EPISODES = 1000  # total number of episodes for training
TEST_EPISODES = 100  # total number of episodes for testing
MAX_STEPS = 200  # total number of steps for each episode

HIDDEN_SIZES = [64, 64]  # hidden layer size
GAMMA = 0.99  # reward discount
DELTA = 0.01  # KL-divergence limit for TRPO update.
VF_LR = 1e-3  # Learning rate for value function optimizer
TRAIN_VF_ITERS = 100  # Number of gradient descent steps to take on value function per epoch
DAMPING_COEFF = 0.1  # Artifact for numerical stability
CG_ITERS = 10  # Number of iterations of conjugate gradient to perform
BACKTRACK_ITERS = 10  # Maximum number of steps allowed in the backtracking line search
BACKTRACK_COEFF = 0.8  # How far back to step during backtracking line search
LAM = 0.97  # lambda for GAE-lambda
SAVE_FREQ = 10  # How often (in terms of gap between epochs) to save the current policy and value function
EPS = 1e-8  # epsilon
BATCH_SIZE = 512  # batch size


class GAE_Buffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.mean_buf = np.zeros(size, dtype=np.float32)
        self.log_std_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, mean, log_std):

        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.mean_buf[self.ptr] = mean
        self.log_std_buf[self.ptr] = log_std
        self.ptr += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def _discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def is_full(self):
        return self.ptr == self.max_size

    def get(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf, self.mean_buf, self.log_std_buf]



class TRPO:
    """
    trpo class
    """

    def __init__(self, state_dim, action_dim, action_bound):
        # critic
        with tf.name_scope('critic'):
            layer = input_layer = tl.layers.Input([None, state_dim], tf.float32)
            for d in HIDDEN_SIZES:
                layer = tl.layers.Dense(d, tf.nn.relu)(layer)  # 隐含层两层，每层64个cell
            v = tl.layers.Dense(1)(layer)     # 输出层全连接，1个cell
        self.critic = tl.models.Model(input_layer, v)
        self.critic.train()

        # actor
        with tf.name_scope('actor'):
            layer = input_layer = tl.layers.Input([None, state_dim], tf.float32)
            for d in HIDDEN_SIZES:
                layer = tl.layers.Dense(d, tf.nn.relu)(layer) # 同样，隐含层也是两层，每层64个cell
            mean = tl.layers.Dense(action_dim, tf.nn.tanh)(layer)
            mean = tl.layers.Lambda(lambda x: x * action_bound)(mean)
            log_std = tf.Variable(np.zeros(action_dim, dtype=np.float32))

        self.actor = tl.models.Model(input_layer, mean)
        self.actor.trainable_weights.append(log_std)
        self.actor.log_std = log_std
        self.actor.train()

        self.buf = GAE_Buffer(state_dim, action_dim, BATCH_SIZE, GAMMA, LAM)
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=VF_LR)
        self.action_bound = action_bound

    def get_action(self, state, greedy=False):
        state = np.array([state], np.float32)
        mean = self.actor(state)
        log_std = tf.convert_to_tensor(self.actor.log_std)
        std = tf.exp(log_std)
        std = tf.ones_like(mean) * std
        pi = tfp.distributions.Normal(mean, std)

        if greedy:
            action = mean
        else:
            action = pi.sample()
        action = np.clip(action, -self.action_bound, self.action_bound)
        logp_pi = pi.log_prob(action)

        value = self.critic(state)
        return action[0], value, logp_pi, mean, log_std

    def pi_loss(self, states, actions, adv, old_log_prob):

        mean = self.actor(states)
        pi = tfp.distributions.Normal(mean, tf.exp(self.actor.log_std))
        log_prob = pi.log_prob(actions)[:, 0]
        ratio = tf.exp(log_prob - old_log_prob)
        surr = tf.reduce_mean(ratio * adv)
        return -surr

    def gradient(self, states, actions, adv, old_log_prob):
        pi_params = self.actor.trainable_weights
        with tf.GradientTape() as tape:
            loss = self.pi_loss(states, actions, adv, old_log_prob)
        grad = tape.gradient(loss, pi_params)
        gradient = self._flat_concat(grad)
        return gradient, loss

    def train_vf(self, states, rewards_to_go):

        with tf.GradientTape() as tape:
            value = self.critic(states)
            loss = tf.reduce_mean((rewards_to_go - value[:, 0])**2)
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.trainable_weights))

    def kl(self, states, old_mean, old_log_std):
        old_mean = old_mean[:, np.newaxis]
        old_log_std = old_log_std[:, np.newaxis]
        old_std = tf.exp(old_log_std)
        old_pi = tfp.distributions.Normal(old_mean, old_std)

        mean = self.actor(states)
        std = tf.exp(self.actor.log_std) * tf.ones_like(mean)
        pi = tfp.distributions.Normal(mean, std)

        kl = tfp.distributions.kl_divergence(pi, old_pi)
        all_kls = tf.reduce_sum(kl, axis=1)
        return tf.reduce_mean(all_kls)

    def _flat_concat(self, xs):
        return tf.concat([tf.reshape(x, (-1, )) for x in xs], axis=0) # 把xs的数据变为一维列向量， 然后再拼接起来。

    def get_pi_params(self):
        pi_params = self.actor.trainable_weights
        return self._flat_concat(pi_params)

    def set_pi_params(self, flat_params):
        pi_params = self.actor.trainable_weights
        flat_size = lambda p: int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars
        splits = tf.split(flat_params, [flat_size(p) for p in pi_params])
        new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(pi_params, splits)]
        return tf.group([p.assign(p_new) for p, p_new in zip(pi_params, new_params)])

    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)

    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)

    def cg(self, Ax, b):
        x = np.zeros_like(b)
        r = copy.deepcopy(b)  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = copy.deepcopy(r)
        r_dot_old = np.dot(r, r)
        for _ in range(CG_ITERS):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def hvp(self, states, old_mean, old_log_std, x):
        pi_params = self.actor.trainable_weights
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape0:
                d_kl = self.kl(states, old_mean, old_log_std)
            g = self._flat_concat(tape0.gradient(d_kl, pi_params))
            l = tf.reduce_sum(g * x)
        hvp = self._flat_concat(tape1.gradient(l, pi_params))

        if DAMPING_COEFF > 0:
            hvp += DAMPING_COEFF * x
        return hvp

    def update(self):
        states, actions, adv, rewards_to_go, logp_old_ph, old_mu, old_log_std = self.buf.get()
        g, pi_l_old = self.gradient(states, actions, adv, logp_old_ph)  # g是梯度，pi_l_old是loss

        Hx = lambda x: self.hvp(states, old_mu, old_log_std, x)   # 这是定义一个 得到KL散度Hessian matrix吗?
        x = self.cg(Hx, g)

        alpha = np.sqrt(2 * DELTA / (np.dot(x, Hx(x)) + EPS))
        old_params = self.get_pi_params()

        def set_and_eval(step):
            params = old_params - alpha * x * step
            self.set_pi_params(params)
            d_kl = self.kl(states, old_mu, old_log_std)
            loss = self.pi_loss(states, actions, adv, logp_old_ph)
            return [d_kl, loss]

        # trpo with backtracking line search, hard kl
        for j in range(BACKTRACK_ITERS):
            kl, pi_l_new = set_and_eval(step=BACKTRACK_COEFF**j)
            if kl <= DELTA and pi_l_new <= pi_l_old:
                # Accepting new params at step of line search
                break
        else:
            # Line search failed! Keeping old params.
            set_and_eval(step=0.)

        # Value function updates
        for _ in range(TRAIN_VF_ITERS):
            self.train_vf(states, rewards_to_go)

    def finish_path(self, done, next_state):
        if not done:
            next_state = np.array([next_state], np.float32)
            last_val = self.critic(next_state)
        else:
            last_val = 0
        self.buf.finish_path(last_val)


if __name__ == '__main__':
    env = gym.make(ENV_ID).unwrapped

    # reproducible
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    agent = TRPO(state_dim, action_dim, action_bound)

    t0 = time.time()
    if args.train:  # train
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            state = env.reset()
            state = np.array(state, np.float32)
            episode_reward = 0
            for step in range(MAX_STEPS):
                if RENDER:
                    env.render()
                action, value, logp, mean, log_std = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.array(next_state, np.float32)
                agent.buf.store(state, action, reward, value, logp, mean, log_std)
                episode_reward += reward
                state = next_state
                if agent.buf.is_full():
                    agent.finish_path(done, next_state)
                    agent.update()
                if done:
                    break
            agent.finish_path(done, next_state)
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TRAIN_EPISODES, episode_reward,
                    time.time() - t0
                )
            )
            if episode % SAVE_FREQ == 0:
                agent.save()
        agent.save()
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    if args.test:
        # test
        agent.load()
        for episode in range(TEST_EPISODES):
            state = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                env.render()
                action, *_ = agent.get_action(state, greedy=True)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                if done:
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward,
                    time.time() - t0
                )
            )