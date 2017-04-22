import random
import numpy
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge
from keras.optimizers import RMSprop
from keras import backend as K
from theano.gradient import disconnected_grad

class DqnAgent:
    def __init__(self, state_size=None, number_of_actions=1,
                 epsilon=0.1, minibatch_size=32, discount=0.9, memory_size=50,
                 save_name='basic', save_freq=10):
        self.state_size = state_size
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.minibatch_size = minibatch_size
        self.discount = discount
        self.memory_size = memory_size
        self.save_name = save_name
        self.memory_states = []
        self.memory_actions = []
        self.memory_rewards = []

        self.episode_id = 1
        self.save_freq = save_freq
        self.build_functions()

    def build_model(self):
        S = Input(shape=self.state_size)
        h = Convolution2D(16, 8, 8, subsample=(4, 4),
            border_mode='same', activation='relu')(S)
        h = Convolution2D(32, 4, 4, subsample=(2, 2),
            border_mode='same', activation='relu')(h)
        h = Flatten()(h)
        h = Dense(256, activation='relu')(h)
        V = Dense(self.number_of_actions)(h)
        self.model = Model(S, V)
        try:
            self.model.load_weights('{}.h5'.format(self.save_name))
            print "loading from {}.h5".format(self.save_name)
        except:
            print "Training a new model"


    def build_functions(self):
        S = Input(shape=self.state_size)
        NS = Input(shape=self.state_size)
        A = Input(shape=(1,), dtype='int32')
        R = Input(shape=(1,), dtype='float32')
        T = Input(shape=(1,), dtype='int32')

        self.build_model()
        self.value_fn = K.function([S], self.model(S))

        VS = self.model(S)
        VNS = disconnected_grad(self.model(NS))
        future_value = (1-T) * VNS.max(axis=1, keepdims=True)
        discounted_future_value = self.discount * future_value
        target = R + discounted_future_value
        cost = ((VS[:, A] - target)**2).mean()
        opt = RMSprop(0.0001)
        params = self.model.trainable_weights
        updates = opt.get_updates(params, [], cost)

        self.train_fn = K.function([S, NS, A, R, T], cost, updates=updates)

    def new_episode(self):
        self.memory_states.append([])
        self.memory_actions.append([])
        self.memory_rewards.append([])
        self.memory_states = self.memory_states[-self.memory_size:]
        self.memory_actions = self.memory_actions[-self.memory_size:]
        self.memory_rewards = self.memory_rewards[-self.memory_size:]
        self.episode_id += 1
        if self.episode_id % self.save_freq == 0:
            self.model.save_weights('{}.h5'.format(self.save_name), True)

    def act(self, state):

        values = self.value_fn([state[None, :]])
        if numpy.random.random() < self.epsilon:
            action = numpy.random.randint(self.number_of_actions)
        else:
            action = values.argmax()

        self.memory_states[-1].append(state)
        self.memory_actions[-1].append(action)
        return action, values

    def observe(self, reward):
        self.memory_rewards[-1].append(reward)
        return self.replay_train()

    def replay_train(self):
        memory_size = len(self.memory_states)

        batch_states = numpy.zeros((self.minibatch_size,) + self.state_size)
        batch_nextStates = numpy.zeros((self.minibatch_size,) + self.state_size)
        batch_actions = numpy.zeros((self.minibatch_size, 1), dtype=numpy.int32)
        batch_rewards = numpy.zeros((self.minibatch_size, 1), dtype=numpy.float32)
        T = numpy.zeros((self.minibatch_size, 1), dtype=numpy.int32)

        for i in xrange(self.minibatch_size):

            episode = random.randint(max(0, memory_size-50), memory_size-1)

            num_frames = len(self.memory_states[episode])
            frame = random.randint(0, num_frames-1)

            batch_states[i] = self.memory_states[episode][frame]
            T[i] = 1 if frame == num_frames - 1 else 0
            if frame < num_frames - 1:
                batch_nextStates[i] = self.memory_states[episode][frame + 1]
            batch_actions[i] = self.memory_actions[episode][frame]
            batch_rewards[i] = self.memory_rewards[episode][frame]

        cost = self.train_fn([batch_states, batch_nextStates, batch_actions, batch_rewards, T])
        return cost
