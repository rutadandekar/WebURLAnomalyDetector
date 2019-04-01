# import tensorflow as tf
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import *
from keras.utils import to_categorical
import keras_metrics

class NeuralNetworkModel:
    def __init__(self,n_inputs,learning_rate=None,batch_size=None,epochs=None,logdir=None,testing_split=0.3):
        self.name = "Neural Network Model"
        self.alpha = learning_rate
        self.n_inputs = n_inputs #input
        self.n_classes = 2 #output
        self.batch_size = batch_size
        self.hm_epochs = epochs
        self.logdir = logdir
        self.testing_split = testing_split
        self.rnd_idx = None
        self.loss = 'categorical_crossentropy'
        self.optimizer = Adam(lr=self.alpha)
        self.training_epoch = self.hm_epochs
        self.validation_split = 0.1
        self.model = None
        self.training_history = None
        self.construct_model()

    def construct_model(self):
        self.model = Sequential()
        self.model.add(Dense(units=100, activation='relu', input_dim=self.n_inputs))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=self.n_classes, activation='softmax'))
        # Calculate precision for the second label.
        pa = keras_metrics.precision(label=1)
        pna = keras_metrics.precision(label=0)
        # Calculate recall for the first label.
        ra = keras_metrics.recall(label=1)
        rna = keras_metrics.recall(label=0)
        self.model.compile(loss=self.loss, \
                      optimizer=self.optimizer, \
                      metrics=['accuracy',pa,pna,ra,rna])

    def process_input(self, data, shuffle=True):
        n_data = len(data[0])
        self.ip_data = np.column_stack(data)
        if shuffle==True:
            if self.rnd_idx is None:
                self.rnd_idx = random.sample(range(0,n_data),n_data)
            self.ip_data = self.ip_data[self.rnd_idx]
        testing_split = int((1.0-self.testing_split)*n_data)
        self.ip_data_training = self.ip_data[:testing_split]
        self.ip_data_testing = self.ip_data[testing_split:]

    def process_output(self, data, shuffle=True):
        n_data = len(data)
        self.op_data = to_categorical(data,num_classes=2)
        if shuffle==True:
            if self.rnd_idx is None:
                self.rnd_idx = random.sample(range(0,n_data),n_data)
            self.op_data = self.op_data[self.rnd_idx]
        testing_split = int((1.0-self.testing_split)*n_data)
        self.op_data_training = self.op_data[:testing_split]
        self.op_data_testing = self.op_data[testing_split:]

    def save_parameters(self,filename):
        if self.model is None:
            raise ValueError("No model!")
        self.model.save_weights(filename+'.h5')

    def load_parameters(self,filename):
        if self.model is None:
            raise ValueError("No model!")
        self.model.load_weights(filename+'.h5')

    def train(self):
        if self.model is None:
            raise ValueError("No model!")
        self.training_history = self.model.fit(
                    self.ip_data_training,
                    self.op_data_training,
                    epochs=self.training_epoch,
                    batch_size=self.batch_size,
                    verbose=2,
                    validation_split=self.validation_split,
                    shuffle=True)

    def test(self):
        if self.model is None:
            raise ValueError("No model!")
        evaluation = self.model.evaluate(
                    self.ip_data_testing,
                    self.op_data_testing,
                    batch_size=self.batch_size,
                    verbose=1)
        print '\nTesting loss:',evaluation[0]
        print 'Testing accuracy',evaluation[1]*100,'%'

    def detect(self):
        if self.model is None:
            raise ValueError("No model!")
        return self.model.predict(self.ip_data_testing)
