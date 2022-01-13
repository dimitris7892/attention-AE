import numpy as np
import pandas as pd
import scipy.stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# lstm autoencoder recreate sequence
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
import numpy as np
import tensorflow.keras.backend as K
from sklearn.ensemble import RandomForestRegressor
import matplotlib as plt
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.linear_model import LinearRegression
import seaborn as sns
from AttentionDecoder import AttentionDecoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from random import randrange, sample
#from pyinform import conditional_entropy
from  tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import matplotlib.patches as patches
import pyearth as sp
#from skgof import ks_test, cvm_test, ad_test
import extractSequentialTasks as extTasks
import openpyxl
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.callbacks import Callback, LambdaCallback
#tf.compat.v1.disable_eager_execution()


extasks = extTasks.extractSequencialTasks()
trData = extasks.getData()

raw_seq = trData#[:4000]
seqLSTM = raw_seq
# split into samples
#seqLSTM = split_sequence(raw_seq, n_steps)
#seqLSTM = seqLSTM.reshape(-1,n_steps,9)
n_steps = 20
dataLength = len(seqLSTM)
tasksA=[]
tasksB=[]
tasksAMem=[]
tasksBMem=[]
lenS = 100
start = 13000
tasksNew = extasks.exctractTasks(10, 20, 1000)

def ApEn(self, U, m, r) -> float:
    """Approximate_entropy."""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))

def entropy(self, Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count / len(Y)
    en = np.sum((-1) * prob * np.log2(prob))
    return en

def pointInRect(self, point, rect):
    x1, y1 = rect.xy
    w, h = rect.get_bbox().width, rect.get_bbox().height
    x2, y2 = x1 + w, y1 + h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False

def boxCountingMethod(self, sequence, ylabel, width=None, height=None):
    width = 100
    height = 1
    ax = plt.gca()
    colors = ['blue', 'red']
    colorSeq = ['green', 'black']
    labels = ['TASKA', 'TASKB']
    counters = []
    rectsDicts = []
    for i in range(0, len(sequence)):

        rectsDict = {'rects': []}
        for k in range(0, lenS, width):  ##columns
            for n in range(0, int(max(sequence[i])) + 1):  # rows
                rect = patches.Rectangle((k, n), width, height, linewidth=1, edgecolor=colors[i], facecolor='none')
                item = {}
                item['visited'] = False
                item['rect'] = rect
                rectsDict['rects'].append(item)

                # ax.add_patch(rect)
        plt.plot(sequence[i], c=colorSeq[i], label=labels[i])
        # Add the patch to the Axes
        rectsVisited = []
        # plt.show()
        counter = 0
        for n in range(0, len(sequence[i])):
            for rect in rectsDict['rects']:
                if pointInRect((n, sequence[i][n]), rect['rect']) and rect['visited'] == False:
                    counter += 1
                    rect['visited'] = True
                    rectsDict['rects'].remove(rect)
                    rectsVisited.append(rect['rect'])
                    break
        counters.append(counter)
        for rect in rectsVisited:
            ax.add_patch(rect)

        plt.xlabel('time')
        plt.ylabel(ylabel)
        rectsDicts.append(rectsVisited)
    plt.title('Fractal dimension with box counting method for task A: ' + str(counters[0]) + " for task B: " + str(
        counters[1]) + " with τ = " + str(width) + " and α = " + str(height))
    plt.legend()

    dictA = rectsDicts[0]
    dictB = rectsDicts[1]
    filteredSeqAs = []
    indices = []

    k = 0
    xis = []
    for i in range(0, len(dictB)):  ##iterate rects of B
        filteredSeqA = []
        indices = []
        xi = int(dictB[i].xy[0])
        xi_hat = int(dictB[i].get_bbox().width)
        xis.append(xi)
        if xi != xis[i - 1] or i == 0:
            for n in range(xi, xi + xi_hat):
                if pointInRect((n, sequence[0][n]), dictB[i]) == False:
                    # ax.add_patch(rect['rect'])
                    flag = True
                    filteredSeqA.append(sequence[0][n])

                    indices.append(n)
        # ax.add_patch(plt.plot(filteredSeqA))
        filteredSeqAs.append(indices)

    print(len(indices))

    # plt.show()
    return tasksA[0][list(np.concatenate(filteredSeqAs).astype(int))]
    # print("Fractal dimension: "+str(counter))

    # Joint Entropy

def jEntropy(self, Y, X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y, X]
    return entropy(YX)

    # Conditional Entropy

def cEntropy(self, Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(Y, X) - entropy(X)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a time series."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class NewCallback(Callback):

   def __init__(self, alpha):
        self.alpha = alpha

   def on_epoch_end(self, epoch, logs = {}):
       K.set_value(self.alpha, K.get_value(self.alpha))

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


    def train_step(self, data):
        with tf.GradientTape() as tape:
            #data = data[0]
            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)
            #print(data[0])
            #print(reconstruction)
            #print(data[0])
            #print(reconstruction[0])
            reconstruction_loss =  keras.losses.mean_squared_error(data, reconstruction)
                #keras.losses.mean_squared_error(data,reconstruction)
            '''reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.sparse_categorical_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )'''

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            #print(kl_loss)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss #+ kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class LSTM_AE_IW(keras.Model):

    def __init__(self, encoder, decoder, **kwargs):
        super(LSTM_AE_IW, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        #self.seqlstma = seqlstma
        #self.seqlstmb = seqlstmb
        #self.mem = mem
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")


    @property
    def metrics(self):
        return [
            self.total_loss_tracker

        ]

    def train_step(self, data):

        with tf.GradientTape() as tape:
            data = data[0]

            encoderOutput = self.encoder(data)
            reconstruction = self.decoder(encoderOutput)

            reconstruction_loss = keras.losses.mean_squared_error(data,reconstruction) #+\
            keras.losses.kullback_leibler_divergence(encoderOutput, reconstruction)
            #keras.losses.mean_squared_error(data,reconstruction) +\


            total_loss = reconstruction_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        #self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "loss": self.total_loss_tracker.result(),

        }

class LSTM_AE(keras.Model):

    def __init__(self, encoder, decoder, **kwargs):
        super(LSTM_AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        #self.seqlstma = seqlstma
        #self.seqlstmb = seqlstmb
        #self.mem = mem
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")



    @property
    def metrics(self):
        return [
            self.total_loss_tracker

        ]

    def train_step(self, data):

        with tf.GradientTape() as tape:
            #data = data[0]

            encoderOutput = self.encoder(data)
            reconstruction = self.decoder(encoderOutput)

            reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction) \
                                  +  keras.losses.kullback_leibler_divergence(encoderOutput,reconstruction)
            #

            total_loss = reconstruction_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        #self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "loss": self.total_loss_tracker.result(),

        }


weightsPrev = tf.convert_to_tensor(np.zeros(shape=(6,)))
loss_tracker = keras.metrics.Mean(name="loss")

class Base_Learner(keras.Model):


    def train_step(self, data):
        global weightsPrev
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # Compute our own loss

            loss = keras.losses.mean_squared_error(y, y_pred) #+ keras.losses.mean_squared_error(weightsPrev, self.trainable_weights[6])

        print(weightsPrev)
        print(self.trainable_weights[6])
        # Compute gradients
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, trainable_weights)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        # Update metrics (includes the metric that tracks the loss)
        loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value

        return {
            "loss": loss_tracker.result(),

        }

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.

        return [loss_tracker]

class AE_detect():


    def __init__(self):

        self.weights = []
        self.weightsDict = {}
        '''initialize window indentification AE '''
        encoderA, decoderA = self.windwowIdentificationModel()
        # print(encoderA.summary())
        # print(decoderA.summary())
        self.windAEa = LSTM_AE_IW(encoderA, decoderA)
        self.windAEa.compile(optimizer=keras.optimizers.Adam())


    def custom_loss(self, y_true, y_pred):
        print(self.weightsDict)
        return tf.keras.losses.mean_squared_error(y_true, y_pred) + tf.keras.losses.mean_squared_error(self.weightsCurr , self.weightsPrev)

    def vae_LSTM_Model(self, ):
        ##ENCODER
        latent_dim = 100

        encoder_inputs = keras.Input(shape=(n_steps,1000))
        x = layers.LSTM(500, return_sequences=True )(encoder_inputs)
        #x = layers.Dense(200, )(x)
        #x = layers.Flatten()(x)
        x = TimeDistributed(layers.Dense(1000,  ))(x)

        # x  = tf.reshape(x,shape=(-1,1,16))
        # x = layers.LSTM(16,name='memory_module')(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        ##DECODER
        latent_inputs = keras.Input(shape=(n_steps, latent_dim))
        x = layers.LSTM(1000,return_sequences=True )(latent_inputs)
        # x = layers.Reshape((7, 7, 64))(x)
        #x = layers.Dense(200,  )(x)
        x = layers.LSTM(500, return_sequences=True )(x)
        decoder_outputs = TimeDistributed(layers.Dense(1000,  ))(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        return encoder, decoder

    def VAE_windwowIdentificationModel(self, ):
        ##ENCODER
        latent_dim = 100
        input_dim = 1000

        encoder_inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(input_dim, )(encoder_inputs)
        x = layers.Dense(500, )(x)
        #x = layers.Flatten()(x)
        x = layers.Dense(latent_dim, )(x)

        # x  = tf.reshape(x,shape=(-1,1,16))
        # x = layers.LSTM(16,name='memory_module')(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        ##DECODE9
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(latent_dim,)(latent_inputs)
        # x = layers.Reshape((7, 7, 64))(x)
        x = layers.Dense(500,  )(x)
        #x = layers.Dense(700, activation="relu", )(x)
        decoder_outputs = layers.Dense(input_dim,  )(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
        return encoder, decoder

    def windwowIdentificationModelALT(self, ):
        timesteps = 1 # Length of your sequences
        input_dim = 1000
        latent_dim = 1000
        features = 7
        output_dim = 2000
        inputs = keras.Input(shape=(n_steps,7 ))

        #encoded = layers.LSTM(700,kernel_initializer='he_uniform',return_sequences=True,activation='relu')(inputs)
        #encoded = layers.Dense(input_dim, activation='relu')(inputs)

        encoded = layers.LSTM(3,return_sequences=True,)(inputs)
        #encoded = layers.LSTM(1500, return_sequences=True, )(encoded)
        #encoded = layers.LSTM(900, return_sequences=True, )(encoded)


        encoded = TimeDistributed(layers.Dense(7,  ))(encoded)
            #TimeDistributed(layers.Dense(latent_dim, ))(encoded)
            #layers.LSTM(latent_dim, return_sequences=True )(encoded)#
            #

        #encoded = layers.Dense(latent_dim, activation='relu')(encoded)

        #encoded = layers.RepeatVector(features)(encoded)

        latent_inputs = keras.Input(shape=( n_steps, 7,))


        decoded = layers.LSTM(3, return_sequences=True,name='dec2',)(latent_inputs)
        #decoded = layers.LSTM(1500, return_sequences=True, name='dec3',)(decoded)
        #decoded = layers.LSTM(700, return_sequences=True, name='dec3')(decoded)
        #decoded = layers.LSTM(900, return_sequences=True, name='dec4')(decoded)
        #decoded = layers.LSTM(700, kernel_initializer='he_uniform', activation='relu', return_sequences=True, name='dec3')(decoded)
        decoded = TimeDistributed(layers.Dense(features, ))(decoded)
        #decoded = layers.Dense(input_dim, activation="relu", )(decoded)

        #sequence_autoencoder = keras.Model(inputs, decoded)
        encoder = keras.Model(inputs, encoded)
        decoder = keras.Model(latent_inputs, decoded, name="decoder")

        return  encoder , decoder

    def windwowIdentificationModelGEN(self, ):
        timesteps = 1 # Length of your sequences
        input_dim = 2000
        latent_dim = 2000
        features = 7
        output_dim = 2000
        inputs = keras.Input(shape=(n_steps,input_dim ))

        #encoded = layers.LSTM(700,kernel_initializer='he_uniform',return_sequences=True,activation='relu')(inputs)
        #encoded = layers.Dense(input_dim, activation='relu')(inputs)

        encoded = layers.LSTM(1000,return_sequences=True,)(inputs)
        #encoded = layers.LSTM(1500, return_sequences=True, )(encoded)
        #encoded = layers.LSTM(900, return_sequences=True, )(encoded)


        encoded = TimeDistributed(layers.Dense(latent_dim,  ))(encoded)
            #TimeDistributed(layers.Dense(latent_dim, ))(encoded)
            #layers.LSTM(latent_dim, return_sequences=True )(encoded)#
            #

        #encoded = layers.Dense(latent_dim, activation='relu')(encoded)

        #encoded = layers.RepeatVector(features)(encoded)

        latent_inputs = keras.Input(shape=( n_steps,latent_dim,))


        decoded = layers.LSTM(1000, return_sequences=True,name='dec2',)(latent_inputs)
        #decoded = layers.LSTM(1500, return_sequences=True, name='dec3',)(decoded)
        #decoded = layers.LSTM(700, return_sequences=True, name='dec3')(decoded)
        #decoded = layers.LSTM(900, return_sequences=True, name='dec4')(decoded)
        #decoded = layers.LSTM(700, kernel_initializer='he_uniform', activation='relu', return_sequences=True, name='dec3')(decoded)
        decoded = TimeDistributed(layers.Dense(input_dim, ))(decoded)
        #decoded = layers.Dense(input_dim, activation="relu", )(decoded)

        #sequence_autoencoder = keras.Model(inputs, decoded)
        encoder = keras.Model(inputs, encoded)
        decoder = keras.Model(latent_inputs, decoded, name="decoder")

        return  encoder , decoder

    def lstmAUTO(self, ):

        timesteps = 1000
        input_dim = 7
        latent_dim = 1000

        inputs = keras.Input(shape=( 1000,input_dim))
        encoded = layers.LSTM(500, return_sequences=True)(inputs)
        encoded = layers.LSTM(1000, return_sequences=True)(encoded)

        #decoded = layers.RepeatVector(input_dim)(encoded)

        decoded = layers.LSTM(500, return_sequences=True)(encoded)
        decoded = layers.LSTM(input_dim, return_sequences=True)(decoded)

        sequence_autoencoder = keras.Model(inputs, decoded)
        encoder = keras.Model(inputs, encoded)
        #decoder = keras.Model(latent_inputs, decoded, name="decoder")
        return sequence_autoencoder , encoder

    def windwowIdentificationModel(self, ):
        timesteps = n_steps # Length of your sequences
        input_dim = lenS - n_steps
        latent_dim = lenS - n_steps
        features = 7
        output_dim = lenS
        #inputs = keras.Input(shape=(features, input_dim))
        inputs = keras.Input(shape=(features, timesteps))#(7,20)

        # keras.layers.Bidirectional(
        encoded = (layers.LSTM(n_steps - int(n_steps/2), return_sequences=True, activation='tanh'))(inputs)

        encoded =  layers.LSTM(timesteps, return_sequences=True, activation='tanh')(encoded)
        #encoded = TimeDistributed(layers.Dense(latent_dim, ))(encoded)
        # TimeDistributed(layers.Dense(latent_dim, ))(encoded)
        # layers.LSTM(latent_dim, return_sequences=True )(encoded)#
        latent_inputs = keras.Input(shape=( features, timesteps))
        # decoded = layers.LSTM(1000, return_sequences=True, name='dec2', activation='tanh')(latent_inputs)

        decoded =  (layers.LSTM(n_steps - int(n_steps/2),  name='dec4', return_sequences=True, activation='tanh'))(latent_inputs)

        decoded =  layers.LSTM(timesteps, return_sequences=True, activation='tanh')(decoded)
        # decoded = layers.Dense(input_dim, activation="relu", )(decoded)

        #sequence_autoencoder = keras.Model(inputs, decoded)
        encoder = keras.Model(inputs, encoded)
        decoder = keras.Model(latent_inputs, decoded, name="decoder")

        #print(encoder.summary())
        #print(decoder.summary())
        return  encoder , decoder

    def baselineModel(self, newDimension=None):
        #latent_dim = 100 if newDimension==None else newDimension
        latent_dim = 7
        features = 7
        input_dim = 1000 if newDimension==None else newDimension

        encoder_inputs = keras.Input(shape=(n_steps,features ))

        encoded = layers.LSTM(500, activation='relu',return_sequences=True, name='lstmENC1')(encoder_inputs)
        #encoded = layers.LSTM(300, activation='relu', return_sequences=True, name='lstmENC2')(encoded)
        encoded = layers.LSTM(latent_dim, activation='relu',name='lstmENC')(encoded)
        encoded = RepeatVector(n_steps)(encoded)

        encoder = keras.Model(encoder_inputs, encoded , name="encoder")


        latent_inputs = keras.Input(shape=(n_steps,latent_dim))


        decoded = layers.LSTM(latent_dim, activation='relu',return_sequences=True, name='lstmDEC')(latent_inputs)
        #decoded = layers.LSTM(300, activation='relu', return_sequences=True, name='lstmDEC1')(decoded)
        decoded = layers.LSTM(500, activation='relu', return_sequences=True, name='lstmDEC2')(decoded)
        #decoded = layers.LSTM(500, activation='relu',  kernel_initializer='he_uniform',return_sequences=True, name='lstmDEC1')(decoded)


        decoded = TimeDistributed(layers.Dense(features, activation="relu", ))(decoded)

        decoder = keras.Model(latent_inputs, decoded, name="decoder")

        return encoder, decoder

    def getMemoryWindowBetweenTaskA_TaskB(self, lenSeq, seqLSTMA, seqLSTMB):

        '''scale seqlLSTMA given the task of incremental training
        of a base learner from taskA=>taskB in a way that the accuracy
        of BL evaluated on both tasks when incorporating the scaled info
        is better when evaluated wihout it'''

        ######################################################################
        min_max_scalerA = MinMaxScaler()
        seqAScaled = []
        for i in range(0, len(seqLSTMA)):
            seqScaled = min_max_scalerA.fit_transform(seqLSTMA[i])
            #seqScaled = seqLSTMA[i]
            seqScaled = seqScaled.transpose()
            seqAScaled.append(seqScaled)

        X_train_normA = np.array(seqAScaled)

        min_max_scalerB = MinMaxScaler()
        seqBScaled = []
        for i in range(0, len(seqLSTMB)):
            seqScaled = min_max_scalerB.fit_transform(seqLSTMB[i])
            #seqScaled = seqLSTMB[i]
            seqScaled = seqScaled.transpose()
            seqBScaled.append(seqScaled)

        X_train_normB = np.array(seqBScaled)

        seqA, seqB = X_train_normA, X_train_normB

        es = keras.callbacks.EarlyStopping(monitor='loss', restore_best_weights=True, mode='min')

        self.windAEa.fit(seqA, seqB, epochs=30, )#batch_size=20


        encodedTimeSeriesA = np.round(self.windAEa.encoder.predict(seqA), 2)


        batches = []
        for i in range(0,len(encodedTimeSeriesA)):
            #batch = min_max_scalerA.inverse_transform((seqA[i] + (seqA[i] * encodedTimeSeriesA[i])).transpose())
            #batches.append(batch)
            seq_a = seqLSTMA[i]#.transpose()
            #seq_a = min_max_scalerA.inverse_transform(seqA[i].transpose())
            seq_a = min_max_scalerA.inverse_transform(encodedTimeSeriesA[i].transpose())
            weight = encodedTimeSeriesA[i].transpose()
            #weight = np.random.random((20, 7))
            #batch = (seqA[i].transpose() + (weight))
            batch = (seq_a)
            batches.append(batch)

        selectiveMemory_ExtractedOfTaskA = np.array(batches)


        return selectiveMemory_ExtractedOfTaskA, None

    def plotDistributions(self, seqA,seqB ,var):

        dfA = pd.DataFrame({'stwA':seqA[:,3]})
        dfB = pd.DataFrame({'stwB': seqB[:, 3]})

        sns.displot(dfA, x='stwA')
        sns.displot(dfB, x='stwB')

        plt.show()

    def plotErrBetweenTasks(self, seqLSTMA, seqLSTMB, lrSpeedSel , lrSpeedFull):

        '''visualizeLen = 1000
        plt.plot(np.linspace(0, visualizeLen, visualizeLen), y_hatsa[:visualizeLen], color='red',label='predicted')
        plt.plot(np.linspace(0, visualizeLen, visualizeLen),ya[:visualizeLen],color='green',label='TASKA')
        plt.plot(np.linspace(visualizeLen, visualizeLen*2, visualizeLen), y_hatsb[:visualizeLen], color='red')
        plt.plot(np.linspace(visualizeLen, visualizeLen*2, visualizeLen),yb[:visualizeLen],'blue',label='TASKB')'''
        plot='B'
        if plot =='A':
            i = np.min(seqLSTMA[:,3])
            maxSpeedA = np.max(seqLSTMA[:,3])
            focsPLot = []
            speedsPlot = []
            ranges = []
            featuresPlot = []
            while i <= maxSpeedA:
                # workbook._sheets[sheet].insert_rows(k+27)
                focArray = np.array([k for k in seqLSTMA if float(k[3]) >= i - 0.25 and float(k[3]) <= i + 0.25])
                # focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')
                '''meanFoc = np.mean(focArray[:, 8])
                stdFoc = np.std(focArray[:, 8])
                speedFoc = np.array([k for k in focArray if k[8] >= (meanFoc - (3 * stdFoc)) and k[8] <= (meanFoc + (3 * stdFoc))])'''

                if focArray.__len__() > 0:
                    focsPLot.append(focArray.__len__())
                    featuresPlot.append(np.mean(focArray[:,:6],axis=0))
                    speedsPlot.append(np.mean(focArray[:,3],axis=0))
                    ranges.append(np.mean(focArray[:, 6]))
                    # lrSpeedFoc.fit(focArray[:,5].reshape(-1, 1), focArray[:,8].reshape(-1, 1))
                i += 0.5
                # k += 1

            xi = np.array(speedsPlot)
            yi = np.array(ranges)
            zi = np.array(focsPLot)

            # p2 = np.poly1d(np.polyfit(xi, yi, 2,w=focsPLot),)
            #xiBlue = np.linspace(9, 20, 20 - 9)

            # plt.plot([], [], '.', xp, p2(xp))
            speedList = [8, 9, 10, 11, 12, 13, 14]

            plt.plot(xi, lrSpeedSel.predict(np.concatenate(featuresPlot).reshape(-1, 6)), c='blue')
            plt.plot(xi, yi, c='red')

            plt.scatter(xi, yi, s=zi / 10, c="red", alpha=0.4, linewidth=4)
            # plt.xticks(np.arange(np.floor(min(xi)), np.ceil(max(xi)) + 1, 1))
            # plt.yticks(np.arange(min(yi), max(yi) + 1, 5))
            plt.xlabel("Speed (knots)")
            plt.ylabel("FOC (MT / day)")
            plt.title("Density plot for taskA", loc="center")
            plt.show()

        i = np.min(seqLSTMB[:, 3])
        maxSpeedB = np.max(seqLSTMB[:, 3])
        focsPLot = []
        speedsPlot = []
        ranges = []
        featuresPlot = []
        while i <= maxSpeedB:
            # workbook._sheets[sheet].insert_rows(k+27)
            focArray = np.array([k for k in seqLSTMB if float(k[3]) >= i - 0.25 and float(k[3]) <= i + 0.25])
            # focsApp.append(str(np.round(focArray.__len__() / focAmount * 100, 2)) + '%')
            '''meanFoc = np.mean(focArray[:, 8])
            stdFoc = np.std(focArray[:, 8])
            speedFoc = np.array([k for k in focArray if k[8] >= (meanFoc - (3 * stdFoc)) and k[8] <= (meanFoc + (3 * stdFoc))])'''

            if focArray.__len__() > 0:
                focsPLot.append(focArray.__len__())
                featuresPlot.append(np.mean(focArray[:, :6], axis=0))
                speedsPlot.append(np.mean(focArray[:, 3], axis=0))
                ranges.append(np.mean(focArray[:, 6]))
                # lrSpeedFoc.fit(focArray[:,5].reshape(-1, 1), focArray[:,8].reshape(-1, 1))
            i += 0.5
            # k += 1

        xi = np.array(speedsPlot)
        yi = np.array(ranges)
        zi = np.array(focsPLot)

        plt.plot(xi, lrSpeedSel.predict(np.concatenate(featuresPlot).reshape(-1, 6)), c='blue',label='Base learner with selected mem')
        plt.plot(xi, lrSpeedFull.predict(np.concatenate(featuresPlot).reshape(-1, 6)), c='green',label='Base learner with full mem')
        plt.plot(xi, yi, c='red')

        plt.scatter(xi, yi, s=zi / 10, c="red", alpha=0.4, linewidth=4)
        # plt.xticks(np.arange(np.floor(min(xi)), np.ceil(max(xi)) + 1, 1))
        # plt.yticks(np.arange(min(yi), max(yi) + 1, 5))
        plt.xlabel("Speed (knots)")
        plt.ylabel("FOC (MT / day)")
        plt.legend()
        plt.title("Density plot for task B", loc="center")
        plt.show()

    def baselineLearner(self, ):
        # create model
        model = keras.models.Sequential()

        model.add(keras.layers.LSTM(50,input_shape=(n_steps, 6,), ))  # return_sequences=True

        model.add(keras.layers.Dense(20, ))

        model.add(keras.layers.Dense(6, ))

        model.add(keras.layers.Dense(1))

        model.compile(loss='mse',
                      optimizer=keras.optimizers.Adam())  # experimental_run_tf_function=False )
        # print(model.summary())

        return model

    def customBaselineLearner(self, ):

        # create model
        inputs = keras.Input(shape=(n_steps, 6,), )

        x1 = keras.layers.LSTM(50,input_shape=(n_steps, 6,), )(inputs)  # return_sequences=True

        x2 = keras.layers.Dense(20, )(x1)

        x3 = keras.layers.Dense(6, )(x2)

        outputs = keras.layers.Dense(1)(x3)

        model = Base_Learner(inputs, outputs)

        model.compile(optimizer=keras.optimizers.Adam())  # experimental_run_tf_function=False )
        # print(model.summary())

        return model

    def trainingBaselinesForFOCestimation(self, seqLSTMmem, memoryA, currInd , memories, methods):

        #tf.compat.v1.disable_eager_execution()
        global weightsPrev
        print("Memory of taskA: "+str(len(memoryA)))
        maesprev_s = [0]
        maescurr_s = [0]
        maes0_s = [0]
        maes_sum = []
        mape_sum = []
        maes_s = []
        mape_s = []

        maesa_f = [0]
        maesb_f = [0]
        maes0_f = [0]
        self.weightsCurr = tf.convert_to_tensor([0.0])
        #self.weightsPrev = tf.convert_to_tensor([0.0])

        if 'seq' in methods:
            print("LSTM training  of tasks sequentially  . . .")
            lrBase = self.baselineLearner()

            for i in range(0,len(seqLSTMmem)):

                currTask = seqLSTMmem[i]

                print("LSTM training  of task "+str(i)+" . . .")

                x = []
                y = []
                for k in range(0, len(currTask)):
                    x.append(currTask[k][:, :6])
                    y.append(currTask[k][n_steps - 1, 6])

                x = np.array(x)
                y = np.array(y)
                lrBase.fit(x, y, epochs=20)

                maes_ = []
                pes = []
                for k in range(0, len(x)):
                    y_hat = lrBase.predict(x[k].reshape(1,n_steps,6))[0][0]
                    err = abs(y_hat - y[k])
                    pe =  abs(( y[k] - y_hat)/ y[k])
                    pes.append(pe)
                    maes_.append(err)


                maes_sum.append(np.mean(maes_))
                mape_sum.append(np.mean(pes))

            for ind in range(0, len(tasksNew)):
                print("Score for task " + str(ind) + " without memory:" + str(maes_sum[ind]))

        if 'mem' in methods:
            ##############################################################
            print("LSTM training of tasks with extracted memory . . ")

            #for i in range(0,len(seqLSTMmem)):

            #currTask = seqLSTMmem[i]
            #lrAB = self.baselineLearner()
            lrAB = self.customBaselineLearner()
            #self.weightsDict = {}
            print_weights = LambdaCallback( on_epoch_end=lambda epoch, logs:  self.weightsDict.update({epoch:tf.ragged.constant(lrAB.get_weights()[6])}) )


            for ind in range(0,len(seqLSTMmem)):

                currTask = seqLSTMmem[ind]
                batchesX = []
                batchesY = []
                for i in range(0, len(currTask)):
                    batchesX.append(currTask[i][:, :6])
                    batchesY.append(currTask[i][n_steps - 1, 6])

                batchesX = np.array(batchesX)
                batchesY = np.array(batchesY)

                if ind > 0:
                    batchesXmem = []
                    batchesYmem = []
                    memory = memories[ind]
                    for i in range(0, len(memory)):
                        batchesXmem.append(memory[i][:, :6])
                        batchesYmem.append(memory[i][n_steps - 1, 6])

                    batchesXmem = np.array(batchesXmem)
                    batchesYmem = np.array(batchesYmem)

                    batchesXsel = np.append(batchesXmem, batchesX, axis=0)
                    batchesYsel = np.append(batchesYmem, batchesY)

                if ind == 0:

                    self.weightsCurr = tf.convert_to_tensor([0.0])
                    weightsPrev = tf.convert_to_tensor(np.zeros(shape=(6,)))
                    lrAB.fit(batchesX, batchesY, epochs=20)

                else:
                    self.weightsCurr = tf.ragged.constant(lrAB.get_weights()[6])
                    weightsPrev = tf.ragged.constant(self.weights[ind - 1][6])
                    #weightsPrev = self.weights[ind - 1][6]
                    print("Before Fit: " + str(weightsPrev))
                    lrAB.fit(batchesXsel,  batchesYsel, epochs=20, callbacks=[print_weights])

                self.weights.append(lrAB.get_weights())
                #print(self.weightsDict)

                maes = []
                pes = []
                for i in range(0, len(batchesX)):
                    y_hat = lrAB.predict(batchesX[i].reshape(1, n_steps, 6))[0][0]
                    err = abs(y_hat - batchesY[i])
                    pe = abs((batchesY[i]  - y_hat)/batchesY[i]  )
                    maes.append(err)
                    pes.append(pe)
                maes_s.append(np.mean(maes))
                mape_s.append(np.mean(pes))
            ####################################################################

            for ind in range(0, len(tasksNew)):
                print("Score for task " + str(ind) + " with selective memory:" + str(maes_s[ind]))

        if 'full' in methods:

            print("LSTM training with full memory of tasks . . ")
            lrAfull = self.baselineLearner()

            xa = []
            ya = []
            prevTask = seqLSTMmem[currInd - 1]
            for i in range(0, len(prevTask)):
                xa.append(prevTask[i][:, :6])
                ya.append(prevTask[i][n_steps - 1, 6])

            xa = np.array(xa)
            ya = np.array(ya)


            xb = []
            yb = []
            currTask = seqLSTMmem[currInd ]
            for i in range(0, len(currTask)):
                xb.append(currTask[i ][:, :6])
                yb.append(currTask[i ][n_steps - 1, 6])

            xb = np.array(xb)
            yb = np.array(yb)

            x0 = []
            y0 = []
            firstTask = seqLSTMmem[0]
            for i in range(0, len(firstTask)):
                x0.append(firstTask[i][:, :6])
                y0.append(firstTask[i][n_steps - 1, 6])

            lstmXfull = np.append(x0, xa, axis=0)
            lstmXfull = np.append(lstmXfull, xb, axis=0)
            lstmYfull = np.append(y0, ya)
            lstmYfull = np.append(lstmYfull, yb)



            lrAfull.fit(lstmXfull, lstmYfull, epochs=20)



            for i in range(0,len(xb)):
                y_hat = lrAfull.predict(xb[i].reshape(1,n_steps,6))[0][0]
                err = abs(y_hat- yb[i])
                maesb_f.append(err)

            for i in range(0,len(xa)):
                y_hat = lrAfull.predict(xa[i].reshape(1,n_steps,6))[0][0]
                err = abs(y_hat - ya[i])
                maesa_f.append(err)

            for i in range(0,len(xa)):
                y_hat = lrAfull.predict(x0[i].reshape(1,n_steps,6))[0][0]
                err = abs(y_hat - y0[i])
                maes0_f.append(err)



            for ind in range(0, len(tasksNew)):
                print("Score for task "+str(ind)+" with full memory:" + str(np.mean(maes_[ind])))


        dict = {"trMethod": {'sel':[], "without": []}}

        for k in range(0,len(tasksNew)):

            #selPerf = np.round(100 - mape_s[k],2)
            #withoutPerf = np.round(100 - mape_sum[k],2)
            dict['trMethod']['sel'].append(maes_s[k])
            dict['trMethod']['without'].append( maes_sum[k])


        df = pd.DataFrame.from_dict(dict['trMethod'])

        df.to_csv('./AE_files/perfLSTM.csv')
        return df

    def plotPerfBetweenTASKS(self,):

        perf = pd.read_csv("./AE_files/perfLSTM.csv")

        perfSel = perf['sel'].values
        perfWithout = perf['without'].values
        tasksNames = []
        for k in range(0,len(perfSel)):

            tasksNames.append("TASK {}".format(k))

        plt.plot(tasksNames, perfSel, c = 'red', label='LSTM Perf with memory between tasks')
        plt.plot(tasksNames, perfWithout, c='blue', label='LSTM Perf without memory between tasks')
        plt.xlabel("TASKS")
        plt.ylabel("MAE")
        plt.grid()
        plt.legend()
        plt.show()

    def runAlgorithmsforEvaluation(self,  alg, seqLen):
        memories = None
        methods = ['seq','mem']
        if alg!='RND':
            dfs = []
            memories = []
            for k in range(0, len(tasksNew)):

                if k==0 :
                    memory, memoryB = self.getMemoryWindowBetweenTaskA_TaskB(seqLen, tasksNew[k], tasksNew[k+1])
                elif k < len(tasksNew) - 1:
                    memory, memoryB = self.getMemoryWindowBetweenTaskA_TaskB(seqLen, memory, tasksNew[k + 1])

                memories.append(memory)

            df = self.trainingBaselinesForFOCestimation(tasksNew, memory, 2, memories, methods)
            dfs.append(df)

            merged = pd.concat(dfs)
            merged.to_csv('./AE_files/' + alg + '.csv')

        if alg =='RND':
            dfs = []
            alg = 'RND'
            for k in range(0, 5):
                randomMemoryofTaskA = seqLSTMA[np.random.randint(seqLSTMA.shape[0], size=len(memories[k])), :]

                df = self.trainingBaselinesForFOCestimation(seqLSTMA, seqLSTMB, randomMemoryofTaskA, alg)
                dfs.append(df)

            merged = pd.concat(dfs)
            merged.to_csv('./AE_files/' + alg + '.csv')

        return memories

def main():

    #plotDistributions(seqLSTMA,seqLSTMB,'foc')
    #return
    # memory = pd.read_csv('./AE_files/selectiveMemory_ExtractedOfTaskA.csv', ).values
    aedetect = AE_detect()

    aedetect.plotPerfBetweenTASKS()
    return

    lenMemories = aedetect.runAlgorithmsforEvaluation('LR', lenS )
    pd.DataFrame({'memories':lenMemories}).to_csv('./AE_files/lenMemories.csv')
    lr = pd.read_csv('./AE_files/LR.csv').values

    fullMemMeanErrA = np.mean(np.array([k for k in lr if k[0] == 'ScoreA with full memory'])[:, 1])
    fullMemMeanErrB = np.mean(np.array([k for k in lr if k[0] == 'ScoreB with full memory'])[:, 1])
    fullMemMeanErrC = np.mean(np.array([k for k in lr if k[0] == 'ScoreC with full memory'])[:, 1])

    selMemMeanErrA = np.mean(np.array([k for k in lr if k[0] == 'ScoreA with selective memory'])[:, 1])
    selMemMeanErrB = np.mean(np.array([k for k in lr if k[0] == 'ScoreB with selective memory'])[:, 1])
    selMemMeanErrC = np.mean(np.array([k for k in lr if k[0] == 'ScoreC with selective memory'])[:, 1])

    withoutMemMeanErrA = np.mean(np.array([k for k in lr if k[0] == 'ScoreA without memory'])[:, 1])
    withoutMemMeanErrB = np.mean(np.array([k for k in lr if k[0] == 'ScoreB without memory'])[:, 1])
    withoutMemMeanErrC = np.mean(np.array([k for k in lr if k[0] == 'ScoreC without memory'])[:, 1])

    print("full MEM error for A, B, C "+str((fullMemMeanErrA + fullMemMeanErrB + fullMemMeanErrC)/3))
    print("sel MEM error for A, B, C " + str((selMemMeanErrA + selMemMeanErrB + selMemMeanErrC) / 3))
    print("without MEM error for A, B, C " + str((withoutMemMeanErrA + withoutMemMeanErrB + withoutMemMeanErrC) / 3))


    #print(str(len(memory)))
    #memory = pd.read_csv('./AE_files/selectiveMemory_ExtractedOfTaskA.csv',).values
    #print(str(len(memory)))
    return
    scoresAE=[]
    scoresAE_mem = []
    '''for k in range(0,5):
       scoreAE =  trainAE()
       scoresAE.append(np.round(scoreAE,2))

    scores = pd.DataFrame({'scoreAE':scoresAE,})
    scores.to_csv('./AE_files/scoresAE.csv')'''

    #return

    '''for k in range(0,5):
        scoreAE_mem = trainAE_withMemoryOfPrevTask(memory)
        scoresAE_mem.append(np.round(scoreAE_mem,2))

    scores = pd.DataFrame({'scoreAE_mem':scoresAE_mem})
    scores.to_csv('./AE_files/scoresAE_mem.csv')'''

    stwMem = memory[:,3]
    stwA = seqLSTMA[:,3]
    while len(stwMem)<=len(stwA):
        stwMem = np.append(stwMem,stwMem)



    foc12 = [(itm, 'stw of task A') for itm in stwA]
    foc23 = [(itm, 'stw of memory extracted task A') for itm in stwMem]

    joinedFoc = foc12 + foc23

    df = pd.DataFrame(data=joinedFoc,
                      columns=['stw', 'Original / Extracted memory'])
    # df.Zip = df.Zip.astype(str).str.zfill(5)

    # plt.title('FOC distributions')

    sns.displot(df, x="stw", hue='Original / Extracted memory', kind="kde", multiple="stack")

    #stwMem = np.append(stwMem, np.nan(len(stwA)-len(stwMem)))

    dfStwDec = pd.DataFrame({"time":np.linspace(0,len(stwA),len(stwA)),
                                "stw of task A":stwA,

                                },)

    dfStwMem = pd.DataFrame({"time": np.linspace(min(stwMem), len(stwMem), len(stwMem)),
                             "stw of memory extracted task A": stwMem,

                             }, )

    #dfStwDec.plot(kind='kde')
    sns.displot(dfStwDec, x="stw of task A", kind="kde")
    sns.displot(dfStwMem, x="stw of memory extracted task A", kind="kde")
    #plt.xlim(min(stwA),max(stwA))

    #dfStwMem.plot(kind='kde')
    plt.xlim(min(stwMem),max(stwMem))
    plt.legend()
    plt.show()

# # ENTRY POINT
if __name__ == "__main__":
    main()