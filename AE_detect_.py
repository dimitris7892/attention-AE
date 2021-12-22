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
lenS = 1000
start = 13000
tasksNew = extasks.exctractTasks(3, 20, 1000)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a time series."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



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


class AE_detect():

    def __init__(self):

        '''initialize window indentification AE '''
        encoderA, decoderA = self.windwowIdentificationModel()
        # print(encoderA.summary())
        # print(decoderA.summary())
        self.windAEa = LSTM_AE_IW(encoderA, decoderA)
        self.windAEa.compile(optimizer=keras.optimizers.Adam())

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
        prob = count/len(Y)
        en = np.sum((-1)*prob*np.log2(prob))
        return en

    def pointInRect(self, point,rect):
        x1, y1 = rect.xy
        w, h = rect.get_bbox().width , rect.get_bbox().height
        x2, y2 = x1+w, y1+h
        x, y = point
        if (x1 < x and x < x2):
            if (y1 < y and y < y2):
                return True
        return False

    def boxCountingMethod(self, sequence,ylabel,width = None, height = None):
        width = 100
        height = 1
        ax = plt.gca()
        colors =['blue','red']
        colorSeq = ['green', 'black']
        labels = ['TASKA','TASKB']
        counters = []
        rectsDicts = []
        for i in  range(0,len(sequence)):

            rectsDict = {'rects':[]}
            for k in range(0,lenS,width): ##columns
                for n in range(0,int(max(sequence[i]))+1): #rows
                    rect = patches.Rectangle((k, n), width, height, linewidth=1, edgecolor=colors[i], facecolor='none')
                    item = {}
                    item['visited'] = False
                    item['rect'] = rect
                    rectsDict['rects'].append(item)

                    #ax.add_patch(rect)
            plt.plot(sequence[i],c=colorSeq[i],label=labels[i])
            # Add the patch to the Axes
            rectsVisited = []
            #plt.show()
            counter = 0
            for n in range(0,len(sequence[i])):
                for rect in rectsDict['rects']:
                    if pointInRect((n,sequence[i][n]),rect['rect']) and rect['visited']==False:
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
        plt.title('Fractal dimension with box counting method for task A: '+ str(counters[0]) +" for task B: "+str(counters[1])+ " with τ = "+str(width)+" and α = " + str(height))
        plt.legend()


        dictA  = rectsDicts[0]
        dictB = rectsDicts[1]
        filteredSeqAs = []
        indices = []


        k=0
        xis = []
        for i in range(0,len(dictB)): ##iterate rects of B
                filteredSeqA = []
                indices = []
                xi = int(dictB[i].xy[0])
                xi_hat = int(dictB[i].get_bbox().width)
                xis.append(xi)
                if xi != xis[i-1] or i ==0:
                    for n in range(xi,xi+xi_hat):
                        if pointInRect((n, sequence[0][n]), dictB[i]) == False :
                             #ax.add_patch(rect['rect'])
                             flag = True
                             filteredSeqA.append(sequence[0][n])

                             indices.append(n)
                #ax.add_patch(plt.plot(filteredSeqA))
                filteredSeqAs.append(indices)

        print(len(indices))

        #plt.show()
        return tasksA[0][list(np.concatenate(filteredSeqAs).astype(int))]
        #print("Fractal dimension: "+str(counter))

    #Joint Entropy
    def jEntropy(self, Y,X):
        """
        H(Y;X)
        Reference: https://en.wikipedia.org/wiki/Joint_entropy
        """
        YX = np.c_[Y,X]
        return entropy(YX)

    #Conditional Entropy
    def cEntropy(self, Y, X):
        """
        conditional entropy = Joint Entropy - Entropy of X
        H(Y|X) = H(Y;X) - H(X)
        Reference: https://en.wikipedia.org/wiki/Conditional_entropy
        """
        return jEntropy(Y, X) - entropy(X)


    def custom_loss(self, y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)\
        +tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)

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

    def VAE_getMemoryWindowBetweenTaskA_TaskB(self, ):

        seqLSTMBtr = seqLSTMB.transpose()#.reshape(n_steps,9,1000)
        seqLSTMAtr = seqLSTMA.transpose()#.reshape(n_steps,9,1000)

        encoder , decoder  = VAE_windwowIdentificationModel()
        windAE = VAE(encoder, decoder)
        windAE.compile(optimizer=keras.optimizers.Adam())


        windAE.fit(seqLSTMAtr,seqLSTMBtr,epochs=200,)

        #windAE_.fit(seqLSTMAtr,seqLSTMAtr,epochs=100,)

        #windEncoder = windAE.layers[:-2]
        encodedTimeSeries = np.round(windAE.encoder.predict(seqLSTMAtr),2)
        encodedTimeSeriesReshaped = encodedTimeSeries.reshape(1000,9)

        selectiveMemory_ExtractedOfTaskA = seqLSTMA[[k for k in range(0,1000) if encodedTimeSeriesReshaped[:,k].all()>0]]

        return selectiveMemory_ExtractedOfTaskA

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
            #batch = (seq_a + (weight))
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

    def trainAE_withRollingWIndowOfPrevTask(self, seqLSTMA, seqLSTMB):
        memWindow = [5, 10, 15, 20]

        memoryEncoderLayers = []
        mem = 0
        decSeqWithDiffWindow = []

        for mem in memWindow:

            encoder, decoder = baselineModel()
            lstm_autoencoderMem = LSTM_AE(encoder, decoder,)
            lstm_autoencoderMem.compile(optimizer=keras.optimizers.Adam())

            memoryOfPrevTask = seqLSTMA[-mem:]
            addMemoryToNewTask = np.append(memoryOfPrevTask, seqLSTMB, axis=0)

            lstm_autoencoderMem.fit(addMemoryToNewTask, addMemoryToNewTask, epochs=30)

            x_test_encoded = lstm_autoencoderMem.encoder.predict(seqLSTMB)
            decSeqMem = lstm_autoencoderMem.decoder.predict(x_test_encoded)

            decSeqWithDiffWindow.append(decSeqMem)
            # score , acc = lstm_autoencoder.evaluate(tasks[i], tasks[i], epochs=10)

            scoreAE_mem = np.linalg.norm(seqLSTMB - decSeqMem)
            print("AE Score with mem window: " + str(mem) + "  " + str(scoreAE_mem))

            #lstm_autoencoder.fit(tasks[i], tasks[i], epochs=10)

        seqLSTMB = seqLSTMB.reshape(seqLSTMB.shape[0],9)
        stwOr = seqLSTMB[:,3]

        stwDec = decSeqInit
        stwDec = stwDec.reshape(stwDec.shape[0],9)
        stwDec = stwDec[:,3]

        #dfStwOr = pd.DataFrame({"time":np.linspace(0,len(stwOr),len(stwOr)),"stwOr":stwOr})
        '''stwDec5 = decSeqWithDiffWindow[0]
        stwDec5 = stwDec5.reshape(1000,9)
        stwDec5 =stwDec5[:,3]
        
        stwDec10 = decSeqWithDiffWindow[1]
        stwDec10 = stwDec10.reshape(1000,9)
        stwDec10 =stwDec10[:,3]
        
        
        stwDec15 = decSeqWithDiffWindow[2]
        stwDec15 = stwDec15.reshape(1000,9)
        stwDec15 =stwDec15[:,3]
        
        
        stwDec20 = decSeqWithDiffWindow[3]
        stwDec20 = stwDec20.reshape(1000,9)
        stwDec20 = stwDec20[:,3]
        
        stwDec25 = decSeqWithDiffWindow[4]
        stwDec25 = stwDec25.reshape(1000,9)
        stwDec25 = stwDec25[:,3]'''

        stwDec30 = decSeqWithDiffWindow[0]
        stwDec30 = stwDec30.reshape(stwDec30.shape[0],9)
        stwDec30 = stwDec30[:,3]

        dfStwDec = pd.DataFrame({"time":np.linspace(0,len(stwDec),len(stwDec)),"stw Original":stwOr,
                                "stw Decoded without memory":stwDec,
                                 "stw Decoded with 30min memory of previous task":stwDec30,
                                },)

        ''' "stw Decoded with memory 10min":stwDec10,
                                 "stw Decoded with memory 15min":stwDec15,
                                 "stw Decoded with memory 20min":stwDec20,
                                 "stw Decoded with memory 25min":stwDec25,
                                 "stw Decoded with memory 30min":stwDec30,'''
        dfStwDec.plot(kind='kde')
        plt.xlim(min(stwDec),max(stwDec))
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

        model.add(keras.layers.Dense(10, ))

        model.add(keras.layers.Dense(1))

        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.Adam())  # experimental_run_tf_function=False )
        # print(model.summary())

        return model

    def trainingBaselinesForFOCestimation(self, seqLSTMmem, memoryA, currInd , memories):

        print("Memory of taskA: "+str(len(memoryA)))

        print("LSTM training  of tasks sequentially  . . .")
        lrBase = self.baselineLearner()
        maes_sum = []
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
            for k in range(0, len(x)):
                y_hat = lrBase.predict(x[k].reshape(1,n_steps,6))[0][0]
                err = abs(y_hat - y[k])
                maes_.append(err)


            scoreAWM = 0
            scoreBWM = 0

            maes_sum.append(maes_)

            print("Score for task "+str(i)+" without memory:" + str(np.mean(maes_)))
            print("R2 Score for A without memory:" + str(scoreAWM))
            print("R2 Score for B without memory:" + str(scoreBWM) + "\n")


        ##############################################################
        print("LSTM training of tasks with extracted memory . . ")

        #for i in range(0,len(seqLSTMmem)):

        #currTask = seqLSTMmem[i]
        lrAB = self.baselineLearner()

        batchesXmem = []
        batchesYmem = []
        for i in range(0,len(memoryA)):

            batchesXmem.append(memoryA[i][:,:6])
            batchesYmem.append(memoryA[i][n_steps-1,6])

        batchesXmem = np.array(batchesXmem)
        batchesYmem = np.array(batchesYmem)


        memory = memories[1]
        batchesXmem1 = []
        batchesYmem1 = []
        for i in range(0, len(memory)):
            batchesXmem1.append(memory[i][:, :6])
            batchesYmem1.append(memory[i][n_steps - 1, 6])

        batchesXmem1 = np.array(batchesXmem1)
        batchesYmem1 = np.array(batchesYmem1)


        batchesXa = []
        batchesYa = []
        prevTask = seqLSTMmem[currInd - 1]
        for i in range(0, len(prevTask)):
            batchesXa.append(prevTask[i][:, :6])
            batchesYa.append(prevTask[i][n_steps - 1, 6])

        batchesXa = np.array(batchesXa)
        batchesYa = np.array(batchesYa)

        batchesXb = []
        batchesYb = []
        currTask = seqLSTMmem[currInd]

        for i in range(0, len(currTask)):
            batchesXb.append(currTask[i][:, :6])
            batchesYb.append(currTask[i][n_steps - 1, 6])

        batchesXb = np.array(batchesXb)
        batchesYb = np.array(batchesYb)


        batchesX0 = []
        batchesY0 = []
        firstTask = seqLSTMmem[0]
        for i in range(0, len(firstTask)):
            batchesX0.append(firstTask[i][:, :6])
            batchesY0.append(firstTask[i][n_steps - 1, 6])

        batchesX0 = np.array(batchesX0)
        batchesY0 = np.array(batchesY0)

        lstmX = np.append(batchesX0, batchesXmem,axis=0)
        lstmX = np.append(lstmX, batchesXa,axis=0)
        lstmX = np.append(lstmX, batchesXb, axis=0)

        lstmY = np.append(batchesY0, batchesYmem)
        lstmY = np.append(lstmY, batchesYa)
        lstmY = np.append(lstmY, batchesYb)

        #tasksMemX = [batchesX0, batchesXmem, batchesXa, batchesXmem1, batchesXb]
        #tasksMemY = [batchesY0, batchesYmem, batchesYa, batchesYmem1, batchesYb]

        #for lstmX, lstmY in zip(tasksMemX, tasksMemY):
            #lrAB.fit(lstmX, lstmY, epochs=20)
        #lrAB.layers[0].set_weights([weights, np.array([0] * (genModelKnots - 1))])
        lrAB.fit(lstmX, lstmY, epochs=20)

        recosntructA = []
        for i in range(0,len(memoryA)):

             if i ==0:
                recon = memoryA[i][0]
             else:
                 recon = memoryA[i][0]
             recosntructA.append(recon)


        maesprev_s = []
        maescurr_s = []
        maes0_s = []
        y_hatsprev = []
        y_hatscurr = []
        y_hats0 = []
        for i in range(0,len(batchesXb)):
            y_hatcurr = lrAB.predict(batchesXb[i].reshape(1,n_steps,6))[0][0]
            y_hatscurr.append(y_hatcurr)
            err = abs(y_hatcurr - batchesYb[i])
            maescurr_s.append(err)

        for i in range(0,len(batchesXa)):
            y_hatprev = lrAB.predict(batchesXa[i].reshape(1,n_steps,6))[0][0]
            y_hatsprev.append(y_hatprev)
            err = abs(y_hatprev - batchesYa[i])
            maesprev_s.append(err)

        for i in range(0,len(batchesX0)):
            y_hat0 = lrAB.predict(batchesX0[i].reshape(1,n_steps,6))[0][0]
            y_hats0.append(y_hat0)
            err = abs(y_hat0 - batchesY0[i])
            maes0_s.append(err)

        maes_s = [maes0_s, maesprev_s, maescurr_s]

        for ind in range(0,len(seqLSTMmem)):
            print("Score for task "+str(ind)+" with selective memory:" + str(np.mean(maes_s[ind])))

        #print("Score for B with selective memory:" + str(np.mean(maesb_s)))
        #print("R2 Score for A with selective memory:" + str(scoreASel))
        #print("R2 Score for B with selective memory:" + str(scoreBSel))
        #print("Score difference: " +str(abs(np.mean(maesb_s) - np.mean(maesa_s)))+"\n")


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


        maesa_f = []
        maesb_f = []
        maes0_f = []
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

        scoreAFull = 0
        scoreBFull = 0

        maes_ = [maes0_f, maesa_f, maesb_f]

        for ind in range(0, len(seqLSTMmem)):
            print("Score for task "+str(ind)+" with full memory:" + str(np.mean(maes_[ind])))




        df = pd.DataFrame.from_dict({"ScoreA without memory": np.mean(maes_sum[0]) ,
                           "ScoreB without memory": np.mean(maes_sum[1]),
                            "ScoreC without memory": np.mean(maes_sum[2]),

                           "ScoreA with selective memory": np.mean(maes_s[0]),
                           "ScoreB with selective memory": np.mean(maes_s[1]),
                            "ScoreC with selective memory": np.mean(maes_s[2]),

                           "ScoreA with full memory": np.mean(maes_[0]),
                           "ScoreB with full memory": np.mean(maes_[1]),
                        "ScoreC with full memory": np.mean(maes_[2]),

                           },orient='index')
        #df.to_csv('./AE_files/'+alg+'.csv')
        return df


    def runAlgorithmsforEvaluation(self,  alg, seqLen):
        memories = None
        if alg!='RND':
            dfs = []
            memories = []
            for k in range(0, len(tasksNew)):

                if k==0 :
                    memory, memoryB = self.getMemoryWindowBetweenTaskA_TaskB(seqLen, tasksNew[k], tasksNew[k+1])
                elif k < len(tasksNew) - 1:
                    memory, memoryB = self.getMemoryWindowBetweenTaskA_TaskB(seqLen, tasksNew[k], tasksNew[k + 1])

                memories.append(memory)

            df = self.trainingBaselinesForFOCestimation(tasksNew, memory, 2, memories)
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