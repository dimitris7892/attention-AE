import numpy as np
import csv
import io
import pandas as pd
from numpy import array

class extractSequencialTasks:

    def __init__(self):

        pass


    def exctractTasks(self, numOfTasks, n_steps, lenS, start = 2000, ):

        trData = self.getData()
        offset = 5000 #5 days

        raw_seq = trData  # [:4000]

        # split into samples
        tasksMem = []

        for i in range(0, numOfTasks):

            seqLSTM = raw_seq[start:lenS + start]
            seqLSTMmem = self.split_sequence(seqLSTM, n_steps)
            seqLSTMmem = seqLSTMmem.reshape(-1, n_steps, 7)


            tasksMem.append(seqLSTMmem)

            start += (lenS + offset)

        return tasksMem


    def getData(self, ):

        data = pd.read_csv('./data/DANAOS/EXPRESS ATHENS/mappedDataNew.csv').values
        draft = data[:, 8].reshape(-1, 1)
        wa = data[:, 10]
        ws = data[:, 11]
        stw = data[:, 12]
        swh = data[:, 22]
        bearing = data[:, 1]
        lat = data[:, 26]
        lon = data[:, 27]
        foc = data[:, 15]

        trData = np.array(np.append(draft, np.asmatrix([wa, ws, stw, swh,bearing, foc]).T, axis=1)).astype(float)

        trData = np.nan_to_num(trData)
        trData = np.array([k for k in trData if
                           str(k[0]) != 'nan' and float(k[2]) > 0 and float(k[4]) > 0 and (float(k[3]) >= 8) and float(
                               k[6]) > 0]).astype(float)

        return trData

    def split_sequence(self, sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            # seq_x, seq_y = sequence[i:end_ix][:, 0:sequence.shape[1] - 1], sequence[end_ix - 1][sequence.shape[1] - 1]
            seq_ = sequence[i:end_ix][:, :]
            X.append(seq_)
            # y.append(seq_y)
        return array(X)

        # define input sequence


def main():

    exttsk = extractSequencialTasks()

    tasks = exttsk.exctractTasks(3, 20, 1000, 2400)

    x=0

# # ENTRY POINT
if __name__ == "__main__":
    main()