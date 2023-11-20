import pandas as pd, matplotlib.pyplot as plt, numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def calculate_lengths(final_data):

    input_lengths = final_data['question_header'].str.split().apply(len)
    output_lengths = final_data['sql'].str.split().apply(len)
    
    return input_lengths, output_lengths


def draw_histogram(lists, titles, output_path='./infographs/input_output_lengths_histogram.png', bins=60, step=10):

    plt.figure(figsize=(12, 6))

    for lst in range(len(lists)):

        plt.subplot(1, len(lists), lst+1)
        plt.hist(lists[lst], bins=bins, color='blue', alpha=0.9)
        plt.title(f'Histogram of List {titles[lst]} Sequence Lengths')  
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(0, max(lists[lst])+1, step=step))  

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

class Dataset:
    def __init__(self, data, tknizer_question, tknizer_sql, max_len_input, max_len_output):
        self.encoder_inps = data.question_header.values
        self.decoder_inps = data.sql_input.values
        self.decoder_outs = data.sql_output.values
        self.tknizer_question = tknizer_question
        self.tknizer_sql = tknizer_sql
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output

    def __getitem__(self, i):
        self.encoder_seq = self.tknizer_question.texts_to_sequences([self.encoder_inps[i]]) 
        self.decoder_inp_seq = self.tknizer_sql.texts_to_sequences([self.decoder_inps[i]])
        self.decoder_out_seq = self.tknizer_sql.texts_to_sequences([self.decoder_outs[i]])

        self.encoder_seq = pad_sequences(self.encoder_seq, maxlen=self.max_len_input, dtype='int32', padding='post')
        self.decoder_inp_seq = pad_sequences(self.decoder_inp_seq, maxlen=self.max_len_output, dtype='int32', padding='post')
        self.decoder_out_seq = pad_sequences(self.decoder_out_seq, maxlen=self.max_len_output, dtype='int32', padding='post')
        return self.encoder_seq, self.decoder_inp_seq, self.decoder_out_seq

    def __len__(self): 
        return len(self.encoder_inps)
    

class Dataloader(tf.keras.utils.Sequence):    
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.dataset.encoder_inps))

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.squeeze(np.stack(samples, axis=1), axis=0) for samples in zip(*data)]
        return tuple([[batch[0],batch[1]],batch[2]])

    def __len__(self):  
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)




