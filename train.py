import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
import os

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.preprocessing.text import Tokenizer
from src.preprocessing import load_data, load_table_info, merge_data, convert_to_sql_strings, \
    concatenate_headers_and_questions, format_inputs, filter_data_by_sequence_length
from utilities.utils import calculate_lengths, draw_histogram, custom_lossfunction, Dataloader, Dataset
from src.model import Encoder_decoder

# Data Preprocessing
train_data = load_data(r"P:/text-to-sql-project/text-to-SQL-using-LSTM/data/train.jsonl")
train_table_info = load_table_info(r"P:/text-to-sql-project/text-to-SQL-using-LSTM/data/train.tables.jsonl")
merged_data = merge_data(train_data, train_table_info)

merged_data_string = convert_to_sql_strings(merged_data, 'merged_data.csv')
question_header = concatenate_headers_and_questions(merged_data)

format_inputs(question_header, merged_data_string, 'final_data.csv')

# Understanding data insights
processed_data = pd.read_csv(r".\final_data.csv")
input_lengths, output_lengths = calculate_lengths(processed_data)

draw_histogram([input_lengths, output_lengths], ["Input Lengths", "Output Lengths"])

# The lengths limit for the imput and output sequence is based on 90th percentile
input_length_lim = int(np.percentile(input_lengths, 90))
output_length_lim = int(np.percentile(output_lengths, 90))

print(input_length_lim, output_length_lim)

filtered_processed_data = filter_data_by_sequence_length(input_lengths, output_lengths, input_length_lim,
                                                         output_length_lim, processed_data)

filtered_processed_data.to_csv("questions_to_sql.csv")

filtered_processed_data['sql_input'] = '<start> ' + filtered_processed_data['sql'].astype(str)
filtered_processed_data['sql_output'] = filtered_processed_data['sql'].astype(str) + ' <end>'

# print(filtered_processed_data)
tknizer_question = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tknizer_question.fit_on_texts(filtered_processed_data['question_header'].values)
tknizer_sql = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tknizer_sql.fit_on_texts(filtered_processed_data['sql_input'].values)

with open('tokenizer_question.pickle', 'wb') as handle:
  pickle. dump(tknizer_question, handle, protocol=pickle. HIGHEST_PROTOCOL)
with open('tokenizer_sql.pickle', 'wb') as handle:
  pickle. dump(tknizer_sql, handle, protocol=pickle. HIGHEST_PROTOCOL)

with open('tokenizer_question.pickle', 'rb') as handle:
  tknizer_question=pickle.load(handle)
with open('tokenizer_sql.pickle', 'rb') as handle:
  tknizer_sql=pickle.load(handle) 

print(filtered_processed_data.shape)

# Maintaining a better relationship between words via using pre-trained word embeddings (Glove)

embeddings_index = dict()

with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embedding_vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding_vector

vocab_sql_size = len(tknizer_sql.word_index.keys())
vocab_word_size = len(tknizer_question.word_index.keys())

embedding_matrix = np.zeros((vocab_sql_size + 1, 100))

for word, index in tknizer_sql.word_index.items():
    pretrained_embedding = embeddings_index.get(word)

    if pretrained_embedding is not None:
        embedding_matrix[index] = pretrained_embedding

print(embedding_matrix.shape)

train_dataset = Dataset(filtered_processed_data, tknizer_question, tknizer_sql, input_length_lim+1, output_length_lim+1)
train_dataloader = Dataloader(train_dataset, batch_size=64)

print(train_dataloader[0][0][0].shape, train_dataloader[0][0][1].shape, train_dataloader[0][1].shape)

# Loading validation data
val_dataset = load_data(r"P:\text-to-sql-project\text-to-SQL-using-LSTM\data\dev.jsonl")
val_table_info = load_table_info(r"P:\text-to-sql-project\text-to-SQL-using-LSTM\data\dev.tables.jsonl")
print(val_table_info.shape)

val_merged_data = merge_data(val_dataset, val_table_info)
print(val_merged_data.shape)

val_merged_data_string = convert_to_sql_strings(val_merged_data, 'val_merged_data.csv')
val_question_header = concatenate_headers_and_questions(val_merged_data)
format_inputs(val_question_header, val_merged_data_string, 'val_final_data.csv')



val_processed_data = pd.read_csv(r".\val_final_data.csv")
input_lengths, output_lengths = calculate_lengths(val_processed_data)


val_filtered_processed_data = filter_data_by_sequence_length(input_lengths, output_lengths, input_length_lim,
                                                         output_length_lim, val_processed_data)

val_filtered_processed_data.to_csv("val_questions_to_sql.csv")

val_filtered_processed_data['sql_input'] = '<start> ' + val_filtered_processed_data['sql'].astype(str)
val_filtered_processed_data['sql_output'] = val_filtered_processed_data['sql'].astype(str) + ' <end>'

print(val_filtered_processed_data.shape)

val_dataset = Dataset(val_filtered_processed_data, tknizer_question, tknizer_sql, input_length_lim+1, output_length_lim+1)
val_dataloader = Dataloader(val_dataset, batch_size=64)


# Training the model
filepath_last = "model_last"
filepath_best = "model_best.hdf5"  
latest_checkpoint = tf.train.latest_checkpoint(os.path.dirname(filepath_last))

if latest_checkpoint:
    model = Encoder_decoder(input_length=36, out_vocab_size=vocab_sql_size, inp_vocab_size=vocab_word_size, 
                            embedding_size=50, lstm_size=128, batch_size=64, score_fun='dot', att_units=64)
    model.load_weights(latest_checkpoint)
    print(f"Model loaded from checkpoint: {latest_checkpoint}\n \n \n")
else:
    model = Encoder_decoder(input_length=36, out_vocab_size=vocab_sql_size, inp_vocab_size=vocab_word_size, 
                            embedding_size=50, lstm_size=128, batch_size=64, score_fun='dot', att_units=64)
    print("Creating a new model. \n \n \n")

checkpoint_last = ModelCheckpoint(filepath=filepath_last, 
                                   monitor='val_loss', 
                                   verbose=1, 
                                   save_best_only=False,  
                                   mode='auto',
                                   save_weights_only=True)

checkpoint_best = ModelCheckpoint(filepath=filepath_best, 
                                   monitor='val_loss', 
                                   verbose=1, 
                                   save_best_only=True,  
                                   mode='auto',
                                   save_weights_only=True)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0.0, 
                          patience=5, 
                          verbose=1, 
                          mode='auto')

optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=optimizer, loss=custom_lossfunction, metrics=['accuracy'])

train_steps = filtered_processed_data.shape[0] // 64
valid_steps = val_filtered_processed_data.shape[0] // 64

model.fit(train_dataloader, 
          steps_per_epoch=train_steps, 
          batch_size=64,
          epochs=50, 
          validation_data=val_dataloader, 
          validation_steps=valid_steps,
          callbacks=[checkpoint_last, checkpoint_best, tensorboard_callback, earlystop])

model.summary()