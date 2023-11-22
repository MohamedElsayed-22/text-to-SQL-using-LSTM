
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense


class Encoder(tf.keras.Model):
    def __init__(self, inp_vocab_size, embedding_size, lstm_size, input_length):
        super().__init__()
        self.vocab_size = inp_vocab_size
        self.embedding_dim = embedding_size   
        self.input_length = input_length
        self.enc_units = lstm_size

    def build(self, input_shape):  
        self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                                   mask_zero=True, name="embedding_layer_encoder")
        self.lstm = LSTM(self.enc_units, return_state=True, return_sequences=True, name="Encoder_LSTM")

    def call(self, input_sequence, states, training=True):

        input_embedd = self.embedding(input_sequence)
        self.lstm_output, self.lstm_state_h, self.lstm_state_c = self.lstm(input_embedd, initial_state=states)
        return self.lstm_output, self.lstm_state_h, self.lstm_state_c

    def initialize_states(self, batch_size):

        state_h = tf.zeros((batch_size, self.enc_units))
        state_c = tf.zeros((batch_size, self.enc_units))
        return state_h, state_c
    

class Attention(tf.keras.layers.Layer):
  def __init__(self, scoring_function, att_units):
    super().__init__()
    self.scoring_function = scoring_function
    self.att_units = att_units
    
    if self.scoring_function == 'dot':
      self.dot = tf.keras.layers.Dot(axes=(1, 2))

    if self.scoring_function == 'general':
      self.dot = tf.keras.layers.Dot(axes=2)
      self.W = tf.keras.layers.Dense(att_units)
    
    elif self.scoring_function == 'concat':
      self.W1 = tf.keras.layers.Dense(att_units)
      self.W2 = tf.keras.layers.Dense(att_units)
      self.V = tf.keras.layers.Dense(1)
  
  def call(self, decoder_hidden_state, encoder_output):

    if self.scoring_function == 'dot':
        score = self.dot([decoder_hidden_state, encoder_output])
        attention_weights = tf.nn.softmax(score)
        context_vector = tf.matmul(attention_weights, encoder_output)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        attention_weights = tf.expand_dims(attention_weights, axis=2)
        return context_vector, attention_weights

    elif self.scoring_function == 'general':
        decoder_hidden_state = tf.expand_dims(decoder_hidden_state, 1)
        enc_dense = self.W(encoder_output)
        similarity = tf.matmul(decoder_hidden_state, enc_dense, transpose_b=True)
        attention_weights = tf.nn.softmax(similarity, axis=2)
        context_vector = tf.matmul(attention_weights, encoder_output)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        attention_weights = tf.transpose(attention_weights, perm=[0, 2, 1])
        return context_vector, attention_weights
    
    elif self.scoring_function == 'concat':
        decoder_hidden_state = tf.expand_dims(decoder_hidden_state, 1)
        score = self.V(tf.nn.tanh(self.W1(decoder_hidden_state)) + self.W2(encoder_output))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    

class One_Step_Decoder(tf.keras.Model):
    def __init__(self, tar_vocab_size, embedding_dim, input_length, dec_units, score_fun, att_units):
        super().__init__()
        self.vocab_size = tar_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length)
        self.lstm = tf.keras.layers.LSTM(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.attention = Attention(self.score_fun, self.att_units)
        self.fc = Dense(self.vocab_size)

    def call(self, input_to_decoder, encoder_output, state_h, state_c):
        x = self.embedding(input_to_decoder)
        context_vector, attention_weights = self.attention(state_h, encoder_output)

        # Print the shape of x for debugging
        print("Shape of x:", x.shape)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        init_states = [state_h, state_c]
        output, state_h, state_c = self.lstm(x, initial_state=init_states)
        output = self.fc(output)
        return tf.reduce_sum(output, 1), state_h, state_c, attention_weights, context_vector

class Decoder(tf.keras.Model):
    def __init__(self, out_vocab_size, embedding_dim, input_length, dec_units, score_fun, att_units):
        super().__init__()
        self.vocab_size = out_vocab_size + 1
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.input_length = input_length
        self.att_units = att_units
        self.score_fun = score_fun
        # Using embedding_matrix and not training the embedding layer
        self.one_step_decoder = One_Step_Decoder(self.vocab_size, self.embedding_dim, self.input_length, self.dec_units, self.score_fun, self.att_units)

    def call(self, input_to_decoder, encoder_output, decoder_hidden_state, decoder_cell_state):
        output_tensor_array = tf.TensorArray(tf.float32, size=tf.shape(input_to_decoder)[1])

        for timestamp in range(tf.shape(input_to_decoder)[1]):
            output, state_h, state_c, attention_weights, context_vector = self.one_step_decoder(
                input_to_decoder[:, timestamp:timestamp + 1], encoder_output, decoder_hidden_state, decoder_cell_state)
            output_tensor_array = output_tensor_array.write(timestamp, output)

        output_tensor_array = tf.transpose(output_tensor_array.stack(), [1, 0, 2])
        return output_tensor_array
    

class Encoder_decoder(tf.keras.Model):
    def __init__(self, inp_vocab_size, embedding_size, input_length, lstm_size, out_vocab_size, batch_size, score_fun, att_units):
        super().__init__()
        self.vocab_size = inp_vocab_size
        self.embedding_size = embedding_size
        self.input_length = input_length
        self.enc_units = lstm_size
        self.out_vocab_size = out_vocab_size
        self.score_fun = score_fun
        self.att_units = att_units
        self.batch_size = batch_size
        
        self.encoder = Encoder(inp_vocab_size=self.vocab_size, embedding_size=self.embedding_size, input_length=61, lstm_size=self.enc_units)
        self.decoder = Decoder(out_vocab_size=self.out_vocab_size, embedding_dim=self.embedding_size, dec_units=self.enc_units, input_length=36, score_fun=self.score_fun, att_units=self.att_units)
        
    def call(self, data, *params):

        input, output = data[0], data[1]
        initial_state = self.encoder.initialize_states(self.batch_size)
        encoder_output, encoder_h, encoder_c = self.encoder(input, initial_state)
        decoder_output = self.decoder(output, encoder_output, encoder_h, encoder_c)
        return decoder_output




