# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class RNNLSTMEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional LSTM.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNLSTMEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

class BiDAFAttn(object):
    """Module for bidirectional attention flow.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states (c_i)
    and the values are the question hidden states (q_i).

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size


    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)
          keys_mask: Tensor shape (batch_size, num_keys).
            1s where there's real input, 0s where there's padding

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BiDAFAttn"):
            N = keys.shape[1]
            M = values.shape[1]
            print("N: ", N, "M: ", M)
            v = self.value_vec_size
            batchDim = keys.shape[0]
            C = keys; # (batch_size, N, v)
            Q = values; # (batch_size, M, v)

            """
            keys: c -> (batch_size, N, v)
            values: q -> (batch_size, M, v)
            """

            ### Create Similarity Matrix: S (NxM)
            C_exp = tf.expand_dims(C, axis=2) # (batch_size, N, 1, v)
            Q_exp = tf.expand_dims(Q, axis=1) # (batch_size, 1, M, v)
            s_layer3 = C_exp * Q_exp # (batch_size, N, M, v)
            s_layer2 = tf.tile(Q_exp, multiples=[1, N, 1, 1]) # (batch_size, N, M, v)
            s_layer1 = tf.tile(C_exp, multiples=[1, 1, M, 1]) # (batch_size, N, M, v)
            s_stack = tf.concat([s_layer1, s_layer2, s_layer3], axis=-1) # (batch_size, N, M, 3v)

            w_sim = tf.get_variable("w_sim", shape=[3*v,], initializer=tf.contrib.layers.xavier_initializer())
            S = tf.tensordot(s_stack,w_sim, axes=[[-1],[0]]) # (batch_size, N, M)

            print("S shape: " + str(S.shape))
            S_mask_pt1 = tf.expand_dims(values_mask, axis=1) # (batch_size, 1, M)
            S_mask_pt2 = tf.expand_dims(keys_mask, axis=-1) # (batch_size, N, 1)
            S_mask = S_mask_pt1 * S_mask_pt2 # (batch_size, N, M)
            print("S_mask shape: ", S_mask.shape)

            ### C2Q - Context To Question Attention
            # Calculate attention distribution
            _,alpha = masked_softmax(S, S_mask, 2) # (batch_size, N, M)
            print("alpha shape: ", alpha.shape)
            a = tf.matmul(alpha,Q) # (batch_size, N, v)
            print("a shape: ", a.shape)

            ### Q2C - Question to Context Attention
            S_mod = tf.cast(S, tf.float32) * tf.cast(S_mask, tf.float32) # (batch_size, N, M)
            m = tf.reduce_max(S_mod,axis=2) # (batch_size, N)
            print("m shape: ", m.shape)

            beta = tf.nn.softmax(m) # (batch_size, N)
            beta = tf.expand_dims(beta,axis=-1) # (batch_size, N, 1)
            print("beta shape: ", beta.shape)

            C_perm = tf.transpose(C, perm=[0, 2, 1]) # (batch_size, v, N)
            print("C_perm shape: ", C_perm.shape)
            c_prime = tf.matmul(C_perm, beta) # (batch_size, v, 1)
            print("c_prime shape: ", c_prime.shape)

            ## compose output
            b_layer1 = C # (batch_size, N, v)
            b_layer2 = a # (batch_size, N, v)
            b_layer3 = C * a # (batch_size, N, v)
            c_prime_perm = tf.transpose(c_prime, perm=[0, 2, 1])
            b_layer4 = C * c_prime_perm # (batch_size, N, v)

            b = tf.concat([b_layer1, b_layer2, b_layer3, b_layer4], axis=-1) # (batch_size, N, 4v)
            print("b shape: ", b.shape)

            output = b
            output = tf.nn.dropout(output, self.keep_prob)

            return None, output

class DCNAttn(object):
    """Module for dynamic coattention attention flow.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states (c_i)
    and the values are the question hidden states (q_i).

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size


    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)
          keys_mask: Tensor shape (batch_size, num_keys).
            1s where there's real input, 0s where there's padding

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("DCNAttn"):
            N = keys.shape[1]
            M = values.shape[1]
            print("N: ", N, "M: ", M)
            v = self.value_vec_size
            batchDim = keys.shape[0].value

            C = keys; # (batch_size, N, v)
            Q = values; # (batch_size, M, v)

            """
            keys: c -> (batch_size, N, v)
            values: q -> (batch_size, M, v)
            """
            #Create adjusted masks for sentinel vector
            row_to_add = tf.ones([tf.shape(keys)[0], 1], tf.int32) # (batch_size, 1)

            adj_values_mask = tf.concat([values_mask, row_to_add], axis=1) # (batch_size, M+1)
            print("adj_values_mask shape: ", adj_values_mask.shape)

            adj_keys_mask = tf.concat([keys_mask, row_to_add], axis=1) # (batch_size, N+1)
            print("adj_keys_mask shape: ", adj_keys_mask.shape)

            # Get Q' = tanh(W_cQ + B_c)

            W_c = tf.get_variable("W_c", shape=[M, M], initializer=tf.contrib.layers.xavier_initializer()) # (M, M)
            print("W_c shape: ", W_c.shape)

            B_c = tf.get_variable("B_c", shape=[M, v], initializer=tf.contrib.layers.xavier_initializer()) # (M, v)
            print("B_c shape: ", B_c.shape)

            W_c_perm = tf.transpose(W_c, perm=[1,0]) # (M, M)
            Q_perm = tf.transpose(Q, perm=[0, 2, 1]) # (batch_size, v, M)
            Q_prime_pt1 = tf.tensordot(Q_perm,W_c_perm, [[2],[1]]) # (batch_size, v, M)
            print("Q_prime_pt1 shape: ", Q_prime_pt1.shape)
            Q_prime_pt1 = tf.transpose(Q_prime_pt1, perm=[0, 2, 1]) # (batch_size, M, v)
            print("Q_prime_pt1 shape: ", Q_prime_pt1.shape)

            Q_prime_pt2 = B_c # (M, v)
            print("Q_prime_pt2 shape: ", Q_prime_pt2.shape)

            Q_prime_sum = Q_prime_pt1 + Q_prime_pt2 # (batch_size, M, v)
            print("Q_prime_sum shape: ", Q_prime_sum.shape)

            Q_prime = tf.nn.tanh(Q_prime_sum) # (batch_size, M, v)
            print("Q_prime shape: ", Q_prime.shape)

            # Create C_new and Q_new
            c_0 = tf.get_variable("c_0", shape=[1, v], initializer=tf.contrib.layers.xavier_initializer()) # (batch_size, 1, v)
            print("c_0 shape: ", c_0.shape)
            q_0 = tf.get_variable("q_0", shape=[1, v], initializer=tf.contrib.layers.xavier_initializer()) # (batch_size, 1, v)
            print("q_0 shape: ", q_0.shape)

            row_zeros = tf.zeros([tf.shape(keys)[0], 1, v], tf.float32) # (batch_size, 1, v)
            c_0 = row_zeros + c_0 # (batch_size, 1, v)
            q_0 = row_zeros + q_0 # (batch_size, 1, v)

            print("c_0 shape: ", c_0.shape)
            C_new = tf.concat([C, c_0], axis=1) # (batch_size, N+1, v)
            print("C_new shape: ", C_new.shape)
            Q_new = tf.concat([Q_prime, q_0], axis=1) # (batch_size, M+1, v)
            print("Q_new shape: ", Q_new.shape)

            # Create Affinity Matrix L (N+1 x M+1)
            Q_new_perm = tf.transpose(Q_new, perm=[0,2,1]) # (batch_size, v, M+1)
            print("Q_new_perm shape: ", Q_new_perm.shape)

            L = tf.matmul(C_new,Q_new_perm) # (batch_size, N+1, M+1)
            print("L shape: ", L.shape)

            L_mask_pt1 = tf.expand_dims(adj_values_mask, axis=1) # (batch_size, 1, M+1)
            print("L_mask_pt1 shape: ", L_mask_pt1.shape)

            L_mask_pt2 = tf.expand_dims(adj_keys_mask, axis=-1) # (batch_size, N+1, 1)
            print("L_mask_pt2 shape: ", L_mask_pt1.shape)

            L_mask = L_mask_pt1 * L_mask_pt2 # (batch_size, N+1, M+1)
            print("L_mask shape: ", L_mask.shape)

            ### C2Q - Context To Question Attention
            # Calculate attention distribution
            _,alpha = masked_softmax(L, L_mask, 2) # (batch_size, N+1, M+1)
            print("alpha shape: ", alpha.shape)

            a = tf.matmul(alpha,Q_new) # (batch_size, N+1, v)
            print("a shape: ", a.shape)

            ### Q2C - Question to Context Attention
            _,beta = masked_softmax(L, L_mask, 1) # (batch_size, N+1, M+1)
            print("beta shape: ", beta.shape)

            C_new_perm = tf.transpose(C_new, perm=[0,2,1]) # (batch_size, v, N+1)
            print("C_new_perm shape: ", C_new_perm.shape)

            b = tf.matmul(C_new_perm,beta) # (batch_size, v, M+1)
            print("b shape: ", b.shape)

            b_perm = tf.transpose(b,perm=[0, 2, 1]) # (batch_size, M+1, v)
            print("b_perm shape: ", b_perm.shape)

            s = tf.matmul(alpha,b_perm) # (batch_size, N+1,v)
            print("s shape: ", s.shape)


            s_out = s[:, :N, :] # (batch_size, N, v)
            print("s_out shape: ", s_out.shape)
            a_out = a[:, :N, :] # (batch_size, N, v)
            print("a_out shape: ", a_out.shape)

            lstm_input = tf.concat([s_out, a_out], 2) # (batch_size, N, 2v)
            print("lstm_input shape: ", lstm_input.shape)


            encoder = RNNLSTMEncoder(2*v, self.keep_prob)
            hiddens = encoder.build_graph(lstm_input, keys_mask) # (batch_size, N, 4v)
            print("hiddens shape: ", hiddens.shape)
            output = hiddens

            output = tf.nn.dropout(output, self.keep_prob)

            return None, output
            
def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
