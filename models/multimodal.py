from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

from models import alexnet
import numpy as np


class MultiModal(object):
    def __init__(self, is_training, image_tensor, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self.alexnet = alexnet.AlexNet({'data': image_tensor}, trainable=False)
        self.image_input = image_tensor

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm_cell = rnn_cell.LSTMCell(size, size, use_peepholes=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]
        outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)

        image_features = self.alexnet.layers['fc7']
        image_features_size = int(image_features.get_shape().num_elements() / batch_size)

        outputs = [tf.concat(1, [o, image_features, i]) for o, i in zip(outputs, inputs)]
        new_size = size + image_features_size + size    # The size of input and output is size

        output = tf.reshape(tf.concat(1, outputs), [-1, new_size])
        self.outputs = output
        softmax_w = tf.get_variable("softmax_w", [new_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        loss = seq2seq.sequence_loss_by_example([logits], [
            tf.reshape(self._targets, [-1])
        ], [tf.ones([batch_size * num_steps])], vocab_size)
        self.logits = logits
        self._cost = cost = tf.reduce_sum(loss) * (1.0 / batch_size)
        self._final_state = states[-1]
        self.probs = tf.nn.softmax(logits)

        if not is_training:
            self._train_op = tf.no_op()
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        tf.scalar_summary('perplexity', cost)
        tf.histogram_summary('loss', loss)
        tf.histogram_summary('probs', self.probs)

    def sample(self, sess, vocab, config, image, prime=('The ', )):
        chars = {v: k for k, v in vocab.items()}
        state = self.initial_state.eval()
        #for char in prime[:-1]:
        #    x = np.zeros((config.batch_size, config.num_steps))
        #    x[0, 0] = vocab[char]
        #    feed = {self.input_data: x, self.initial_state: state}
        #    [state] = sess.run([self.final_state], feed)

        ret = ' '.join(prime) + ' '
        char = prime[-1]
        x = np.zeros((config.batch_size, config.num_steps))
        for n in xrange(config.num_steps):
            # x = np.zeros((config.batch_size, config.num_steps))
            x[:, n] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state, self.image_input: image}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            # sample = int(np.random.choice(len(p), p=p))
            p = probs[n]
            sample = weighted_pick(p)
            #sample = np.argmax(p)
            try:
                pred = chars[sample]
            except KeyError:
                print('Could not find key {}'.format(sample))
                pred = '<EOS>'
            # pred = chars.get(sample, '<eos>')
            if pred == '<EOS>':
                ret += '. '
            else:
                ret += pred + ' '
            char = pred
        return ret

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    def load_alexnet(self, path, session):
        self.alexnet.load(path, session)


def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    p = np.random.rand(1) * s
    i = int(np.searchsorted(t, p))
    if i == len(t):
        return i - 1
    return i
