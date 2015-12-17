import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import reader
from models.complete_model import MultiModal

image_size = 227

class Configs(object):
    learning_rate = 0.25
    max_grad_norm = 3
    num_layers = 2
    num_steps = 10
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 0.8
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

def main(rnn_config, eval_config):
    print('Loading data')
    data, vocab = reader.flickr_raw_data(10000, rnn_config.num_steps, image_size)
    test_image = np.zeros((eval_config.batch_size, image_size, image_size, 3))
    test_image[0, :, :] = plt.imread('data/test_image.jpg')/255.

    rnn_config.vocab_size = len(vocab)
    eval_config.vocab_size = len(vocab)
    print('Vocab size of: {}'.format(rnn_config.vocab_size))

    with tf.Graph().as_default(), tf.Session() as sess:
        print('Creating rnn model')
        initializer = tf.uniform_unit_scaling_initializer()

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            train_image_tensor = tf.placeholder(np.float32, (rnn_config.batch_size, image_size, image_size, 3), 'input_image')
            m = MultiModal(is_training=True, config=rnn_config, image_tensor=train_image_tensor)
            m.load_alexnet('models/alexnet_weights.npy', sess)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            # mvalid = MultiModal(is_training=False, config=config, image_features=net.layers['fc8'])
            test_image_tensor = tf.placeholder(np.float32, (eval_config.batch_size, image_size, image_size, 3), 'test_input_image')
            mtest = MultiModal(is_training=False, config=eval_config, image_tensor=test_image_tensor)

        tf.initialize_all_variables().run()
        print('Importing cnn weights')

        for epoch in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
            m.assign_lr(sess, config.learning_rate * lr_decay)
            print('Running epoch {} with learning rate {}'.format(epoch, config.learning_rate * lr_decay))
            loss = run_epoch(sess, data, m.train_op, config, m, mtest, test_image, vocab, verbose=True)

            print('Sampling rnn')
            s = sample(sess, vocab, eval_config, mtest, test_image, ['<BOS>'])
            print(s)
            print('Loss : {}'.format(loss))


def run_epoch(session, data, eval_op, config, train_model, test_model, test_image, vocab, verbose=False):
    """Runs the model on the given data."""

    epoch_size = len(data['dataset']) / config.batch_size  # ((len(data) // batch_size) - 1) // num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = train_model.initial_state.eval()

    for step, (x, im, y) in enumerate(reader.flickr_iterator(data, config.batch_size, config.num_steps, (image_size, image_size, 3))):
        cost, _, _ = session.run([train_model.cost, train_model.logits, eval_op],
                                 {train_model.input_data: x, train_model.targets: y,
                                  train_model.initial_state: state, train_model.image_input: im})
        costs += cost
        iters += config.num_steps

        if verbose and step % 10 == 0 and step != 0:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * config.batch_size / (time.time() - start_time)))
        if verbose and step % 30 == 0 and step != 0:
            print(sample(session, vocab, eval_config, test_model, test_image, prime=['<BOS>']))
    return np.exp(costs / iters)


def sample(sess, vocab, config, model, image, prime=('The ',)):
    chars = {v: k for k, v in vocab.items()}
    state = model.initial_state.eval()
    for char in prime[:-1]:
        x = np.zeros((config.batch_size, config.num_steps))
        x[0, 0] = vocab[char]
        feed = {model.input_data: x, model.initial_state: state}
        [state] = sess.run([model.final_state], feed)

    def weighted_pick(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        p = np.random.rand(1) * s
        i = int(np.searchsorted(t, p))
        if i == len(t):
            return i - 1
        return i

    ret = ' '.join(prime) + ' '
    char = prime[-1]
    x = np.zeros((config.batch_size, config.num_steps))
    for n in xrange(config.num_steps):
        #x = np.zeros((config.batch_size, config.num_steps))
        x[:, n] = vocab[char]
        feed = {model.input_data: x, model.initial_state: state, model.image_input: image}
        [probs, state] = sess.run([model.probs, model.final_state], feed)
        # sample = int(np.random.choice(len(p), p=p))
        p = probs[n]
        sample = weighted_pick(p)
        try:
            pred = chars[sample]
        except KeyError:
            print('Could not find key {}'.format(sample))
            pred = '<eos>'
        # pred = chars.get(sample, '<eos>')
        if pred == '<fill>':
            ret += '? '
        elif pred == '<eos>':
            ret += '. '
            break
        else:
            ret += pred + ' '
        char = pred
    return ret


if __name__ == "__main__":
    config = Configs()
    eval_config = Configs()
    eval_config.batch_size = 1
    main(config, eval_config)
