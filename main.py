import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import reader
from models.multimodal import MultiModal

flags = tf.flags
flags.DEFINE_integer(
    'N', -1, 'The number of samples of the Flickr8k dataset. -1 for all the samples.')
logging = tf.logging

image_size = 227


class Configs(object):
    learning_rate = 0.00001    #0.25
    max_grad_norm = 5
    num_layers = 2
    num_steps = 10
    hidden_size = 400
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 0.7
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def main(rnn_config, eval_config):
    print('Loading data')
    train_data, valid_data, vocab = reader.flickr_raw_data(flags.FLAGS.N, rnn_config.num_steps,
                                                           image_size)
    test_image = np.zeros((eval_config.batch_size, image_size, image_size, 3))
    test_image[0, :, :] = plt.imread('data/test_image.jpg') / 255.

    rnn_config.vocab_size = len(vocab)
    eval_config.vocab_size = len(vocab)
    print('Vocab size of: {}'.format(rnn_config.vocab_size))

    with tf.Graph().as_default(), tf.Session() as sess:
        print('Creating rnn model')
        initializer = tf.uniform_unit_scaling_initializer()

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            train_image_tensor = tf.placeholder(np.float32, (rnn_config.batch_size, image_size,
                                                             image_size, 3), 'input_image')
            m = MultiModal(is_training=True, config=rnn_config, image_tensor=train_image_tensor)
            m.load_alexnet('models/alexnet_weights.npy', sess)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = MultiModal(is_training=False,
                                config=rnn_config,
                                image_tensor=train_image_tensor)
            test_image_tensor = tf.placeholder(np.float32, (eval_config.batch_size, image_size,
                                                            image_size, 3), 'test_input_image')
            mtest = MultiModal(is_training=False,
                               config=eval_config,
                               image_tensor=test_image_tensor)

        merged = tf.merge_all_summaries()
        tf.initialize_all_variables().run()
        print('Importing cnn weights')

        writer = tf.train.SummaryWriter("logs/")

        for epoch in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
            m.assign_lr(sess, config.learning_rate * lr_decay)
            print('Running epoch {} with learning rate {}'.format(epoch,
                                                                  config.learning_rate * lr_decay))
            train_loss = run_epoch(sess, train_data, m.train_op, config, m, mtest, test_image,
                                   vocab,
                                   summary=merged,
                                   summary_writer=writer,
                                   verbose=True)
            print('Training Loss : {:.3f}'.format(train_loss))
            valid_loss = run_epoch(sess, valid_data, mvalid.train_op, config, mvalid, None,
                                   test_image, vocab,
                                   verbose=False)
            print('Valid Loss: {:.3f}'.format(valid_loss))

            print('Sampling rnn')
            s = mtest.sample(sess, vocab, eval_config, test_image, ['<BOS>'])
            print(s)


def run_epoch(session, data, eval_op, config, train_model, test_model, test_image, vocab,
              summary_writer=None,
              summary=None,
              verbose=False):
    """Runs the model on the given data."""
    epoch_size = len(data['dataset']) / config.batch_size
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = train_model.initial_state.eval()

    image_shape = (image_size, image_size, 3)
    eval_ops = [train_model.cost, train_model.logits, eval_op]
    if summary is not None:
        eval_ops.append(summary)
    flickr_iter = reader.flickr_iterator(data, config.batch_size, config.num_steps, image_shape)
    for step, (x, im, y) in enumerate(flickr_iter):
        feed_dict = {
            train_model.input_data: x,
            train_model.targets: y,
            train_model.initial_state: state,
            train_model.image_input: im
        }
        results = session.run(eval_ops, feed_dict)
        cost = results[0]
        if summary is not None:
            summary_result = results[3]

        costs += cost
        iters += config.num_steps
        if verbose:
            completed = step * 1.0 / epoch_size
            perplexity = np.exp(costs / iters)
            wps = iters * config.batch_size / (time.time() - start_time)
            print("{0} perplexity: {1:.3f} speed: {2:.0f} wps".format(completed, perplexity, wps))
            summary_writer.add_summary(summary_result)
        if verbose and step % 10 == 0 and step != 0:
            print(test_model.sample(session, vocab, eval_config, test_image, prime=['<BOS>']))

    return np.exp(costs / iters)


if __name__ == "__main__":
    config = Configs()
    eval_config = Configs()
    eval_config.batch_size = 1
    main(config, eval_config)
