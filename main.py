import time

import numpy as np
import skimage
import tensorflow as tf
import os

import reader
from models.multimodal import MultiModal

flags = tf.flags
flags.DEFINE_integer(
    'N', -1, 'The number of samples of the Flickr8k dataset. -1 for all the samples.')
flags.DEFINE_boolean('restore', False, "Wether we restore the graph from the checkpoint or not")
logging = tf.logging



class Configs(object):
    learning_rate = 0.0001    #0.25
    max_grad_norm = 5
    num_layers = 3
    num_steps = 12
    hidden_size = 1200
    max_epoch = 4
    max_max_epoch = 20
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    image_size = 227


def main(rnn_config, eval_config):
    print('Loading data')
    train_data, valid_data, vocab = reader.flickr_raw_data(flags.FLAGS.N, rnn_config.num_steps,
                                                           rnn_config.image_size)
    test_image = np.zeros((eval_config.batch_size, eval_config.image_size, eval_config.image_size, 3))
    im = skimage.img_as_float(skimage.io.imread('data/test_image.jpg'))#plt.imread('data/test_image.jpg') / 255.
    test_image[0, :, :] = skimage.transform.rescale(im, rnn_config.image_size / 227.0)

    rnn_config.vocab_size = len(vocab)
    eval_config.vocab_size = len(vocab)
    print('Vocab size of: {}'.format(rnn_config.vocab_size))


    with tf.Graph().as_default(), tf.Session() as sess:
        global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        print('Creating rnn model')
        if flags.FLAGS.restore:
            initializer = None
        else:
            initializer = tf.uniform_unit_scaling_initializer()

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            train_image_tensor = tf.placeholder(np.float32, (rnn_config.batch_size, rnn_config.image_size,
                                                             rnn_config.image_size, 3), 'input_image')
            m = MultiModal(is_training=True, config=rnn_config, image_tensor=train_image_tensor, global_step_tensor=global_step_tensor)
            m.load_alexnet('models/alexnet_weights.npy', sess)
            variables_to_save = tf.trainable_variables() + [global_step_tensor]

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = MultiModal(is_training=False, config=rnn_config,
                                image_tensor=train_image_tensor, global_step_tensor=global_step_tensor)
            test_image_tensor = tf.placeholder(np.float32, (eval_config.batch_size, eval_config.image_size,
                                                            eval_config.image_size, 3), 'test_input_image')
            mtest = MultiModal(is_training=False,
                               config=eval_config,
                               image_tensor=test_image_tensor, global_step_tensor=global_step_tensor)

        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("logs/")

        saver = tf.train.Saver(variables_to_save)
        if flags.FLAGS.restore:
            tf.initialize_all_variables().run()
            checkpoint = tf.train.latest_checkpoint(os.path.abspath('ckpts/'))
            print('Restoring from checkpoint: {}'.format(checkpoint))
            saver.restore(sess, checkpoint)
        else:
            tf.initialize_all_variables().run()

        print('Getting the global step : {}'.format(tf.train.global_step(sess, global_step_tensor)))
        #for epoch in range(config.max_max_epoch):
        epoch = get_epoch(global_step_tensor, sess, flags.FLAGS.N, config.batch_size)
        while epoch <= config.max_max_epoch:
            lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
            m.assign_lr(sess, config.learning_rate * lr_decay)
            print('Running epoch {} with learning rate {}'.format(epoch,
                                                                  config.learning_rate * lr_decay))
            train_loss = run_epoch(sess, train_data, m.train_op, config, m, mtest, test_image,
                                   vocab, global_step_tensor,
                                   summary=merged,
                                   summary_writer=writer,
                                   verbose=True)
            print('Training Loss : {:.3f}'.format(train_loss))
            valid_loss = run_epoch(sess, valid_data, mvalid.train_op, config, mvalid, None,
                                   test_image, vocab, global_step_tensor,
                                   verbose=False)
            print('Valid Loss: {:.3f}'.format(valid_loss))

            print('Sampling rnn')
            s = mtest.sample(sess, vocab, eval_config, test_image, ['<BOS>'])
            print(s)
            saver.save(sess, os.path.abspath('ckpts/captionning'), global_step_tensor)
            epoch = get_epoch(global_step_tensor, sess, flags.FLAGS.N, config.batch_size)


def get_epoch(global_step_tensor, sess, N, batch_size):
    return tf.train.global_step(sess, global_step_tensor) // (N // batch_size)


def run_epoch(session, data, eval_op, config, train_model, test_model, test_image, vocab, global_step,
              summary_writer=None,
              summary=None,
              verbose=False):
    """Runs the model on the given data."""
    epoch_size = len(data['dataset']) / config.batch_size
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = train_model.initial_state.eval()

    image_shape = (config.image_size, config.image_size, 3)
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
            summary_writer.add_summary(summary_result, global_step=tf.train.global_step(session, global_step))
        if verbose and step % 50 == 0 and step != 0:
            print(test_model.sample(session, vocab, eval_config, test_image, prime=['<BOS>']))

    return np.exp(costs / iters)


if __name__ == "__main__":
    config = Configs()
    eval_config = Configs()
    eval_config.batch_size = 1
    main(config, eval_config)
