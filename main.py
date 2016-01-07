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
flags.DEFINE_boolean('restore', True, "Wether we restore the graph from the checkpoint or not")
logging = tf.logging


class Configs(object):
    learning_rate = 0.001    #0.25
    max_grad_norm = 5
    num_layers = 2
    num_steps = 14
    hidden_size = 400
    image_features_size = 500
    max_epoch = 4
    max_max_epoch = 3
    keep_prob = 0.5
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    image_size = 227
    pad_symbol = 0.


def main(rnn_config, eval_config):
    print('Loading data')
    train_data, valid_data, vocab = reader.flickr_raw_data(flags.FLAGS.N, rnn_config.num_steps,
                                                           rnn_config.image_size)
    rnn_config.pad_symbol = vocab['<PAD>']
    eval_config.pad_symbol = vocab['<PAD>']
    test_image = np.zeros(
        (eval_config.batch_size, eval_config.image_size, eval_config.image_size, 3))
    im = skimage.img_as_float(skimage.io.imread('data/test_image.jpg')
                              )    #plt.imread('data/test_image.jpg') / 255.
    test_image[0, :, :] = skimage.transform.rescale(im, rnn_config.image_size / 227.0)

    rnn_config.vocab_size = len(vocab)
    eval_config.vocab_size = len(vocab)
    print('Vocab size of: {}'.format(rnn_config.vocab_size))

    with tf.Graph().as_default(), tf.Session() as sess:
        global_step_tensor, m, merged, mtest, mvalid, mgen, gen_image_tensor, image_train_op, saver, writer = init_models(eval_config,
                                                                                  rnn_config, sess)

        train_model(eval_config, global_step_tensor, m, merged, mtest, mvalid, saver, sess,
                    test_image, train_data, valid_data, vocab, writer)

        create_image(sess, mgen, gen_image_tensor, image_train_op, ['<BOS>', 'a', 'woman', 'wear', 'a', 'blue', 'coat', '<EOS>'], eval_config, vocab, writer)


def create_image(session, mgen, image_tensor, image_train_op, input_sentence, eval_config, vocab, writer):
    input = np.zeros((eval_config.batch_size, eval_config.num_steps))
    for i, word in enumerate(input_sentence):
        input[:, i] = vocab[word]
    target = np.zeros((eval_config.batch_size, eval_config.num_steps))
    target[:,:-1] = input[:,:-1]
    feed_dict = {mgen.input_data: input, mgen.targets:target}

    image_sum = tf.image_summary('picture', image_tensor)

    for i in range(10000):
        cost, _, summary = session.run([mgen.cost, image_train_op, image_sum], feed_dict=feed_dict)
        if i % 10 == 0:
            writer.add_summary(summary, global_step=i)
            print(cost)


def train_model(eval_config, global_step_tensor, m, merged, mtest, mvalid, saver, sess, test_image,
                train_data, valid_data, vocab, writer):
    print('Getting the global step : {}'.format(tf.train.global_step(sess, global_step_tensor)))
    epoch = get_epoch(global_step_tensor, sess, flags.FLAGS.N, config.batch_size)
    while epoch <= config.max_max_epoch:
        lr_decay = config.lr_decay ** max(epoch - config.max_epoch, 0.0)
        m.assign_lr(sess, config.learning_rate * lr_decay)
        print('Running epoch {} with learning rate {}'.format(epoch,
                                                              config.learning_rate * lr_decay))
        train_loss = run_epoch(sess, train_data, m.train_op, config, m, mtest, test_image, vocab,
                               global_step_tensor,
                               summary=merged,
                               summary_writer=writer,
                               verbose=True)
        print('Training Loss : {:.3f}'.format(train_loss))
        valid_loss = run_epoch(sess, valid_data, mvalid.train_op, config, mvalid, None, test_image,
                               vocab, global_step_tensor,
                               verbose=False)
        print('Valid Loss: {:.3f}'.format(valid_loss))

        print('Sampling rnn')
        s = mtest.sample(sess, vocab, eval_config, test_image, ['<BOS>'])
        print(s)
        saver.save(sess, os.path.abspath('ckpts/captionning'), global_step_tensor)
        epoch = get_epoch(global_step_tensor, sess, flags.FLAGS.N, config.batch_size)


def init_models(eval_config, rnn_config, sess):
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
    print('Creating rnn model')
    if flags.FLAGS.restore:
        initializer = None
    else:
        initializer = tf.uniform_unit_scaling_initializer()
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        train_image_tensor = tf.placeholder(np.float32,
                                            (rnn_config.batch_size, rnn_config.image_size,
                                             rnn_config.image_size, 3), 'input_image')
        m = MultiModal(is_training=True,
                       config=rnn_config,
                       image_tensor=train_image_tensor,
                       global_step_tensor=global_step_tensor)
        m.load_alexnet('models/alexnet_weights.npy', sess)
        variables_to_save = tf.trainable_variables() + [global_step_tensor]
        #print(variables_to_save)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = MultiModal(is_training=False,
                            config=rnn_config,
                            image_tensor=train_image_tensor,
                            global_step_tensor=global_step_tensor)
        test_image_tensor = tf.placeholder(np.float32,
                                           (eval_config.batch_size, eval_config.image_size,
                                            eval_config.image_size, 3), 'test_input_image')
        mtest = MultiModal(is_training=False,
                           config=eval_config,
                           image_tensor=test_image_tensor,
                           global_step_tensor=global_step_tensor)

        initial_value = np.zeros((eval_config.batch_size, eval_config.image_size, eval_config.image_size, 3)).astype(np.float32)
        initial_value[0,:,:,:] = skimage.img_as_float(skimage.io.imread('data/test_image.jpg'))
        image_gen = tf.Variable(initial_value, trainable=True)
        mgen = MultiModal(is_training=False,
                          config=eval_config,
                          image_tensor=image_gen,
                          global_step_tensor=global_step_tensor)

        gradients = tf.gradients(mgen.cost, [image_gen])
        print(gradients)
        optimizer = tf.train.AdamOptimizer(0.1)
        image_train = optimizer.apply_gradients(zip(gradients, [image_gen]))

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("logs/")
    saver = tf.train.Saver()
    tf.initialize_all_variables().run()
    if flags.FLAGS.restore:
        checkpoint = tf.train.latest_checkpoint(os.path.abspath('ckpts/'))
        if checkpoint:
            print('Restoring from checkpoint: {}'.format(checkpoint))
            saver.restore(sess, checkpoint)
    return global_step_tensor, m, merged, mtest, mvalid, mgen, image_gen, image_train, saver, writer


def get_epoch(global_step_tensor, sess, N, batch_size):
    return (tf.train.global_step(sess, global_step_tensor) + 1) // (N // batch_size)


def run_epoch(session, data, eval_op, config, train_model, test_model, test_image, vocab,
              global_step,
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
            print("{0:.2%} perplexity: {1:.3f} speed: {2:.0f} wps".format(completed, perplexity,
                                                                          wps))
            summary_writer.add_summary(summary_result,
                                       global_step=tf.train.global_step(session, global_step))
        if verbose and step % 25 == 0 and step != 0:
            print(test_model.sample(session, vocab, eval_config, test_image, prime=['<BOS>']))
            print(test_model.sample(session, vocab, eval_config, test_image, prime=['<BOS>'], sampling_func=np.argmax))

    return np.exp(costs / iters)


if __name__ == "__main__":
    config = Configs()
    eval_config = Configs()
    eval_config.batch_size = 1
    main(config, eval_config)
