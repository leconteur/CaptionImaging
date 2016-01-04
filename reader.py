from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import skimage.data
import skimage.transform
from tensorflow.python.platform import gfile


def _build_vocab(data):
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(data, word_to_id):
    return [word_to_id[word] for word in data]


def flickr_raw_data(N, num_steps, image_size):
    all_text = []

    with gfile.GFile('data/Flickr8k_text/Flickr_8k.trainImages.txt', 'r') as f:
        train_images = f.readlines()
        train_images = [i.strip() for i in train_images]
    with gfile.GFile('data/Flickr8k_text/Flickr_8k.testImages.txt', 'r') as f:
        test_images = f.readlines()
        test_images = [i.strip() for i in test_images]

    with gfile.GFile('data/Flickr8k_text/Flickr8k.lemma.token.txt', 'r') as f:
        train = []
        valid = []
        images = {}
        for i, line in enumerate(f):
            if 0 < N <= len(train):
                break
            if i % 500 == 0:
                complete = len(train) / N if N > 0 else i
                print('Loading dataset : {:.1%}'.format(complete))
            image_filename, sentence = line.split('\t')
            image_filename, number = image_filename.split('#')
            sentence = sentence.lower().replace('\n', '').translate(None, '.,')
            sentence = '<BOS> ' + sentence + ' <EOS>'
            sentence = sentence.split()
            sentence.extend(['<PAD>'] * ((num_steps - len(sentence)) + 1))
            if int(number) == 0:
                image_path = os.path.join('data', 'Flicker8k_Dataset', image_filename)
                try:
                    im = skimage.data.imread(image_path) / 255.
                    im = skimage.transform.rescale(im, image_size / 500.0)
                    image = np.zeros((image_size, image_size, 3))
                    image[:im.shape[0], :im.shape[1], :] = im
                    images[image_filename] = image
                except IOError as e:
                    print(e)
                    image = None
                    continue
            if image is None:
                continue
            all_text.extend(sentence)
            # if int(number) == 4: # The first time, sentences is an empty list
            if image_filename in test_images:
                valid.append((image_filename, sentence[:num_steps + 1]))
            else:
                train.append((image_filename, sentence[:num_steps + 1]))
    vocab = _build_vocab(all_text)
    for dataset in [train, valid]:
        for _, sentence in dataset:
            for i, word in enumerate(sentence):
                sentence[i] = vocab[word]
    print(len(train), len(valid))
    train = {'images': images, 'dataset': train}
    valid = {'images': images, 'dataset': valid}
    return train, valid, vocab


def flickr_iterator(data, batch_size, num_steps, imshape):
    data_len = len(data['dataset'])
    np.random.shuffle(data['dataset'])
    for i in range(0, data_len - batch_size, batch_size):
        sentences = np.zeros((batch_size, num_steps))
        images = np.zeros((batch_size, imshape[0], imshape[1], imshape[2]))
        targets = np.zeros((batch_size, num_steps))
        for j, d in enumerate(data['dataset'][i:i + batch_size]):
            sentences[j, :] = d[1][:-1]
            images[j] = data['images'][d[0]]
            targets[j, :] = d[1][1:]
        yield (sentences, images, targets)
