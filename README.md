# CaptionImaging
This project is an experiment to see if it is possible to create or modify images from a caption in natural language. I will also take this project as an occasion to try out the Tensorflow library. Keep in mind that this is a toy project and it may not lead to any interesting results other than having fun.

## Strategy
To generate images from caption, my strategy is to train a multimodal neural network to do image captioning. This neural network is trained to predict the next word in a caption, using the previous words in the sentence and the image as an input. This network is a multimodal network composed of a RNN into which we feed the caption, and a convnet into which we feed the picture. This network then tries to minimize the error on the prediction of the next word in the caption.

Once this network is trained, we will then "train" the picture pixels to minimize the perplexity of the caption conditioned on the picture.

## Some details

This project is a toy project. As such, I used a relatively simple, pretrained convnet which is fast to run. It would probably lead to better results to exchange AlexNet for GoogleNet or another good convolutionnal neural network architecture.

I trained this project on the flickr8k dataset. This is a relatively small dataset. To train on a larger dataset, I would probably have to use a better input scheme to reduce the stress on memory.

## Results

Coming soon ...


## Inspiration and sources

This project is inspired from other works. The first work on which it is inspired is image captioning. I based my project on my interpretation of this article: http://arxiv.org/pdf/1412.6632v5.pdf

Another inspiration for this work is the article "A Neural Algorithm of Artistic Style": http://arxiv.org/abs/1508.06576

The flickr8k dataset used to train the network can be found at the following address: http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html

The weights and pretrained network can be found in the caffe model-zoo: https://github.com/BVLC/caffe/wiki/Model-Zoo

The importation of the AlexNet is done using the following project: https://github.com/ethereon/caffe-tensorflow

The sampling of the captions is inspired by the code in the following project: https://github.com/sherjilozair/char-rnn-tensorflow

The rnn part of the code is taken from the tensorflow tutorial on rnn: https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html

## Warning
This code really needs more comment and better variable names. I will try to find time to do a good cleanup soon.

I tried to be explicit about the code I used in my project. If I have made a mistake in my citation or the use of your code, please advise me and I will remediate the situation.
