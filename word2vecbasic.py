#-*-coding:utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pickle
import reader




def load():
  with open("static/train_routes.pkl","r") as file:
    data = pickle.load(file)
  print("train data length:", len(data))
  with open("static/vocabulary/id_to_url.pkl", "r") as file:
    reverse_dictionary = pickle.load(file)
  return data, reverse_dictionary

data1, data2, data3, _, voc = reader.ptb_raw_data("data")
print(voc)
data = data1 + data2 + data3
key, val = [], []
for k, v in voc.iteritems():
  key.append(k)
  val.append(v)
reverse_dictionary = dict(zip(val, key))
del key, val, voc
# data = [x for x in data if x != 1]
vocabulary_size = len(reverse_dictionary)
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_sidze, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      for term in data[0:span]:
        buffer.append(term)
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
print("###########################################")
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.
basic_lr = 0.5
batch_size = 72
embedding_size = 200  # Dimension of the embedding vector.
skip_window = 3       # 窗口单边大小
num_skips = 6         # 每个中心词产生多少个样本
num_sampled = 64      # Number of negative examples to sample.
words_per_epoch = len(data) // (2* skip_window +1)
print("step per epoch:", words_per_epoch)
num_epoch = 15
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)



class word2vec:
  def __init__(self):
    self._train_inputs = tf.placeholder(tf.int32, shape=[batch_size], name='x')
    self._train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name='y')
    self._valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
    self._global_step = tf.Variable(0, trainable=False, name="global_step")
    with tf.device('/cpu:0'):
      # Look up embeddings for inputs.
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -0.5/embedding_size, 0.5/embedding_size),
          name="embedding")
      embed = tf.nn.embedding_lookup(embeddings, self._train_inputs)
      self._embeddings = embeddings
      # Construct the variables for the NCE loss
      nce_weights = tf.Variable(
          tf.truncated_normal([vocabulary_size, embedding_size],
                              stddev=1.0 / math.sqrt(embedding_size)),
          name="nce_weight")
      self.variable_summaries(nce_weights, "nce_weight")
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="nce_biases")
      self.variable_summaries(nce_biases, "nce_biases")

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    self._loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=self._train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size), name="nce_loss")
    tf.summary.scalar("loss", self._loss)

    learning_rate = tf.Variable(basic_lr, trainable=False, dtype=tf.float32)
    self._update_lr = tf.assign(learning_rate, self._new_lr)
    tf.summary.scalar("lr", learning_rate)
    # Construct the SGD optimizer using a learning rate of 1.0.
    self._optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self._loss,
                                                                                global_step=self._global_step)
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, self._valid_dataset)
    self._similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
    # Add variable initializer.
    self._init = tf.global_variables_initializer()
    self._summary_op = tf.summary.merge_all()

  def variable_summaries(self, var, name):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean/' + name, mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.summary.scalar('stddev/' + name, stddev)
      tf.summary.scalar('max/' + name, tf.reduce_max(var))
      tf.summary.scalar('min/' + name, tf.reduce_min(var))
      tf.summary.histogram(name, var)


  def train(self):

    filewirter = tf.summary.FileWriter("model/")
    # Step 5: Begin training.
    saver = tf.train.Saver()

    with tf.Session() as session:
      # We must initialize all variables before we use them.
      session.run(self._init)
      print('Initialized')
      sim = self._similarity.eval(session=session)
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)

      average_loss = 0
      for epoch in range(num_epoch):
        lr = max(0.0001, basic_lr * pow(0.5, (epoch+1) // 2))
        print(lr)
        session.run(self._update_lr, feed_dict={self._new_lr: lr})
        for step in xrange(words_per_epoch):
          batch_inputs, batch_labels = generate_batch(
              batch_size, num_skips, skip_window)
          feed_dict = {self._train_inputs: batch_inputs, self._train_labels: batch_labels}

          # We perform one update step by evaluating the optimizer op (including it
          # in the list of returned values for session.run()
          _, loss_val = session.run([self._optimizer, self._loss], feed_dict=feed_dict)
          average_loss += loss_val
          if step % 1000 == 0:
            summaries, g_step = session.run([self._summary_op,self._global_step], feed_dict=feed_dict)
            filewirter.add_summary(summaries, g_step)
          if step % 10000 == 0:
            if step > 0:
              average_loss /= 10000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0
            # saver.save(session, "model/model.ckpt", global_step=self._global_step.eval(session))

        # Note that this is expensive (~20% slowdown if computed every 500 steps)


      # final_embeddings = normalized_embeddings.eval(session=session)

      final_embed = self._embeddings.eval(session=session)
      with open("static/vocabulary/embedding.pkl","wb") as file:
        pickle.dump(final_embed, file)


  def eval(self):
    last_checkpoint = tf.train.latest_checkpoint("model/")
    print(last_checkpoint)
    saver = tf.train.Saver()
    with tf.Session() as session:
      saver.restore(sess=session, save_path=last_checkpoint)
      sim = self._similarity.eval(session=session)
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
      final_embed = self._embeddings.eval(session=session)
      with open("static/vocabulary/embedding.pkl", "wb") as file:
        pickle.dump(final_embed, file)
# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
# def plot_with_labels(low_dim_embs, labels, filename):
#   assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
#   plt.figure(figsize=(18, 18))  # in inches
#   for i, label in enumerate(labels):
#     x, y = low_dim_embs[i, :]
#     plt.scatter(x, y)
#     plt.annotate(label,
#                  xy=(x, y),
#                  xytext=(5, 2),
#                  textcoords='offset points',
#                  ha='right',
#                  va='bottom')
#
#   plt.savefig(filename)

# try:
#   # pylint: disable=g-import-not-at-top
#   from sklearn.manifold import TSNE
#   import matplotlib.pyplot as plt
#
#   tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
#   plot_only = 500
#   low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
#   labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#   plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))
#
# except ImportError as ex:
#   print('Please install sklearn, matplotlib, and scipy to show embeddings.')
#   print(ex)
def main(_):
  with tf.Graph().as_default():
    model = word2vec()
    model.train()

if __name__ == "__main__":
  tf.app.run()

