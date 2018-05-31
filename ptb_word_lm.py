# coding=utf-8
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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os

import numpy as np
import tensorflow as tf

import reader_single as reader
import util

from tensorflow.python.client import device_lib
from tensorflow.python.saved_model import builder as saved_model_builder

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("model_path", "/tmp/ptb_test/",
                    "SaveModel path.")
flags.DEFINE_string("model_version", 1,
                    "the version of model")
flags.DEFINE_string("save_path", "log_test/",
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")

FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, vocab_size):

    self._is_training = is_training
    self._cell = None
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps
    size = config.hidden_size

    self._input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
    self._targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    self._embed_lookup = inputs

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    output, state = self._build_rnn_graph(inputs, config, is_training)
    # state_is_tuple=True, so state=(c,h), and the state of multiRNN is n-tuples

    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

    # define predict for tensorflow serving
    self._predict = logits[:, self.num_steps-1:self.num_steps, :]

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        self._targets,
        tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    # gradient在BP过程中，很容易出现vanish或explode现象，尤其是RNN这种back很多个timesteps的结构，因此要使用clip来对gradient进行调节
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    # 既然要调节梯度，那么就不能简单的使用optimizer.minimize(loss)，而是需要显式的计算gradients，然后进行clip，将clip后的gradient进行apply
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _build_rnn_graph(self, inputs, config, is_training):
      return self._build_rnn_graph_lstm(inputs, config, is_training)

  def _get_lstm_cell(self, config, is_training):
    return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    cell = self._get_lstm_cell(config, is_training)
    if is_training and config.keep_prob < 1:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=config.keep_prob)

    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def predict(self):
    return self._predict

  @property
  def embed_lookup(self):
    return self._embed_lookup

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


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1      # 相关参数的初始值为随机均匀分布，范围是[-init_scale,+init_scale]
  learning_rate = 1.0
  max_grad_norm = 5    # 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小
  num_layers = 2
  num_steps = 10       # 分隔句子的粒度大小，每次会把num_steps个单词划分为一句话
  hidden_size = 200    # 隐层单元数目，每个词会表示成[hidden_size]大小的向量
  max_epoch = 4        # epoch<max_epoch时，lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
  max_max_epoch = 1   # 完整的文本要循环的次数
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 25      # 和num_steps共同作用，要把原始训练数据划分为batch_size组，每组划分为n个长度为num_steps的句子。
  # vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  # vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  # vocab_size = 10000


class Config_Test(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  # vocab_size = 10000


def run_epoch(session, model, data, config, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)
  epoch_size = ((len(data) // config.batch_size) - 1) // config.num_steps

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
      "logit": model.predict,
      "embed_lookup": model.embed_lookup
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step, (x, y) in enumerate(reader.ptb_iterator(data, batch_size=config.batch_size, num_steps=config.num_steps)):
    vals = session.run(fetches, feed_dict={
      model.input_data: x,
      model.targets: y,
      model.initial_state: state
    })

    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += config.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * config.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = Config_Test()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  return config


def main(_):
  data_path = "/home/zyt/data/JD/"

  raw_data = reader.ptb_raw_data(data_path)
  train_data, valid_data, test_data, vocab_size = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default(), tf.Session() as session:
    # 相关参数的初始值为随机均匀分布，范围是[-init_scale,+init_scale]
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      # train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, vocab_size=vocab_size)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      # valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, vocab_size=vocab_size)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      # test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config, vocab_size=vocab_size)

    init_op = tf.global_variables_initializer()
    session.run(init_op)
    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity = run_epoch(session, m, train_data, config, eval_op=m.train_op,
                                     verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid, valid_data, config)
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    test_perplexity = run_epoch(session, mtest, test_data, eval_config)
    print("Test Perplexity: %.3f" % test_perplexity)

    # Export tensorflow serving
    export_path = os.path.join(tf.compat.as_bytes(FLAGS.model_path),
                                 tf.compat.as_bytes(str(FLAGS.model_version)))
    builder = saved_model_builder.SavedModelBuilder(export_path)
    prediction_inputs = {'input': tf.saved_model.utils.build_tensor_info(mtest.input_data)}
    # 'cell_state': tf.saved_model.utils.build_tensor_info(mtest.state),
    # 'embed_lookup': tf.saved_model.utils.build_tensor_info(mtest.embed_lookup)
    prediction_outputs = {'output': tf.saved_model.utils.build_tensor_info(mtest.predict),
                          'cell_state': tf.saved_model.utils.build_tensor_info(mtest.final_state[-1].c),
                          'embed_lookup': tf.saved_model.utils.build_tensor_info(mtest.embed_lookup)}
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
      inputs=prediction_inputs,
      outputs=prediction_outputs,
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                           'predict_signature': prediction_signature,
                                         })
    builder.save()
    print("Done export!")


if __name__ == "__main__":
  tf.app.run()
