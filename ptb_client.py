# coding=utf-8

from __future__ import print_function

import sys
import threading
import pickle

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import reader_single as reader

tf.app.flags.DEFINE_string('server', 'localhost:6007',
                           'predictionService host:port')

tf.app.flags.DEFINE_integer('batch_size', 1, '')
tf.app.flags.DEFINE_integer('num_steps', 5, '')

FLAGS = tf.app.flags.FLAGS


def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Generate test data
    test_path = "/home/zyt/data/JD/test.csv"
    # load url_to_id from file
    input_file = open("data/processed_data/url_to_id.pkl", 'rb')
    url_to_id = pickle.load(input_file)
    vocab_size = len(url_to_id)
    input_file.close()
    test_data = reader._get_route_from_data(test_path, url_to_id)
    n_tests = len(test_data) - 1

    # Send request
    for step, (x, y) in enumerate(reader.ptb_iterator(test_data, FLAGS.batch_size, FLAGS.num_steps)):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'ptb_serving'
        request.model_spec.signature_name = 'predict_signature'
        # input_path, _ = reader.ptb_producer(test_data, FLAGS.batch_size, FLAGS.num_steps)
        request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(x,
                                                                           shape=[FLAGS.batch_size, FLAGS.num_steps]))
        result = stub.Predict(request, 10.0)
        if step % 2 == 0:
            print("------------------------------------output---------------------------")
            logits = result.outputs['output'].float_val
            print(logits[(FLAGS.num_steps-1)*vocab_size:])
            print("------------------------------------cell_state---------------------------")
            print(result.outputs['cell_state'].float_val)
            print("------------------------------------embed_lookup---------------------------")
            print(result.outputs['embed_lookup'].float_val)


if __name__ == '__main__':
    tf.app.run()
