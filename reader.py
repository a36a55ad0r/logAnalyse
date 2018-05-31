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


"""测试按网页类别进行完全归并"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pickle
import os
import re

import pandas as pd
import numpy as np
import tensorflow as tf


def test_search(url):
    # Search Engines Reg
    SousouReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*sousou\.com'
    SouGouReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*sogou\.com'
    Searcg360Reg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*so\.360\.cn/'
    BaiduReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*baidu\.com'
    BingReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*bing\.com'
    AolReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*aol\.com'
    AskReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*ask\.com'
    DaumReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*daum\.net'
    GoogleReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*google\.'
    MailReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*go\.mail\.ru'
    WebCrawlerReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*webcrawler\.com'
    WowReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*us\.wow\.com'
    YahooReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*yahoo\.(com|co)'
    YandexReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*yandex\.(com|by)'
    MySearchReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*zxyt\.cn'
    BingIEReg = '^([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)*bing\.ie'
    SearchLockReg = '^www\.searchlock\.com'
    SoSoReg = '^www\.soso\.com'
    SoReg = '^www\.so\.com'
    GoogleWebLightReg = '^googleweblight\.com'
    result = re.search(
        SouGouReg + '|' + SousouReg + '|' + Searcg360Reg + '|' + BaiduReg + '|' + BingReg + '|' + AolReg + '|' + AskReg + '|' + DaumReg + '|' +
        GoogleReg + '|' + MailReg + '|' + WebCrawlerReg + '|' + WowReg + '|' + YahooReg + '|' + YandexReg + '|' + MySearchReg + '|' + BingIEReg + '|' +
        SearchLockReg + '|' + SoSoReg + '|' + SoReg + '|' + GoogleWebLightReg, url)
    if result:
        return result.group()
    else:
        return None


def _filter_url(url):
    if re.search(r'(((&|\?)(key)?word=)|(^membercenter\.))', url) is not None:
        return -1

    return 0


def _build_vocab(filename):
    data = pd.read_csv(filename)

    request_url = data['request_url'].tolist()
    # filtered_request_url = [url for url in request_url if _filter_url(url) != -1]
    referer_url = data[data['referer_url'] != '-1']['referer_url'].tolist()
    # filtered_referer_url = [url for url in referer_url if _filter_url(url) != -1]
    '''zhang jie:添加url->url type'''
    # ============================================================
    request_id = data['request_id'].tolist()
    referer_id = data[data['referer_url'] != '-1']['referer_id'].tolist()
    url_2_type = request_id + referer_id
    counter = collections.Counter(url_2_type)
    sorted_pair = sorted(counter.iteritems(), key=lambda x: (-x[1], x[0]))
    ranked_urls = [k for k, v in sorted_pair if v >= 5]
    print(len(ranked_urls))

    type_to_id = dict(zip(ranked_urls, [i+2 for i in range(len(ranked_urls))]))
    id_to_type = dict(zip([i+2 for i in range(len(ranked_urls))], ranked_urls))
    type_to_id['UNK'] = 0
    type_to_id['<eos>'] = 1
    id_to_type[0] = "UNK"
    id_to_type[1] = "<eos>"

    # =============================================================
    print("build vocabulary successfully!")

    output = open("static/vocabulary/type_to_id.pkl", 'wb')
    pickle.dump(type_to_id, output)
    output.close()
    output = open("static/vocabulary/id_to_type.pkl", 'wb')
    pickle.dump(id_to_type, output)
    output.close()
    print("Save vocabulary successfully!")

    return type_to_id, id_to_type


def _path_to_ids(path, url_to_id):
    path_ids = list()
    for url in path:
        if url in url_to_id:
            path_ids.append(url_to_id[url])
        else:
            path_ids.append(url_to_id['UNK'])
    return path_ids


def _get_route_from_data(filename, url_to_id, min_path_len=3):
    routes = list()

    data = pd.read_csv(filename, chunksize=100000, encoding='utf-8')

    print("process data...")
    cur_ipto = 0  # current iptonumber
    path, url_path = [], []
    count = 0
    for chunk in data:
        print("process chunk_{}".format(count))
        sorted_chunk = chunk.sort_values(['iptonumber', 'visit_time'])
        filter_chunk = sorted_chunk[['visit_time', 'iptonumber', 'request_url', 'referer_url',
                                     'request_id', 'referer_id']]

        rows_iter = filter_chunk.itertuples()

        # row[0]: index, row[1]: visit_time, row[2]: iptonumber, row[3]: request_url, row[4]: referer_url

        for row in rows_iter:
            if row[2] != cur_ipto:
                if (len(path) >= min_path_len):
                    path_ids = _path_to_ids(path + ['<eos>'], url_to_id)
                    map(lambda x: routes.append(x), path_ids)
                url_path, path = [], []
                request_url = _filter_url(row[3])
                referer_url = _filter_url(row[4])
                if (referer_url != '-1'):
                    url_path.append(referer_url)
                    path.append(row[6])
                if(request_url !=-1):
                    url_path.append(request_url)
                    path.append(row[5])
                if (len(path) > 0):
                    cur_ipto = row[2]
            else:
                request_url = _filter_url(row[3])
                referer_url = _filter_url(row[4])
                if (referer_url != '-1' and (referer_url != path[len(path) - 1] or len(path) == 0)):
                    url_path.append(referer_url)
                    path.append(row[6])
                if (request_url != -1):
                    url_path.append(request_url)
                    path.append(row[5])
        count += 1

    if (len(path) >= min_path_len):
        path_ids = _path_to_ids(path + ['<eos>'], url_to_id)
        map(lambda x: routes.append(x), path_ids)

    print("process data successfully!")

    return routes





def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    The PTB dataset comes from Tomas Mikolov's webpage:

    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "20160720.csv")
    test_path = os.path.join(data_path, "20160727.csv")
    if os.path.exists("static/vocabulary/type_to_id.pkl"):
        with open("static/vocabulary/type_to_id.pkl") as file:
            url_to_id = pickle.load(file)
    else:
        url_to_id, _ = _build_vocab(train_path)

    # with open("static/train_routes.pkl") as file:
    #     datas = pickle.load(file)
    vocabulary = len(url_to_id)
    datas = _get_route_from_data(train_path, url_to_id)
    # train_data = datas[0:1000000]
    train_data = [x if x < vocabulary else 0 for x in datas[len(datas) // 5:]]
    print("train data length:", len(train_data))

    # with open("static/routes_contain_target_url.pkl", "rb") as file:
    #     valid_data = pickle.load(file)
    valid_data = [x if x < vocabulary else 0 for x in datas[:len(datas) // 5]]
    # valid_data = _get_route_contain_particular_url(train_path, url_to_id, "bigtree.en.made-in-china.com/")
    print("valid_data length:", len(valid_data))
    # valid_data = datas[1000000:]
    test_data = _get_route_from_data(test_path, url_to_id)
    # test_data = valid_data[0:len(valid_data) // 2]

    print("vocabulary length:", vocabulary)
    return train_data, valid_data, test_data, vocabulary, url_to_id


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).

    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.

    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y


def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size==0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)


def main(_):
    # for test
    data_path = "data/"
    train_path = os.path.join(data_path, "20160720.csv")
    train_data, valid_data, test_data, vocabulary=ptb_raw_data(data_path)
    print(test_data)


if __name__ == "__main__":
    tf.app.run()




