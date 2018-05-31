#-*-coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle, os

from tensorflow.contrib.tensorboard.plugins import projector
LOG_DIR = "embedding/"
def load_session_embedding(filepath):
    with open(filepath) as file:
      session_embedding = pickle.load(file)
    session_embedding = np.array(session_embedding)
    shape = session_embedding.shape
    session_embedding = np.reshape(session_embedding, newshape=(shape[0] * shape[1], shape[2]))
    return session_embedding
def embedding_visualiztion(_):

    session_embed = load_session_embedding("static/cell_state/train_cell_state.pkl")
    shape1 = [min(100000, session_embed.shape[0]), session_embed.shape[1]]
    sess_embed = tf.Variable(session_embed[:shape1[0]],name="session_embed")
    del session_embed
    session_embed_with_url = load_session_embedding("static/cell_state/valid_cell_state.pkl")
    shape2 = [min(100000, session_embed_with_url.shape[0]), session_embed_with_url.shape[1]]
    sess_embed_contain_url = tf.Variable(session_embed_with_url[:shape2[0]],
                                         name="session_embed_contain_particular_url")
    del session_embed_with_url
    with open("static/vocabulary/embedding.pkl","r") as inputfile:
        url_embedding = pickle.load(inputfile)
    url_embed = tf.Variable(url_embedding, name="url_embedding")
    del url_embedding

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)


    root_dir = os.path.dirname(os.path.abspath(__file__))
    summary_wirter = tf.summary.FileWriter(logdir=LOG_DIR)
    config = projector.ProjectorConfig()
    embedding1 = config.embeddings.add()
    embedding1.tensor_name = url_embed.name
    embedding1.metadata_path = os.path.join(root_dir, "static/vocabulary/embedding.tsv")
    embedding2 = config.embeddings.add()
    embedding2.tensor_name = sess_embed_contain_url.name
    embedding2.metadata_path = os.path.join(root_dir,"static/vocabulary/session2.tsv")
    embedding3 = config.embeddings.add()
    embedding3.tensor_name = sess_embed.name
    embedding3.metadata_path = os.path.join(root_dir,"static/vocabulary/session1.tsv")

    projector.visualize_embeddings(summary_wirter, config=config)

    with open("static/ctynumber2.pkl", "r") as inputfile:
        session2cty = pickle.load(inputfile)
    with open("static/cell_state_to_countrynumber_contain_particular_url.pkl", mode="r") as inputfile:
        part_session2cty = pickle.load(inputfile)
    with open("static/vocabulary/id_to_url.pkl", 'r') as input_file:
        id_to_url = pickle.load(input_file)
    with open("static/vocabulary/url_to_type.pkl", 'r') as input_file:
        url_to_type = pickle.load(input_file)
    url_label = get_url_label(id_to_url,url_to_type)
    create_meta(url_label, "URLTypeId", name="embedding.tsv")
    create_meta(session2cty, "CountryId", name="session1.tsv")
    create_meta(part_session2cty, "CountryId",name="session2.tsv")


def create_meta(y, attribute_name="label", name="default.tsv"):
    with open("static/vocabulary/" + name, mode="wb") as outputfile:
        outputfile.write("Index\t%s\n" % attribute_name)
        for index, cnid in enumerate(y):
            outputfile.write("%d\t%s\n" %(index, str(cnid)))

def get_url_label(id2url,url2type):
    label =[]
    for i in xrange(len(id2url)):
        url = id2url[i]
        type = url2type.get(url)
        if type is not None:
            label.append(type)
        else:
            label.append("-1") # out of vocabulary
    return label

if __name__ =="__main__":
    tf.app.run(main=embedding_visualiztion)
