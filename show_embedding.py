# coding=utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import numpy as np
from sklearn.manifold import TSNE
from multiprocessing import Pool, Manager
import time
import os.path as osp
import pickle
import os, collections

import ptb_server as ptb
file_name = tf.train.latest_checkpoint("model/") #file_name: Name of the checkpoint file.
print(file_name)



def print_tensors_in_checkpoint_file(tensor_name):
    """Prints tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:

    tensor_name: Name of the tensor in the checkpoint file to print.
    """
    try:
        reader = tf.train.NewCheckpointReader(file_name)
        if not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            return reader.get_tensor(tensor_name)
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                "with SNAPPY.")

def plot_with_labels_by_color(low_dim_embs, labels=None, color=None, scalarmap=None, filename='tsne.png'):

    fig = plt.figure(figsize=(32, 32))  #in inches
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(fig)
    if labels is not None:
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    for i in range(len(low_dim_embs)):
        if low_dim_embs.shape[1] == 2:
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y, c=scalarmap.to_rgba(color[i]))
            if labels is not None:
                plt.annotate(labels[i],
                             xy=(x, y),
                             xytext=(5, 2),
                             textcoords='offset points',
                             ha='right',
                             va='bottom')
        else:
            x, y, z = low_dim_embs[i, :]
            ax.scatter(x, y, z, c=scalarmap.to_rgba(color[i]))
            if labels is not None:
                ax.text(x,y,z,str(labels[i]))
    plt.savefig(filename)
    angles = np.linspace(0,360,90)
    files = Rotation(ax, angles)
    make_gif(files, filename.split('.')[0]+".gif")


def Rotation(ax, angles):
    files = []
    for i, angles in enumerate(angles):
        ax.view_init(elev=None, azim=int(angles))
        fname = "tmp%d.jpg" % i
        ax.figure.savefig(fname)
        files.append(fname)
    return files

def make_gif(files, output, delay=80, repeat=True):
    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'
              % (delay, loop, " ".join(files), output))
    for file in files:
        os.remove(file)


def plot_with_labels_1(low_dim_embs, labels=None, filename='tsne-500.png'):
    plt.figure(figsize=(18, 18))  #in inches
    if labels is not None:
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        for i, label in zip(range(len(labels)), labels):
            x, y= low_dim_embs[i, :]
            plt.scatter(x, y,c='r')
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
    else:
        for i in range(len(low_dim_embs)):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y,c='r')
    plt.savefig(filename)


def looking_url_by_types_in_chunk(valid_id, start, id_to_url, url_to_type, url_dict, lock):
    for key, url in id_to_url.items():
        url_type = url_to_type.get(url)
        if url_type in valid_id:
            with lock:
                print("1")
                url_dict[url_type] += [start + key]

def get_words_of_types(valid_ids, id_to_url, url_to_type):
    valid_sets = {x: [] for x in valid_ids}

    for key, url in id_to_url.items():
        url_type = url_to_type.get(url)
        if url_type in valid_ids:
            valid_sets[url_type] += [key]
    # for key in valid_sets.keys():
    #     print(key, len(valid_sets[key]))
    # pool = Pool(8)
    # manager = Manager()
    # lock = manager.Lock()
    # vacob_len = len(id_to_url)
    # print "vacob length %d" % vacob_len
    # valid_sets = manager.dict({x: [] for x in valid_ids})
    # start = time.time()
    # print "start loading  valid_sets"
    # for i in range(8):
    #     step_size = int(vacob_len / 8)
    #     start_key, end_key = step_size * i, min(step_size * (i + 1), vacob_len)
    #     sub_id_to_url = dict([(i, id_to_url[i]) for i in range(start_key, end_key)])
    #     pool.apply_async(looking_url_by_types_in_chunk, args=(valid_ids,
    #                                                           start_key,
    #                                                           sub_id_to_url,
    #                                                           url_to_type,
    #                                                           valid_sets,
    #                                                           lock))
    # pool.close()
    # pool.join()
    # print "load valid_sets end, spending %g min" % ((time.time() - start) / 60)
    return dict(valid_sets)

def visual_embs_by_tsne(embeding, labels=None, color=None, scalarmap=None, name="tsne.jpg"):

    tsne = TSNE(perplexity=40, n_components=3, init='pca', n_iter=2000)
    low_dims_emb = tsne.fit_transform(embeding)
    if color is not None:
        plot_with_labels_by_color(low_dims_emb, labels, color, scalarmap, name)
    else:
        plot_with_labels_1(low_dims_emb, labels, name)


def random_sample_url_by_type(type_set, num_per_type, id_to_url, url_to_type):
    # 按类别随机采样url
    print("start sampling")
    vocabulary_len = len(id_to_url)
    valid = get_words_of_types(type_set, id_to_url, url_to_type)
    category, color =0, []
    labels = []
    sample = []
    for key in type_set:
        print(len(valid[key]))
        for j in list(np.random.randint(low=len(valid[key])//2+1, size=num_per_type)):
            sample.append(valid[key][j])
            labels.append(key)
        color += num_per_type * [category]
        category += 1
    # print(sample)
    cNorm = cmx.colors.Normalize(vmin=0, vmax=category)
    hot = plt.get_cmap('hot')
    scalarmap = cmx.ScalarMappable(cNorm, hot)
    embedding = np.array(print_tensors_in_checkpoint_file("Model/embedding"))
    print(embedding.shape)
    visual_embs_by_tsne(embedding[sample],
                        # labels=labels,
                        color=color,
                        scalarmap=scalarmap,
                        name="url_embeding_10category_100samples_2.jpg")

def sample_session_by_country(filename, selected_countries=None, sample_per_country=300, limit=None):
    with open(filename) as file:
        session2country = pickle.load(file)
    counter = collections.Counter(session2country)
    if selected_countries is not None:
        for c in selected_countries:
            if not counter.has_key(c):
                print("not contain country: %s" % str(c))
                return None
    else:
        n_top_countries = [x for x, _ in counter.most_common(20)]
        selected_countries = np.random.choice(n_top_countries[:5], 3)
    valid_set = {x:[] for x in selected_countries}
    lens = min(limit, len(session2country)) if limit is not None else len(session2country)
    for i in xrange(lens):
        if session2country[i] in selected_countries:
            valid_set[session2country[i]].append(i)
    print("selected country:", valid_set.keys())
    samples = []
    category, color = 0, []
    for key in valid_set.keys():
        samples += list(np.random.choice(valid_set[key], size=sample_per_country))
        # for i in list(np.random.randint(len(valid_set[key]),size=sample_per_country)):
        #     samples.append(valid_set[key][i])
        color += [category] * sample_per_country
        category+=1
    return samples, color, selected_countries

def draw_session_embedding(selected_countries=None):
    with open("static/train_cell_state.pkl") as file:
        session_embedding = pickle.load(file)
    session_embedding = np.array(session_embedding)
    shape = session_embedding.shape
    session_embedding = np.reshape(session_embedding, newshape=(shape[0]*shape[1],shape[2]))
    # if selected_countries is not None:
    sample, color, selected_countries = sample_session_by_country(
        "static/whole_session_ctynumber.pkl",
        limit=shape[0]*shape[1])
    cNorm = cmx.colors.Normalize(vmin=0, vmax=len(selected_countries))
    hot = plt.get_cmap('hot')
    scalarmap = cmx.ScalarMappable(cNorm, hot)
    # sample = np.random.randint(session_embedding.shape[0]//2,session_embedding.shape[0],size=1000)
    session_embedding_sampled = session_embedding[sample]
    visual_embs_by_tsne(session_embedding_sampled,
                        color=color,
                        scalarmap=scalarmap,
                        name="session_embedding_1000_2.png")

def url_type_set(id_to_url, url_to_type):
    if osp.exists("static/vocabulary/url_type_analyse.pkl"):
        with open("static/vocabulary/url_type_analyse.pkl") as file:
            return pickle.load(file)
    url_type_dict = {}
    for i in xrange(len(id_to_url)):
        url = id_to_url[i]
        url_type = url_to_type.get(url,-1)
        if url_type != -1:
            url_type_dict[url_type] = url_type_dict.get(url_type,0)+1
    with open("static/vocabulary/url_type_analyse.pkl", 'wb') as file:
        pickle.dump(url_type_dict, file)
    return url_type_dict

def get_url_embedding_tsv(id_to_url, url_to_type):
    tsv = open("static/vocabulary/url_embeding.tsv", "w")
    for i in xrange(100000):
        url = id_to_url[i]
        type = url_to_type.get(url)
        if type is not None:
            tsv.write(str(type)+"\n")
        else:
            tsv.write("unk\n")


def draw_cell_state_embedding(selected_countries=None):
    with open("static/valid_cell_state.pkl") as file:
        session_embedding = pickle.load(file)
    session_embedding = np.array(session_embedding)
    shape = session_embedding.shape
    session_embedding = np.reshape(session_embedding, newshape=(shape[0]*shape[1],shape[2]))
    sample, color, selected_countries = sample_session_by_country("static/cell_state_to_countrynumber_contain_particular_url.pkl")
    # sample = np.random.randint(session_embedding.shape[0],size=1000)
    # sample = np.random.randint(session_embedding.shape[0]//2,session_embedding.shape[0],size=1000)
    session_embedding_sampled = session_embedding[sample]
    cNorm = cmx.colors.Normalize(vmin=0, vmax=len(selected_countries))
    hot = plt.get_cmap('hot')
    scalarmap = cmx.ScalarMappable(cNorm, hot)
    visual_embs_by_tsne(session_embedding_sampled,
                        color=color,
                        scalarmap=scalarmap,
                        name="cell_embedding_1000_2.png")

def main():



    id_to_url, url_to_type = {},{}
    with open("static/vocabulary/id_to_url.pkl", 'r') as input_file:
        id_to_url = pickle.load(input_file)
    with open("static/vocabulary/url_to_type.pkl", 'r') as input_file:
        url_to_type = pickle.load(input_file)
    # type_set = [200,1165,7008,2462,227,2488,2211,2554]
    type_set = [7008, 6143, 2210, 200]
    random_sample_url_by_type(type_set, 300, id_to_url, url_to_type)

    draw_cell_state_embedding()
    draw_session_embedding()
    # url_types = url_type_set(id_to_url, url_to_type)
    # dict = sorted(url_types.iteritems(), key=lambda d: d[1], reverse=True)
    # print dict
    # get_url_embedding_tsv(id_to_url, url_to_type)

if __name__ == "__main__":
    main()