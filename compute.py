import numpy as np
import cPickle as pickle
def load_session_embedding(filepath):
    with open(filepath) as file:
      session_embedding = pickle.load(file)
    session_embedding = np.array(session_embedding)
    shape = session_embedding.shape
    session_embedding = np.reshape(session_embedding, newshape=(shape[0] * shape[1], shape[2]))
    return session_embedding

def compute_means(x):
    return np.mean(x, axis=0)

def compute_stddev(x):
    return np.std(x, axis=0)

def compute_range_between_stddev(x, alpha):
    mean = compute_means(x)
    stddev = compute_stddev(x)
    count=0;
    for i in xrange(x.shape[0]):
        low = x[i] - (mean-alpha*stddev)
        high = mean + alpha*stddev - x[i];
        if(np.where(low>=0)[0].shape[0] == low.shape[0] and
           np.where(high>=0)[0].shape[0] == low.shape[0]):
            count+=1;

    return 1.0*count / x.shape[0]
def compute_range_between(x):
    alphas = [1,2,3,5,7,10]
    percentages = []
    for alpha in alphas:
        percent = compute_range_between_stddev(x,alpha)
        if(len(percentages) == 0):
            percentages.append(percent)
        else:
            percentages.append(percent)
    # print(percentages)
    return percentages
def show(data):
    labels = "1","2","3","5","7","10"
    x = [data[i] if (i==0) else data[i]-data[i-1] for i in range(len(data))]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()
    ax.pie(x,labels=labels, autopct="%3.1f %%")
    plt.savefig("f1.png")

if __name__ == "__main__":
    session_embed = load_session_embedding("static/cell_state/train_cell_state.pkl")
    # print(compute_means(session_embed))
    # print(compute_stddev(session_embed))
    print(compute_range_between(session_embed))