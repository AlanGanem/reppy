import numpy as np
from scipy.sparse import csr_matrix
try:
    from sparse_dot_topn import awesome_cossim_topn
except ImportError:
    print('install sparse_dot_topn in order to use "predict" method more efficiently. run "pip install sparse-dot-topn" on prompt.'/
          'make sure you have a C compiler installed.')

def normalize(X):
    norm = np.linalg.norm(X, axis=-1).reshape(-1, 1)
    norm[norm <=0] = 1
    X = X/norm
    return X

def get_similarity(matrix1,matrix2, n_top, min_similarity, zero_diagonal = False):
    matrix1 = normalize(matrix1)
    matrix2 = normalize(matrix2)

    matrix1 = csr_matrix(matrix1).astype(float)
    matrix2 = csr_matrix(matrix2).astype(float)
    similarity_matrix = awesome_cossim_topn(matrix1, matrix2.T, ntop=n_top, lower_bound=min_similarity)
    # set diagonal to zero
    if zero_diagonal == True:
        similarity_matrix.setdiag(0)

    return similarity_matrix

def query(query_matrix,search_matrix,sim_top_n, sim_thresh, return_ranking):
    similarity_matrix = get_similarity(query_matrix,search_matrix,sim_top_n, sim_thresh)
    sim_ranking = []
    if return_ranking:
        for idx in range(query_matrix.shape[0]):

            row = similarity_matrix[idx].A
            row = row.flatten()

            # get mixed sim and topn thresh
            if (sim_top_n != None) and (sim_thresh != None):
                ind1 = row.argsort()[-sim_top_n:][::-1].flatten()
                ind2 = np.argwhere(row >= sim_thresh).flatten()
                ind = np.array([i for i in ind1 if i in ind2])
                #ind = np.array(list(set(ind1).intersection(set(ind2))))
            # get mixed sim thresh
            elif (sim_top_n == None):
                ind = np.argwhere(row >= sim_thresh).flatten()
            # get mixed top_n thresh
            elif (sim_thresh == None):
                ind = row.argsort()[-sim_top_n:][::-1].flatten()

            try:
                sim_ranking.append([list(i) for i in zip(ind, row[ind])])
            except:
                sim_ranking.append([])
        return sim_ranking

    else:
        return similarity_matrix


def batch_generator(X, y, batch_size):
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    X =  X[shuffle_index, :]
    y =  y[shuffle_index]
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[index_batch,:].todense()
        y_batch = y[index_batch]
        counter += 1
        yield(np.array(X_batch),y_batch)
        if (counter < number_of_batches):
            np.random.shuffle(shuffle_index)
            counter=0