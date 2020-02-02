import gensim
import pickle
import umap
import numpy as np

import graph2vec

class Node2Vec(graph2vec.Node2Vec):
    @classmethod
    def load(cls, out_file):
        """
        Load embeddings from gensim.models.KeyedVectors format
        """
        self.model = gensim.wv.load_word2vec_format(out_file)

    @classmethod
    def load_full(cls, loading_path, **pickleargs):
        """
        loads full Node2Vec model, including umap embeddings

        :param loading_path: loading path with extension
        :param pickleargs: kwargs to pickle.load
        :return:
        """
        with open(loading_path, 'rb') as file:
            loaded_model = pickle.load(file, **pickleargs)
        return loaded_model

    def save_full(self, saving_path, **pickleargs):
        with open(saving_path, 'wb') as file:
            pickle.dump(self, file, **pickleargs)

    def __init__(
            self,
            walklen = 10,
            epochs = 10,
            return_weight = 1.,
            neighbour_weight = 1.,
            threads = 0,
            keep_walks=True,
            w2v_size = 32,
            w2v_negative = 20,
            w2v_iter = 5,
            w2v_batch_words = 128
    ):


        self.w2vparams = {
            "window": walklen,
            "size": w2v_size,
            "negative": w2v_negative,
            "iter": w2v_iter,
            "batch_words": w2v_batch_words
        }

        self.sampling_params = {
            'walklen':walklen,
            'epochs':epochs,
            'return_weight':return_weight,
            'neighbour_weight':neighbour_weight,
            'threads':threads,
            'keep_walks':keep_walks
        }
        super().__init__(
            w2vparams = self.w2vparams
            **self.sampling_params,
        )

        return

    def transform(self, data, entity_classes, agg = 'sum'):

        row_to_nodes = GraphTools.df_to_node(data, entity_classes)
        embeddings = [getattr(self.node2vec.model.wv[i], agg)(axis=0) for i in tqdm.tqdm(row_to_nodes)]
        embeddings = np.array(embeddings)

        return embeddings

    def umap_fit_transform(self, **umapargs):
        ump = umap.UMAP(**umapargs)
        n2v_vectors = self.model.wv.vectors
        embeddings = ump.fit_transform(n2v_vectors)
        self.ump = ump
        return embeddings

    def umap_transform(self,X):
        embeddings = self.ump.transform(X)
        return embeddings



######### viz

df_dict = {
    'Description': index2word,
    'X': n2v_2d[:, 0].flatten(),
    'y': n2v_2d[:, 1].flatten(),
    'classes': classes
}

df = pd.DataFrame(df_dict)

class_enumerator = {v: i for i, v in enumerate(set(classes))}
class_num = [class_enumerator[cl] for cl in classes]
colors = color_map(class_num, 'Paired')
df['colors'] = colors
df['radius'] = 0.1

dir(mpl.cm)

bokeh_reduce_scatter(
    df,
    nonselection_alpha=0.5,
    colors_column='colors',
    hover_info=['Description'],
    file_name='node2vec_umap',
    file_title='node2vec_umap',
    x_axis_label='X',
    y_axis_label='y',
    plot_height=800,
    plot_width=1200,
    mpl_color_map=['inferno'],
    plot_title='node2vec map',
    toolbar_location='below',
    #    radii_column = 'radius',
    #    colors_column = 'colors'
)

