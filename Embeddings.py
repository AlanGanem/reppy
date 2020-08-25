import gensim
import pickle
import umap
import numpy as np
import tqdm
import warnings

import graph2vec
from .GraphTools import df_to_node


class Node2Vec(graph2vec.Node2Vec):
    @classmethod
    def load(cls, out_file):
        """
        Load embeddings from gensim.models.KeyedVectors format
        """
        cls.model = gensim.wv.load_word2vec_format(out_file)

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
            neighbor_weight = 1.,
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
            'neighbor_weight':neighbor_weight,
            'threads':threads,
            'keep_walks':keep_walks
        }
        super().__init__(
            w2vparams = self.w2vparams,
            **self.sampling_params,
        )

        return

    def transform(self, data, entity_classes, agg = 'sum',handler = 'handle'):
        assert handler in ['warn', 'handle', 'raise']
        row_to_nodes = df_to_node(data, entity_classes)
        entities_set = set([ent for ent in self.model.wv.vocab.keys()])
        embeddings = []
        for i in tqdm.tqdm(row_to_nodes):
            try:
                embeddings.append(getattr(self.model.wv[i], agg)(axis=0))
            except KeyError:
                if handler == 'raise':
                    raise KeyError('unknown entity in: {}'.format(i))
                else:
                    list_i = list(i)
                    missing_entities = set(i) - entities_set
                    for ent in missing_entities:
                        if handler == 'warn':
                            class_ = ent.split('__')[0]
                            list_i[list_i.index(ent)] = class_
                            warnings.warn('{} is an unknown entity. class {} embedding returned instead'.format(ent,class_))
                        elif handler == 'handle':
                            class_ = ent.split('__')[0]
                            list_i[list_i.index(ent)] = class_
                    if agg:
                        embeddings.append(getattr(self.model.wv[list_i], agg)(axis=0))
                    else:
                        embeddings.append(self.model.wv[list_i])

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

    # def plot_embeddings(self):
    #     data =
    #     self.ump.embeddings_

    # def predict(self,data,features,unknown_features,query_matrix,agg,handler):
    #     '''
    #     predicts top_n similar pred_features. Can be used to fill missing data.
    #     requires C compiler
    #     :return:
    #     '''
    #     for feature in unknown_features:
    #         data.loc[:,feature] = feature
    #
    #     embeddings = self.transform(
    #         data = data,
    #         entity_classes = features,
    #         agg = agg,
    #         handler  = handler
    #     )

