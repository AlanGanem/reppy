from typing import List, Tuple, Dict, Set, Union, Optional, Callable
import pickle

import Embeddings
import GraphTools

from .Utils import bokeh_reduce_scatter, DfScaler, Viz
from umap import UMAP

class Estimator():

    @classmethod
    def load(cls, loading_path, **pickleargs):
        with open(loading_path, 'rb') as file:
            loaded_model = pickle.load(file, **pickleargs)
        return loaded_model

    def save_full(self, saving_path, **pickleargs):
        with open(saving_path, 'wb') as file:
            pickle.dump(self, file, **pickleargs)

    def __init__(
            self,
            categorical_features: List[str],
            numerical_features: List[str],
            use_global_info: bool = True,
            global_info_agg_function: Union[str, Callable] = 'idf',
            entity_agg_function: Union[str, Callable] = 'sum',
            idf_log_base: int = 2,
            n2v_walklen=10,
            n2v_epochs=10,
            n2v_return_weight=1.,
            n2v_neighbour_weight=1.,
            n2v_threads=0,
            n2v_keep_walks=True,
            w2v_size=32,
            w2v_negative=20,
            w2v_iter=5,
            w2v_batch_words=128,
            scale_numerical = True,
            scale_method = 'RobustScaler',
            scale_columns=None,
            scale_method_columns_dict=None,

    ):

        self.node2vec = Embeddings.Node2Vec(
            walklen=n2v_walklen,
            epochs=n2v_epochs,
            return_weight=n2v_return_weight,
            neighbour_weight=n2v_neighbour_weight,
            threads=n2v_threads,
            keep_walks=n2v_keep_walks,
            w2v_size=w2v_size,
            w2v_negative=w2v_negative,
            w2v_iter=w2v_iter,
            w2v_batch_words=w2v_batch_words
        )

        if not scale_columns:
            scale_columns = numerical_features
        self.numerical_features_scaler = DfScaler(
            method=scale_method,
            columns=scale_columns,
            method_columns=scale_method_columns_dict
        )

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.use_global_info = use_global_info
        self.global_info_agg_function = global_info_agg_function
        self.entity_agg_function = entity_agg_function
        self.idf_log_base = idf_log_base
        self.scale_numerical = scale_numerical

        return

    def fit(
            self,
            data: pd.DataFrame,
    ):
        G = GraphTools.build_graph(
            data = data,
            entity_classes = self.categorical_features,
            use_global_info = self.use_global_info,
            global_info_agg_function = self.global_info_agg_function,
            entity_agg_function = self.entity_agg_function,
            idf_log_base = self.idf_log_base
        )

        self.node2vec.fit(G)
        if self.scale_numerical:
            self.numerical_features_scaler.fit(data)

        return



    def transform(
            self,
            data: pd.DataFrame,
    ):
        '''

        :param data: data frame containing all categorical and numerical features
        :return: np.array containing categorical embeddings and scaled numerical variables.
                 catgorical embeddings are return_array[0:embeddings_size]
                 numerical scaled values are return_array[embeddings_size:len(numerical_features)]
        '''
        row_to_nodes = GraphTools.df_to_nodes(data, self.categorical_features)

        categorical_embeddings = self.node2vec.transform(row_to_nodes)
        if self.scale_numerical:
            numerical_values = self.numerical_features_scaler.transform(data)
            numerical_values = numerical_values[self.numerical_features].values
        else:
            numerical_values = data[self.numerical_features].values

        return_array = np.concatenate([categorical_embeddings,numerical_values], axis = 1)

        return return_array

