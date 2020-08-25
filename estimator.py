from typing import List, Tuple, Dict, Set, Union, Optional, Callable
import pickle
import pandas as pd
import numpy as np
import unidecode
import warnings
import tqdm
import gensim
import keras

from umap import UMAP

from .Embeddings import Node2Vec
from .GraphTools import build_graph, df_to_node
from .Utils import DfScaler, Viz, linalg


#IMPLEMENT NORMALIZED SEARCH
#IMPLEMENT QUERY BY CLASS (CLOSER ENTITYES FROM A SPECIFIC CLASS GIVEN A QUERY)
#IMPLEMENT TOPN SEARCH
#IMPLEMENT SIMILARITY THRESHOLD SEARCH
#IMPLEMENT MAX VARAINCE VECTOR BY CLASS
#IMPLEMENT SVM FOR ENTITY SEPARATION AND NORMAL VECTOR DEFINITION
#STUDY FACEBOOKS PYTORCH BIGGRAPH SUPPORT VIABILTY

def preprocess_data(data, categorical_features= None):
    # preprocess feature names
    data = data.copy()
    if not categorical_features:
        categorical_features = data.dtypes[data.dtypes == 'object'].index

    data[categorical_features]


    data = data.rename(
        columns={i: unidecode.unidecode(' '.join(i.split()).replace(' ', '_').lower()) for i in categorical_features})
    categorical_features = [i.lower() for i in categorical_features]

    print(data.columns)

    categorical_features = [unidecode.unidecode(' '.join(i.split()).replace(' ', '_')) for i in categorical_features]
    # preprocess feature values


    data.loc[:, categorical_features] = data.loc[:, categorical_features].astype(str)

    for feature in categorical_features:
        data.loc[:, feature] = data.loc[:, feature].str.replace(pat=' +', repl=' ')
        data.loc[:,feature] = data.loc[:,feature].str.strip()
        data.loc[:, feature] = data.loc[:, feature].str.lower()

    return data

class Var2Node2Vec():

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
            entity_classes: List[str],
            feature_classes: Optional[List[str]] = None,
            numerical_features: List[str] = None,
            use_global_info: bool = True,
            global_info_agg_function: Union[str, Callable] = 'idf',
            entity_agg_function: Union[str, Callable] = 'sum',
            idf_log_base: int = 2,
            n2v_walklen=10,
            n2v_epochs=10,
            n2v_return_weight=1.,
            n2v_neighbor_weight=1.,
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
            categorical_centroids_method = 'mean',
            numerical_centroids_method = 'median'

    ):

        self.node2vec = Node2Vec(
            walklen=n2v_walklen,
            epochs=n2v_epochs,
            return_weight=n2v_return_weight,
            neighbor_weight=n2v_neighbor_weight,
            threads=n2v_threads,
            keep_walks=n2v_keep_walks,
            w2v_size=w2v_size,
            w2v_negative=w2v_negative,
            w2v_iter=w2v_iter,
            w2v_batch_words=w2v_batch_words
        )

        if not scale_columns:
            scale_columns = numerical_features

        if numerical_features:
            self.numerical_features_scaler = DfScaler.DfScaler(
                method=scale_method,
                columns=scale_columns,
                method_columns=scale_method_columns_dict
            )

        self.entity_classes = entity_classes
        self.feature_classes = feature_classes if feature_classes else []
        self.numerical_features = numerical_features
        self.use_global_info = use_global_info
        self.global_info_agg_function = global_info_agg_function
        self.entity_agg_function = entity_agg_function
        self.idf_log_base = idf_log_base
        self.scale_numerical = scale_numerical
        self.categorical_centroids_method = categorical_centroids_method
        self.numerical_centroids_method = numerical_centroids_method

        return

    def build_graph(self,data):
        G = build_graph(
            data = data[self.entity_classes+self.feature_classes],
            entity_classes = self.entity_classes,
            feature_classes= self.feature_classes,
            use_global_info = self.use_global_info,
            global_info_agg_function = self.global_info_agg_function,
            entity_agg_function = self.entity_agg_function,
            idf_log_base = self.idf_log_base
        )
        return G
        
    def fit(
            self,
            data: pd.DataFrame,
    ):

        data = data.copy()

        assert data.__class__ == pd.DataFrame
        if self.numerical_features:
            if self.scale_numerical:
                self.numerical_features_scaler.fit(data)

        G = self.build_graph(data)
            
        self.node2vec.fit(G)

        self.categorical_class_map = {
            class_: {
                entity: vector for entity, vector in self.node2vec.model.wv.vocab.items() if
                '{}__'.format(class_) in entity
            }
            for class_ in self.feature_classes + self.entity_classes
        }

        self.update_categorical_class_centroids(
            model= self.node2vec.model.wv,
            method = self.categorical_centroids_method,
            categorical_class_map = self.categorical_class_map
        )


        self.nv = self.node2vec.model.wv #access model wv methods

        return self
    
    def keep_fitting(self, data):

        data = data.copy()

        assert data.__class__ == pd.DataFrame
        if self.numerical_features:
            if self.scale_numerical:
                self.numerical_features_scaler.fit(data)

        G = self.build_graph(data)

        self.node2vec.keep_fitting(G)

        self.categorical_class_map = {
            class_: {
                entity: vector for entity, vector in self.node2vec.model.wv.vocab.items() if
                '{}__'.format(class_) in entity
            }
            for class_ in self.feature_classes + self.entity_classes
        }

        self.update_categorical_class_centroids(
            model=self.node2vec.model.wv,
            method=self.categorical_centroids_method,
            categorical_class_map=self.categorical_class_map
        )

        self.nv = self.node2vec.model.wv  # access model wv methods

        return

    def transform(
            self,
            data: pd.DataFrame,
            categorical_embeddings_agg = 'sum',
            handler = 'warn',
            features_subset = None,
            normalize = False
    ):
        """
        :param featues_subset: subset of features to compose the embedding.
        composition is done by "categorical_embeddings_agg" operation.
        None will return embeddings of all self.categorical_features (unknow values
        will be filled with generical class vector)
        
        :param categorical_embeddings_agg: aggregation function of embeddings of 
        each feature
        
        :param data: data frame containing all categorical and numerical features
        
        :param categorical_embeddings_agg: how to aggregate entity embeddings
        
        :return: np.array containing categorical embeddings and scaled numerical variables.
                 catgorical embeddings are return_array[0:embeddings_size]
                 numerical scaled values are return_array[embeddings_size:len(numerical_features)]
        """
        data = data.copy()

        if not features_subset:
            categorical_features_subset = self.feature_classes + self.entity_classes
            numerical_features_subset = self.numerical_features
        else:
            categorical_features_subset = [i for i in self.feature_classes + self.entity_classes if i in features_subset]
            if self.numerical_features:
                numerical_features_subset = [i for i in self.numerical_features if i in features_subset]
            else:
                numerical_features_subset = []

        if self.numerical_features:
            if self.scale_numerical:
                numerical_values = self.numerical_features_scaler.transform(data)
                numerical_values = numerical_values[numerical_features_subset].values
            else:
                numerical_values = data[numerical_features_subset].values

        categorical_embeddings = self.node2vec.transform(
            data,
            entity_classes = categorical_features_subset,
            agg = categorical_embeddings_agg,
            handler = handler
            )

        if self.numerical_features:
            return_array = np.concatenate([categorical_embeddings,numerical_values], axis = 1)
        else:
            return_array = categorical_embeddings

        if normalize:
            return_array = linalg.normalize(return_array)
        return return_array

    def umap_fit(self,X,y = None, **umapargs):
        self.umap = UMAP(**umapargs)
        self.umap.fit_transform(X,y)
        return self

    def umap_transform(self,X):
        embeddings = self.umap.transform(X)
        return embeddings

    def update_categorical_class_centroids(self,model,categorical_class_map, method = 'mean'):
        """
        :param method: numpy aggregation function to calculate centroids. default is mean
        :return:
        """
        for class_ in categorical_class_map:
            class_entities = [ent for ent in categorical_class_map[class_]]
            class_centroid = getattr(self.node2vec.model.wv[class_entities],method)(axis=0)
            model.add(class_, class_centroid, replace = True)

        return

    def predict_feature(
            self,
            query_data,
            feature,
            features_subset = [],
            categorical_embeddings_agg = 'sum',
            handler = 'warn'
    ):
        '''
        predicts top_n similar pred_features. Can be used to fill missing data.
        requires C compiler
        :return:
        '''
        assert all([i in self.entity_classes+self.feature_classes for i in features_subset])
        query_data = query_data.copy()

        query_data = query_data[features_subset]
        query_tokens = self.df_to_node(query_data,query_data.columns)
        query_tokens = self.check_nodes(query_tokens,handler = handler)
        query_embeddings = np.array([getattr(self.node2vec.model.wv[i],categorical_embeddings_agg)(axis = 0) for i in query_tokens])

        search_space_tokens = [i for i in self.categorical_class_map[feature]]
        search_space_embeddings = np.array([self.node2vec.model.wv[i] for i in search_space_tokens])
        search_space_index_map = {i: v for i, v in enumerate(search_space_tokens)}        
        
        query_result = self.predict_sim(
            query_embeddings,
            search_space_embeddings,
            top_n = 1,
            return_ranking = False
        )

        nonzero_query = query_result.nonzero()[1]        
        preds = [search_space_index_map[i] for i in nonzero_query]
        
        preds = np.array(preds)
        return preds

    def predict_sim(
            self,
            query_embeddings,
            search_space_embeddings,
            top_n = 5,
            min_sim = 0.0,
            return_ranking = True
    ):
        '''
        predicts top_n similar pred_features. Can be used to fill missing data.
        requires C compiler
        :return:
        '''

        print('calculating similarity matrices...')
        query_result = linalg.query(query_embeddings, search_space_embeddings, top_n, min_sim,return_ranking)
        
        #returns
        
        return query_result

    def check_nodes(self, nodes, handler = 'warn'):
        assert handler in ['warn', 'handle', 'raise']        
        nodes = np.array(list(nodes))
        nodes_set = set(nodes.flatten())

        entities_set = set([ent for ent in self.node2vec.model.wv.vocab.keys()])
        missing_nodes = nodes_set - entities_set

        for i in tqdm.tqdm(missing_nodes):

            if handler == 'warn':
                class_ = i.split('__')[0]
                nodes[nodes == i] = class_
                warnings.warn(
                    '{} is an unknown entity. class {} embedding returned instead'.format(i, class_))
            elif handler == 'handle':
                nodes[nodes == i] = i.split('__')[0]
            elif handler == 'raise':
                raise KeyError('unknown entity in: {}'.format(i))

        return nodes.tolist()

    def df_to_node(self,data,entity_classes):
        return df_to_node(data,entity_classes)

    def get_embeddings_from_array(self,array, agg = 'sum', normalize = False):

        embeddings = np.array([getattr(self.nv[i],agg)(axis=0) for i in array])
        if normalize == True:
            embeddings = linalg.normalize(embeddings)
        return embeddings


class Var2Vec():

    @classmethod
    def load(cls, loading_path, **pickleargs):
        with open(loading_path, 'rb') as file:
            loaded_model = pickle.load(file, **pickleargs)
        return loaded_model

    def save_full(self, saving_path, **pickleargs):
        with open(saving_path, 'wb') as file:
            pickle.dump(self, file, **pickleargs)

    def __init__(self,data, features, model = 'Word2Vec', categorical_centroids_method = np.median, corpus_file=None, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3,batch_words = 100, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, compute_loss=False, callbacks=(),
                 max_final_vocab=None):

        if features.__class__ not in [list,tuple,set]:
            features = [features]

        if not model in ["FastText","Word2Vec"]:
            raise ValueError('model should be one of ["FastText","Word2Vec"]')
        if model == 'FastText':
            model = gensim.models.FastText
        if model == 'Word2Vec':
            model = gensim.models.Word2Vec
        '''
        all params description are avalible in gensim.model.Word2Vec documentation
        '''
        sentences = df_to_node(data,features)
        sentences = sentences.values.tolist()
        self.model = model(sentences, corpus_file=corpus_file, size=size, alpha=alpha, window=window, min_count=min_count,
                 max_vocab_size=max_vocab_size, sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
                 sg=sg, hs=hs, negative=negative, ns_exponent=ns_exponent, cbow_mean=cbow_mean, iter=iter, null_word=null_word,
                 trim_rule=trim_rule, sorted_vocab=sorted_vocab, callbacks=callbacks,
                 batch_words = batch_words)
        #compute_loss = compute_loss, max_final_vocab=max_final_vocab

        self.size = size
        self.entity_types = features
        self.categorical_centroids_method = categorical_centroids_method

        self.categorical_class_map = {
            class_: {
                entity: vector for entity, vector in self.model.wv.vocab.items() if
                '{}__'.format(class_) in entity
            }
            for class_ in self.entity_types
        }

        self.update_categorical_class_centroids(
            model=self.model.wv,
            method= self.categorical_centroids_method,
            categorical_class_map=self.categorical_class_map
        )

    def transform(self, data,features, handler = 'raise', normalize = False, agg = None):

        if features.__class__ not in [list,tuple,set]:
            features = [features]

        if not handler in ["raise","warn","handle"]:
            raise ValueError('handler should be in ["raise","warn","handle"]')

        words = df_to_node(data,features, sep = '__')

        for word in words:
            if word.__class__ == str:
                word = [word]

        embeddings = []
        for i in tqdm.tqdm(words):
            sentence = []
            for j in i:
                try:
                    sentence.append(self.model.wv[j])
                except KeyError:
                    if handler == 'raise':
                        raise
                    elif handler == 'warn':
                        warnings.warn('{} not in vocabulary. Vector of zeros was passed'.format(j))
                        sentence.append(np.zeros((self.size)))
                    elif handler == 'handle':
                        sentence.append(np.zeros((self.size)))


            sentence = np.array(sentence)
            if agg:
                sentence = getattr(sentence, agg)(axis=0)
            else:
                pass
            if normalize == True:
                sentence = linalg.normalize(sentence)

            embeddings.append(sentence)

        embeddings = np.array(embeddings)
        if agg:
            embeddings = embeddings.reshape(embeddings.shape[0],embeddings.shape[2])

        return embeddings



    def update_categorical_class_centroids(self,model,categorical_class_map, method = np.median):
        """
        :param method: numpy aggregation function to calculate centroids. default is mean
        :return:
        """
        for class_ in categorical_class_map:
            class_entities = [ent for ent in categorical_class_map[class_]]
            if method.__class__ == str:
                class_centroid = getattr(model[class_entities],method)(axis=0)
            else:
                class_centroid = method(model[class_entities],axis=0)
            model.add(class_, class_centroid, replace = True)

    def similar_by_class(self, entity1, features, topn = 10):
        """
        queries entities similar to entity1 on subset of features
        :param entity1: entity to search
        :param features: features subset
        :param topn: top n similar entities
        :return: list of tuples (entity, similarity)
        """

        if features.__class__ not in [list, tuple, set]:
            features = [features]

        valid_features = list(self.categorical_class_map.keys())
        for feature in features:
            if not feature in valid_features:
                raise ValueError('feature must be one of {} not "{}"'.format(valid_features,feature))

        entities_list = [word for word in self.categorical_class_map[feature] for feature in features]
        sims = np.array([self.model.wv.similarity(entity1, entity) for entity in tqdm.tqdm(entities_list)]).flatten()
        n_tops_idx = sims.argsort()[-topn:][::-1]
        item_sims = list(zip(np.array(entities_list)[n_tops_idx],sims[n_tops_idx]))
        return item_sims

class SupervisedEmbedder(keras.Model):

    def __init__(self):
        super().__init__()
        return
    def build_model(self, output_dim, regreession_type = 'classification' ,**layerargs):
        """
        builds the model layers
        :param layerargs: layer args from tf.keras.layers.Dense
        :return: None
        """
        if regreession_type.lower() == 'classification':
            pass
        elif regreession_type.lower() == 'regression':
            pass
        else:
            raise TypeError('regression_type should be one of ["classification, regression"] not {}'.format(regreession_type))

        entry_layer = keras.layers.Dense(**layerargs)
        out_layer = keras.layers.Dense(output_dim)


import gensim


class OneVsOneEmbeddings:

    def __int__(self):
        return

    def fit(self, data, entity, explainer_sizes, exp_gensim_args={}):
        explainers = {}
        for explainer_feature in explainer_sizes:
            if not explainer in exp_gensim_args:
                exp_gensim_args[explainer] = {}
            phrases = self._make_phrases(data, entity, explainer)
            explainers[explainer_feature] = self._fit_one_explainer(
                phrases, explainer_sizes[explainer_feature], **exp_gensim_args[explainer])

        self.explainers = explainers
        self.entity = entity
        self.explainer_sizes = explainer_sizes
        return self

    def transform(entity):
        if entity.__class__ in [list, tuple, set]:
            preds = []
            for ent in entity:
                preds.append(self._transform_one_entity(entity))
        else:
            preds = self._transform_one_entity(entity)
        return preds

        def _transform_one_entity(self, entity):
            embeddings_dict = {}

        for explainer in self.explainers:
            embeddings_dict[explainer] = self.explainers[explainer].wv[entity]
        return embeddings_dict

    def _make_phrases(self, data, entity, explainer):
        data = data[[entity, explainer]].copy()
        data = data.astype(str)
        phrases = data.groupby(explainer)[entity].apply(
            lambda x: ' '.join(list(x))).values.tolist()
        return phrases

        def _fit_one_explainer(self, phrases, size, **gensimargs):

        if not model in ["FastText", "Word2Vec"]:
            raise ValueError(
                'model should be one of ["FastText","Word2Vec"]')
        if model == 'FastText':
            model = gensim.models.FastText
        if model == 'Word2Vec':
            model = gensim.models.Word2Vec

        explainer_model = model(phrases, size=size, **gensimargs)
        return explainer_model