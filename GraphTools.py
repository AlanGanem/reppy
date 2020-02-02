# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 01:43:45 2020

@author: User Ambev
"""

import itertools
import numpy as np
import pandas as pd
import networkx as nx
import tqdm
from typing import List, Tuple, Dict, Set, Union, Optional, Callable


def inverse_logn_frequency(X: float, base: float) -> float:
    if base > 0:
        result = np.log(1 + X) / np.log(base)
    else:
        result = X

    return 1 / result


def build_graph(
        data: pd.DataFrame,
        entity_classes: List[str],
        use_global_info: bool = True,
        global_info_agg_function: Union[str, Callable] = 'idf',
        entity_agg_function: Union[str, Callable] = 'sum',
        idf_log_base: int = 2,

) -> nx.Graph:
    # _LOCALINFO is the column name on local (edges) information, usualy the frequency of links between nodes (tf)
    # _GLOBALINFO is the column name of global (node) information accross the network, usually the inverse of frequency log (idf)
    # _INDEXINFO is the column containing the index value (edge) used to map from graph back to tabular data
    SPECIAL_NAMES = ['_LOCALINFO', '_INDEXINFO', '_GLOBALINFO']
    # VALID_DEFAULT_GLOBAL_INFO= ['tf', 'tf_idf']

    # perform input checks
    var_checks(
        data=data,
        SPECIAL_NAMES=SPECIAL_NAMES,
        entity_classes=entity_classes
    )

    # assert entities are strings
    data.loc[:, entity_classes] = data.loc[:, entity_classes].astype(str)

    # create combinations to build multipartite(by class) entities graph
    combinations = list(itertools.combinations(entity_classes, 2))
    data['_LOCALINFO'] = 1

    # create global_information df
    global_property_df = get_global_entity_properties(
        data=data,
        entity_classes=entity_classes,
        global_info_agg_function=global_info_agg_function,
        idf_log_base=2
    )

    # create edges between entities assigning attirbute "weight" as the product of both edges idf and _LOCALINFO of edge frequency
    # edge values are going to be normalized (locally) in the Node2Vec during transition probability calculation
    # edges nodes be labelled as follows : ("<entity_class>__<entity>")

    edges = make_edges(
        data=data,
        combinations=combinations,
        global_property_df=global_property_df,
        use_global_info=use_global_info,
        entity_agg_function=entity_agg_function,
    )

    # create empty graph
    G = nx.Graph()
    # ad edges from edge list
    G.add_edges_from(edges)

    attrs = {}
    for node in G.nodes:
        attrs[node] = {}
        class_, value = node.split('__')
        attrs[node]['class'] = class_
        attrs[node]['value'] = value
        attrs[node]['global_info'] = global_property_df.loc[(class_,value),'_GLOBALINFO']

    nx.set_node_attributes(G,attrs)

    return G


def get_local_edge_properties(
        data: pd.DataFrame,
        combination: Tuple[str, str],
        dummy_count_col: str = '_LOCALINFO',
        entity_agg_function: Union[Callable, str] = 'sum'
) -> pd.DataFrame:
    cter = data[list(combination) + [dummy_count_col]].groupby(list(combination), as_index=False)

    if entity_agg_function.__class__ == str:
        cter = getattr(cter, entity_agg_function)()

    # custom entity_agg_function takes a dataframe of single edges and outputs a float
    elif callable(entity_agg_function):
        cter = cter.apply(entity_agg_function)

    if entity_agg_function == 'binary':
        cter[dummy_count_col] = 1

    cter['_INDEXINFO'] = cter.index

    return cter


def make_edges(
        data,
        combinations: List[tuple],
        global_property_df: pd.DataFrame,
        use_global_info: bool = True,
        entity_agg_function: Union[str, Callable] = 'sum',

) -> List[Tuple[str, str, Dict[str, float]]]:
    edges = []
    for combination in tqdm.tqdm(combinations):

        local_edge_properties_df = get_local_edge_properties(
            data=data,
            combination=combination,
            dummy_count_col='_LOCALINFO',
            entity_agg_function=entity_agg_function
        )

        if use_global_info == True:
            merge1 = pd.merge(local_edge_properties_df, global_property_df.loc[combination[0]], right_index=True,
                              left_on=combination[0])
            merge2 = pd.merge(merge1, global_property_df.loc[combination[1]], right_index=True, left_on=combination[1])
            local_edge_properties_df['edge_weight'] = merge2['_LOCALINFO'] * merge2['_GLOBALINFO_x'] * merge2[
                '_GLOBALINFO_y']
        else:
            local_edge_properties_df['edge_weight'] = local_edge_properties_df['_LOCALINFO']

        zipper = [v.to_list() for i, v in
                  local_edge_properties_df[list(combination) + ['edge_weight', '_LOCALINFO', '_INDEXINFO']].iterrows()]

        edges_ = [
            (
                '{}__{}'.format(combination[0], u),
                '{}__{}'.format(combination[1], v),
                {
                    'weight': w,
                    'counts': c,
                    'df_index': i,
                    # 'global_property':(g0,g1) _GLOBALINFO,
                    # 'local_property(link)': _LOCALINFO //Future implementation*
                }  # node attributes should be passed inside this dict
            )
            for u, v, w, c, i in tqdm.tqdm(zipper)
        ]

        edges += edges_

    return edges


def var_checks(
        data,
        SPECIAL_NAMES,
        entity_classes,

) -> None:
    # check for entity classes that are note str type
    wrong_types = [i for i in entity_classes if data[i].dtypes != 'object']
    ok_types = [i for i in entity_classes if data[i].dtypes == 'object']
    wrong_ok_types = [col for col in ok_types if not all([i == str for i in map(type, data[col])])]
    wrong_types = wrong_ok_types + wrong_types

    if wrong_types:
        raise ValueError(
            '"data" "entity_classes" should contain only strings. Cast {} values to "str" to proceed'.format(
                wrong_types))

    # check wheter dataframe has columns with special names
    check_special_names = [i for i in SPECIAL_NAMES if i in data.columns]
    if check_special_names:
        raise ValueError(
            '"data" should not contain {} as columns. Rename the columns to proceed'.format(check_special_names))

    # valid edge_weights are one of these \or a custom function for global (idf) and local (tf) edge information\*to implement
    # if edge_weights not in VALID_DEFAULT_GLOBAL_INFO:
    #    raise ValueError('"edge_weights" must be one of {}'.format(VALID_DEFAULT_GLOBAL_INFO))

    # check if theres blank spaces or dunders in entities and classes
    invalid_entity_classes = [i for i in entity_classes if (' ' in i) or ('__' in i)]
    if invalid_entity_classes:
        raise ValueError(
            'an entity name should not contain blank spaces or double underscores "__". rename "{}"'.format(
                invalid_entity_classes))

    entities_check = set(data[entity_classes].values.flatten())
    invalid_entities = [i for i in entities_check if (' ' in i) or ('__' in i)]
    if invalid_entities:
        raise ValueError(
            'an entity name should not contain blank spaces or double underscores "__". rename "{}"'.format(
                invalid_entities))


def get_global_entity_properties(
        data: pd.DataFrame,
        entity_classes: List[str],
        global_info_agg_function: Union[str, Callable] = 'idf',
        idf_log_base: float = 2
) -> pd.DataFrame:
    if global_info_agg_function == 'idf':
        # create inverse document frequency (global property of node) series by entity
        val_counts = {e_class: data[e_class].value_counts().apply(lambda x: inverse_logn_frequency(x, idf_log_base)) for
                      e_class in entity_classes}
        globalinfo = pd.concat(val_counts.values(), keys=val_counts.keys())
        globalinfo = pd.DataFrame(globalinfo)
        globalinfo.columns = ['_GLOBALINFO']


    elif callable(global_info_agg_function):

        # global_info_agg_function takes a DataFrame (item of df.Groupby iterator) containing all incidences
        # of one single entity(node) and performs some operation.
        # it can also perform operations with featuers that are not necessarily in "entity_classes"
        # the outout of function SHOULD be a float

        global_info = {e_class: data.groupby([e_class]).apply(global_info_agg_function) for e_class in entity_classes}
        # assert output is series
        assert global_info.__class__ == pd.Series
        globalinfo = pd.DataFrame(globalinfo)
        globalinfo.columns = ['_GLOBALINFO']

    else:
        raise ValueError('global_info_agg_function must be "idf" or callable, not {}'.format(global_info_agg_function))

    return globalinfo

def df_to_node(data, entity_classes):

    SPECIAL_NAMES = ['_LOCALINFO', '_INDEXINFO', '_GLOBALINFO']
    var_checks(
            data,
            SPECIAL_NAMES,
            entity_classes,
    )

    data = data.copy()
    for col in entity_classes:
        data.loc[:,col] = '{}__'.format(col) + data[col]
    nodes = data[entity_classes].apply(tuple,axis = 1)
    return nodes

