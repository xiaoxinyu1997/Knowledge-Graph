from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import logging
import numpy as np
# import nni

LOG = logging.getLogger('sklearn_classification')

def load_data():

    data = pd.read_csv('../train_for_relations.csv')
    X = data.iloc[:, :1668]
    Y = data.iloc[:, 1668:]

    X_train, X_test, Y_train ,Y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, X_test, Y_train, Y_test
    
def get_default_parameters():
    '''get default parameters'''
    params = {
        'criterion': 'gini',
        'splitter': 'best',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.0,
        'max_features': None,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'ccp_alpha': 0.0,
    }
    return params

def get_model(PARAMS):
    '''Get saved_model according to parameters'''
    model = DecisionTreeClassifier()
    model.criterion = PARAMS.get('criterion')
    model.splitter = PARAMS.get('splitter')
    model.max_depth = PARAMS.get('max_depth')
    model.min_samples_split = PARAMS.get('min_samples_split')
    model.min_weight_fraction_leaf = PARAMS.get('min_weight_fraction_leaf')
    model.max_features = PARAMS.get('max_features')
    model.max_leaf_nodes = PARAMS.get('max_leaf_nodes')
    model.min_impurity_decrease = PARAMS.get('min_impurity_decrease')
    model.ccp_alpha = PARAMS.get('ccp_alpha')

    return model

def run(X_train, X_test, y_train, y_test, model):
    '''Train saved_model and predict result'''
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    LOG.debug('score: %s', score)
    print(score)
    # nni.report_final_result(score)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    try:
        # # get parameters from tuner
        # RECEIVED_PARAMS =  # nni.get_next_parameter()
        # LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        # PARAMS.update(RECEIVED_PARAMS)
        # LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(X_train, X_test, y_train, y_test, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
