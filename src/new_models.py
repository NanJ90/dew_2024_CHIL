from collections import Counter
import copy
import inspect
from itertools import combinations, product
import json
from concurrent.futures import ProcessPoolExecutor
import os
import pickle
import sys
from turtle import distance
from typing import Iterable, Mapping, Tuple, Union
import warnings

# from src.new_base import BaseNonmissingSubspaceClassifier
# from src.utils import get_classification_metrics
warnings.filterwarnings("ignore")

# from deslib_missingness.des import KNORAU, meta_des
# from deslib_missingness.base import DroppingClassifier
# from deslib_missingness.static import Oracle
# from deslib.dcs import LCA, MCB
# from deslib.des import KNOP, KNORAE, KNORAU
from imblearn.under_sampling import RandomUnderSampler
import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist
from sklearn.experimental import enable_iterative_imputer, enable_hist_gradient_boosting
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaseEnsemble, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.impute._base import _BaseImputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss, max_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import torch
from torch import nn, softmax
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from tqdm import tqdm
import xgboost as xgb

sys.path.append('.')

from new_base import BaseInheritanceClassifier, BaseNonmissingSubspaceClassifier,\
    ClassifierWithImputation, InheritanceCompatibleClassifier
from data_loaders import Dataset, MedicalDataset, PlacentalAnalytesTests
from data_loaders import load_numom2b_analytes_dataset
from data_loaders import load_wisconsin_diagnosis_dataset, load_wisconsin_prognosis_dataset
from utils import Test, get_n_tests_missing, label_encoded_data,\
    find_optimal_threshold, get_device, get_prediction_method, get_classification_metrics
from utils import get_sample_indices_with_optional_tests, plot_prediction_errors
from utils import get_prediction_method


__all__ = [
    'FullInheritance', 'StackedHeterogenousClassifier',
    'StackedHeterogenousRegressor', 'StackedInheritanceEstimator',
    'StackedParametricClassifier'
]


# class NonmissingSubspaceInheritanceClassifier(BaseNonmissingSubspaceClassifier):
#     def __init__(
#         self,
#         data: Union[Dataset, MedicalDataset],
#         base_estimator: BaseEstimator,
#         base_estimator_params: dict = {},
#         prediction_method: str = 'auto',
#         voting='soft',
#         use_optimal_threshold=False,
#         threshold=0.5
#     ):
#         super().__init__(
#             data=data,
#             base_estimator=base_estimator,
#             base_estimator_params=base_estimator_params,
#             prediction_method=prediction_method,
#             use_optimal_threshold=use_optimal_threshold,
#             threshold=threshold
#         )
#         self.name = type(base_estimator).__name__ + 'NonmissingSubspaceInheritanceClassifier'
#         # just hard-code for now to be compatible with 
#         # InheritanceCompatibleEstimator. Can refactor later.
#         self.prediction_method = get_prediction_method(base_estimator())
#         self.voting = voting

#     def fit(self, X=None, y=None, cv_fold_index=None):
#         if cv_fold_index is not None:
#             super().fit_node_estimators(cv_fold_index)
#             self._classes = list(set(self.data[0][0][self.data.target_col]))
#         elif X is not None and y is not None:
#             super().fit_node_estimators(X, y)
#             self._classes = list(set(y))
#         else:
#             raise Exception('either provide CV fold index or (X,y)')
#         self.is_fit = True

#     @property
#     def classes_(self):
#         try: 
#             return self._classes
#         except:
#             raise Exception('Model has not been fit.')

#     def predict_proba(self, X=None, cv_fold_index=None):
#         if cv_fold_index is not None:
#             test = self.data[cv_fold_index][1]
#             X_test = test.drop(self.target_col, axis=1)
#         elif X is not None:
#             X_test = X
#         else:
#             raise Exception('Please provide either `X` input array or a CV fold index')
    
#         if isinstance(X_test, pd.DataFrame):
#             cols = [c for c in X_test.columns if c != self.target_col]

#         assert len(X_test.shape) == 2

#         predictions = []
#         nodes_seen = []
        
#         for i in tqdm(range(X_test.shape[0])):

#             sample = X_test.iloc[i,:].astype(np.float32)
#             raw_sample = sample.copy()
#             indices = np.ravel(np.argwhere(np.isnan(np.array(sample))))
#             valid_indices = [x for x in range(len(sample)) if x not in indices]
#             level = len(set(list(self.feature_to_test_map.values()))) \
#                 - get_n_tests_missing(cols, self.feature_to_test_map, indices)
#             sample_1d = np.array([
#                 sample[i] for i in range(len(sample))
#                 if i not in indices
#             ])
#             # sample = sample_1d.reshape(1,-1).astype(np.float32)
#             # sample = sample.dropna(axis=1)
#             sample = sample.iloc[list(valid_indices)].to_frame().T
#             # print(sample)
#             # print(type(sample))
#             # print(sample.shape)
#             # create node: pair[level, tuple[relevant indices]]
#             node = (
#                 level, 
#                 tuple(
#                     # [x for x in range(X_test.shape[1]) if x not in indices]
#                     valid_indices
#                 )
#             )
#             # print('node features: ' + str(self.dag.nodes[node]['features']))
#             # print('initial sample nonmissing features: ' + str(list(sample.columns)))
#             nodes_seen.append(node)
#             model = self.dag.nodes[node]['model']
#             # print(sample)
            
#             # ancestor_predictions includes prediction at current node as well
#             # so initialize ancestor predictions with current node prediction
            
#             # output a single value in [0,1]
#             try:
#                 first_prediction = model.predict_proba(sample)[0][1]
#             except:
#                 first_prediction = model.predict_proba(sample)[0][0]
#             self.dag.nodes[node]['passthrough_predictions'][i] = first_prediction
            
#             if self.voting == 'soft':
#                 ancestor_predictions = [first_prediction]
#             else:
#                 first_prediction = 0 if first_prediction < model.threshold else 1
#                 ancestor_predictions = [first_prediction]

#             df = self.data[0][0]
#             df = df[[c for c in df.columns if c != self.data.target_col]]

#             if level != 0:
#                 for a in nx.ancestors(self.dag, node):
#                     current_features = self.dag.nodes[a]['features']
#                     # print('new node features: ' + str(list(current_features)))
#                     current_indices = np.array([
#                         df.columns.get_loc(c) for c in current_features
#                     ]).astype(np.int32)
#                     current_sample = raw_sample.iloc[current_indices].to_frame().T
#                     # print('selected features: ' + str(list(current_sample.columns)))
#                     # print(current_sample)
#                     ancestor_model = self.dag.nodes[a]['model']
#                     # predict with ancestor model 
#                     # and add to list of ancestor predictions
#                     current_prediction = getattr(
#                         ancestor_model, 'predict_proba'
#                     )(current_sample)[0][1] # output a single value in [0,1]
#                     self.dag.nodes[a]['passthrough_predictions'][i] = current_prediction
#                     if self.voting == 'soft':
#                         ancestor_predictions.append(current_prediction)
#                     else:
#                         prediction = 0 if current_prediction < ancestor_model.threshold else 1
#                         ancestor_predictions.append(prediction)
#             prediction = sum(ancestor_predictions) / len(ancestor_predictions)

#             predictions.append(prediction)
#             self.dag.nodes[node]['predictions'][i] = prediction
            
#         predictions = [[1 - p, p] for p in predictions]
#         return np.vstack(predictions)

#     def predict(self, X=None, cv_fold_index=None):
#         if X is not None:
#             probas = self.predict_proba(X=X)[:,1]
#         elif cv_fold_index is not None:
#             probas = self.predict_proba(cv_fold_index=cv_fold_index)[:,1]
#         predictions = [1 if p > self.threshold else 0 for p in probas]
#         return predictions

#     def get_nodewise_classification_metrics(self, y_test=None, cv_fold_index=None):
#         if cv_fold_index is not None:
#             test = self.data[cv_fold_index][1]
#             y_test = test[self.target_col]
#         elif y_test is None:
#             raise Exception('Please provide either `y_test` or `cv_fold_index`')

#         for node in self.dag.nodes:
#             try:
#                 probs_dict = self.dag.nodes[node]['passthrough_predictions']
#                 proba_predictions = list(probs_dict.values())
#                 test_indices = list(probs_dict.keys())
#                 ground_truth = np.array(y_test)[test_indices]
#                 # t = find_optimal_threshold(proba_predictions, ground_truth)
#                 # predictions = [1 if p > t else 0 for p in proba_predictions]
#                 t = self.dag.nodes[node]['optim_threshold']
#                 predictions = [1 if p > t else 0 for p in proba_predictions]
#                 predictions_baseline_thresh = [1 if p > 0.5 else 0 for p in proba_predictions]
#                 metrics = get_classification_metrics(predictions, ground_truth)
#                 metrics_baseline = get_classification_metrics(
#                     predictions_baseline_thresh, ground_truth
#                 )
#                 roc_auc_value = roc_auc_score(ground_truth, proba_predictions)
#                 metrics['roc_auc'] = roc_auc_value
#                 accuracy = 1 - (np.sum(np.logical_xor(predictions, ground_truth)) / len(predictions))
#                 metrics['accuracy'] = accuracy
#                 accuracy_baseline_thresh = 1 - (np.sum(
#                     np.logical_xor(
#                         predictions_baseline_thresh, ground_truth)
#                     ) / len(predictions)
#                 )
#                 metrics_baseline['accuracy_0.5_threshold'] = accuracy_baseline_thresh
#                 metrics_baseline['roc_auc'] = roc_auc_value
#                 self.dag.nodes[node]['individual_classification_metrics'] = metrics
#                 self.dag.nodes[node]['individual_classification_metrics 0.5 threshold'] = metrics_baseline
#                 # print('level: ' + str(node[0]) + ' | gmean sensitivity vs spec: ' + str(metrics['gmean_sens_spec']))
#                 print('level: ' + str(node[0])
#                     + ' | metrics with optimal threshold: ' + str(metrics)
#                     + ' \n metrics with 0.5 threshold: ' + str(metrics_baseline)
#                 )
#             except:
#                 continue


# class NonmissingSubspaceStratificationClassifier(BaseNonmissingSubspaceClassifier):
#     def __init__(
#         self,
#         data: Union[Dataset, MedicalDataset],
#         base_estimator: BaseEstimator,
#         base_estimator_params: dict = {},
#         prediction_method: str = 'auto',
#     ):
#         super().__init__(
#             data=data,
#             base_estimator=base_estimator,
#             base_estimator_params=base_estimator_params,
#             prediction_method=prediction_method
#         )
#         self.name = type(base_estimator).__name__ + 'NonmissingSubspaceStratificationClassifier'
#         # just hard-code for now to be compatible with 
#         # InheritanceCompatibleEstimator. Can refactor later.
#         self.prediction_method = 'predict'

#     def fit(self, cv_fold_index=0):
#         super().fit_node_estimators(cv_fold_index)
#         self.is_fit = True

#     @property
#     def classes_(self):
#         try: 
#             return list(set(self.data.y_train))
#         except:
#             raise Exception('Model has not been fit.')

#     def predict(self, X=None, cv_fold_index=None):
#         if cv_fold_index is not None:
#             test = self.data[cv_fold_index][1]
#             X_test = test.drop(self.target_col, axis=1)
#         elif X is not None:
#             X_test = X
#         else:
#             raise Exception('Please provide either `X` input array or a CV fold index')

    
#         if isinstance(X_test, pd.DataFrame):
#             cols = [c for c in X_test.columns if c != self.target_col]
#             # X_test = X_test.to_numpy()

#         assert len(X_test.shape) == 2

#         predictions = []
#         nodes_seen = []
        
#         for i in tqdm(range(X_test.shape[0])):

#             sample = X_test.iloc[i,:].astype(np.float32)
#             raw_sample = sample.copy()
#             indices = np.ravel(np.argwhere(np.isnan(np.array(sample))))
#             valid_indices = [x for x in range(len(sample)) if x not in indices]
#             level = len(set(list(self.feature_to_test_map.values()))) \
#                 - get_n_tests_missing(cols, self.feature_to_test_map, indices)
#             sample_1d = np.array([
#                 sample[i] for i in range(len(sample))
#                 if i not in indices
#             ])
#             sample = sample.iloc[list(valid_indices)].to_frame().T
            
#             # create node: pair[level, tuple[relevant indices]]
#             node = (
#                 level, 
#                 tuple(
#                     # [x for x in range(X_test.shape[1]) if x not in indices]
#                     valid_indices
#                 )
#             )
#             nodes_seen.append(node)
#             model = self.dag.nodes[node]['model']
            
#             # ancestor_predictions includes prediction at current node as well
#             # so initialize ancestor predictions with current node prediction
            
#             # output a single value in [0,1]
#             prediction = getattr(model, self.prediction_method)(sample)[0][1]

#             predictions.append(prediction)
#             self.dag.nodes[node]['predictions'][i] = prediction
            
#         return np.array(predictions)


# class FullInheritance(BaseInheritanceClassifier):
#     def __init__(
#         self,
#         data: Union[Dataset, MedicalDataset],
#         base_estimator: BaseEstimator,
#         base_estimator_params: dict = {},
#         prediction_method: str = 'auto'
#     ):
#         super().__init__(
#             data=data,
#             base_estimator=base_estimator,
#             base_estimator_params=base_estimator_params,
#             prediction_method=prediction_method
#         )
#         self.name = type(base_estimator).__name__ + '_FullInheritance'
#         # just hard-code for now to be compatible with 
#         # InheritanceCompatibleEstimator. Can refactor later.
#         self.prediction_method = 'predict'

#     def fit(self):
#         super().fit_node_estimators()
#         self.is_fit = True

#     @property
#     def classes_(self):
#         try: 
#             return list(set(self.data.y_train))
#         except:
#             raise Exception('Model has not been fit.')

#     def predict(self, X_test: pd.DataFrame):
    
#         if isinstance(X_test, pd.DataFrame):
#             cols = [c for c in X_test.columns if c != self.target_col]
#             X_test = X_test.to_numpy()

#         assert len(X_test.shape) == 2

#         predictions = []
#         nodes_seen = []
        
#         for i in tqdm(range(X_test.shape[0])):

#             sample = X_test[i,:].astype(np.float32)
#             raw_sample = sample.copy()
#             indices = np.ravel(np.argwhere(np.isnan(sample)))
#             level = len(set(list(self.feature_to_test_map.values()))) \
#                 - get_n_tests_missing(cols, self.feature_to_test_map, indices)
#             sample_1d = np.array([
#                 sample[i] if i not in indices else np.nan
#                 for i in range(len(sample))
#             ])
#             sample = sample_1d.reshape(1,-1).astype(np.float32)
#             # create node: pair[level, tuple[relevant indices]]
#             node = (
#                 level, 
#                 tuple(
#                     [x for x in range(X_test.shape[1]) if x not in indices]
#                 )
#             )
#             nodes_seen.append(node)
#             model = self.dag.nodes[node]['model']
            
#             # ancestor_predictions includes prediction at current node as well
#             # so initialize ancestor predictions with current node prediction
#             # first_prediction = getattr(model, self.prediction_method)(sample)[0]
#             # first_prediction = getattr(model, 'predict')(raw_sample)[0]
#             first_prediction = getattr(model, 'predict')(sample)[0]
#             self.dag.nodes[node]['passthrough_predictions'][i] = first_prediction
#             ancestor_predictions = [first_prediction]

#             df = self.data.drop(columns=[self.target_col])

#             if level != 0:
#                 for a in nx.ancestors(self.dag, node):
                    
#                     ancestor_model = self.dag.nodes[a]['model']
#                     # current_prediction = ancestor_model.predict(raw_sample)[0]
#                     current_prediction = ancestor_model.predict(sample)[0]
#                     self.dag.nodes[a]['passthrough_predictions'][i] = current_prediction
#                     ancestor_predictions.append(current_prediction)
#             prediction = sum(ancestor_predictions) / len(ancestor_predictions)

#             predictions.append(prediction)
#             self.dag.nodes[node]['predictions'][i] = prediction
            
#         return np.array(predictions)

#     def find_each_model_val_threshold(self):
#         """
#         for each model in the DAG, find its optimal threshold for discerning
#         positive vs. negative class.
#         """
#         for node in self.dag.nodes:
#             model = self.dag.nodes[node]['model']
#             #TODO{construct X_val, y_val}
#             proba_predictions = model.predict(self.X_val)
#             threshold = find_optimal_threshold(
#                 proba_predictions, self.y_test, step=0.1
#             )
#             self.dag.nodes[node]['optim_threshold'] = threshold


# class StratificationOnly(BaseInheritanceClassifier):

#     def __init__(
#         self,
#         data: Union[Dataset, MedicalDataset],
#         base_estimator: BaseEstimator,
#         base_estimator_params: dict = {},
#         prediction_method: str = 'auto'
#     ):
#         super().__init__(
#             data=data,
#             base_estimator=base_estimator,
#             base_estimator_params=base_estimator_params,
#             prediction_method=prediction_method
#         )
#         self.name = type(base_estimator).__name__ + '_StratificationOnly'
           

#     def fit(self):
#         super().fit_node_estimators()

#     def predict(self, X_test: pd.DataFrame):

#         if isinstance(X_test, pd.DataFrame):
#             cols = [c for c in X_test.columns if c != self.target_col]
#             X_test = X_test.to_numpy()

#         assert len(X_test.shape) == 2

#         predictions = []
#         nodes_seen = []
        
#         for i in tqdm(range(X_test.shape[0])):

#             sample = X_test[i,:].astype(np.float32)
#             # raw_sample = sample.copy()
#             indices = np.ravel(np.argwhere(np.isnan(sample)))
#             level = len(set(list(self.feature_to_test_map.values()))) \
#                 - get_n_tests_missing(cols, self.feature_to_test_map, indices)
#             # sample = np.array([
#                 # sample[i] 
#                 # if i not in indices else np.nan
#                 # for i in range(len(sample))
#             # ])
#             # sample = sample.reshape(1,-1).astype(np.float32)
#             # create node: pair[level, tuple[relevant indices]]
#             node = (
#                 level, 
#                 tuple(
#                     [x for x in range(X_test.shape[1]) if x not in indices]
#                 )
#             )
#             nodes_seen.append(node)
#             model = self.dag.nodes[node]['model']
            
#             # prediction = getattr(model, 'predict')(raw_sample)[0]
#             prediction = getattr(model, 'predict')(sample)[0]
#             predictions.append(prediction)
#             self.dag.nodes[node]['predictions'][i] = prediction

#         return np.vstack(predictions)


class StackedHeterogenousClassifier(StackingClassifier):
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        passthrough=False,
        verbose=0
    ):
        super().__init__(
            estimators,
            final_estimator,
            cv,
            stack_method,
            n_jobs,
            passthrough,
            verbose
        )
        self.name = type(final_estimator).__name__ + \
            '_StackedHeterogeneousClassifier'

    


class StackedHeterogenousRegressor(StackingRegressor):
    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        passthrough=False,
        verbose=0
    ):
        super().__init__(
            estimators,
            final_estimator,
            cv,
            stack_method,
            n_jobs,
            passthrough,
            verbose
        )
        self.name = type(final_estimator).__name__ + \
            '_StackedHeterogeneousRegressor'


# class StackedInheritanceEstimator(BaseInheritanceClassifier):
#     def __init__(
#         self,
#         data: pd.DataFrame,
#         base_estimator: BaseEstimator,
#         meta_estimator: BaseEstimator,
#         target_col: str,
#         tests: Iterable[Test],
#         base_features: Iterable[str],
#         feature_to_test_map: dict,
#         base_estimator_params: dict = {},
#         meta_estimator_params: dict = {},
#         prediction_method: str = 'auto',
#         prediction_type: str = 'auto'
#     ):
#         super().__init__(
#             data=data,
#             base_estimator=base_estimator,
#             target_col=target_col,
#             tests=tests,
#             base_features=base_features,
#             feature_to_test_map=feature_to_test_map,
#             base_estimator_params=base_estimator_params,
#             prediction_method=prediction_method
#         )
#         self.meta_estimator_params = meta_estimator_params
#         if not inspect.isclass(meta_estimator):
#             meta_estimator = type(meta_estimator)
#         else:
#             self.meta_estimator = meta_estimator
        
#         if prediction_type == 'auto':
#             # infer whether classification or regression
#             if len(Counter(self.data[target_col])) < np.sqrt(len(self.data)):
#                 self.prediction_type = 'classification'
#             else:
#                 self.prediction_type = 'regression'
#         else:
#             assert prediction_type in ['classification', 'regression']
#             self.prediction_type = prediction_type
#         self.name = type(base_estimator).__name__ + \
#             '_StackedInheritance_Meta(' + self.meta_estimator.__name__ + ')'

#     def fit(self):
#         super().fit_node_estimators()
#         # fit stacking classifier at each node using ancestor models
#         with tqdm(total=len(self.dag.nodes)) as pbar:
#             for node in self.dag.nodes:
#                 estimators = [(str(node), self.dag.nodes[node]['model'])]
#                 if node[0] != 0:    
#                     for ancestor in nx.ancestors(self.dag, node):
#                         ancestor_model = self.dag.nodes[ancestor]['model']
#                         estimators.append((str(ancestor), (ancestor_model)))
                    
#                 if self.prediction_type == 'classification':
#                     stacking_model = StackingClassifier(
#                         estimators=estimators,
#                         final_estimator=self.meta_estimator(**self.meta_estimator_params),
#                         cv='prefit'
#                     )
#                 else:
#                     stacking_model = StackingRegressor(
#                         estimators=estimators,
#                         final_estimator=self.meta_estimator(**self.meta_estimator_params),
#                         cv='prefit'
#                     )
#                 indices = node[1]
#                 X = self.data[[c for c in self.data.columns if c != self.target_col]]
#                 y = self.data[self.target_col]
#                 # current_node_rows = self.data.iloc[:, list(indices)].dropna(
#                 #     axis=0
#                 # ).index
#                 # current_data = self.data.loc[current_node_rows, :]
#                 # X = current_data[
#                 #     [c for c in current_data.columns if c != self.target_col]
#                 # ]
#                 # y = current_data[self.target_col]
#                 stacking_model.fit(X, y)
#                 self.dag.nodes[node]['stacking_model'] = stacking_model
#                 pbar.update(1)

#     def predict(self, X_test: pd.DataFrame):

#         if isinstance(X_test, pd.DataFrame):
#             cols = X_test.columns
#             X_test = X_test.to_numpy()

#         assert len(X_test.shape) == 2

#         predictions = []
#         nodes_seen = []
        
#         for i in tqdm(range(X_test.shape[0])):

#             sample = X_test[i,:].astype(np.float32)
#             raw_sample = sample.copy()
#             indices = np.ravel(np.argwhere(np.isnan(sample)))
#             level = len(set(list(self.feature_to_test_map.values()))) \
#                 - get_n_tests_missing(cols, self.feature_to_test_map, indices)
#             sample = np.array([
#                 sample[i] 
#                 if i not in indices else np.nan
#                 for i in range(len(sample))
#             ])
#             sample = sample.reshape(1,-1).astype(np.float32)
#             # create node: pair[level, tuple[relevant indices]]
#             node = (
#                 level, 
#                 tuple(
#                     [x for x in range(X_test.shape[1]) if x not in indices]
#                 )
#             )
#             nodes_seen.append(node)
#             stacking_model = self.dag.nodes[node]['stacking_model']
            
#             prediction = getattr(
#                 stacking_model, self.prediction_method
#             )(sample)
#             predictions.append(prediction)
#             self.dag.nodes[node]['predictions'][i] = prediction

#         return np.vstack(predictions)


# class StackedParametricClassifier(nn.Module):
#     def __init__(
#         self,
#         estimators: Union[
#             Iterable[BaseEstimator], Mapping[str, BaseEstimator]
#         ],
#         input_dim: int
#     ) -> None:
#         super().__init__()

#         if isinstance(estimators, Mapping):
#             self.estimators = estimators
#         else:
#             estimators_dict = {}
#             for i, model in enumerate(estimators):
#                 estimators_dict[str(i) + '_' + type(model).__name__] = model
#             self.estimators = estimators_dict

#         self.nn_layer1 = nn.Linear(input_dim, input_dim)
#         self.nn_layer2 = nn.Linear(input_dim, len(self.estimators))
#         self.softmax = nn.Softmax()

#     def forward(self, X):
#         device = get_device()
#         X = torch.nan_to_num(X)
#         estimators_predictions = []
#         for model in self.estimators.values():
#             estimators_predictions.append(
#                 getattr(model, get_prediction_method(model))(X)
#             )
#         estimators_predictions = torch.from_numpy(
#             np.array(estimators_predictions).astype(np.float32)
#         ).float().squeeze()[:, 1].to(device)

#         X = self.nn_layer1(X)
        
#         X = self.nn_layer2(X)
#         weights = self.softmax(X).squeeze()
        
#         weighted_sum = torch.dot(weights, estimators_predictions)
        
#         return weighted_sum

#     def predict(self, X):
#         return self.forward(X)

#     def fit(self, X, y):
#         device = get_device()
#         if isinstance(X, pd.DataFrame):
#             X = torch.from_numpy(X.values.astype(np.float32)).float().to(device)
#         elif isinstance(X, np.ndarray):
#             X = torch.tensor(X.astype(np.float32)).float().to(device)
#         y = torch.from_numpy(
#             np.array(y).astype(np.float32)
#         ).float().squeeze()
#         if len(y.shape) == 2:
#             y = y[:, 1]
#         assert len(y.shape) == 1
#         y = y.to(device)

#         train_tensor = data_utils.TensorDataset(X, y)
#         trainloader = data_utils.DataLoader(dataset = train_tensor, batch_size = 1, shuffle = True)
#         criterion = nn.L1Loss()
#         optimizer = optim.SGD(self.parameters(), lr=0.005, momentum=0.9)
#         for epoch in range(5):  # loop over the dataset multiple times

#             running_loss = 0.0
#             for i, data in enumerate(trainloader, 0):
#                 # get the inputs; data is a list of [inputs, labels]
#                 inputs, labels = data

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward + backward + optimize
#                 outputs = self.forward(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#                 # print statistics
#                 running_loss += loss.item()
#                 if i % 200 == 199:    # print every 2000 mini-batches
#                     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
#                     running_loss = 0.0

#         print('Finished Training')
#         return self


# class DynamicEnsembleSelectorInheritance:
#     def __init__(
#         self,
#         inheritance_model,
#         des_method
#     ) -> None:
#         self.inheritance_model = inheritance_model
#         self.des_method = des_method

#     def fit_each_node(self, X=None, y=None, cv_fold_index=None):
#         if X is not None and y is not None:
#             X_train, y_train = X.copy(), y.copy()
#         elif cv_fold_index is not None:
#             train, test = self.inheritance_model.data[cv_fold_index].copy()
#             y_train = train[self.inheritance_model.data.target_col]
#             X_train = train.drop(self.inheritance_model.data.target_col, axis=1)
#         else:
#             raise Exception('You must provide either (X,y) or a CV fold index')

#         for node in self.inheritance_model.dag.nodes:
#             node_attributes = self.inheritance_model.dag.nodes[node]
#             classifier_pool = [node_attributes['model']]
#             if node[0] != 0:
#                 for a in nx.ancestors(self.inheritance_model.dag, node):
#                     classifier_pool.append(self.inheritance_model.dag.nodes[a]['model'])
            
#             current_features = node_attributes['features']
        
#     def fit(self, X=None, y=None, cv_fold_index=None):
#         self.fit_each_node(X, y, cv_fold_index)

#     def predict(self, X):
#         pass


class DEWClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        classifier_pool:    Union[Mapping[str, BaseEstimator],\
                            Iterable[Tuple[str, BaseEstimator]]],
        n_neighbors=5,
        n_top_to_choose=[1,3,5,None],
        competence_threshold = 0.5
    ) -> None:
        super().__init__()
        self.classifier_pool = dict(classifier_pool)
        self.n_top_to_choose=n_top_to_choose
        self.n_neighbors = n_neighbors
        self.competence_threshold = competence_threshold
        self.samplewise_clf_errors = pd.DataFrame({})
        self.weight_assignments = pd.DataFrame({})
        self._temp_weights = []

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.train_samples = copy.deepcopy(X)

        for clf_name, model in self.classifier_pool.items():
            probas = model.predict_proba(X)
            # errors = np.max(np.clip(y - probas, 0, 1), axis=1)
            errors = [log_loss(y[i, :], probas[i, :]) for i in range(len(y))]
            self.samplewise_clf_errors[clf_name] = errors

    def predict_proba_one_sample(self, sample):
        predictions = {}
        query = np.array(sample).reshape(1,-1)
        pipeline_competences: dict = self.estimate_competences(
            q=query, samples_df=self.train_samples
        )
        competences = list(pipeline_competences.values())
        full_weights = scipy.special.softmax(competences)
        
        sample = sample.to_frame().T
        
        for top_n in self.n_top_to_choose:
            if top_n not in [None, -1, 0]:
                # rank from highest competence --> lowest_competence
                ranked = np.argsort(competences)[::-1][0: top_n]
                top_n_clf = [
                    list(self.classifier_pool.items())[i] 
                    for i in ranked
                ]
                top_n_clf_competences = np.array([
                    competences[i]
                    for i in ranked
                ])
            else:
                top_n_clf = list(self.classifier_pool.items())
                top_n_clf_competences = competences

            weights = scipy.special.softmax(top_n_clf_competences)
            probas = np.array([
                model.predict_proba(sample)[0]
                for clf_name, model in top_n_clf
            ])
            prediction = np.dot(weights, probas)
            predictions[top_n] = prediction

        return predictions
            

    def predict_proba(self, X):
        top_n_prediction_sets = {}

        predictions = {top_n: [] for top_n in self.n_top_to_choose}
        all_samples = [X.iloc[i, :].astype(np.float32) for i in range(X.shape[0])]

        with tqdm(total=len(all_samples)) as pbar:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
                for result in pool.map(self.predict_proba_one_sample, all_samples):
                    for top_n in result.keys():
                        predictions[top_n].append(result[top_n])
                    pbar.update(1)

        for top_n in predictions.keys():
            predictions[top_n] = np.vstack(predictions[top_n])

        return predictions

    def predict(self, X) -> dict:
        predictions = {}
        probas_sets = self.predict_proba(X)
        for top_n, probas in probas_sets.items():
            a = (probas == probas.max(axis=1, keepdims=True)).astype(int)
            predictions[top_n] = a

        return predictions

    def get_nearest_neighbors(self, q, samples_df):
        pipeline_specific_neighbor_indices = {}
        for idx, p in self.classifier_pool.items():
            imputed_q = p.imputer.transform(q).reshape(1,-1)
            imputed_samples = p.imputer.transform(samples_df)
            distances = cdist(imputed_q, imputed_samples)[0]
            # we will use numerical indices simply for facilitating y_true indexing in competence esitmation
            dist_df = pd.DataFrame(data=distances, columns=['distance'], index=range(len(samples_df)))
            sorted_distances_df = dist_df.sort_values('distance')
            pipeline_specific_neighbor_indices[idx] = sorted_distances_df.index[0: self.n_neighbors]

        return pipeline_specific_neighbor_indices


    def estimate_competences(self, q, samples_df) -> dict:
        
        pipeline_competences = {}
        pipeline_specific_neighbor_indices = self.get_nearest_neighbors(
            q, samples_df
        )
        for pipeline_idx, indices in pipeline_specific_neighbor_indices.items():
            # predictions = pipelines[pipeline_idx].predict_proba(samples_df.loc[indices, :])
            # errors = np.abs(predictions - y_true[indices])
            errors = self.samplewise_clf_errors[pipeline_idx]
            competences = 1 - errors
            competence = np.mean(competences)
            competence = competence / (np.std(competences) + 0.01) if competence > self.competence_threshold else 0
            pipeline_competences[pipeline_idx] = competence

        return pipeline_competences


    def get_pipeline_weights(self, q, samples_df) -> dict:
        pipeline_weights = {}
        pipeline_competences = self.estimate_competences(q, samples_df)
        all_competence_scores = list(pipeline_competences.values())
        weights = scipy.special.softmax(all_competence_scores)
        for idx, pipeline_type in enumerate(list(pipeline_competences.keys())):
            pipeline_weights[pipeline_type] = weights[idx]

        return pipeline_weights






# ======================================================== #

# def run_experiment(data, cv_output_dir=None, prediction_error_viz_outfile=None):

#     inheritance_metrics = []
#     knn_metrics = []
#     mice_metrics = []
#     vanilla_metrics = []
#     stacked_metrics = []
#     ds_metrics = []
#     oracle_metrics = []

#     proba_predictions_dfs_list = []
#     errors_dfs_list = []
#     y_trues = []

#     weights = []

#     for i in range(data.n_folds):
#         proba_predictions_df = pd.DataFrame({})
#         errors_df = pd.DataFrame({})

#         # initialize classifiers
#         clf_inheritance = NonmissingSubspaceInheritanceClassifier(
#             data=data, 
#             base_estimator= xgb.XGBClassifier,
#             base_estimator_params={'n_jobs': 1, 'max_depth': 3, 'n_estimators': 60, 'verbosity': 0},
#             voting='soft',
#             use_optimal_threshold=False
#         )
        
#         clf_knn = ClassifierWithImputation(
#             estimator=xgb.XGBClassifier(
#                 **{'n_jobs': 1, 'max_depth': 3, 'n_estimators': 60, 'verbosity': 0}
#             ),
#             imputer=KNNImputer
#         )

#         clf_mice = ClassifierWithImputation(
#             estimator=xgb.XGBClassifier(
#                 **{'n_jobs': 1, 'max_depth': 3, 'n_estimators': 60, 'verbosity': 0}
#             ),
#             imputer=IterativeImputer
#         )

#         clf_vanilla = xgb.XGBClassifier(**{'n_jobs': 1, 'max_depth': 3, 'n_estimators': 60, 'verbosity': 0})

#         clf_stacked = StackingClassifier(
#             estimators=[
#                 ('inheritance_xgb', clf_inheritance), 
#                 ('knn-imputation_xgb', clf_knn),
#                 ('mice-imputation_xgb', clf_mice),
#                 ('off_shelf_xgb', clf_vanilla)
#             ],
#             # final_estimator=RandomForestClassifier(
#             #     n_estimators=20, max_depth=3, max_features=0.5, 
#             #     max_samples=0.7, n_jobs=1
#             # ),
#             final_estimator=xgb.XGBClassifier(
#                 **{'n_jobs': 1, 'max_depth': 3, 'n_estimators': 60, 'verbosity': 0}
#             ),
#             cv='prefit',
#             n_jobs=1
#         )

#         clf_ds = DEWClassifier(
#             classifier_pool={
#                 'inheritance': clf_inheritance,
#                 'knn_imputation': clf_knn,
#                 'mice_imputation': clf_mice,
#                 'xgb_baseline': clf_vanilla
#             },
#             n_neighbors=5
#         )

#         clf_oracle = Oracle(
#             pool_classifiers=[clf_inheritance, clf_knn, clf_mice, clf_vanilla]
#         )

#         df = data.raw_data.copy()

#         train, val, test = data[i]
        
#         y_test = test[data.target_col]
#         y_trues += list(y_test)

#         y_train = train[data.target_col]
#         y_val = val[data.target_col]
        
#         X_test = test.drop(data.target_col, axis=1)
#         X_val = val.drop(data.target_col, axis=1)
#         X_train = train.drop(data.target_col, axis=1)
        
#         #################################################################

#         print('with full inheritance')
#         clf_inheritance.fit(X=X_train, y=y_train)
#         proba_predictions = clf_inheritance.predict_proba(X=X_test)[:,1]
#         proba_predictions_df['full_inheritance'] = proba_predictions
#         errors_df['full_inheritance'] = np.abs(proba_predictions - y_test)

#         roc_auc = roc_auc_score(y_test, proba_predictions)
#         predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
#         # print(classification_report(y_test, predictions))
#         metrics = get_classification_metrics(predictions, y_test)
#         metrics['roc_auc'] = round(roc_auc, 4)
#         accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
#         metrics['accuracy'] = round(accuracy, 4)
#         print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
#         inheritance_metrics.append(list(metrics.values()))
#         # with open('../results/inheritance_dag.pkl', 'wb') as f:
#         #     pickle.dump(clf_inheritance.dag, f)
#         print('\n-----------------------------\n')

#         #################################################################

#         print('with knn imputation')
#         clf_knn.fit(X_train, y_train)
#         proba_predictions = clf_knn.predict_proba(X_test)[:,1]
#         proba_predictions_df['knn_imputation'] = proba_predictions
#         errors_df['knn_imputation'] = np.abs(proba_predictions - y_test)
#         predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
#         # t = find_optimal_threshold(proba_predictions, y_test)
#         # predictions = [1 if p > t else 0 for p in proba_predictions]
#         # y_test = data[0][1][data.target_col]
#         roc_auc = roc_auc_score(y_test, proba_predictions)
#         metrics = get_classification_metrics(predictions, y_test)
#         # print(classification_report(y_test, predictions))
#         metrics['roc_auc'] = round(roc_auc, 4)
#         accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
#         metrics['accuracy'] = round(accuracy, 4)
#         knn_metrics.append(list(metrics.values()))
#         print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
#         print('\n-----------------------------\n')

#         #################################################################

#         print('with MICE imputation')
#         clf_mice.fit(X_train, y_train)
#         proba_predictions = clf_mice.predict_proba(X_test)[:,1]
#         proba_predictions_df['mice_imputation'] = proba_predictions
#         errors_df['mice_imputation'] = np.abs(proba_predictions - y_test)
#         predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
#         # t = find_optimal_threshold(proba_predictions, y_test)
#         # predictions = [1 if p > t else 0 for p in proba_predictions]
#         # y_test = data[0][1][data.target_col]
#         roc_auc = roc_auc_score(y_test, proba_predictions)
#         metrics = get_classification_metrics(predictions, y_test)
#         # print(classification_report(y_test, predictions))
#         metrics['roc_auc'] = round(roc_auc, 4)
#         accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
#         metrics['accuracy'] = round(accuracy, 4)
#         mice_metrics.append(list(metrics.values()))
#         print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
#         print('\n-----------------------------\n')

#         #################################################################

#         print('off-the-shelf xgboost:')
#         clf_vanilla.fit(X_train, y_train)
#         proba_predictions = clf_vanilla.predict_proba(X_test)[:,1]
#         proba_predictions_df['off-the-shelf xgb'] = proba_predictions
#         errors_df['off-the-shelf xgb'] = np.abs(proba_predictions - y_test)
#         # t = find_optimal_threshold(proba_predictions, y_test)
#         # predictions = [1 if p > t else 0 for p in proba_predictions]
#         # y_test = data[0][1][data.target_col]
#         roc_auc = roc_auc_score(y_test, proba_predictions)
#         predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
#         # print(classification_report(y_test, predictions))
#         metrics = get_classification_metrics(predictions, y_test)
#         metrics['roc_auc'] = round(roc_auc, 4)
#         accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
#         metrics['accuracy'] = round(accuracy, 4)
#         vanilla_metrics.append(list(metrics.values()))
#         print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
#         print('\n-----------------------------\n')

#         ######################################################################

#         print('stacked model:')
#         clf_stacked.fit(X_train, y_train)
#         proba_predictions = clf_stacked.predict_proba(X_test)[:,1]
#         proba_predictions_df['stacked'] = proba_predictions
#         errors_df['stacked'] = np.abs(proba_predictions - y_test)
#         predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
#         # t = find_optimal_threshold(proba_predictions, y_test)
#         # predictions = [1 if p > t else 0 for p in proba_predictions]
#         # y_test = data[0][1][data.target_col]
#         roc_auc = roc_auc_score(y_test, proba_predictions)
#         metrics = get_classification_metrics(predictions, y_test)
#         # print(classification_report(y_test, predictions))
#         metrics['roc_auc'] = round(roc_auc, 4)
#         accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
#         metrics['accuracy'] = round(accuracy, 4)
#         stacked_metrics.append(list(metrics.values()))
#         print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
#         print('\n======================\n===================\n')

#         ###########################################################

#         print('DS model:')
#         # clf_ds.fit(X_train, y_train)
        
#         # use validation set for nearest-neighbor search + prediction;
#         # avoids train set leakage

#         clf_ds.fit(X_val, y_val)
#         proba_predictions = clf_ds.predict_proba(X_test)[:, 1]
#         proba_predictions_df['DEW'] = proba_predictions
#         errors_df['DEW'] = np.abs(proba_predictions - y_test)
#         predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
#         weights.append(clf_ds.weight_assignments)
        
#         # t = find_optimal_threshold(proba_predictions, y_test)
#         # predictions = [1 if p > t else 0 for p in proba_predictions]
#         # y_test = data[0][1][data.target_col]
#         roc_auc = roc_auc_score(y_test, proba_predictions)
#         metrics = get_classification_metrics(predictions, y_test)
#         # print(classification_report(y_test, predictions))
#         metrics['roc_auc'] = round(roc_auc, 4)
#         accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
#         metrics['accuracy'] = round(accuracy, 4)
#         ds_metrics.append(list(metrics.values()))
#         print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
#         print('\n======================\n===================\n')

#         ###########################################################

#         print('Oracle Classifier:')
#         clf_oracle.fit(X_train, y_train)
#         proba_predictions = clf_oracle.predict_proba(X_test, y_test)[:, 1]
#         proba_predictions_df['Oracle'] = proba_predictions
#         errors_df['Oracle'] = np.abs(proba_predictions - y_test)
#         predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
#         roc_auc = roc_auc_score(y_test, proba_predictions)
#         metrics = get_classification_metrics(predictions, y_test)
#         metrics['roc_auc'] = round(roc_auc, 4)
#         accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
#         metrics['accuracy'] = round(accuracy, 4)
#         oracle_metrics.append(list(metrics.values()))
#         print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
#         print('\n======================\n===================\n')

#         ##############################################################

#         proba_predictions_dfs_list.append(proba_predictions_df)
#         errors_dfs_list.append(errors_df)

#     proba_predictions_df_total = pd.concat(proba_predictions_dfs_list)
#     proba_predictions_df_total.index = range(len(proba_predictions_df_total))
    
#     errors_df_total = pd.concat(errors_dfs_list)
#     errors_df_total.index = range(len(errors_df_total))

#     weights_df = pd.concat(weights)

#     plot_prediction_errors(
#         y_true=np.array(y_trues), 
#         proba_predictions_df=proba_predictions_df_total,
#         title='Example Comparison of Model Prediction Errors',
#         xlabel='Index of Samples',
#         ylabel='Class Probability Prediction Error',
#         outfile=prediction_error_viz_outfile
#     )

#     inheritance_metrics = np.vstack(inheritance_metrics)
#     knn_metrics = np.vstack(knn_metrics)
#     mice_metrics = np.vstack(mice_metrics)
#     vanilla_metrics = np.vstack(vanilla_metrics)
#     stacked_metrics = np.vstack(stacked_metrics)
#     ds_metrics = np.vstack(ds_metrics)
#     oracle_metrics = np.vstack(oracle_metrics)

#     mean_inheritance_metrics = np.mean(inheritance_metrics, axis=0)
#     mean_knn_metrics = np.mean(knn_metrics, axis=0)
#     mean_mice_metrics = np.mean(mice_metrics, axis=0)
#     mean_vanilla_metrics = np.mean(vanilla_metrics, axis=0)
#     mean_stacked_metrics = np.mean(stacked_metrics, axis=0)
#     mean_ds_metrics = np.mean(ds_metrics, axis=0)
#     mean_oracle_metrics = np.mean(oracle_metrics, axis=0)

#     print('\n=========\n==========\n')

#     print('cross-validated mean metrics:\n----------------\n')

#     metric_types = [
#             'sensitivity', 'specificity', 'ppv', 'npv', 'gmean_sens_spec',
#             'gmean_all_metrics', 'roc_auc', 'accuracy'
#         ]
#     cv_results_means = {}

#     for t in [
#         'mean_inheritance_metrics', 'mean_knn_metrics', 
#         'mean_mice_metrics','mean_vanilla_metrics', 
#         'mean_stacked_metrics', 'mean_ds_metrics', 'mean_oracle_metrics'
#     ]:
#         cv_results_means[t] = {}
#         print(t + str(': '))
#         for i, metric in enumerate(metric_types):
#             cv_results_means[metric + '_' + t] = eval(t)[i]
#             print(metric + ': ' + str(eval(t)[i]))
        
#         print('-------------')

#     df = pd.DataFrame({k: [v] for k, v in cv_results_means.items()})
#     return df, weights_df, errors_df_total


# def run_wisconsin_bc_experiment(data, missingness_amounts, masked_features, 
#     out_dir, experiment_type
# ):
#     DIAGNOSIS_KEYS = ['wisconsin_bc_diagnosis', 'diagnosis']
#     PROGNOSIS_KEYS = ['wisconsin_bc_prognosis', 'prognosis']

#     if experiment_type in DIAGNOSIS_KEYS:
#         data = load_wisconsin_diagnosis_dataset(
#             MASKED_FEATURE_TYPES=masked_features,
#             missingness_amounts=missingness_amounts
#         )
#         experiment_type = 'diagnosis'
#     elif experiment_type in PROGNOSIS_KEYS:
#         data = load_wisconsin_prognosis_dataset(
#             MASKED_FEATURE_TYPES=masked_features,
#             missingness_amounts=missingness_amounts
#         )
#         experiment_type = 'prognosis'
#     else:
#         raise AssertionError('Please use diagnosis or prognosis.')
    
#     err_dir = '../results/wisconsin_bc_' + experiment_type
#     err_viz_file = 'amount(' + str(missingness_amounts[0]) + \
#         ')__features(' + str(masked_features) + ').png'
    
#     err_viz_file = os.path.join(err_dir, err_viz_file)
#     results_df, weights_df, errors_df = run_experiment(data=data, cv_output_dir=out_dir, 
#         prediction_error_viz_outfile=err_viz_file)
#     return results_df, weights_df, errors_df


# def run_randomized_wisconsin_bc_experiments(experiment_type):
#     import random
#     DIAGNOSIS_KEYS = ['wisconsin_bc_diagnosis', 'diagnosis']
#     PROGNOSIS_KEYS = ['wisconsin_bc_prognosis', 'prognosis']
#     FEATURE_TYPES=[
#         'radius', 'texture','perimeter','area','smoothness','compactness',
#         'concavity','concave points','symmetry','fractaldimension'
#     ] # there are 10 feature types; choose anywhere from 1-9 as optional tests
#     missingness_amounts = [
#         [miss_fraction / 10] * x 
#         for x in range(1, 10) #########TODO{turn this back into range(1,10)}
#         for miss_fraction in range(1, 10) #########TODO{turn this back into range(1,10)}
#     ]

#     running_results = {}

#     for amounts in missingness_amounts:
#         n_missing_tests = len(amounts)
#         masked_features = random.sample(FEATURE_TYPES, k=n_missing_tests)
#         if experiment_type in DIAGNOSIS_KEYS:
#             data = load_wisconsin_diagnosis_dataset(
#                 MASKED_FEATURE_TYPES=masked_features,
#                 missingness_amounts=amounts
#             )
#         elif experiment_type in PROGNOSIS_KEYS:
#             data = load_wisconsin_prognosis_dataset(
#                 MASKED_FEATURE_TYPES=masked_features,
#                 missingness_amounts=amounts
#             )
#         out_dir = '../results/missing_' + str(amounts[0]) + '__features_'\
#             + str(masked_features)
#         results, weights_df, errors_df = run_wisconsin_bc_experiment(
#             data=data, missingness_amounts=amounts, 
#             masked_features=masked_features, out_dir=out_dir, 
#             experiment_type=experiment_type
#         )
#         print(results)
#         results.index = ['amount(' + str(amounts[0]) + ')__features(' + str(masked_features) + ')']
#         weights_df.to_csv(
#             '../results/wisconsin_bc_' + experiment_type + '/dew_weights/' + \
#                 'amount(' + str(amounts[0]) + ')__features(' + str(masked_features) + ')'
#         )
#         errors_df.to_csv(
#             '../results/wisconsin_bc_' + experiment_type + '/errors/' + \
#                 'amount(' + str(amounts[0]) + ')__features(' + str(masked_features) + ')'
#         )
#         running_results[
#             'amount(' + str(amounts[0]) + ')__features(' + str(masked_features) + ')'
#         ] = results

#     final_results = pd.concat(list(running_results.values()))

#     if experiment_type in DIAGNOSIS_KEYS:
#         final_results.to_csv('../results/final_results_wisconsin_bc_diagnosis_FROZEN.csv')
#     elif experiment_type in PROGNOSIS_KEYS:
#         final_results.to_csv('../results/final_results_wisconsin_bc_prognosis_FROZEN.csv')

#     return final_results


    #################################################################

    


if __name__ == '__main__':
    # results = run_randomized_wisconsin_bc_experiments(experiment_type='prognosis')
    # del results
    # results = run_randomized_wisconsin_bc_experiments(experiment_type='diagnosis')
    # del results
    # data = load_numom2b_analytes_dataset(target_col='PEgHTN')
    # numom2b_viz_file = '../results/numom2b_hypertension/error_predictions.png'
    # numom2b_inheritance_results_file = '../results/numom2b_hypertension/numom2b_inheritance_results.csv'
    # results, weights_df, errors_df = run_experiment(data, prediction_error_viz_outfile=numom2b_viz_file)
    # results.to_csv(numom2b_inheritance_results_file)
    # weights_df.to_csv('../results/numom2b_hypertension/dew_weights.csv')
    # errors_df.to_csv('../results/numom2b_hypertension/errors.csv')
    pass