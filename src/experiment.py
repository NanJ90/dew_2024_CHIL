"""
This file houses the `Experiment` class.
It serves as a template for constructing reproducible experiments.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import inspect
from itertools import product
import json
import os
from random import random
from typing import Iterable, Tuple
import sys

from deslib.base import BaseDS
from deslib.des import DESClustering, KNORAE, KNORAU, KNOP, METADES
from deslib_missingness.static import Oracle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_gaussian_quantiles
from sklearn.ensemble import RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import classification_report, max_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import xgboost as xgb

sys.path.append('.')

from data_loaders import Dataset, MedicalDataset, MissDataset
from new_base import ClassifierWithImputation, IdentityImputer
from new_models import NonmissingSubspaceInheritanceClassifier, DEWClassifier
from utils import get_classification_metrics, plot_prediction_errors, produce_NA


class ExperimentBase:
    def __init__(
        self,
        dataset,
        exp_type='mcar',
        name='Experiment_' + str(datetime.now()),
        base_dir=None,
        random_state=42
    ):
        self.dataset = dataset
        self.n_folds = len(dataset.train_val_test_triples)

        self.name = name
        
        if base_dir is None:
            base_dir = '../Experiment_Trials/' + name

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        self.base_dir = os.path.join(base_dir, exp_type)
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        
        self.results_dir = os.path.join(self.base_dir, 'results')
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        
        self.cv_results = {
            i: {} for i in range(self.n_folds)
        }

        self.results = {}
        self.random_state = random_state

    def run(self):
        raise NotImplementedError

    def write_results(self, outfile, write_method='w') -> None:
        """
        write_method :: either 'w' for overwriting file, 
                        or 'a' for appending to file
        """
        results_json = json.dump(self.results)
        with open(outfile, write_method) as f:
            f.write(results_json)


class DynamicEstimatorSelectionExperiment(ExperimentBase):
    def __init__(
        self,
        dataset: MedicalDataset,
        estimators: Iterable[BaseEstimator],
        dynamic_selection_method: BaseDS,

        name='Experiment_Type(DES)_' + str(datetime.now()),
        base_dir=None,
        random_state=42,
        **des_args
    ):
        super().__init__(
            dataset=dataset,
            name=name,
            base_dir=base_dir,
            random_state=random_state
        )
        if inspect.isclass(dynamic_selection_method):
            self.model = dynamic_selection_method(
                pool_classifiers=estimators,
                **des_args
            )
        else:
            self.model = type(dynamic_selection_method)(
                pool_classifiers=estimators,
                **des_args
            )

        
class Numom2bExperiment(ExperimentBase):
    def __init__(
        self, 
        dataset,
        base_estimator=xgb.XGBClassifier,
        base_estimator_param_grid = {
            'n_estimators': [5, 10, 20, 40, 60, 80, 100, 150, 200, 300],
            'max_depth': list(range(3, 12))
        }

    ):
        self.data = dataset
        self.base_estimator = base_estimator
        self.base_estimator_param_grid = base_estimator_param_grid
        self.n_folds = len(self.data.train_test_pairs)

        # cross-validation results: keys ~ folds; values ~ fold results dict 
        self.cv_results = {
            i: {} 
            for i in range(self.n_folds)
        }

    def run_baselines(self):
        """
        baselines:
            * kNN-imputation
            * MICE-imputation
            * mean-imputation
            * off-the-shelf XGBoost NaN-handling

        param_grids:
            + XGBoost models
                * n_estimators [5, 10, 20, 40, 60, 80, 100, 150, 200, 300]
                * max_depth [3-11]
        """

        for i, (train, test) in self.data:
            y_train = train[self.data.target_col]
            y_test = test[self.data.target_col]
            X_train = train.drop(self.data.target_col, axis=1)
            X_test = test.drop(self.data.target_col, axis=1)

            knn_imp_classifier = ClassifierWithImputation(
                estimator=self.base_estimator(),
                imputer=KNNImputer()
            )

            ####################################################

            mice_imp_classifier = ClassifierWithImputation(
                estimator=self.base_estimator(),
                imputer=IterativeImputer()
            )

            ####################################################

            mean_imp_classifier = ClassifierWithImputation(
                estimator=self.base_estimator(),
                imputer=SimpleImputer(strategy='mean')
            )

            ####################################################

            vanilla_classifier = self.base_estimator()

            ####################################################

            nonmissing_subspace_inheritance_classifier = NonmissingSubspaceInheritanceClassifier(
                data=self.data,
                base_estimator=xgb.XGBClassifier(),
                base_estimator_params={
                    'n_jobs': 1,
                    'n_estimators': 40,
                    'max_depth': 3
                },
                voting='hard'
            )
            nonmissing_subspace_inheritance_classifier.fit(cv_fold_index=i)
            proba_predictions = nonmissing_subspace_inheritance_classifier.predict(cv_fold_index=i)
            predictions = [1 if x > 0.5 else 0 for x in proba_predictions]

            ####################################################


            


    def run_nonmissing_subspaces_full_inheritance(self):
        pass

    def run_stratification_only(self):
        """
        stratification-only model CV
        """
        pass

    def run_nonmissing_subspaces_des(self):
        pass

    def run_heterogeneous_des(self):
        pass


class SyntheticClassificationExperiment:
    def __init__(
        self,
        synthetic_data_mechanism='make_classification',
        nan_type='MCAR',
        miss_rate=0.5,
        mnar_method=None,
        percent_features_without_missing_values=None,
        **make_classification_kwargs
    ) -> None:

        assert nan_type in ['MCAR', 'MAR', 'MNAR']
        if synthetic_data_mechanism == 'make_classification':
            X, y = make_classification(**make_classification_kwargs)
        elif synthetic_data_mechanism == 'make_gaussian_quantiles':
            X, y = make_gaussian_quantiles(**make_classification_kwargs)
        else:
            raise Exception('Choose either `make_classification` or `make_gaussian_quantiles`')

        matrices_dict = produce_NA(
            X=X, p_miss=miss_rate, mecha=nan_type, opt=mnar_method, 
            p_obs=percent_features_without_missing_values, q=None
        )
        X = matrices_dict['X_incomp']
        df = pd.DataFrame(X)
        df['target'] = y
        
        self.data = Dataset(data=df, n_folds=5, target_col='target')


    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        inheritance_metrics = []
        knn_metrics = []
        mice_metrics = []
        vanilla_metrics = []
        stacked_metrics = []
        dew_metrics = []
        oracle_metrics = []

        proba_predictions_dfs_list = []
        errors_dfs_list = []
        y_trues = []

        weights = []

        for i in range(self.data.n_folds):
            proba_predictions_df = pd.DataFrame({})
            errors_df = pd.DataFrame({})

            # initialize classifiers
            clf_inheritance = NonmissingSubspaceInheritanceClassifier(
                data=self.data, 
                base_estimator= xgb.XGBClassifier,
                base_estimator_params={'n_jobs': 1, 'max_depth': 3, 'n_estimators': 60, 'verbosity': 0},
                voting='soft',
                use_optimal_threshold=False
            )
            
            clf_knn = ClassifierWithImputation(
                estimator=xgb.XGBClassifier(
                    **{'n_jobs': 1, 'max_depth': 3, 'n_estimators': 60, 'verbosity': 0}
                ),
                imputer=KNNImputer
            )

            clf_mice = ClassifierWithImputation(
                estimator=xgb.XGBClassifier(
                    **{'n_jobs': 1, 'max_depth': 3, 'n_estimators': 60, 'verbosity': 0}
                ),
                imputer=IterativeImputer
            )

            clf_vanilla = xgb.XGBClassifier(**{'n_jobs': 1, 'max_depth': 3, 'n_estimators': 60, 'verbosity': 0})

            clf_stacked = StackingClassifier(
                estimators=[
                    ('inheritance_xgb', clf_inheritance), 
                    ('knn-imputation_xgb', clf_knn),
                    ('mice-imputation_xgb', clf_mice),
                    ('off_shelf_xgb', clf_vanilla)
                ],
                # final_estimator=RandomForestClassifier(
                #     n_estimators=20, max_depth=3, max_features=0.5, 
                #     max_samples=0.7, n_jobs=1
                # ),
                final_estimator=xgb.XGBClassifier(
                    **{'n_jobs': 1, 'max_depth': 3, 'n_estimators': 60, 'verbosity': 0}
                ),
                cv='prefit',
                n_jobs=1
            )

            clf_ds = DEWClassifier(
                classifier_pool={
                    'inheritance': clf_inheritance,
                    'knn_imputation': clf_knn,
                    'mice_imputation': clf_mice,
                    'xgb_baseline': clf_vanilla
                },
                n_neighbors=3
            )

            clf_oracle = Oracle(
                pool_classifiers=[clf_inheritance, clf_knn, clf_mice, clf_vanilla]
            )

            df = self.data.raw_data.copy()

            train, val, test = self.data[i]
            
            y_test = test[self.data.target_col]
            y_trues += list(y_test)

            y_train = train[self.data.target_col]
            y_val = val[self.data.target_col]
            
            X_test = test.drop(self.data.target_col, axis=1)
            X_val = val.drop(self.data.target_col, axis=1)
            X_train = train.drop(self.data.target_col, axis=1)
            
            #################################################################

            print('with full inheritance')
            clf_inheritance.fit(X=X_train, y=y_train)
            proba_predictions = clf_inheritance.predict_proba(X=X_test)[:,1]
            proba_predictions_df['full_inheritance'] = proba_predictions
            errors_df['full_inheritance'] = np.abs(proba_predictions - y_test)

            roc_auc = roc_auc_score(y_test, proba_predictions)
            predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
            # print(classification_report(y_test, predictions))
            metrics = get_classification_metrics(predictions, y_test)
            metrics['roc_auc'] = round(roc_auc, 4)
            accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
            metrics['accuracy'] = round(accuracy, 4)
            print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
            inheritance_metrics.append(list(metrics.values()))
            # with open('../results/inheritance_dag.pkl', 'wb') as f:
            #     pickle.dump(clf_inheritance.dag, f)
            print('\n-----------------------------\n')

            #################################################################

            print('with knn imputation')
            clf_knn.fit(X_train, y_train)
            proba_predictions = clf_knn.predict_proba(X_test)[:,1]
            proba_predictions_df['knn_imputation'] = proba_predictions
            errors_df['knn_imputation'] = np.abs(proba_predictions - y_test)
            predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
            # t = find_optimal_threshold(proba_predictions, y_test)
            # predictions = [1 if p > t else 0 for p in proba_predictions]
            # y_test = data[0][1][data.target_col]
            roc_auc = roc_auc_score(y_test, proba_predictions)
            metrics = get_classification_metrics(predictions, y_test)
            # print(classification_report(y_test, predictions))
            metrics['roc_auc'] = round(roc_auc, 4)
            accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
            metrics['accuracy'] = round(accuracy, 4)
            knn_metrics.append(list(metrics.values()))
            print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
            print('\n-----------------------------\n')

            #################################################################

            print('with MICE imputation')
            clf_mice.fit(X_train, y_train)
            proba_predictions = clf_mice.predict_proba(X_test)[:,1]
            proba_predictions_df['mice_imputation'] = proba_predictions
            errors_df['mice_imputation'] = np.abs(proba_predictions - y_test)
            predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
            # t = find_optimal_threshold(proba_predictions, y_test)
            # predictions = [1 if p > t else 0 for p in proba_predictions]
            # y_test = data[0][1][data.target_col]
            roc_auc = roc_auc_score(y_test, proba_predictions)
            metrics = get_classification_metrics(predictions, y_test)
            # print(classification_report(y_test, predictions))
            metrics['roc_auc'] = round(roc_auc, 4)
            accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
            metrics['accuracy'] = round(accuracy, 4)
            mice_metrics.append(list(metrics.values()))
            print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
            print('\n-----------------------------\n')

            #################################################################

            print('off-the-shelf xgboost:')
            clf_vanilla.fit(X_train, y_train)
            proba_predictions = clf_vanilla.predict_proba(X_test)[:,1]
            proba_predictions_df['off-the-shelf xgb'] = proba_predictions
            errors_df['off-the-shelf xgb'] = np.abs(proba_predictions - y_test)
            # t = find_optimal_threshold(proba_predictions, y_test)
            # predictions = [1 if p > t else 0 for p in proba_predictions]
            # y_test = data[0][1][data.target_col]
            roc_auc = roc_auc_score(y_test, proba_predictions)
            predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
            # print(classification_report(y_test, predictions))
            metrics = get_classification_metrics(predictions, y_test)
            metrics['roc_auc'] = round(roc_auc, 4)
            accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
            metrics['accuracy'] = round(accuracy, 4)
            vanilla_metrics.append(list(metrics.values()))
            print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
            print('\n-----------------------------\n')

            ######################################################################

            print('stacked model:')
            clf_stacked.fit(X_train, y_train)
            proba_predictions = clf_stacked.predict_proba(X_test)[:,1]
            proba_predictions_df['stacked'] = proba_predictions
            errors_df['stacked'] = np.abs(proba_predictions - y_test)
            predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
            # t = find_optimal_threshold(proba_predictions, y_test)
            # predictions = [1 if p > t else 0 for p in proba_predictions]
            # y_test = data[0][1][data.target_col]
            roc_auc = roc_auc_score(y_test, proba_predictions)
            metrics = get_classification_metrics(predictions, y_test)
            # print(classification_report(y_test, predictions))
            metrics['roc_auc'] = round(roc_auc, 4)
            accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
            metrics['accuracy'] = round(accuracy, 4)
            stacked_metrics.append(list(metrics.values()))
            print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
            print('\n======================\n===================\n')

            ###########################################################

            print('DEW model:')
            # clf_ds.fit(X_train, y_train)
            
            # use validation set for nearest-neighbor search + prediction;
            # avoids train set leakage

            clf_ds.fit(X_val, y_val)
            proba_predictions = clf_ds.predict_proba(X_test)[:, 1]
            proba_predictions_df['DEW'] = proba_predictions
            errors_df['DEW'] = np.abs(proba_predictions - y_test)
            predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
            weights.append(clf_ds.weight_assignments)
            
            # t = find_optimal_threshold(proba_predictions, y_test)
            # predictions = [1 if p > t else 0 for p in proba_predictions]
            # y_test = data[0][1][data.target_col]
            roc_auc = roc_auc_score(y_test, proba_predictions)
            metrics = get_classification_metrics(predictions, y_test)
            # print(classification_report(y_test, predictions))
            metrics['roc_auc'] = round(roc_auc, 4)
            accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
            metrics['accuracy'] = round(accuracy, 4)
            dew_metrics.append(list(metrics.values()))
            print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
            print('\n======================\n===================\n')

            ###########################################################

            print('Oracle Classifier:')
            clf_oracle.fit(X_train, y_train)
            proba_predictions = clf_oracle.predict_proba(X_test, y_test)[:, 1]
            proba_predictions_df['Oracle'] = proba_predictions
            errors_df['Oracle'] = np.abs(proba_predictions - y_test)
            predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
            roc_auc = roc_auc_score(y_test, proba_predictions)
            metrics = get_classification_metrics(predictions, y_test)
            metrics['roc_auc'] = round(roc_auc, 4)
            accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
            metrics['accuracy'] = round(accuracy, 4)
            oracle_metrics.append(list(metrics.values()))
            print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
            print('\n======================\n===================\n')

            ##############################################################

            proba_predictions_dfs_list.append(proba_predictions_df)
            errors_dfs_list.append(errors_df)

        proba_predictions_df_total = pd.concat(proba_predictions_dfs_list)
        proba_predictions_df_total.index = range(len(proba_predictions_df_total))
        self.proba_predictions_df_total = proba_predictions_df_total
        
        errors_df_total = pd.concat(errors_dfs_list)
        errors_df_total.index = range(len(errors_df_total))
        self.errors_df_total = errors_df_total

        weights_df = pd.concat(weights)
        self.weights_df = weights_df

        # plot_prediction_errors(
        #     y_true=np.array(y_trues), 
        #     proba_predictions_df=proba_predictions_df_total,
        #     title='Example Comparison of Model Prediction Errors',
        #     xlabel='Index of Samples',
        #     ylabel='Class Probability Prediction Error',
        #     outfile=prediction_error_viz_outfile
        # )

        inheritance_metrics = np.vstack(inheritance_metrics)
        knn_metrics = np.vstack(knn_metrics)
        mice_metrics = np.vstack(mice_metrics)
        vanilla_metrics = np.vstack(vanilla_metrics)
        stacked_metrics = np.vstack(stacked_metrics)
        dew_metrics = np.vstack(dew_metrics)
        oracle_metrics = np.vstack(oracle_metrics)

        mean_inheritance_metrics = np.mean(inheritance_metrics, axis=0)
        mean_knn_metrics = np.mean(knn_metrics, axis=0)
        mean_mice_metrics = np.mean(mice_metrics, axis=0)
        mean_vanilla_metrics = np.mean(vanilla_metrics, axis=0)
        mean_stacked_metrics = np.mean(stacked_metrics, axis=0)
        mean_dew_metrics = np.mean(dew_metrics, axis=0)
        mean_oracle_metrics = np.mean(oracle_metrics, axis=0)

        print('\n=========\n==========\n')

        print('cross-validated mean metrics:\n----------------\n')

        metric_types = [
                'sensitivity', 'specificity', 'ppv', 'npv', 'gmean_sens_spec',
                'gmean_all_metrics', 'roc_auc', 'accuracy'
            ]
        cv_results_means = {}

        for t in [
            'mean_inheritance_metrics', 'mean_knn_metrics', 
            'mean_mice_metrics','mean_vanilla_metrics', 
            'mean_stacked_metrics', 'mean_dew_metrics', 'mean_oracle_metrics'
        ]:
            cv_results_means[t] = {}
            print(t + str(': '))
            for i, metric in enumerate(metric_types):
                cv_results_means[metric + '_' + t] = eval(t)[i]
                print(metric + ': ' + str(eval(t)[i]))
            
            print('-------------')

        df = pd.DataFrame({k: [v] for k, v in cv_results_means.items()})
        return df, weights_df, errors_df_total


def run_randomized_synthetic_clf_experiments():
    """
    
    """
    results_dfs = []
    results_cols = [
        'missingness_type', 'n_clusters_per_class', 'n_informative_features',
        'n_redundant_features', 'n_features_masked', 'missingness_rate',
        'synthetic_data_mechanism', 'dew_weights_file', 
        'prediction_errors_file'
    ]
    
    # == params for dataset generator
    synthetic_data_mechanisms = ['make_classification']
    mnar_methods = ['logistic']
    miss_rates = [x / 10 for x in range(2, 9)]
    features_without_missing_values_percentages = [x / 10 for x in range(2, 9)]
    

    # == make_classification params
    n_samples = [1500]
    n_features = [8]
    n_classes = [2]
    n_clusters_per_class = [2]
    n_informative = [2,3,4]
    n_redundant = [2,3,4]
    flip_y = [0.03]
    random_state = [0]

    dataset_generator_params = product(
        n_samples, n_features, n_classes, 
        n_clusters_per_class, n_informative, n_redundant, flip_y, random_state
    )

    keywords_ = [
        'n_samples', 'n_features', 'n_classes', 'n_clusters_per_class',
        'n_informative', 'n_redundant', 'flip_y', 'random_state'
    ]
    dataset_generator_kwargs_list = []
    for params in dataset_generator_params:
        kwargs_ = {
            k: arg_ for k, arg_ in zip(keywords_, params)
        }
        dataset_generator_kwargs_list.append(kwargs_)


    param_grid_names = [
        'synthetic_data_mechanism', 'miss_rate', 'mnar_method', 
        'percent_features_without_missing_values'
    ]

    mcar_param_grid = product(
        synthetic_data_mechanisms, miss_rates, [None], [None]
    )

    mar_param_grid = product(
        synthetic_data_mechanisms, miss_rates, [None], 
        features_without_missing_values_percentages
    )

    mnar_param_grid = product(
        synthetic_data_mechanisms, miss_rates, mnar_methods, 
        features_without_missing_values_percentages
    )

    # == MCAR experiments
    # for param_idx, params in enumerate(mcar_param_grid):
    #     for data_gen_idx, data_gen_kwargs in enumerate(dataset_generator_kwargs_list):
            
    #         param_dict = {k: v for k, v in zip(param_grid_names, params)}
    #         param_dict['nan_type'] = 'MCAR'
            
    #         exp = SyntheticClassificationExperiment(**param_dict, **data_gen_kwargs)
            
    #         df, weights_df, errors_df = exp.run()
            
            
    #         filename = str((param_idx, data_gen_idx)) + '.csv'
    #         weights_filename = '../results/synthetic_classification/mcar/dew_weights/' + filename
    #         errors_filename = '../results/synthetic_classification/mcar/prediction_errors/' + filename
    #         weights_df.to_csv(weights_filename)
    #         errors_df.to_csv(errors_filename)

    #         param_dict['dew_weights_filename'] = weights_filename
    #         param_dict['prediction_errors_filename'] = errors_filename
            
            
    #         params_df = pd.DataFrame({k: [v] for k, v in param_dict.items()})
    #         params_df.index = [0]
            
    #         data_gen_args_df = pd.DataFrame({k: [v] for k, v in data_gen_kwargs.items()})
    #         data_gen_args_df.index = [0]
            
    #         df.index = [0]
            
    #         experiment_setup_df = pd.merge(
    #             left=params_df, right=data_gen_args_df, how='outer',
    #             left_index=True, right_index=True
    #         )
    #         result_df = pd.merge(
    #             left=experiment_setup_df, right=df, how='outer',
    #             left_index=True, right_index=True
    #         )
    #         results_dfs.append(result_df)
    #         results_filename = '../results/synthetic_classification/mcar/results/' + filename
    #         result_df.to_csv(results_filename)
            


    # == MAR experiments
    # for param_idx, params in enumerate(mar_param_grid):
    #     for data_gen_idx, data_gen_kwargs in enumerate(dataset_generator_kwargs_list):
            
    #         param_dict = {k: v for k, v in zip(param_grid_names, params)}
    #         param_dict['nan_type'] = 'MAR'
            
    #         exp = SyntheticClassificationExperiment(**param_dict, **data_gen_kwargs)
            
    #         df, weights_df, errors_df = exp.run()
            
            
    #         filename = str((param_idx, data_gen_idx)) + '.csv'
    #         weights_filename = '../results/synthetic_classification/mar/dew_weights/' + filename
    #         errors_filename = '../results/synthetic_classification/mar/prediction_errors/' + filename
    #         weights_df.to_csv(weights_filename)
    #         errors_df.to_csv(errors_filename)

    #         param_dict['dew_weights_filename'] = weights_filename
    #         param_dict['prediction_errors_filename'] = errors_filename
            
            
    #         params_df = pd.DataFrame({k: [v] for k, v in param_dict.items()})
    #         params_df.index = [0]
            
    #         data_gen_args_df = pd.DataFrame({k: [v] for k, v in data_gen_kwargs.items()})
    #         data_gen_args_df.index = [0]
            
    #         df.index = [0]
            
    #         experiment_setup_df = pd.merge(
    #             left=params_df, right=data_gen_args_df, how='outer',
    #             left_index=True, right_index=True
    #         )
    #         result_df = pd.merge(
    #             left=experiment_setup_df, right=df, how='outer',
    #             left_index=True, right_index=True
    #         )
    #         results_dfs.append(result_df)
    #         results_filename = '../results/synthetic_classification/mar/results/' + filename
    #         result_df.to_csv(results_filename)



    # == MNAR experiments
    for param_idx, params in enumerate(mnar_param_grid):
        for data_gen_idx, data_gen_kwargs in enumerate(dataset_generator_kwargs_list):
            
            param_dict = {k: v for k, v in zip(param_grid_names, params)}
            param_dict['nan_type'] = 'MNAR'
            
            exp = SyntheticClassificationExperiment(**param_dict, **data_gen_kwargs)
            
            df, weights_df, errors_df = exp.run()
            
            
            filename = str((param_idx, data_gen_idx)) + '.csv'
            weights_filename = '../results/synthetic_classification/mnar/dew_weights/' + filename
            errors_filename = '../results/synthetic_classification/mnar/prediction_errors/' + filename
            weights_df.to_csv(weights_filename)
            errors_df.to_csv(errors_filename)

            param_dict['dew_weights_filename'] = weights_filename
            param_dict['prediction_errors_filename'] = errors_filename
            
            
            params_df = pd.DataFrame({k: [v] for k, v in param_dict.items()})
            params_df.index = [0]
            
            data_gen_args_df = pd.DataFrame({k: [v] for k, v in data_gen_kwargs.items()})
            data_gen_args_df.index = [0]
            
            df.index = [0]
            
            experiment_setup_df = pd.merge(
                left=params_df, right=data_gen_args_df, how='outer',
                left_index=True, right_index=True
            )
            result_df = pd.merge(
                left=experiment_setup_df, right=df, how='outer',
                left_index=True, right_index=True
            )
            results_dfs.append(result_df)
            results_filename = '../results/synthetic_classification/mnar/results/' + filename
            result_df.to_csv(results_filename)

    final_results_df = pd.concat(results_dfs)
    final_results_df.to_csv('../results/synthetic_classification/final_results.csv')

    return final_results_df


class CustomExperiment():

    def _init_pipelines(self, classifier_pool=None):
                # construct pipelines
        if isinstance(classifier_pool, list):
            pipelines = {}
            for p in classifier_pool:
                if isinstance(p, ClassifierWithImputation):
                    pipelines[
                        'Estim(' + p.estimator_name + ')_Imputer(' + p.imputer_name + ')'
                    ] = p
                else:
                    pipelines[
                        'Estim(' + p.estimator_name + ')_Imputer(Identity)'
                    ] = p
        elif classifier_pool is None:
            xgb_params = {'n_jobs': 1, 'max_depth': 4, 'n_estimators': 50, 'verbosity': 0}
            rf_params = {'n_jobs': 1, 'max_depth': 4, 'n_estimators': 50, 'verbose': 0}
            knn_params = {'n_jobs': 1, 'weights': 'distance', 'n_neighbors': 5}

            models = [
                xgb.XGBClassifier(**xgb_params),
                RandomForestClassifier(**rf_params),
            ]
            imputers = [
                IterativeImputer(BayesianRidge()),
                IterativeImputer(xgb.XGBRegressor(**xgb_params)),
                IterativeImputer(RandomForestRegressor(**rf_params)),
                KNNImputer(n_neighbors=5)
            ]
            clf_imputer_pairs = product(models, imputers)
            pipelines_list = [
                ClassifierWithImputation(
                    estimator=clf, 
                    imputer=imp
                )
                for clf, imp in clf_imputer_pairs
            ]
            pipelines = {
                'Estim(' + p.estimator_name + ')_Imputer(' + p.imputer_name + ')': p
                for p in pipelines_list
            }
            vanilla_xgb = ClassifierWithImputation(
                xgb.XGBClassifier(**xgb_params),
                IdentityImputer()
            )
            pipelines['Estim(' + str(type(vanilla_xgb)) + ')_Imputer(Identity)'] = vanilla_xgb
            # so we have 9 total pipelines

        else:
            pipelines = classifier_pool

        assert isinstance(pipelines, dict), 'The Classifier Pool (Pipelines) must be a dictionary, not ' + str(type(pipelines))
        self.pipelines = pipelines
        self.unfitted_pipelines = deepcopy(self.pipelines)

        self.clf_stacked = StackingClassifier(
            estimators=list(self.pipelines.items()),
            final_estimator=xgb.XGBClassifier(
                **{'n_jobs': 1, 'max_depth': 3, 'n_estimators': 60, 'verbosity': 0}
            ),
            cv='prefit',
            n_jobs=1
        )

        self.clf_dew_models = [
            DEWClassifier(
                classifier_pool=self.pipelines,
                n_neighbors=3,
                n_top_to_choose=n
            )
            for n in [1,3,5, None]
        ]

    def __init__(
        self,
        dataset: MissDataset,
        exp_type='mcar',
        dataset_name='',
        name='Experiment_' + str(datetime.now()),
        base_dir=None,
        classifier_pool=None,
        random_state=42
    ):
        self.dataset = dataset
        self.n_folds = len(dataset.train_val_test_triples)

        self.name = name
        
        if base_dir is None:
            base_dir = '../Experiment_Trials/' + dataset_name + '/' + name

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        self.base_dir = os.path.join(base_dir, exp_type)
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        
        self.results_dir = os.path.join(self.base_dir, 'results')
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        
        self.cv_results = {
            i: {} for i in range(self.n_folds)
        }

        self.results = {}
        self.random_state = random_state

        self._init_pipelines(classifier_pool=classifier_pool)
        
        self.metrics = {p: [] for p in self.pipelines}
        self.metrics['Uniform Model Averaging'] = []
        self.metrics[str(type(self.clf_stacked))] = []
        for clf_dew in self.clf_dew_models:
            self.metrics[
                str(type(clf_dew)) + '_top_' + str(clf_dew.n_top_to_choose)
            ] = []

    def _run_one_pipeline(self, pipeline, pipeline_name, X_train, y_train, X_test, y_test):
        # proba_predictions_df = pd.DataFrame({})
        # errors_df = pd.DataFrame({})
        pipeline.fit(X_train, y_train)
        proba_predictions = pipeline.predict_proba(X_test)[:,1]
        # proba_predictions_df[pipeline_name] = proba_predictions
        errors = np.abs(proba_predictions - y_test)
        # errors_df[pipeline_name] = errors
        predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
        roc_auc = roc_auc_score(y_test, proba_predictions)
        metrics = get_classification_metrics(predictions, y_test)
        # print(classification_report(y_test, predictions))
        metrics['roc_auc'] = round(roc_auc, 4)
        accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
        metrics['accuracy'] = round(accuracy, 4)
        self.metrics[pipeline_name].append(list(metrics.values()))

        return proba_predictions, errors



    def do_experiment_one_fold(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # run baselines
        proba_predictions_df = pd.DataFrame({})
        errors_df = pd.DataFrame({})
        with tqdm(total=len(list(self.pipelines))) as pbar:
            for pipeline_type in self.pipelines:
                print('running ' + pipeline_type)
                pipeline = self.pipelines[pipeline_type]
                proba_predictions, errors = self._run_one_pipeline(
                    pipeline,
                    pipeline_type,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test
                )
                proba_predictions_df[pipeline_type] = proba_predictions
                errors_df[pipeline_type] = errors
                print(pipeline_type + ' complete')
                pbar.update(1)

        # run uniform model averaging over baselines
        print('running uniform averaging')
        proba_predictions = proba_predictions_df.mean(axis=1)
        errors = errors_df.mean(axis=1)
        proba_predictions_df['Uniform Model Averaging'] = proba_predictions
        errors_df['Uniform Model Averaging'] = errors
        predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
        roc_auc = roc_auc_score(y_test, proba_predictions)
        metrics = get_classification_metrics(predictions, y_test)
        # print(classification_report(y_test, predictions))
        metrics['roc_auc'] = round(roc_auc, 4)
        accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
        metrics['accuracy'] = round(accuracy, 4)
        self.metrics['Uniform Model Averaging'].append(list(metrics.values()))
        # need to store metrics names somewhere accessible; not ideal way to do this but it works
        self.metric_type_cols = list(metrics.keys())
        print('uniform averaging complete')

        # run stacked classifier
        print('running stacked generalization')
        proba_predictions, errors = self._run_one_pipeline(
            self.clf_stacked,
            str(type(self.clf_stacked)),
            X_train=X_val, y_train=y_val,
            X_test=X_test, y_test=y_test
        )
        proba_predictions_df[str(type(self.clf_stacked))] = proba_predictions
        errors_df[str(type(self.clf_stacked))] = errors
        print('stacked generalization complete')
        
        # run DEW classifiers
        print('running DEW models')
        for clf_dew in self.clf_dew_models:
            proba_predictions, errors = self._run_one_pipeline(
                clf_dew,
                str(type(clf_dew)) + '_top_' + str(clf_dew.n_top_to_choose),
                X_train=X_val, y_train=y_val, # use val (second-stage training set)
                X_test=X_test, y_test=y_test
            )
            weights = clf_dew.weight_assignments # this will be the same for each trial
            proba_predictions_df[
                str(type(clf_dew)) + '_top_' + str(clf_dew.n_top_to_choose)
            ] = proba_predictions
            errors_df[
                str(type(clf_dew)) + '_top_' + str(clf_dew.n_top_to_choose)
            ] = errors
        print('DEW complete')

        return proba_predictions_df, errors_df, weights

    def do_kfold_experiments(self):
        y_trues = []
        preds_dfs = []
        errors_dfs = []
        all_dew_weights = []
        for fold in range(self.dataset.n_folds):
            self._init_pipelines()
            train, val, test = self.dataset[fold]
            y_test = test[self.dataset.target_col]
            y_trues += list(y_test)

            y_train = train[self.dataset.target_col]
            y_val = val[self.dataset.target_col]
            
            X_test = test.drop(self.dataset.target_col, axis=1)
            X_val = val.drop(self.dataset.target_col, axis=1)
            X_train = train.drop(self.dataset.target_col, axis=1)

            proba_predictions_df, errors_df, weights = self.do_experiment_one_fold(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test
            )
            preds_dfs.append(proba_predictions_df)
            errors_dfs.append(errors_df)
            all_dew_weights.append(weights)

        all_cols = list(self.pipelines.keys()) + [
            'Uniform Model Averaging',
            str(type(self.clf_stacked)),
        ]
        all_cols += [
            str(type(clf_dew)) + '_top_' + str(clf_dew.n_top_to_choose)
            for clf_dew in self.clf_dew_models
        ]
        print(len(preds_dfs))
        for df in preds_dfs:
            print(df.columns)

        print('concatenating predictions')
        preds_df = pd.concat(preds_dfs)
        # preds_df.index = range(len(preds_df))
        # preds_df.columns = all_cols
        self.proba_predictions_df_total = preds_df
        
        print('concatenating errors')
        errors_df_total = pd.concat(errors_dfs)
        # errors_df_total.index = range(len(errors_df_total))
        # errors_df_total.columns = all_cols
        self.errors_df_total = errors_df_total

        print('concatenating weights')
        weights_df = pd.concat(all_dew_weights)
        # weights_df.columns = list(self.pipelines.keys())
        self.weights_df = weights_df
        
    def run(self):
        self.do_kfold_experiments()
        print('making metrics df')
        
        metrics_df = pd.DataFrame({})
        for m in self.metrics:
            metrics_df[m] = np.mean(self.metrics[m], axis=0)
        print(metrics_df)
        metrics_df.index = self.metric_type_cols
        print(metrics_df)
        print('experimental run complete.')
        return metrics_df, self.weights_df, self.errors_df_total


def run_custom_experiments(data, dataset_name, miss_param_dict, target_col):

    miss_param_grid = list(product(*tuple(miss_param_dict.values())))

    param_lookup_dict = {}
    metrics_dfs = []
    miss_type = miss_param_dict['missing_mechanism']
    name = miss_type[0] + '_Experiment_' + str(datetime.now())

    with tqdm(total=len(miss_param_grid)) as pbar:
        for i, params in enumerate(miss_param_grid):
            data_copy = deepcopy(data)
            params = {
                k: p 
                for k, p in zip(list(miss_param_dict.keys()), params)
            }
            param_lookup_dict[i] = params

            # name = 'Experiment_' + str(datetime.now())
            dataset = MissDataset(
                data=data_copy,
                target_col=target_col,
                n_folds=5,
                **params,
            )
            experiment = CustomExperiment(
                dataset=dataset, dataset_name=dataset_name, 
                exp_type=params['missing_mechanism'],
                name=name
            )
            metrics_df, weights_df, errors_df = experiment.run()

            filename = str(i) + '.csv'
            weights_filename = os.path.join(experiment.results_dir, 'weights_' + filename)
            errors_filename = os.path.join(experiment.results_dir, 'errors_' + filename)
            
            print('writing weights')
            weights_df.to_csv(weights_filename)
            
            print('errors df:')
            print(errors_df)
            print('writing errors')
            errors_df.to_csv(errors_filename)

            print('appending metrics df')
            metrics_dfs.append(metrics_df)
            print('making metrics filename')
            metrics_filename = os.path.join(experiment.results_dir, 'metrics_' + filename)
            print('writing metrics to csv')
            metrics_df.to_csv(metrics_filename)
            print('updating progress bar after index ' + str(i))
            pbar.update(1)
    
    final_results = pd.concat(metrics_dfs)
    final_results.to_csv(os.path.join(experiment.base_dir, 'final_results.csv'))

    param_lookup_dict_json = json.dumps(param_lookup_dict)
    with open(os.path.join(experiment.base_dir, 'params_lookup.json'), 'w') as f:
        f.write(param_lookup_dict_json)


SUPPORTED_DATASET_NAMES = [
    'Breast Cancer Wisconsin Diagnostic',
    'Breast Cancer Wisconsin Prognostic',
    'Thyroid',
    'Heart Disease',
    'Iris',
    'Dermatology',
    'Parkinsons',
    'Diabetes 130 hospitals',
    'EEG Eye State Data Set',
    'Cervical Cancer',
    'Parkinson Speech',
    'Myocardial infarction complications'
]

if __name__ == '__main__':
    # run_randomized_synthetic_clf_experiments()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str
    )
    parser.add_argument(
        '--target_column', type=str,
    )
    parser.add_argument(
        '--dataset_name', type=str,
        default=''
    )
    args = parser.parse_args()
    # PATH='../data/heart_cleveland_upload.csv'
    # data_object = pd.read_csv(PATH)
    # TARGET_COL = 'condition'

    PATH = args.data_path
    data_object = pd.read_csv(PATH)
    TARGET_COL = args.target_column
    dataset_name = args.dataset_name
    if len(dataset_name) == 0:
        dataset_name = PATH.split('/')[-1]

    dataset_name = dataset_name.lower()

    MCAR_PARAM_DICT = {
        'p_miss': [x/10 for x in range(3,9)], 
        'missing_mechanism': ["MCAR"], 
        'opt': [None], 
        'p_obs': [None], 
        'q': [None],
    }

    MAR_PARAM_DICT = {
        'p_miss': [x/10 for x in range(3,9)], 
        'missing_mechanism': ["MAR"], 
        'opt': [None], 
        'p_obs': [x/10 for x in range(3,9)], 
        'q': [None],
    }

    MNAR_PARAM_DICT = {
        'p_miss': [x/10 for x in range(3,9)], 
        'missing_mechanism': ["MNAR"], 
        'opt': ['logistic'], 
        'p_obs': [x/10 for x in range(3,9)], 
        'q': [None],
    }

    # MCAR_PARAM_GRID = product(list(MCAR_PARAM_DICT.values()))
    # MAR_PARAM_GRID = product(list(MAR_PARAM_DICT.values()))
    # MNAR_PARAM_GRID = product(list(MNAR_PARAM_DICT.values()))

    for d in [MCAR_PARAM_DICT, MAR_PARAM_DICT, MNAR_PARAM_DICT]:
        run_custom_experiments(
            data=data_object, 
            dataset_name=dataset_name,
            miss_param_dict=d, 
            target_col=TARGET_COL
        )
