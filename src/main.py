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
import pickle
# from deslib.base import BaseDS
# from deslib.des import DESClustering, KNORAE, KNORAU, KNOP, METADES
# from deslib_missingness.static import Oracle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_gaussian_quantiles
from sklearn.ensemble import RandomForestRegressor, StackingClassifier, StackingRegressor, VotingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import classification_report, f1_score, log_loss, max_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import xgboost as xgb

sys.path.append('.')

from data_loaders import Dataset, MedicalDataset, MissDataset, DataLoadersEnum, split_parkinsons_data
from new_base import ClassifierWithImputation, IdentityImputer, RegressorWithImputation
# from new_models import NonmissingSubspaceInheritanceClassifier, DEWClassifier
from new_models import DEWClassifier, DEWRegressor
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


class CustomClassificationExperiment():

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
            # vanilla_xgb = ClassifierWithImputation(
            #     xgb.XGBClassifier(**xgb_params),
            #     IdentityImputer()
            # )
            # pipelines['Estim(' + str(type(vanilla_xgb)) + ')_Imputer(Identity)'] = vanilla_xgb
            # so we have [---9---] __8__ total baselines

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
            # final_estimator=DecisionTreeClassifier(max_depth=5),
            cv='prefit',
            n_jobs=1
        )

        self.clf_dew = DEWClassifier(
            classifier_pool=self.pipelines,
            n_neighbors=7,
            n_top_to_choose=[1,3,None],
            competence_threshold=0.5
        )

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
        self.label_enc = OneHotEncoder()#convert categorical target value to numerical binary ones
        self.label_enc.fit(np.array(self.dataset.data[self.dataset.target_col]).reshape(-1, 1))

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
        
        self.metrics = {p: [] for p in self.pipelines}#where is the pipeline??
        self.metrics['Uniform Model Averaging'] = []
        self.metrics[str(type(self.clf_stacked))] = []
        for top_n in self.clf_dew.n_top_to_choose:
            self.metrics['dew_top_' + str(top_n)] = []
        self.weights_dfs = {}

    def _run_one_pipeline(self, pipeline, pipeline_name, X_train, y_train, X_test, y_test):
        pipeline.fit(X_train, y_train)
        proba_predictions = pipeline.predict_proba(X_test)[:, 1]
        if isinstance(proba_predictions, list):
            proba_predictions = proba_predictions[0]
        
        y_test_2d = np.array(self.label_enc.transform(y_test.reshape(-1, 1)).todense())
        errors = np.abs(y_test - proba_predictions)
        predictions = np.round(proba_predictions)
        single_label_y_test = np.argmax(y_test_2d, axis=1)
        roc_auc = roc_auc_score(y_test, proba_predictions)
        
        metrics = {}
        metrics['roc_auc'] = round(roc_auc, 4)
        accuracy = 1 - (np.sum(np.logical_xor(predictions, single_label_y_test)) / len(predictions))
        metrics['accuracy'] = round(accuracy, 4)
        metrics['f1_score'] = f1_score(single_label_y_test, predictions)
        self.metrics[pipeline_name].append(list(metrics.values()))

        return proba_predictions, errors



    def do_experiment_one_fold(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # run baselines
        self._init_pipelines()
        proba_predictions_per_pipeline = {}
        errors_df = pd.DataFrame({})
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)
        

        pipelines_start = datetime.now()
        for pipeline_type in self.pipelines:
            pipeline = self.pipelines[pipeline_type]
            proba_predictions, errors = self._run_one_pipeline(
                pipeline,
                pipeline_type,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test
            )
            proba_predictions_per_pipeline[pipeline_type] = proba_predictions
            errors_df[pipeline_type] = errors
        print('pipelines completed in ' + str(datetime.now() - pipelines_start))
        proba_predictions_per_pipeline_2d = {
            k: np.hstack([1 - np.ravel(probas).reshape(-1,1), np.ravel(probas).reshape(-1,1)])
            for k, probas in proba_predictions_per_pipeline.items()
        }
        baseline_test_predictions_dict = deepcopy(proba_predictions_per_pipeline_2d)
        
        dstacked_predictions = np.dstack(list(proba_predictions_per_pipeline.values())) 
        dstacked_predictions = dstacked_predictions[0] # shape[1,x,y] -> shape[x,y]
        proba_predictions = np.mean(
            dstacked_predictions,
            axis=-1
        )
        y_test_2d = np.array(self.label_enc.transform(y_test.reshape(-1, 1)).todense())
        errors = np.abs(y_test - proba_predictions)
        proba_predictions_per_pipeline['Uniform Model Averaging'] = proba_predictions
        errors_df['Uniform Model Averaging'] = errors
        predictions = np.round(proba_predictions)
        single_label_y_test = np.argmax(y_test_2d, axis=1)
        roc_auc = roc_auc_score(y_test, proba_predictions)
        
        metrics = {}
        metrics['roc_auc'] = round(roc_auc, 4)
        accuracy = 1 - (np.sum(np.logical_xor(predictions, single_label_y_test)) / len(predictions))
        metrics['accuracy'] = round(accuracy, 4)
        metrics['f1_score'] = f1_score(y_test, predictions)
        self.metrics['Uniform Model Averaging'].append(list(metrics.values()))
        # need to store metrics names somewhere accessible; not ideal way to do this but it works
        self.metric_type_cols = list(metrics.keys())

        # run stacked classifier
        proba_predictions, errors = self._run_one_pipeline(
            self.clf_stacked,
            str(type(self.clf_stacked)),
            X_train=X_val, y_train=y_val,
            X_test=X_test, y_test=y_test
        )
        proba_predictions_per_pipeline[str(type(self.clf_stacked))] = proba_predictions
        errors_df[str(type(self.clf_stacked))] = errors
        
        # run DEW classifiers
        print('running DEW models')
        dew_start = datetime.now()
        self.clf_dew.set_baseline_test_predictions(baseline_test_predictions_dict)

        self.clf_dew.fit(X_val, y_val)
        proba_predictions_sets, weights_sets, distances = self.clf_dew.predict_proba(X_test)
        compiled_models = '../compiled_models'
        if not os.path.exists(compiled_models):
            os.mkdir(compiled_models)
        
        with open('dew_model.pkl', 'wb') as f:
            pickle.dump(self.clf_dew, f)

        distances_df = pd.DataFrame(distances)
        
        for top_n, proba_predictions in proba_predictions_sets.items():
            proba_predictions = proba_predictions[:, 1] # remove slice if multiclass
            proba_predictions_per_pipeline['dew_top_' + str(top_n)] = proba_predictions
            y_test_2d = np.array(self.label_enc.transform(y_test.reshape(-1, 1)).todense())
            
            errors = np.abs(proba_predictions - y_test)
            errors_df['dew_top_' + str(top_n)] = errors
            predictions = np.round(proba_predictions)
            single_label_y_test = np.argmax(y_test_2d, axis=1)
            roc_auc = roc_auc_score(y_test, proba_predictions)
            metrics = {}
            metrics['roc_auc'] = round(roc_auc, 4)
            accuracy = 1 - (np.sum(np.logical_xor(predictions, single_label_y_test)) / len(predictions))
            metrics['accuracy'] = round(accuracy, 4)
            metrics['f1_score'] = f1_score(single_label_y_test, predictions)
            self.metrics['dew_top_' + str(top_n)].append(list(metrics.values()))
        
        # == the following two lines are only necessary if multiclass
        # for k in proba_predictions_per_pipeline.keys():
        #     proba_predictions_per_pipeline[k] = proba_predictions_per_pipeline[k][:, 1]
        # TODO{i don't think those lines are correct, should do another way. but not doing multiclass now so later?}
        print('DEW completed in ' + str(datetime.now() - dew_start))
        print(self.metrics)

        return errors_df, weights_sets, pd.DataFrame(proba_predictions_per_pipeline), distances_df

    def do_kfold_experiments(self):
        y_trues = []
        errors_dfs = []
        proba_predictions_dfs = []
        distances_dfs = []
        all_dew_weights = {top_n: [] for top_n in self.clf_dew.n_top_to_choose}
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

            errors_df, weights_sets, proba_predictions_df, distances_df = self.do_experiment_one_fold(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test
            )
            errors_dfs.append(errors_df)
            proba_predictions_dfs.append(proba_predictions_df)
            for top_n in weights_sets.keys():
                all_dew_weights[top_n].append(weights_sets[top_n])
            distances_dfs.append(distances_df)

        all_cols = list(self.pipelines.keys()) + [
            'Uniform Model Averaging',
            str(type(self.clf_stacked)),
        ]
        all_cols += [
            str(type(self.clf_dew)) + '_top_' + str(top_n)
            for top_n in self.clf_dew.n_top_to_choose
        ]

        errors_df_total = pd.concat(errors_dfs)
        self.errors_df_total = errors_df_total

        for top_n in all_dew_weights.keys():
            weights = np.vstack(all_dew_weights[top_n])
            self.weights_dfs[top_n] = pd.DataFrame(weights)

        self.proba_predictions_df_total = pd.concat(proba_predictions_dfs)
        self.distances_df_total = pd.concat(distances_dfs)
        
    def run(self):
        self.do_kfold_experiments()
        
        metrics_df = pd.DataFrame({})
        for m in self.metrics:
            metrics_df[m] = np.mean(self.metrics[m], axis=0)
        metrics_df.index = self.metric_type_cols
        print('experimental run complete.')
        return metrics_df, self.errors_df_total, self.weights_dfs, self.proba_predictions_df_total, self.distances_df_total


class CustomRegressionExperiment():

    def _init_pipelines(self, regressor_pool=None):
                # construct pipelines
        if isinstance(regressor_pool, list):
            pipelines = {}
            for p in regressor_pool:
                if isinstance(p, RegressorWithImputation):
                    pipelines[
                        'Estim(' + p.estimator_name + ')_Imputer(' + p.imputer_name + ')'
                    ] = p
                else:
                    pipelines[
                        'Estim(' + p.estimator_name + ')_Imputer(Identity)'
                    ] = p
        elif regressor_pool is None:
            xgb_params = {'n_jobs': 1, 'max_depth': 4, 'n_estimators': 50, 'verbosity': 0}
            rf_params = {'n_jobs': 1, 'max_depth': 4, 'n_estimators': 50, 'verbose': 0}
            knn_params = {'n_jobs': 1, 'weights': 'distance', 'n_neighbors': 5}

            models = [
                xgb.XGBRegressor(**xgb_params),
                RandomForestRegressor(**rf_params),
            ]
            imputers = [
                IterativeImputer(BayesianRidge()),
                IterativeImputer(xgb.XGBRegressor(**xgb_params)),
                IterativeImputer(RandomForestRegressor(**rf_params)),
                KNNImputer(n_neighbors=5)
            ]
            clf_imputer_pairs = product(models, imputers)
            pipelines_list = [
                RegressorWithImputation(
                    estimator=clf, 
                    imputer=imp
                )
                for clf, imp in clf_imputer_pairs
            ]
            pipelines = {
                'Estim(' + p.estimator_name + ')_Imputer(' + p.imputer_name + ')': p
                for p in pipelines_list
            }

        else:
            pipelines = regressor_pool

        assert isinstance(pipelines, dict), 'The Classifier Pool (Pipelines) must be a dictionary, not ' + str(type(pipelines))
        self.pipelines = pipelines
        self.unfitted_pipelines = deepcopy(self.pipelines)

        self.reg_stacked = StackingRegressor(
            estimators=list(self.pipelines.items()),
            final_estimator=xgb.XGBRegressor(
                **{'n_jobs': 1, 'max_depth': 3, 'n_estimators': 60, 'verbosity': 0}
            ),
            # final_estimator=DecisionTreeClassifier(max_depth=5),
            cv='prefit',
            n_jobs=1
        )

        self.reg_dew = DEWRegressor(
            regressor_pool=self.pipelines,
            n_neighbors=7,
            n_top_to_choose=[1,3,None]
        )

    def __init__(
        self,
        dataset: MissDataset,
        exp_type='mcar',
        dataset_name='',
        name='Experiment_' + str(datetime.now()),
        base_dir=None,
        regressor_pool=None,
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

        self._init_pipelines(regressor_pool=regressor_pool)
        
        self.metrics = {p: [] for p in self.pipelines}
        self.metrics['Uniform Model Averaging'] = []
        self.metrics[str(type(self.reg_stacked))] = []
        for top_n in self.reg_dew.n_top_to_choose:
            self.metrics['dew_top_' + str(top_n)] = []
        self.weights_dfs = {}

    def _run_one_pipeline(self, pipeline, pipeline_name, X_train, y_train, X_test, y_test):
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        if isinstance(predictions, list):
            predictions = predictions[0]
        
        errors = np.abs(y_test - predictions)
        metrics = {}
        metrics['mae'] = round(np.mean(errors), 4)
        self.metrics[pipeline_name].append(list(metrics.values()))

        return predictions, errors



    def do_experiment_one_fold(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # run baselines
        self._init_pipelines()
        predictions_per_pipeline = {}
        errors_df = pd.DataFrame({})
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)
        
        pipelines_start = datetime.now()
        for pipeline_type in tqdm(self.pipelines, total=len(self.pipelines)):
            pipeline = self.pipelines[pipeline_type]
            predictions, errors = self._run_one_pipeline(
                pipeline,
                pipeline_type,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test
            )
            predictions_per_pipeline[pipeline_type] = predictions
            errors_df[pipeline_type] = errors
        print('pipelines completed in ' + str(datetime.now() - pipelines_start))
        baseline_test_predictions_dict = deepcopy(predictions_per_pipeline)
        # run uniform model averaging over baselines
        dstacked_predictions = np.dstack(list(predictions_per_pipeline.values())) 
        dstacked_predictions = dstacked_predictions[0] # shape[1,x,y] -> shape[x,y]
        predictions = np.mean(
            dstacked_predictions,
            axis=-1
        ).squeeze()
        errors = np.abs(y_test.squeeze() - predictions)
        predictions_per_pipeline['Uniform Model Averaging'] = predictions
        errors_df['Uniform Model Averaging'] = errors
        metrics = {}
        metrics['mae'] = round(np.mean(errors), 4)
        self.metrics['Uniform Model Averaging'].append(list(metrics.values()))
        # need to store metrics names somewhere accessible; not ideal way to do this but it works
        self.metric_type_cols = list(metrics.keys())

        # run stacked regressor
        predictions, errors = self._run_one_pipeline(
            self.reg_stacked,
            str(type(self.reg_stacked)),
            X_train=X_val, y_train=y_val,
            X_test=X_test, y_test=y_test
        )
        predictions_per_pipeline[str(type(self.reg_stacked))] = predictions
        errors_df[str(type(self.reg_stacked))] = errors
        # print('stacked generalization complete')
        
        # run DEW regressors
        print('running DEW models')
        dew_start = datetime.now()
        self.reg_dew.set_baseline_test_predictions(baseline_test_predictions_dict)

        self.reg_dew.fit(X_val, y_val)
        predictions_sets, weights_sets, distances = self.reg_dew.predict(X_test)

        distances_df = pd.DataFrame(distances)
        
        for top_n, predictions in predictions_sets.items():
            predictions = predictions.squeeze()
            predictions_per_pipeline['dew_top_' + str(top_n)] = predictions
            
            errors = np.abs(predictions - y_test.squeeze())
            errors_df['dew_top_' + str(top_n)] = errors
            metrics = {}
            
            metrics['mae'] = np.round(np.mean(errors), 4)
            self.metrics['dew_top_' + str(top_n)].append(list(metrics.values()))
        
        # == the following two lines are only necessary if multiclass
        # for k in proba_predictions_per_pipeline.keys():
        #     proba_predictions_per_pipeline[k] = proba_predictions_per_pipeline[k][:, 1]
        # TODO{i don't think those lines are correct, should do another way. but not doing multiclass now so later?}
        print('DEW completed in ' + str(datetime.now() - dew_start))
        print(self.metrics)

        return errors_df, weights_sets, pd.DataFrame(predictions_per_pipeline), distances_df

    def do_kfold_experiments(self):
        y_trues = []
        errors_dfs = []
        predictions_dfs = []
        distances_dfs = []
        all_dew_weights = {top_n: [] for top_n in self.reg_dew.n_top_to_choose}
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

            errors_df, weights_sets, predictions_df, distances_df = self.do_experiment_one_fold(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test
            )
            errors_dfs.append(errors_df)
            predictions_dfs.append(predictions_df)
            for top_n in weights_sets.keys():
                all_dew_weights[top_n].append(weights_sets[top_n])
            distances_dfs.append(distances_df)

        all_cols = list(self.pipelines.keys()) + [
            'Uniform Model Averaging',
            str(type(self.reg_stacked)),
        ]
        all_cols += [
            str(type(self.reg_dew)) + '_top_' + str(top_n)
            for top_n in self.reg_dew.n_top_to_choose
        ]

        errors_df_total = pd.concat(errors_dfs)
        self.errors_df_total = errors_df_total

        for top_n in all_dew_weights.keys():
            weights = np.vstack(all_dew_weights[top_n])
            self.weights_dfs[top_n] = pd.DataFrame(weights)

        self.predictions_df_total = pd.concat(predictions_dfs)
        self.distances_df_total = pd.concat(distances_dfs)
        
    def run(self):
        self.do_kfold_experiments()
        
        metrics_df = pd.DataFrame({})
        for m in self.metrics:
            metrics_df[m] = np.mean(self.metrics[m], axis=0)
        metrics_df.index = self.metric_type_cols
        print('experimental run complete.')
        return metrics_df, self.errors_df_total, self.weights_dfs, self.predictions_df_total, self.distances_df_total



def run_custom_experiments(data, dataset_name, miss_param_dict, target_col, task_type='classification'):

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

            dataset = MissDataset(
                data=data_copy,
                target_col=target_col,
                n_folds=5,
                **params,
            )
            if dataset_name == 'parkinsons':
                dataset.split_dataset_hook(
                    splitting_function=split_parkinsons_data, df=dataset.data, n_folds=5
                )
            if task_type == 'classification':
                experiment = CustomClassificationExperiment(
                    dataset=dataset, dataset_name=dataset_name, 
                    exp_type=params['missing_mechanism'],
                    name=name
                )
            else:
                experiment = CustomRegressionExperiment(
                    dataset=dataset, dataset_name=dataset_name, 
                    exp_type=params['missing_mechanism'],
                    name=name
                )
            metrics_df, errors_df, weights_dfs, preds_df, distances_df = experiment.run() 

            filename = str(i) + '.csv'
            errors_filename = os.path.join(experiment.results_dir, 'errors_' + filename)
            errors_df.to_csv(errors_filename)

            metrics_dfs.append(metrics_df)
            metrics_filename = os.path.join(experiment.results_dir, 'metrics_' + filename)
            metrics_df.to_csv(metrics_filename)

            preds_filename = os.path.join(experiment.results_dir, 'predictions_' + filename)
            preds_df.to_csv(preds_filename)

            for top_n in weights_dfs.keys():
                weights_filename = os.path.join(experiment.results_dir, 'weights_top_' + str(top_n) + '_' + filename)
                weights_dfs[top_n].to_csv(weights_filename)

            distances_filename = os.path.join(experiment.results_dir, 'distances_' + filename)
            distances_df.to_csv(distances_filename)
            print('updating progress bar after index ' + str(i))
            pbar.update(1)
    
    final_results = pd.concat(metrics_dfs)
    final_results.to_csv(os.path.join(experiment.base_dir, 'final_results.csv'))

    param_lookup_dict_json = json.dumps(param_lookup_dict)
    with open(os.path.join(experiment.base_dir, 'params_lookup.json'), 'w') as f:
        f.write(param_lookup_dict_json)


def run_parkinsons_experiment(exp_type='mcar'):
    train_file = '../data/parkinsons/train_data.txt'
    test_file = '../data/parkinsons/test_data.txt'
    train_df = pd.read_csv(train_file, header=None, index_col=0)
    train_ids = set(train_df.index)
    test_df = pd.read_csv(test_file, header=None, index_col=0)
    # make sure train/test ids do not overlap
    n_train_ids = len(train_ids)
    test_df.index += n_train_ids
    train_colnames = [
        'Jitter (local)', 'Jitter (local, absolute)', 'Jitter (rap)', 
        'Jitter (ppq5)', 'Jitter (ddp)',  'Shimmer (local)', 
        'Shimmer (local, dB)', 'Shimmer (apq3)', 'Shimmer (apq5)', 
        'Shimmer (apq11)', 'Shimmer (dda)', 'AC', 'NTH', 'HTN', 
        'Median pitch', 'Mean pitch', 'Standard deviation', 'Minimum pitch',
        'Maximum pitch',  'Number of pulses', 'Number of periods', 
        'Mean period', 'Standard deviation of period',  
        'Fraction of locally unvoiced frames', 'Number of voice breaks', 
        'Degree of voice breaks', 'UPDRS', 'targets'
    ]
    test_colnames = [
        'Jitter (local)', 'Jitter (local, absolute)', 'Jitter (rap)', 
        'Jitter (ppq5)', 'Jitter (ddp)',  'Shimmer (local)', 
        'Shimmer (local, dB)', 'Shimmer (apq3)', 'Shimmer (apq5)', 
        'Shimmer (apq11)', 'Shimmer (dda)', 'AC', 'NTH', 'HTN', 
        'Median pitch', 'Mean pitch', 'Standard deviation', 'Minimum pitch',
        'Maximum pitch',  'Number of pulses', 'Number of periods', 
        'Mean period', 'Standard deviation of period',  
        'Fraction of locally unvoiced frames', 'Number of voice breaks', 
        'Degree of voice breaks', 'targets'
    ] # all from train_colnames __except UPDRS__
    train_df.columns = train_colnames
    test_df.columns = test_colnames

    train_df = train_df.drop(columns='UPDRS')
    train_ids = list(set(train_df.index))
    train_ids, val_ids = (
        train_ids[0 : int(len(train_ids)/2)], 
        train_ids[int(len(train_ids)/2) : ]
    )
    train_df = train_df.loc[train_ids, :]
    val_df = train_df.loc[val_ids, :]

    y_train = train_df.targets
    y_val = val_df.targets
    y_test = test_df.targets
    print(y_train.value_counts(), y_val.value_counts(), y_test.value_counts())

    X_train = train_df.drop(columns='targets')
    X_val = val_df.drop(columns='targets')
    X_test = test_df.drop(columns='targets')

    data = MissDataset(data=train_df, p_miss=0.3, target_col='targets', missing_mechanism=exp_type, n_folds=5)

    experiment = CustomClassificationExperiment(
        dataset=data,
        exp_type='mcar',
        dataset_name='parkinsons',
        name='Experiment_' + str(datetime.now()),
        base_dir=None,
        classifier_pool=None,
        random_state=42
    )

    errors_df, weights_sets, probas_df, distances_df = experiment.do_experiment_one_fold(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test
    )
    metrics_df = pd.DataFrame({})
    for m in experiment.metrics:
        metrics_df[m] = np.mean(experiment.metrics[m], axis=0)
    metrics_df.index = experiment.metric_type_cols

    filename = str(0) + '.csv'
    # weights_filename = os.path.join(experiment.results_dir, 'weights_' + filename)
    errors_filename = os.path.join(experiment.results_dir, 'errors_' + filename)
    errors_df.to_csv(errors_filename)

    metrics_filename = os.path.join(experiment.results_dir, 'metrics_' + filename)
    metrics_df.to_csv(metrics_filename)

    probas_filename = os.path.join(experiment.results_dir, 'predictions_' + filename)
    probas_df.to_csv(probas_filename)

    for top_n in weights_sets.keys():
        weights_filename = os.path.join(experiment.results_dir, 'weights_top_' + str(top_n) + '_' + filename)
        weights_sets[top_n].to_csv(weights_filename)

    distances_filename = os.path.join(experiment.results_dir, 'distances_' + filename)
    distances_df.to_csv(distances_filename)

SUPPORTED_DATASET_NAMES = [
    'Breast Cancer Wisconsin Diagnostic',
    'Breast Cancer Wisconsin Prognostic',
    'Thyroid',
    'Cleveland Heart Disease',
    'Iris',
    'Dermatology',
    'Parkinsons',
    'Diabetes 130 hospitals',
    'EEG Eye State Data Set',
    'Cervical Cancer',
    'Parkinson Speech',
    'myocardial_infarction'
]

CURRENT_SUPPORTED_DATALOADERS = {
    'eeg_eye_state': DataLoadersEnum.prepare_eeg_eye_data,
    'Cleveland Heart Disease': DataLoadersEnum.prepare_cleveland_heart_data,
    'diabetic_retinopathy': DataLoadersEnum.prepare_diabetic_retinopathy_dataset,
    'wpbc': DataLoadersEnum.prepare_wpbc_data,
    'wdbc': DataLoadersEnum.prepare_wdbc_data,
    'parkinsons': DataLoadersEnum.prepare_parkinsons_data,
    'cervical_cancer': DataLoadersEnum.prepare_cervical_cancer_data,
    'myocardial_infarction': DataLoadersEnum.prepare_myocardial_infarction_data,
    'student': DataLoadersEnum.prepare_student_data
}


def run(custom_experiment_data_object, task_type='classification'):
    
    MCAR_PARAM_DICT = {
        # 'p_miss': [x/10 for x in range(3,9)], 
        'p_miss': [0.3], 
        'missing_mechanism': ["MCAR"], 
        'opt': [None], 
        'p_obs': [None], 
        'q': [None],
    }

    MAR_PARAM_DICT = {
        # 'p_miss': [x/10 for x in range(3,9)], 
        'p_miss': [0.3], 
        'missing_mechanism': ["MAR"], 
        'opt': [None], 
        'p_obs': [0.3], 
        'q': [None],
    }

    MNAR_PARAM_DICT = {
        'p_miss': [0.3], 
        'missing_mechanism': ["MNAR"], 
        'opt': ['logistic'], 
        'p_obs': [0.3], 
        'q': [None],
    }

    for d in [MCAR_PARAM_DICT, MAR_PARAM_DICT, MNAR_PARAM_DICT]:
        run_custom_experiments(
            data=custom_experiment_data_object.data, 
            dataset_name=custom_experiment_data_object.dataset_name,
            miss_param_dict=d, 
            target_col=custom_experiment_data_object.target_col,
            task_type=task_type
        )


if __name__ == '__main__':
    # run_randomized_synthetic_clf_experiments()
    # TASK_TYPE = 'regression'
    # def cervical_cancer():
    #     for target_col in ['Hinselmann', 'Schiller', 'Citology', 'Biopsy']:
    #         data_preparation_function_object = CURRENT_SUPPORTED_DATALOADERS['cervical_cancer']
    #         data_object = data_preparation_function_object(target_col=target_col) # call function
    #         # for miss_type in ['mcar', 'mnar', 'mar']:
    #         #     run_parkinsons_experiment(miss_type)
    #     run(data_object, TASK_TYPE)
    
    # get function object
    # data_preparation_function_object = CURRENT_SUPPORTED_DATALOADERS['Cleveland Heart Disease']
    # data_object = data_preparation_function_object()
    # run(data_object)
    # datapaths = ['../data/student/student-mat.csv', '../data/student/student-por.csv']
    # cols = ['G1', 'G2', 'G3']
    # for p in datapaths:
    #     for c in cols:
    #         data_preparation_function_object = CURRENT_SUPPORTED_DATALOADERS['student']
    #         data_object = data_preparation_function_object(path_to_data=p, target_col=c)
    #         run(data_object, task_type=TASK_TYPE)
    data_preparation_function_object = CURRENT_SUPPORTED_DATALOADERS['eeg_eye_state']
    data_object = data_preparation_function_object()
    run(data_object)
