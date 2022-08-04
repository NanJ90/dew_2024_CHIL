def main(
    experiment_type='wisconsin_bc_prognosis', 
    missingness_amounts=[0.5]*7,
    MASKED_FEATURE_TYPES=[
                'radius', 
                'texture',
                'perimeter',
                'area',
                'smoothness',
                'compactness',
                'concavity',
                # 'concave points',
                # 'symmetry',
                # 'fractaldimension'
            ],
    prediction_error_outfile = None
):
    if experiment_type == 'numom2b_hypertension':
        data = load_numom2b_analytes_dataset()
    elif experiment_type == 'wisconsin_bc_prognosis':
        data = load_wisconsin_prognosis_dataset(MASKED_FEATURE_TYPES=MASKED_FEATURE_TYPES,
        missingness_amounts=missingness_amounts
        )
    elif experiment_type == 'wisconsin_bc_diagnosis':
        
        data = load_wisconsin_diagnosis_dataset(
            MASKED_FEATURE_TYPES=MASKED_FEATURE_TYPES,
            missingness_amounts=missingness_amounts
        )
        print('missingness amounts: ' + str(missingness_amounts))
    # tests = PlacentalAnalytesTests().tests
    # pa = ['ADAM12','ENDOGLIN','SFLT1','VEGF','AFP','fbHCG',
    #     'INHIBINA','PAPPA','PLGF']

    inheritance_metrics = []
    knn_metrics = []
    mice_metrics = []
    vanilla_metrics = []
    stacked_metrics = []
    ds_metrics = []
    oracle_metrics = []

    proba_predictions_dfs_list = []
    y_trues = []

    for i in range(data.n_folds):
        proba_predictions_df = pd.DataFrame({})

        # initialize classifiers
        clf_inheritance = NonmissingSubspaceInheritanceClassifier(
            data=data, 
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
            final_estimator=RandomForestClassifier(
                n_estimators=20, max_depth=3, max_features=0.5, 
                max_samples=0.7, n_jobs=1
            ),
            cv='prefit',
            n_jobs=1
        )

        clf_ds = DynamicSelector(
            classifier_pool={
                'inheritance': clf_inheritance,
                'knn_imputation': clf_knn,
                'mice_imputation': clf_mice,
                'xgb_baseline': clf_vanilla
            },
            n_neighbors=5
        )

        clf_oracle = Oracle(
            pool_classifiers=[clf_inheritance, clf_knn, clf_mice, clf_vanilla]
        )

        df = data.raw_data.copy()
        # df = df.dropna(axis=0)
        # df = df[[c for c in df.columns if c not in pa]]
        # rus = RandomUnderSampler()
        # X_train, y_train = rus.fit_resample(X_train, y_train)
        # X_test, y_test = rus.fit_resample(X_test, y_test)

        train, test = data[i]
        # test_indices = get_sample_indices_with_optional_tests(test, test_features=data.test_features)
        # test = test.loc[test_indices, :]
        y_test = test[data.target_col]
        y_trues += list(y_test)
        y_train = train[data.target_col]
        X_test = test.drop(data.target_col, axis=1)
        X_train = train.drop(data.target_col, axis=1)
        
        #################################################################

        print('with full inheritance')
        clf_inheritance.fit(X=X_train, y=y_train)
        proba_predictions = clf_inheritance.predict(X=X_test)
        proba_predictions_df['full_inheritance'] = proba_predictions
        # t = find_optimal_threshold(proba_predictions, y_test)
        # predictions = [1 if p > t else 0 for p in proba_predictions]
        # y_test = data[0][1][data.target_col]

        # nodwise_metrics = clf_inheritance.get_nodewise_classification_metrics(y_test=y_test)


        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_test, proba_predictions)
        predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
        # print(classification_report(y_test, predictions))
        metrics = get_classification_metrics(predictions, y_test)
        metrics['roc_auc'] = round(roc_auc, 4)
        accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
        metrics['accuracy'] = round(accuracy, 4)
        print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
        inheritance_metrics.append(list(metrics.values()))
        with open('../results/inheritance_dag.pkl', 'wb') as f:
            pickle.dump(clf_inheritance.dag, f)
        print('\n-----------------------------\n')

        #################################################################

        print('with knn imputation')
        clf_knn.fit(X_train, y_train)
        proba_predictions = clf_knn.predict_proba(X_test)[:,1]
        proba_predictions_df['knn_imputation'] = proba_predictions
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

        print('DS model:')
        # clf_knop.fit(X_train, y_train)
        # proba_predictions = clf_knop.predict_proba(X_test)[:,1]
        # proba_predictions_df['KNOP'] = proba_predictions
        clf_ds.fit(X_train, y_train)
        proba_predictions = clf_ds.predict_proba(X_test)[:, 1]
        proba_predictions_df['DS_new'] = proba_predictions
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
        ds_metrics.append(list(metrics.values()))
        print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
        print('\n======================\n===================\n')

        ###########################################################

        print('Oracle Classifier:')
        clf_oracle.fit(X_train, y_train)
        proba_predictions = clf_oracle.predict_proba(X_test, y_test)[:, 1]
        proba_predictions_df['Oracle'] = proba_predictions
        predictions = [1 if p > 0.5 else 0 for p in proba_predictions]
        roc_auc = roc_auc_score(y_test, proba_predictions)
        metrics = get_classification_metrics(predictions, y_test)
        metrics['roc_auc'] = round(roc_auc, 4)
        accuracy = 1 - (np.sum(np.logical_xor(predictions, y_test)) / len(predictions))
        metrics['accuracy'] = round(accuracy, 4)
        oracle_metrics.append(list(metrics.values()))
        print('\n'.join([str(a) + ': ' + str(b) for a, b in metrics.items()]))
        print('\n======================\n===================\n')

        ###############################################################

        proba_predictions_dfs_list.append(proba_predictions_df)

    proba_predictions_df_total = pd.concat(proba_predictions_dfs_list)
    proba_predictions_df_total.index = range(len(proba_predictions_df_total))
    print(len(proba_predictions_df_total))
    plot_prediction_errors(
        y_true=np.array(y_trues), 
        proba_predictions_df=proba_predictions_df_total,
        title='Example Comparison of Model Prediction Errors',
        xlabel='Index of Samples',
        ylabel='Class Probability Prediction Error'
        outfile=prediction_error_outfile
    )
    # plot_prediction_errors(y_true=np.array(y_test), proba_predictions_df=proba_predictions_df)

    inheritance_metrics = np.vstack(inheritance_metrics)
    knn_metrics = np.vstack(knn_metrics)
    mice_metrics = np.vstack(mice_metrics)
    vanilla_metrics = np.vstack(vanilla_metrics)
    stacked_metrics = np.vstack(stacked_metrics)
    ds_metrics = np.vstack(ds_metrics)
    oracle_metrics = np.vstack(oracle_metrics)

    median_inheritance_metrics = np.median(inheritance_metrics, axis=0)
    median_knn_metrics = np.median(knn_metrics, axis=0)
    median_mice_metrics = np.median(mice_metrics, axis=0)
    median_vanilla_metrics = np.median(vanilla_metrics, axis=0)
    median_stacked_metrics = np.median(stacked_metrics, axis=0)
    median_ds_metrics = np.median(ds_metrics, axis=0)
    median_oracle_metrics = np.median(oracle_metrics, axis=0)

    print('\n=========\n==========\n')

    print('cross-validated median metrics:\n----------------\n')

    metric_types = [
            'sensitivity', 'specificity', 'ppv', 'npv', 'gmean_sens_spec',
            'gmean_all_metrics', 'roc_auc', 'accuracy'
        ]
    cv_results_medians = {}

    for t in [
        'median_inheritance_metrics', 'median_knn_metrics', 
        'median_mice_metrics','median_vanilla_metrics', 
        'median_stacked_metrics', 'median_ds_metrics', 'median_oracle_metrics'
    ]:
        cv_results_medians[t] = {}
        print(t + str(': '))
        for i, metric in enumerate(metric_types):
            cv_results_medians[metric + '_' + t] = eval(t)[i]
            print(metric + ': ' + str(eval(t)[i]))
        
        print('-------------')

    # with open('../results/sample_wisc_bc_diagnosis.json', 'w') as f:
    #     json.dump(cv_results_medians, f)

    return pd.DataFrame(cv_results_medians)