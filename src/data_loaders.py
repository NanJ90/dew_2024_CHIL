from collections import namedtuple
from collections.abc import Sequence
from enum import Enum
from itertools import product
import sys
from typing import Callable, Iterable, Union


from imblearn.base import BaseSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


sys.path.append('.')
from utils import create_missing_values, get_cols_without_missing_values, produce_NA, Test


def label_encoded_data(data, ignore_columns):
    # save data type of the columns a object datatype would indicated 
    # a categorical feature
    data_dict = dict(data.dtypes)
    # data = data.fillna(0)

    features = list(data.columns)

    for feature in ignore_columns:
         features.remove(feature)

    le = LabelEncoder()

    for labels in features:
        # check if the column is categorical 
        if data_dict[labels] == np.object:
            try:
                data.loc[:, labels] = le.fit_transform(data.loc[:, labels])
            except:
                print(labels)

    return data


CustomExperimentDataObject = namedtuple(
    'CustomExperimentDataObject',
    ['data', 'dataset_name', 'target_col']
)


class MedicalDataset(Sequence):
    def __init__(
        self,
        data,
        n_folds=None,
        test_size=None,
        target_col: str = '',
        tests={},
        feature_to_test_map: dict = {},
        sampling_strategy: Union[None, Callable, BaseSampler] = None,
        sampling_strategy_kwargs: Iterable = {},
        cv_random_state: int = 42
    ):
        """
        if a `callable` sampling strategy is chosen, it must take a DataFrame
        as its first argument.
        """
        self.raw_data = data.copy()
        self.all_columns = data.columns
        self.target_col = target_col
        self.n_folds = n_folds
        
        if sampling_strategy is not None:
            y = data[target_col]
            X = data.drop(target_col, axis=1)

            if isinstance(sampling_strategy, Callable):
                X, y = sampling_strategy(
                    **sampling_strategy_kwargs
                ).fit_resample(X, y)
            elif isinstance(sampling_strategy, BaseSampler):
                X, y = sampling_strategy.fit_resample(X, y)
            data = X
            data[target_col] = y

        self.data = data
        self.targets = data[self.target_col]
        
        folder_ = StratifiedKFold(
            n_splits=n_folds,
            random_state=cv_random_state,
            shuffle=True
        )
        train_test_pairs = list(folder_.split(self.data, self.targets))
        train_sets = [train for (train, test) in train_test_pairs]
        for t in train_sets:
            np.random.shuffle(t)

        train_val_pairs = [
            (
                train_set[0: int(len(train_set) * 0.8)], 
                train_set[int(len(train_set) * 0.8):]
            )
            for train_set in train_sets
        ]
        final_train_sets = [train for (train, val) in train_val_pairs]
        final_val_sets = [val for (train, val) in train_val_pairs]
        final_test_sets = [test_set for (_, test_set) in train_test_pairs]

        self.train_val_test_triples = list(zip(
            final_train_sets, final_val_sets, final_test_sets
        ))


        # self.train_test_pairs = train_test_pairs

        self.feature_to_test_map = feature_to_test_map
        self.tests = tests
        
        test_features = []
        for x in list(self.tests.values()):
            test_features += x.features
        
        self.test_features = test_features
        
        self.base_features = [
            c 
            for c in self.data.columns 
            if c not in test_features + [self.target_col]
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        train_idx, val_idx, test_idx = self.train_val_test_triples[i]
        train, val, test = (
            self.data.iloc[train_idx, :], 
            self.data.iloc[val_idx, :], 
            self.data.iloc[test_idx, :]
        )

        train, val, test = (
            pd.DataFrame(train, columns=self.all_columns), 
            pd.DataFrame(val, columns=self.all_columns),
            pd.DataFrame(test, columns=self.all_columns)
        )
        return train, val, test
            

class Dataset(Sequence):
    def __init__(
        self,
        data,
        n_folds=None,
        test_size=None,
        target_col: str = '',
        sampling_strategy: Union[None, Callable, BaseSampler] = None,
        sampling_strategy_kwargs: Iterable = {},
        cv_random_state: int = 42
    ):
        """
        if a `callable` sampling strategy is chosen, it must take a DataFrame
        as its first argument.
        """
        self.raw_data = data.copy()
        self.all_columns = data.columns
        self.target_col = target_col
        self.n_folds = n_folds
        
        if sampling_strategy is not None:
            y = data[target_col]
            X = data.drop(target_col, axis=1)

            if isinstance(sampling_strategy, Callable):
                X, y = sampling_strategy(
                    **sampling_strategy_kwargs
                ).fit_resample(X, y)
            elif isinstance(sampling_strategy, BaseSampler):
                X, y = sampling_strategy.fit_resample(X, y)
            data = X
            data[target_col] = y

        self.data = data
        self.targets = data[self.target_col]
        
        folder_ = StratifiedKFold(
            n_splits=n_folds,
            random_state=cv_random_state,
            shuffle=True
        )
        train_test_pairs = list(folder_.split(self.data, self.targets))
        train_sets = [train for (train, test) in train_test_pairs]
        for t in train_sets:
            np.random.shuffle(t)

        train_val_pairs = [
            (
                train_set[0: int(len(train_set) * 0.8)], 
                train_set[int(len(train_set) * 0.8):]
            )
            for train_set in train_sets
        ]
        final_train_sets = [train for (train, val) in train_val_pairs]
        final_val_sets = [val for (train, val) in train_val_pairs]
        final_test_sets = [test_set for (_, test_set) in train_test_pairs]

        self.train_val_test_triples = list(zip(
            final_train_sets, final_val_sets, final_test_sets
        ))

        self.base_features = get_cols_without_missing_values(data[[c for c in data.columns if c != self.target_col]])
        self.tests = {
            f: Test(name=f, features=[f])
            for f in data.columns
            if f not in self.base_features + [self.target_col]
        }
        self.feature_to_test_map = self.tests

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        train_idx, val_idx, test_idx = self.train_val_test_triples[i]
        train, val, test = (
            self.data.iloc[train_idx, :], 
            self.data.iloc[val_idx, :], 
            self.data.iloc[test_idx, :]
        )

        train, val, test = (
            pd.DataFrame(train, columns=self.all_columns), 
            pd.DataFrame(val, columns=self.all_columns),
            pd.DataFrame(test, columns=self.all_columns)
        )
        return train, val, test


class MissDataset(Sequence):
    def __init__(
        self,
        data,
        n_folds=None,
        test_size=None,
        target_col: str = '',
        p_miss=0.1, 
        missing_mechanism = "MCAR", 
        opt = None, 
        p_obs = None, 
        q = None,
        sampling_strategy: Union[None, Callable, BaseSampler] = None,
        sampling_strategy_kwargs: Iterable = {},
        cv_random_state: int = 42
    ):
        """
        if a `callable` sampling strategy is chosen, it must take a DataFrame
        as its first argument.
        """
        self.raw_data = data.copy()
        self.all_columns = data.columns
        self.target_col = target_col
        self.n_folds = n_folds
        
        if sampling_strategy is not None:
            y = data[target_col]
            X = data.drop(target_col, axis=1)

            if isinstance(sampling_strategy, Callable):
                X, y = sampling_strategy(
                    **sampling_strategy_kwargs
                ).fit_resample(X, y)
            elif isinstance(sampling_strategy, BaseSampler):
                X, y = sampling_strategy.fit_resample(X, y)
            data = X
            data[target_col] = y

        self.targets = data[self.target_col]
        X = data.drop(columns=[self.target_col])
        X_cols = X.columns
        X_index = X.index
        miss_dict = produce_NA(
            X=X,
            p_miss=p_miss, 
            mecha=missing_mechanism, 
            opt=opt, p_obs=p_obs, q=q
        )
        X = miss_dict['X_incomp']

        self.data = pd.DataFrame(X, columns=X_cols, index=X_index)
        self.data[self.target_col] = self.targets
        
        folder_ = StratifiedKFold(
            n_splits=n_folds,
            random_state=cv_random_state,
            shuffle=True
        )
        train_test_pairs = list(folder_.split(self.data, self.targets))
        train_sets = [train for (train, test) in train_test_pairs]
        for t in train_sets:
            np.random.shuffle(t)

        train_val_pairs = [
            (
                train_set[0: int(len(train_set) * 0.8)], 
                train_set[int(len(train_set) * 0.8):]
            )
            for train_set in train_sets
        ]
        final_train_sets = [train for (train, val) in train_val_pairs]
        final_val_sets = [val for (train, val) in train_val_pairs]
        final_test_sets = [test_set for (_, test_set) in train_test_pairs]

        self.train_val_test_triples = list(zip(
            final_train_sets, final_val_sets, final_test_sets
        ))

        self.base_features = get_cols_without_missing_values(data[[c for c in data.columns if c != self.target_col]])
        self.tests = {
            f: Test(name=f, features=[f])
            for f in data.columns
            if f not in self.base_features + [self.target_col]
        }
        self.feature_to_test_map = self.tests

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        train_idx, val_idx, test_idx = self.train_val_test_triples[i]
        train, val, test = (
            self.data.iloc[train_idx, :], 
            self.data.iloc[val_idx, :], 
            self.data.iloc[test_idx, :]
        )

        train, val, test = (
            pd.DataFrame(train, columns=self.all_columns), 
            pd.DataFrame(val, columns=self.all_columns),
            pd.DataFrame(test, columns=self.all_columns)
        )
        return train, val, test




class PlacentalAnalytesTests:
    def __init__(self) -> None:
        self.filename = 'placental_analytes'
        self.tests = {
            'ADAM12': Test(name='ADAM12', filename=self.filename, features=['ADAM12']),
            'ENDOGLIN': Test(name='ENDOGLIN', filename=self.filename, features=['ENDOGLIN']),
            'SFLT1': Test(name='SFLT1', filename=self.filename, features=['SFLT1']),
            'VEGF': Test(name='VEGF', filename=self.filename, features=['VEGF']),
            'AFP': Test(name='AFP', filename=self.filename, features=['AFP']),
            'fbHCG': Test(name='fbHCG', filename=self.filename, features=['fbHCG']),
            'INHIBINA': Test(name='INHIBINA', filename=self.filename, features=['INHIBINA']),
            'PAPPA': Test(name='PAPPA', filename=self.filename, features=['PAPPA']),
            'PLGF': Test(name='PLGF', filename=self.filename, features=['PLGF'])
        }

    def get_test(self, test_name):
        return self.tests[test_name]


def load_numom2b_analytes_dataset(target_col='PEgHTN') -> MedicalDataset:
    
    pa = [
        'ADAM12',
        'ENDOGLIN',
        'SFLT1',
        'VEGF',
        'AFP',
        'fbHCG',
        'INHIBINA',
        'PAPPA',
        'PLGF'
    ]

    df_pa = pd.read_csv('/volumes/identify/placental_analytes.csv', na_values=np.nan)
    df_pa = df_pa.drop(columns=[x for x in df_pa.columns if x[-1] == 'c'])
    df_pa = df_pa[df_pa['Visit'] == 2]
    df_pa = label_encoded_data(df_pa, ignore_columns=[])

    df_outcomes = pd.read_csv('/volumes/identify/pregnancy_outcomes.csv')[['StudyID', target_col]]
    df_outcomes = label_encoded_data(df_outcomes, ignore_columns=[])

    df_extra_base = pd.read_csv('/volumes/no name/new/l1_visits/Visit1_l1.csv')
    df_extra_base = df_extra_base.rename(columns={'STUDYID': 'StudyID'})
    lowercase_cols = [c for c in df_extra_base.columns if c == c.lower()]
    df_extra_base = df_extra_base.drop(columns=['AGE_AT_V1', 'GAWKSEND', 'BIRTH_TYPE', 'Unnamed: 0', 'PEGHTN', 'CHRONHTN'] + lowercase_cols)
    df_extra_base = label_encoded_data(df_extra_base, ignore_columns=[])

    df = pd.merge(left=df_pa, left_on='StudyID', right=df_outcomes,
                right_on='StudyID', how='outer'
        )

    df = pd.merge(left=df, left_on='StudyID', right=df_extra_base,
                right_on='StudyID', how='outer')

    df['StudyID'] = list(range(len(df)))

    df = df.drop(columns=['PublicID', 'VisitDate', 'ref_date', 'VisitDate_INT'])
    df = df.drop(columns=['Visit', 'StudyID'])
    if target_col == 'PEgHTN':
        df['PEgHTN'] = [0 if x == 7 else 1 for x in df['PEgHTN']]

    tests_dict = PlacentalAnalytesTests().tests
    RANDOM_STATE = 42
    lowercase_cols = [c for c in df.columns if c == c.lower()]
    exclude_cols = [
        'Unnamed: 0', 'Unnamed: 0.1', 'STUDYID', 'StudyID', 'GAWKSEND', 'BIRTH_TYPE', 
        'AGE_AT_V1', 'PEGHTN', 'CHRONHTN', 'OUTCOME'
    ] + lowercase_cols
    df = df[[c for c in df.columns if c not in exclude_cols]]
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    X_cols = X.columns
    rus = RandomUnderSampler()
    X, y = rus.fit_resample(X, y)
    X = pd.DataFrame(data=X, columns=X_cols)
    df = X
    df[target_col] = y

    return MedicalDataset(
        data=df, target_col=target_col, 
        feature_to_test_map=tests_dict, tests=tests_dict, 
        n_folds=5, test_size=0.2
    )


def load_wisconsin_diagnosis_dataset(
        MASKED_FEATURE_TYPES = [
            'smoothness', 'compactness', 'concavity',
            'symmetry', 'fractaldimension', 'area'
    ],
        missingness_amounts = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    ) -> MedicalDataset:

    assert len(MASKED_FEATURE_TYPES) == len(missingness_amounts)

    colnames = [
        'radius', 
        'texture',
        'perimeter',
        'area',
        'smoothness',
        'compactness',
        'concavity',
        'concave points',
        'symmetry',
        'fractaldimension'
    ]
    feature_types = ['mean', 'stderr', 'worst']
    cols = [x[0] + '_' + x[1] for x in product(feature_types, colnames)]
    cols = ['id', 'diagnosis'] + cols
    df = pd.read_csv('../data/wdbc.data', header=None, 
                    names=cols, na_values='?')
    df = df.dropna(how='any')
    df = df.drop(columns='id')

    feature_groups = {
        c: []
        for c in colnames
    }

    for col in cols:
        splitcol = col.split('_')
        if len(splitcol) < 2:
            continue
        if splitcol[1] in colnames:
            feature_groups[splitcol[1]].append(col)

    target_col = 'diagnosis'

    df[target_col] = LabelEncoder().fit_transform(df[target_col])

    copy_df = df.__copy__()

    tests = {}
    num_tests = len(MASKED_FEATURE_TYPES)

    for i, f, amount in zip(list(range(num_tests)), MASKED_FEATURE_TYPES, missingness_amounts):
        tests[f] = Test(name=f, filename='', features=feature_groups[f])
        copy_df = create_missing_values(copy_df, feature_groups[f], num_samples=amount)
        
    base_features = []
    for f in feature_groups:
        if f not in MASKED_FEATURE_TYPES:
            base_features += feature_groups[f]

    tests_dict = {}
    for t in copy_df.columns:
        if '_' in t:
            test_type = t.split('_')[1]
            if test_type in MASKED_FEATURE_TYPES:
                tests_dict[t] = tests[test_type]

    return MedicalDataset(
        data=copy_df, target_col=target_col, 
        feature_to_test_map=tests_dict, tests=tests, 
        n_folds=5, test_size=0.2, cv_random_state=42,
        sampling_strategy=RandomUnderSampler()
    )


def load_wisconsin_prognosis_dataset(
        MASKED_FEATURE_TYPES = [
            'smoothness', 'compactness', 'concavity',
            'symmetry', 'fractaldimension', 'area'
    ],
        missingness_amounts = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    ) -> MedicalDataset:

    assert len(MASKED_FEATURE_TYPES) == len(missingness_amounts)

    colnames = [
        'radius', 
        'texture',
        'perimeter',
        'area',
        'smoothness',
        'compactness',
        'concavity',
        'concave points',
        'symmetry',
        'fractaldimension'
    ]
    feature_types = ['mean', 'stderr', 'worst']
    cols = [x[0] + '_' + x[1] for x in product(feature_types, colnames)]
    cols = ['id', 'prognosis', 'time'] + cols
    cols += ['tumorsize', 'lymphstatus']

    df = pd.read_csv('/Users/adamcatto/Dropbox/src/nuMoM2b/visit_test_sched/data/wpbc.data', header=None, 
                    names=cols, na_values='?')
    df = df.dropna(how='any')
    df = df.drop(columns='id')

    feature_groups = {
        c: []
        for c in colnames
    }

    for col in cols:
        splitcol = col.split('_')
        if len(splitcol) < 2:
            continue
        if splitcol[1] in colnames:
            feature_groups[splitcol[1]].append(col)
            
    feature_groups['time'] = ['time']
    feature_groups['tumorsize'] = ['tumorsize']
    feature_groups['lymphstatus'] = ['lymphstatus']

    target_col = 'prognosis'

    df[target_col] = LabelEncoder().fit_transform(df[target_col])

    copy_df = df.__copy__()

    tests = {}
    num_tests = len(MASKED_FEATURE_TYPES)

    for i, f, amount in zip(list(range(num_tests)), MASKED_FEATURE_TYPES, missingness_amounts):
        tests[f] = Test(name=f, filename='', features=feature_groups[f])
        copy_df = create_missing_values(copy_df, feature_groups[f], num_samples=amount)
        
    base_features = []
    for f in feature_groups:
        if f not in MASKED_FEATURE_TYPES:
            base_features += feature_groups[f]

    tests_dict = {}
    for t in copy_df.columns:
        if '_' in t:
            test_type = t.split('_')[1]
            if test_type in MASKED_FEATURE_TYPES:
                tests_dict[t] = tests[test_type]
        elif t in ['tumorsize', 'lymphstatus']:
            if t in MASKED_FEATURE_TYPES:
                tests_dict[t] = tests[t]

    return MedicalDataset(
        data=copy_df, target_col=target_col, 
        feature_to_test_map=tests_dict, tests=tests, 
        n_folds=5, test_size=0.2, cv_random_state=42,
        sampling_strategy=RandomUnderSampler()
    )


def normality_test_wisconsin():
    colnames = [
        'radius', 
        'texture',
        'perimeter',
        'area',
        'smoothness',
        'compactness',
        'concavity',
        'concave points',
        'symmetry',
        'fractaldimension'
    ]
    feature_types = ['mean', 'stderr', 'worst']
    cols = [x[0] + '_' + x[1] for x in product(feature_types, colnames)]
    cols = ['id', 'diagnosis'] + cols

    df = pd.read_csv('../data/wdbc.data', header=None, 
                    names=cols, na_values='?')
    print(df.head())
    df = df.dropna(how='any', axis=0)
    df = df.drop(columns=['id','diagnosis'])

    for i in df.columns:
        t = stats.normaltest(df[i])
        print(t)


class DataLoadersEnum(Enum):

    def prepare_eeg_eye_data(
        path_to_eeg_data: str = '../data/eeg_eye_state.csv'
    ) -> CustomExperimentDataObject:
        df = pd.read_csv(path_to_eeg_data, header=None)
        df.columns = [str(c) for c in df.columns]
        target_col = df.columns[-1]
        return CustomExperimentDataObject(
            data=df, 
            dataset_name='eeg_eye_state', 
            target_col=target_col
        )

    def prepare_cleveland_heart_data(
        path_to_data: str = '../data/heart_cleveland_upload.csv'
    ) -> CustomExperimentDataObject:
        df = pd.read_csv(path_to_data)
        target_col = 'condition'
        dataset_name = 'cleveland_heart_disease'
        return CustomExperimentDataObject(
            data=df,
            dataset_name=dataset_name,
            target_col=target_col
        )

    def prepare_diabetic_retinopathy_dataset(
        path_to_data: str = '../data/diabetic_retinopathy_dataset.csv'
    ) -> CustomExperimentDataObject:
        df = pd.read_csv(path_to_data, header=None)
        df.columns = [str(c) for c in df.columns]
        target_col = df.columns[-1]
        return CustomExperimentDataObject(
            data=df, 
            dataset_name='diabetic_retinopathy', 
            target_col=target_col
        )