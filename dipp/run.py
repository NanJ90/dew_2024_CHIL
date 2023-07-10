import math
import os
import pathlib

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from data import TabularDataModule
from models import DIPP

import warnings

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    data_path = '/Users/adamcatto/Dropbox/CUNY/Research/dynamime/data/diabetes_vcu.csv'
    df = pd.read_csv(data_path, na_values='?')
    TARGET_COL = 'readmitted'

    USE_DIAG = False

    
    numeric_cols = [
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_outpatient',
        'number_emergency',
        'number_inpatient',
        'number_diagnoses'
    ]
    if USE_DIAG:
        diagnoses = sorted(list(set(list(df['diag_1']) + list(df['diag_2']) + list(df['diag_3']))))
        diagnoses = {d: idx for idx, d in enumerate(diagnoses)}
        diagnosis_df = df[['diag_1', 'diag_2', 'diag_3']]
        diagnosis_df = diagnosis_df.applymap(lambda x: diagnoses[x])

        d1 = pd.get_dummies(diagnosis_df['diag_1'])
        d2 = pd.get_dummies(diagnosis_df['diag_2'])
        d3 = pd.get_dummies(diagnosis_df['diag_3'])

        for d in [d1, d2, d3]:
            current_codes = d.columns
            for col in list(diagnoses.values()):
                if col not in current_codes:
                    d[col] = [0] * len(d)

        onehot_diagnosis_df = np.logical_or(np.logical_or(d1, d2), d3)
        onehot_diagnosis_df.columns = [f'diag_{x}' for x in onehot_diagnosis_df.columns]
        
        df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])
        df = pd.concat([df, onehot_diagnosis_df], axis=1)
    else:
        df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])

    df = df.drop(
        columns=[
            'encounter_id',
            'patient_nbr',
            'weight',
            'payer_code',
            'medical_specialty',
            'max_glu_serum',
            'A1Cresult'
        ]
    )
    df = df.dropna(how='any')

    cat_cols = [c for c in df.columns if c not in numeric_cols]

    for c in cat_cols:
        label_enc = LabelEncoder()
        df[c] = label_enc.fit_transform(df[c])

    datamodule = TabularDataModule(
        df=df,
        target_col=TARGET_COL,
        batch_size=256,
        n_folds=5,
        eval_fold=0,
        cat_columns=cat_cols, # auto-discovery of cat cols.
        erase_frac=0.3
    )
    model = DIPP(
        datamodule.train_dataset.data,
        in_dim=datamodule.train_dataset.data.shape[1] * 2,
        categorical_features=datamodule.cat_cols,
        task='classification',
        lr=3e-4,
        tree_depth=6,
        num_classes=3,
        out_dim=1
    )

    EXPERIMENT_NAME = pathlib.Path(data_path).stem
    logger = TensorBoardLogger(save_dir='.', name=EXPERIMENT_NAME)
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}_{val_avg_precision}',
        save_top_k=10,
        monitor='val_avg_precision',
        mode='max'
    )
    callbacks = [checkpoint_callback]
    MAX_EPOCHS = 500

    trainer = pl.Trainer(
        accelerator='mps',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        log_every_n_steps=1,
        max_epochs=MAX_EPOCHS,
        num_sanity_val_steps=2
    )

    trainer.fit(
        model=model,
        datamodule=datamodule
    )