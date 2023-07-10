import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def random_erase(row, erase_frac=0.3):
    num_elem = len(row)
    all_indices = np.array(list(range(num_elem)))
    num_to_take = int(num_elem * erase_frac)
    missing_indices = np.random.choice(a=all_indices, size=num_to_take)
    
    missing_row = row.copy()
    indicator = np.zeros(row.shape)
    for idx in missing_indices:
        indicator[idx] = 1
        missing_row[idx] = 0
    
    return missing_row, indicator


class TabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_col, cat_columns=None, erase_frac=0.3):
        """
        The one-hot encoding will take place in the DataModule.
        """
        super().__init__()
        self.targets = df[target_col]
        self.data = df.drop(columns=[target_col])
        self.cat_columns = cat_columns
        self.erase_frac = erase_frac

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx].to_numpy().astype(np.float32)
        row = np.squeeze(row)
        missing_row, indicator = random_erase(row, erase_frac=self.erase_frac)
        target = int(np.squeeze(self.targets.iloc[idx]))

        missing_row = missing_row.astype(np.float32)
        indicator = indicator.astype(np.float32)

        return row, missing_row, indicator, target
    

class TabularDataModule(pl.LightningDataModule):
    def __init__(self, df, target_col, batch_size=4096, n_folds=5, eval_fold=0, cat_columns=None, erase_frac=0.3):
        super().__init__()
        assert 0 <= eval_fold < n_folds
        self.batch_size = batch_size
        self.n_folds = n_folds
        
        self.targets = df[target_col]

        nunique = df.nunique()
        # heuristic: set column to categorical if less than 10 unique vals in col.
        # this is just for quick experiment and should be discarded for future.
        if cat_columns is None:
            cat_cols = nunique[nunique < 10].index
        else:
            cat_cols = cat_columns
        cat_cols = [c for c in cat_cols if c != target_col]
        self.data = df.drop(columns=[target_col])
        
        onehot_cols = []
        for c in cat_cols:
            onehot = pd.get_dummies(df[c]).astype(int)
            onehot.columns = [f'{c}_{onehot_c}' for onehot_c in onehot.columns]
            onehot_cols.append(onehot)


        onehot_df = pd.concat(onehot_cols, axis=1)
        self.cat_cols = onehot_df.columns
        continuous_df = df.drop(columns=cat_cols)

        processed_df = pd.concat([continuous_df, onehot_df], axis=1)
        processed_df[target_col] = self.targets

        folder_ = StratifiedKFold(
            n_splits=self.n_folds,
            random_state=42,
            shuffle=True
        )
        train_val_pairs = list(folder_.split(processed_df.drop(columns=[target_col]), self.targets))
        train_val_df_pairs = [
            (processed_df.iloc[train, :], processed_df.iloc[test, :]) 
            for train, test in train_val_pairs
        ]

        train_df = train_val_df_pairs[eval_fold][0]
        val_df = train_val_df_pairs[eval_fold][1]

        continuous_cols = list(continuous_df.columns)
        try:
            continuous_cols.remove(target_col)
        except:
            continuous_cols = continuous_cols

        for c in continuous_cols:
            mean_ = train_df[c].mean()
            std_ = train_df[c].std()
            train_df[c] = (train_df[c] - mean_) / std_
            val_df[c] = (val_df[c] - mean_) / std_

        self.train_dataset = TabularDataset(
            train_df, 
            target_col=target_col,
            cat_columns=self.cat_cols,
            erase_frac=erase_frac
        )
        self.val_dataset = TabularDataset(
            val_df, 
            target_col=target_col,
            cat_columns=self.cat_cols,
            erase_frac=erase_frac
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True, drop_last=False)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False, drop_last=False)
        