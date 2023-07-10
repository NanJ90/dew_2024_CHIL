from typing import Any
import pandas as pd
import pytorch_lightning as pl
import pytorch_tabular as ptab
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.models.gate.gate_model import CustomHead
from pytorch_tabular.models.common.layers import Embedding1dLayer
import torch
import torch.nn as nn
from torchmetrics import AUROC, AveragePrecision
from omegaconf import DictConfig


class PrintLayer(nn.Module):
    """
    For debugging.
    """
    def __init__(self, msg=''):
        super(PrintLayer, self).__init__()
        self.msg = msg

    def forward(self, x):
        # Do your print / debug stuff here
        print(f'{self.msg}: {x}, {x.shape}')
        return x
    

class GradDims(nn.Module):
    """
    This is ugly but gets the job done
    """
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, X):
        slice_str = '['
        for idx in range(len(X.shape)):
            if idx in self.dims:
                slice_str += ': , '
            else:
                slice_str += '0, '
        slice_str += ']'
        return eval('X' + slice_str)
    

class RemoveDims(nn.Module):
    """
    This is ugly but gets the job done
    """
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, X):
        slice_str = '['
        for idx in range(len(X.shape)):
            if idx not in self.dims:
                slice_str += ': , '
            else:
                slice_str += '0, '
        slice_str += ']'
        return eval('X' + slice_str)


class GATEHead(nn.Module):
    def __init__(self, task, backbone_out_dim, num_trees, output_dim=2):
        super().__init__()
        self.task = task
        self.backbone_out_dim = backbone_out_dim
        self.num_trees = num_trees
        self.output_dim = output_dim
        self.eta = nn.Parameter(torch.rand(self.num_trees, requires_grad=True))
        if self.task == "regression":
            self.T0 = nn.Parameter(torch.rand(self.output_dim), requires_grad=True)

        self.head = nn.Linear(self.backbone_out_dim, output_dim)

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        # B x L x T
        # https://discuss.pytorch.org/t/how-to-pass-a-3d-tensor-to-linear-layer/908/6
        # B x T x L -> B x T x Output
        
        y_hat = self.head(backbone_features.transpose(2, 1))

        # applying weights to each tree and summing up
        # ETA
        y_hat = y_hat * self.eta.reshape(1, -1, 1)
        # summing up
        y_hat = y_hat.sum(dim=1)

        if self.task == "regression":
            y_hat = y_hat + self.T0
        return y_hat

class AutoEncodingImputer(nn.Module):
    def __init__(self, in_dim, cat_indices, bottleneck_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.cat_indices = cat_indices
        self.bottleneck_dim = bottleneck_dim
        if bottleneck_dim is None:
            bottleneck_dim = in_dim // 8
        self.autoencoder = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),

            nn.Linear(in_dim // 2, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),

            nn.Linear(in_dim // 4, bottleneck_dim),
            nn.BatchNorm1d(in_dim // 8),
            nn.ReLU(),

            nn.Linear(bottleneck_dim, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),

            nn.Linear(in_dim // 4, in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU()
        )
        self.categorical_scaling_function = nn.Sigmoid()

    def forward(self, X):
        out_map: torch.Tensor
        out_map = self.autoencoder(X)
        # for idx in range(out_map.shape[1]):
        #     if idx in self.cat_indices:
        #         # out_map[:, idx].apply_(self.categorical_scaling_function)
        #         out_map[:, idx] = self.categorical_scaling_function(out_map[:, idx])
        num_original = self.in_dim // 2
        # shape: (B, C)
        imputed = out_map + X[:, 0 : num_original]
        return imputed




class DIPP(pl.LightningModule):
    def __init__(self, data_df: pd.DataFrame, in_dim, categorical_features, task, lr, tree_depth, num_trees=24, bottleneck_dim=None, num_classes=None, out_dim=1):
        super().__init__()

        self.categorical_column_indices = [data_df.columns.get_loc(f) for f in categorical_features]
        self.imputer = AutoEncodingImputer(
            in_dim=in_dim,
            bottleneck_dim=bottleneck_dim, 
            cat_indices=self.categorical_column_indices
        )
        # self.categorical_column_indices = []
        # n_continuous_features = data_df.shape[1] - len(self.categorical_column_indices)
        
        # Use categorical probability estimates ==> treat all as continuous.
        n_continuous_features = data_df.shape[1]
        self.lr = lr
        self.num_trees = num_trees
        self.task = task
        if task == 'classification':
            assert num_classes is not None
            self.auroc_score = AUROC(num_classes=num_classes)
            self.avg_precision_score = AveragePrecision(num_classes=num_classes)
            final_activation_fn = nn.Softmax(dim=-1)
        else:
            final_activation_fn = nn.Identity()
        
        gate_prediction_backbone = ptab.models.gate.GatedAdditiveTreesBackbone(
            # cat_embedding_dims=[(1,1)] * len(categorical_features),
            cat_embedding_dims=[],
            n_continuous_features=n_continuous_features,
            gflu_stages=4, num_trees=self.num_trees, tree_depth=tree_depth,
            tree_wise_attention=False, binning_activation='sparsemoid',
            feature_mask_function='softmax', chain_trees=False
        )
        head_config = LinearHeadConfig().__dict__
        head_config['share_head_weights'] = True
        head_config = DictConfig(content=head_config, key='share_head_weights')
        gate_prediction_head = GATEHead(
            task=self.task, 
            backbone_out_dim=gate_prediction_backbone.output_dim, 
            num_trees=self.num_trees,
            output_dim=num_classes
        )
        self.gate_prediction_model = nn.Sequential(
            gate_prediction_backbone,
            # RemoveDims([2]),
            # PrintLayer(),
            gate_prediction_head,
            final_activation_fn
        )
        
        self.prediction_loss_fn = nn.L1Loss() if task == 'regression' else nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def forward(self, X):
        imputed = self.imputer(X)
        prediction = self.gate_prediction_model(imputed)
        return imputed, prediction
    
    def training_step(self, batch, batch_idx):
        nonmissing_samples, missing_samples, missing_indicator, targets = tuple(batch)
        missing_vector = torch.cat([missing_samples, missing_indicator], axis=-1)
        imputed, predictions = self.forward(missing_vector)

        cat_reconstruction_losses = []
        cont_reconstruction_losses = []

        for idx in range(imputed.shape[1]):
            if idx in self.categorical_column_indices:
                reconstruction_loss_cat = nn.CrossEntropyLoss()(
                    imputed[:, idx], nonmissing_samples[:, idx]
                ) / len(imputed)
                cat_reconstruction_losses.append(reconstruction_loss_cat)
            else:
                reconstruction_loss_cont = nn.L1Loss()(
                    imputed[:, idx], nonmissing_samples[:, idx]
                ) / len(imputed)
                cont_reconstruction_losses.append(reconstruction_loss_cont)
        
        reconstruction_loss_cont = sum(cont_reconstruction_losses) / len(cont_reconstruction_losses)
        reconstruction_loss_cat = sum(cat_reconstruction_losses) / len(cat_reconstruction_losses)
        reconstruction_loss = (reconstruction_loss_cat + reconstruction_loss_cont) / 2

        prediction_loss = self.prediction_loss_fn(predictions, targets)#.unsqueeze(dim=1).to(torch.float32))

        loss = (reconstruction_loss + prediction_loss) / 2

        self.log('train_reconstruction_cat_loss', reconstruction_loss_cat, on_step=True, on_epoch=True)
        self.log('train_reconstruction_cont_loss', reconstruction_loss_cont, on_step=True, on_epoch=True)
        self.log('train_prediction_loss', prediction_loss, on_step=True, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        nonmissing_samples, missing_samples, missing_indicator, targets = tuple(batch)
        missing_vector = torch.cat([missing_samples, missing_indicator], axis=-1)
        imputed, predictions = self.forward(missing_vector)

        cat_reconstruction_losses = []
        cont_reconstruction_losses = []

        for idx in range(imputed.shape[1]):
            if idx in self.categorical_column_indices:
                reconstruction_loss_cat = nn.CrossEntropyLoss()(
                    imputed[:, idx], 
                    nonmissing_samples[:, idx]
                ) / len(imputed)
                cat_reconstruction_losses.append(reconstruction_loss_cat)
            else:
                reconstruction_loss_cont = nn.L1Loss()(
                    imputed[:, idx], 
                    nonmissing_samples[:, idx]
                ) / len(imputed)
                cont_reconstruction_losses.append(reconstruction_loss_cont)
        
        reconstruction_loss_cont = sum(cont_reconstruction_losses) / len(cont_reconstruction_losses)
        reconstruction_loss_cat = sum(cat_reconstruction_losses) / len(cat_reconstruction_losses)
        reconstruction_loss = (reconstruction_loss_cat + reconstruction_loss_cont) / 2

        # print(predictions.shape, targets.shape)
        prediction_loss = self.prediction_loss_fn(predictions, targets)#.unsqueeze(dim=1).to(torch.float32))

        loss = (reconstruction_loss + prediction_loss) / 2

        self.log('val_reconstruction_cat_loss', reconstruction_loss_cat, on_step=True, on_epoch=True)
        self.log('val_reconstruction_cont_loss', reconstruction_loss_cont, on_step=True, on_epoch=True)
        self.log('val_prediction_loss', prediction_loss, on_step=True, on_epoch=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        if self.task == 'classification':
            # print(predictions.shape, targets.shape)
            self.auroc_score.update(predictions.cpu(), targets.cpu())#.unsqueeze(dim=1).cpu().int())
            self.avg_precision_score.update(predictions.cpu(), targets.cpu())#.unsqueeze(dim=1).cpu().int())

        return loss
    
    def on_validation_epoch_end(self):
        if self.task == 'classification':
            auroc = self.auroc_score.compute()
            avg_precision = self.avg_precision_score.compute()
            self.log('val_AUROC', auroc)
            self.log('val_avg_precision', avg_precision)

            self.auroc_score.reset()
            self.avg_precision_score.reset()

        
    
