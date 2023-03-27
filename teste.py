from utils.tsDataset import TimeseriesDataset, dataModule

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint

import pandas as pd

from argparse import ArgumentParser



class LSTMRegressor(pl.LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''
    def __init__(self, 
                 n_features, 
                 hidden_size, 
                 seq_len, 
                 batch_size,
                 num_layers, 
                 dropout, 
                 learning_rate,
                 criterion):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:,-1])
        return y_pred
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=False, logger = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=False, logger = True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, prog_bar=False, logger = True)
        return loss
    
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--src", default=None, type=str)
    args = parser.parse_args()
    seed_everything(1)
    
    
    p = dict(
    seq_len = 24,
    batch_size = 70, 
    criterion = nn.MSELoss(),
    max_epochs = 50,
    n_features = 1,
    hidden_size = 100,
    num_layers = 1,
    dropout = 0.2,
    learning_rate = 0.001,
    )
    
    mc = ModelCheckpoint(
        dirpath = 'checkpoints',
        save_last = True,
        save_top_k = 1,
        verbose = True,
        monitor = 'train_loss', 
        mode = 'min'
        )

    mc.CHECKPOINT_NAME_LAST = f'DeepAnt-best-checkpoint'

    csv_logger = CSVLogger('./', name='lstm', version='0'),

    trainer = Trainer(
        max_epochs=p['max_epochs'],
        logger=csv_logger,
        accelerator="auto",
        callbacks=[mc], 
        enable_progress_bar=False
    )

    model = LSTMRegressor(
        n_features = p['n_features'],
        hidden_size = p['hidden_size'],
        seq_len = p['seq_len'],
        batch_size = p['batch_size'],
        criterion = p['criterion'],
        num_layers = p['num_layers'],
        dropout = p['dropout'],
        learning_rate = p['learning_rate']
    )

    dm = dataModule(
        path = args.path,
        src =args.src,
        seq_len = 100,
        batch_size = p['batch_size']
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, verbose = False)
