import argparse
from utils.tsDataset import dataModuleDeepAnt
from utils.DeepAnt import AnomalyDetector, DeepAnt
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger

import torch
import torch.nn as nn
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd 
from utils.scores import scorer

if __name__ =="__main__":
    
    #read the parser

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type = str, required=True)
    parser.add_argument('--WL',type=int, required=True)
    parser.add_argument('--n',type=int, required=True)
    parser.add_argument('--i', type=int, required = True)

    args = parser.parse_args()

    print(args)
    

    SEQ_LEN = args.WL * args.n
    model = DeepAnt(SEQ_LEN, 1)
    anomaly_detector = AnomalyDetector(model)
    dm = dataModuleDeepAnt(args.path, 
                            src ='UCR',
                            seq_len = SEQ_LEN,
                            batch_size = 32)

    mc = ModelCheckpoint(
        dirpath = 'checkpoints',
        save_last = True,
        save_top_k = 1,
        verbose = True,
        monitor = 'train_loss', 
        mode = 'min'
        )

    mc.CHECKPOINT_NAME_LAST = f'DeepAnt-best-checkpoint'

    trainer = pl.Trainer(max_epochs=50,
                    accelerator="auto",
                    #devices=1, 
                    callbacks=[mc]
                    )
    trainer.fit(anomaly_detector, dm)

    output = trainer.predict(anomaly_detector, dm)
    preds_losses = torch.tensor([item[1] for item in output]).numpy()
    preds_losses = preds_losses[:-1]


    _, _, res = scorer(dm.df.df['is_anomaly'].loc[dm.df.train_split+SEQ_LEN:], preds_losses[dm.df.train_split:])

    res['dataset'] = dm.df.name
    res['WL'] = args.WL 
    res['n'] = args.n 
    res['id'] = args.i 



    if os.path.exists('res.csv'):
        print('existe')
        df = pd.read_csv('res.csv')
        df = pd.concat([df, pd.DataFrame([res])]).to_csv('res.csv', index = False)

    else:

        pd.DataFrame([res]).to_csv('res.csv', index = False)
    print(res)
