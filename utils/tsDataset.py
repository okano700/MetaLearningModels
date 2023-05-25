from torch.utils.data import Dataset
import pytorch_lightning as pl
import numpy as np
import torch
from utils.TSds import TSds

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

class TimeseriesDataset(Dataset):   
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs. 
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len], self.y[index+self.seq_len-1])
    
    
    
class dataModule(pl.LightningDataModule):

    def __init__(self, path:str, src:str, seq_len:int, batch_size = 128, num_workers=0):
        super().__init__()
        self.path = path
        self.src = src
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None
        self.columns = None
        self.preprocessing = None
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == 'fit' and self.X_train is not None:
            return 
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:  
            return
        if self.src == 'UCR':
            self.df = TSds.read_UCR(self.path)
        if self.src == 'NAB':
            self.df = TSds.read_NAB(self.path)
        if self.src == 'YAHOO':
            self.df = TSds.read_YAHOO(self.path)
        X = self.df.ts
        y = np.array(self.df.df[['value']].shift(-1).ffill())

        X_cv = self.df.ts[:self.df.train_split].reshape(-1,1)
        y_cv = y[:self.df.train_split]
        X_test = self.df.ts[self.df.train_split:].reshape(-1,1)
        y_test = y[self.df.train_split:]
        
    
        X_train, X_val, y_train, y_val = train_test_split(
            X_cv, y_cv, test_size=0.20, shuffle=False
        )

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        y_val = y_cv
        y_train = self.scaler.transform(y_train)
        y_val = self.scaler.transform(y_val)
        y_test = self.scaler.transform(y_test)

        if stage == 'fit' or stage is None:
            self.X_train = self.scaler.transform(X_train)
            self.y_train = y_train
            self.X_val = self.scaler.transform(X_val)
            self.y_val = y_val

        if stage == 'test' or stage is None:
            self.X_test = self.scaler.transform(X_test)
            self.y_test = y_test
        

    def train_dataloader(self):
        train_dataset = TimeseriesDataset(self.X_train, 
                                          self.y_train, 
                                          seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset, 
                                  batch_size = self.batch_size, 
                                  shuffle = True, 
                                  num_workers = self.num_workers)
        
        return train_loader

    def val_dataloader(self):
        val_dataset = TimeseriesDataset(self.X_val, 
                                        self.y_val, 
                                        seq_len=self.seq_len)
        val_loader = DataLoader(val_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(self.X_test, 
                                         self.y_test, 
                                         seq_len=self.seq_len)
        test_loader = DataLoader(test_dataset, 
                                 batch_size = self.batch_size, 
                                 shuffle = False, 
                                 num_workers = self.num_workers)

        return test_loader
    
    def predict_dataloader(self):
        return DataLoader(self.scaler.transform(self.df.ts.reshape(-1,1)), batch_size = 1, num_workers = 1, pin_memory = True, shuffle = False)
    

    
    
    
    
    
    
class TimeseriesDatasetDeepAnt(Dataset):   
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs. 
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len].permute(1,0), self.y[index+self.seq_len-1])
    
    
    
    
class dataModuleDeepAnt(pl.LightningDataModule):

    def __init__(self, path:str, src:str, seq_len:int, batch_size = 128, num_workers=0):
        super().__init__()
        self.path = path
        self.src = src
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.X_test = None
        self.columns = None
        self.preprocessing = None
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == 'fit' and self.X_train is not None:
            return 
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:  
            return
        if self.src == 'UCR':
            self.df = TSds.read_UCR(self.path)
        if self.src == 'NAB':
            self.df = TSds.read_NAB(self.path)
        if self.src == 'YAHOO':
            self.df = TSds.read_YAHOO(self.path)
        X = self.df.ts
        y = np.array(self.df.df[['value']].shift(-1).ffill())

        X_cv = self.df.ts[:self.df.train_split].reshape(-1,1)
        y_cv = y[:self.df.train_split]
        X_test = self.df.ts[self.df.train_split:].reshape(-1,1)
        y_test = y[self.df.train_split:]
        
    
        #X_train, X_val, y_train, y_val = train_test_split(
        #    X_cv, y_cv, test_size=0.20, shuffle=False
        #)
        X_train = X_cv
        y_train = y_cv

        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        y_train = self.scaler.transform(y_train)
        #y_val = self.scaler.transform(y_val)
        y_test = self.scaler.transform(y_test)

        if stage == 'fit' or stage is None:
            self.X_train = self.scaler.transform(X_train)
            self.y_train = y_train
            #self.X_val = self.scaler.transform(X_val)
            #self.y_val = y_val

        if stage == 'test' or stage is None:
            self.X_test = self.scaler.transform(X_test)
            self.y_test = y_test
        

    def train_dataloader(self):
        train_dataset = TimeseriesDatasetDeepAnt(self.X_train, 
                                          self.y_train, 
                                          seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset, 
                                  batch_size = self.batch_size, 
                                  shuffle = True, 
                                  num_workers = self.num_workers)
        
        return train_loader

    def val_dataloader(self):
        val_dataset = TimeseriesDatasetDeepAnt(self.X_val, 
                                        self.y_val, 
                                        seq_len=self.seq_len)
        val_loader = DataLoader(val_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDatasetDeepAnt(self.X_test, 
                                         self.y_test, 
                                         seq_len=self.seq_len)
        test_loader = DataLoader(test_dataset, 
                                 batch_size = self.batch_size, 
                                 shuffle = False, 
                                 num_workers = self.num_workers)

        return test_loader
    
    def predict_dataloader(self):
        test_dataset = TimeseriesDatasetDeepAnt(self.scaler.transform(self.df.ts.reshape(-1, 1)), 
                                 self.scaler.transform(np.array(self.df.df[['value']].shift(-1).ffill())), 
                                 seq_len=self.seq_len)
        return DataLoader(test_dataset, batch_size = 1, shuffle = False)