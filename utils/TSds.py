import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import json


class TSds():

    def __init__(self, df:pd.DataFrame, name:str, ts:np.array, source:str, training:int):
        self.df = df
        self.name = name
        self.ts = ts
        self.source = source
        self.train_split = training

    @classmethod
    def read_UCR(cls, path:str):
        
        split_name = str(path).split('/')
        ds_name = '/'.join(split_name[-2:])

        split_name = str(split_name[-1]).split('.')[0]
        name_aux = str(split_name).split('_')

        ts = np.genfromtxt(path)
        df = pd.DataFrame(ts, columns = ['value'])
        #self._features['DS_name'] = ds_name
        
        anomaly = np.zeros(len(df), dtype = np.int32)
        anomaly[int(name_aux[5]):int(name_aux[6]) + 1] = 1
        df['is_anomaly'] = anomaly

        return cls(df = df, name = ds_name, ts = ts, source = "UCR", training = int(name_aux[4]))


    @classmethod
    def read_YAHOO(cls,path:str):
        
        split_name = str(path).split('/')
        ds_name = '/'.join(split_name[-2:])
        df = pd.read_csv(path)
        df.set_index('timestamp', inplace = True)
        ts = np.array(df['value'])

        return cls(df = df, name = ds_name, ts = ts, source ="YAHOO")

    @classmethod
    def read_NAB(cls, path:str):
        
        split_name = str(path).split('/')
        ds_name = '/'.join(split_name[-2:])
        df = pd.read_csv(path, parse_dates=[0], index_col= 0)
        ts = np.array(df.value)
        df['is_anomaly'] = cls._get_NAB_anomaly(df, ds_name)

        return cls(df = df, name = ds_name, ts = ts, source ="NAB")

    @staticmethod
    def _get_NAB_anomaly(df:pd.DataFrame, ds_name:str = None, path:str = None):
        if path == None:
            with urllib.request.urlopen("https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json") as url:
                an = json.load(url)
        else:
            with open(path, "r") as jsonF:
                an = json.load(jsonF)

        aux = np.zeros(len(df), dtype = np.int32)
        for start, end in an[ds_name]:
            aux[df.index.get_loc(pd.to_datetime(start)): df.index.get_loc(pd.to_datetime(end))] = 1
        return aux



    @staticmethod
    def __get_anomaly_window(df:pd.DataFrame):
        edges = np.diff(np.concatenate([[0],df['is_anomaly'],[0]])).nonzero()[0]
        edges = edges.reshape((-1,2)) + np.array([0,-1])
        if type(df.index) == pd.core.indexes.datetimes.DatetimeIndex:
            return np.array(df.index)[edges]
        else:
            return edges


    def plot(self, width:int = 25, height:int = 8):

        my_alpha = 0.4
        plt.figure(figsize=(width,height))
        real_anoms = self.__get_anomaly_window(self.df)

        if self.source in ['YAHOO','UCR']:

            extend_window = 2
            for anom in real_anoms:
                plt.axvspan(anom[0]-extend_window,anom[1]+extend_window, ymin=0.0, ymax=50, alpha=my_alpha, color='red')
            plt.plot(self.df['value'], zorder=1)
            plt.ylim((self.df['value'].values.min(),self.df['value'].values.max()));
        else:

            for anom in real_anoms:
                plt.axvspan(anom[0],anom[1], ymin=0.0, ymax=50, alpha=my_alpha, color='red')
            plt.plot(self.df['value'], zorder=1)
            plt.ylim((self.df['value'].values.min(),self.df['value'].values.max()));
        plt.draw()
