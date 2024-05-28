import pandas as pd
import numpy as np
import math
import random
import datetime
from datetime import datetime
from copy import copy, deepcopy

import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import norm, skewnorm, gennorm, t, nct, genpareto, genextreme, genhyperbolic, chi2, ncx2

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        seq_x, seq_y = data[i:i+n_steps], data[i+n_steps]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_model_lstm(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(units=100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.25))
    model.add(LSTM(units = 50, activation='relu'))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    return model

class VaR_calculation():
    def __init__(self, returns, name, confidence_level = 99):
        self.name = name
        self.returns = returns # ряд доходностей
        self.T = self.returns.shape[0]

        self.conf_level = confidence_level
        self.alpha = (100 - confidence_level) / 100

        self.models_info = {
            'HS': {'func': self.VAR_historical, 'dist': None, 'name': 'Исторический метод'},
            'norm': {'func': self.VAR_normal, 'dist': norm, 'name': 'Нормальное распределение'},
            'skew norm': {'func': self.VAR_skewnorm, 'dist': skewnorm, 'name': 'Скошенное нормальное распределение'},
            'GGD' : {'func': self.VAR_ggd, 'dist': gennorm, 'name': 'Обобщенное нормальное распределение'},
            't': {'func': self.VAR_students, 'dist': t, 'name': 'Распределение Стьюдента'},
            'nct': {'func': self.VAR_nct, 'dist': nct, 'name': 'Нецентральное распределение Стьюдента'},
            'GPD': {'func': self.VAR_gpd, 'dist': genpareto, 'name': 'Обобщенное Парето распределение'},
            'GEV': {'func': self.VAR_gev, 'dist': genextreme, 'name': 'Обобщенное распределение экстремальных значений'},
            'GHYP': {'func': self.VAR_ghyp, 'dist': genhyperbolic, 'name': 'Обобщенное гиперболическое распределение'},
            'LSTM': {'func': self.VAR_lstm, 'dist': None, 'name': 'LSTM'}
        }

        self.models_backtest = None # список моделей, для которых проводится бэктестирование
        self.df_var, self.N = None, None

        # Параметры распределений
        self.parametrs = {
            'norm': {'loc': 0, 'scale': 0},
            'skew norm': {'a': 0, 'loc': 0, 'scale': 0},
            'GGD' : {'beta': 0, 'loc': 0, 'scale': 0},
            't': {'df': 0, 'loc': 0, 'scale': 0},
            'nct': {'df': 0, 'nc': 0, 'loc': 0, 'scale': 0},
            'GPD': {'c': 0, 'loc': 0, 'scale': 0},
            'GEV': {'c': 0, 'loc': 0, 'scale': 0},
            'GHYP': {'p': 0, 'a': 0, 'b': 0, 'loc': 0, 'scale': 0}
            }

        self.scaler = MinMaxScaler()
        self.model_lstm, self.lstm_param = None, None

    def get_methods_list(self):
        """
            Ф-я возвращает список методов, доступных для расчета VaR.
        """
        print(self.models_info.keys())
        lst = [self.models_info[i]['name'] for i in self.models_info.keys()]
        print('\n'.join(lst))

    def data_for_lstm(self, ret):
        ret = np.array(ret).reshape(-1, 1)
        returns_scaled = self.scaler.fit_transform(ret)
        X, y = prepare_data(returns_scaled, self.lstm_param['n_steps'])
        X = X.reshape((X.shape[0], X.shape[1], self.lstm_param['n_features']))
        return X, y

    # Функция перекалибровки параметров
    def calculate_parametrs(self, df, model=None):
        if model is None:
            models = deepcopy(self.models_backtest)
            if 'HS' in self.models_backtest:
                models.remove('HS')
        else:
            models=[model]
        for dist in models:
            try:
                if dist != "LSTM":
                    new_param = self.models_info[dist]['dist'].fit(df)
                    for ind, p in enumerate(list(self.parametrs[dist].keys())):
                        self.parametrs[dist][p] = new_param[ind]
                else:
                    self.model_lstm = create_model_lstm(n_steps=self.lstm_param['n_steps'], n_features=self.lstm_param['n_features'])
                    X, y = self.data_for_lstm(ret=df, n_steps=self.lstm_param['n_steps'], n_features=self.lstm_param['n_features'])
                    self.model_lstm.fit(X, y, epochs=self.lstm_param['n_epochs'], verbose=0)
            except: pass

    # Исторический метод
    def VAR_historical(self, returns=None):
        returns = self.returns if returns is None else returns
        return np.quantile(returns, self.alpha)

    # Нормальное распределение
    def VAR_normal(self, returns=None):
        returns = self.returns if returns is None else returns
        return stats.norm.ppf(self.alpha, loc=self.parametrs['norm']['loc'], scale=self.parametrs['norm']['scale'])

    # Скошенное нормальное распределение
    def VAR_skewnorm(self, returns=None):
        returns = self.returns if returns is None else returns
        return stats.skewnorm.ppf(self.alpha, a=self.parametrs['skew norm']['a'], loc=self.parametrs['skew norm']['loc'], scale=self.parametrs['skew norm']['scale'])

    # Обощенное нормальное распределение
    def VAR_ggd(self, returns=None):
        returns = self.returns if returns is None else returns
        return stats.gennorm.ppf(self.alpha, beta=self.parametrs['GGD']['beta'], loc=self.parametrs['GGD']['loc'], scale=self.parametrs['GGD']['scale'])

    # Распределение Стьюдента
    def VAR_students(self, returns=None, degrees_of_freedom=10):
        returns = self.returns if returns is None else returns
        return stats.t.ppf(self.alpha, degrees_of_freedom, loc=self.parametrs['t']['loc'], scale=self.parametrs['t']['scale'])

    # Нецентральное распределение Стьюдента
    def VAR_nct(self, returns=None, degrees_of_freedom=10):
        returns = self.returns if returns is None else returns
        return stats.nct.ppf(self.alpha, degrees_of_freedom, nc=self.parametrs['nct']['nc'], loc=self.parametrs['nct']['loc'], scale=self.parametrs['nct']['scale'])

    # Обощенное распределение Парето
    def VAR_gpd(self, returns=None):
        returns = self.returns if returns is None else returns
        return stats.genpareto.ppf(self.alpha, c=self.parametrs['GPD']['c'], loc=self.parametrs['GPD']['loc'], scale=self.parametrs['GPD']['scale'])

    # Обобщенное распределение экстремальных значений
    def VAR_gev(self, returns=None):
        returns = self.returns if returns is None else returns
        return stats.genextreme.ppf(self.alpha, c=self.parametrs['GEV']['c'], loc=self.parametrs['GEV']['loc'], scale=self.parametrs['GEV']['scale'])

    # Обобщенное гиперболическое распределение
    def VAR_ghyp(self, returns=None):
        returns = self.returns if returns is None else returns
        return stats.genhyperbolic.ppf(self.alpha, p=self.parametrs['GHYP']['p'], a=self.parametrs['GHYP']['a'], b=self.parametrs['GHYP']['b'],
                                       loc=self.parametrs['GHYP']['loc'], scale=self.parametrs['GHYP']['scale'])

    # LSTM
    def VAR_lstm(self, returns=None):
        returns = self.returns if returns is None else returns
        X, y = self.data_for_lstm(ret=returns)
        #y = self.scaler.inverse_transform(y)
        y_pred = self.model_lstm.predict(X, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred)
        errors = y_pred.flatten()
        return np.quantile(errors, self.alpha)

    # Бэктестинг
    def backtest(self, window_size=250, recalibration_freq=50,
                 models=None,
                 lstm_dict={'n_steps':25,
                            'n_features':1,
                            'n_epochs':50
                            },
                 return_=0):
        """
            Функция обратного тестирования.
            Если return_ = 1, возвращает DataFrame с результатми вычисления VaR для каждого момента времени различными моделями.
            models - cписок тестируемых моделей. По умолчанию ['HS', 'norm', 'skew norm', 'GGD', 't', 'nct', 'GPD', 'GEV', 'GHYP'].
            window_size - размер окна обучения.
            recalibration_freq - частота наблюдений, с которой переопределеяются параметры распределений.
            lstm_dict содержит параметры, необходимые для обучения LSTM.
        """
        self.models_backtest = list(self.models_info.keys()) if models is None else models

        if 'LSTM' in self.models_backtest:
            self.lstm_param = lstm_dict

        self.df_var = deepcopy(self.returns).to_frame()
        c = 0

        for i in range(window_size, self.T):
            date = self.returns.index[i]
            ret = self.returns[i-window_size : i]
            if c % recalibration_freq == 0:
                self.calculate_parametrs(ret)
            c += 1
            for method in self.models_backtest:
                var_value = self.models_info[method]['func'](ret)
                self.df_var.loc[date, method] = var_value
                # Дополнительно проводим перекалибровку параметров после пробоя
                #if self.df_var.loc[date, self.name] < var_value:
                #    self.calculate_parametrs(ret, model=method)

        self.df_var.dropna(inplace = True)
        self.N = self.df_var.shape[0]

        if return_:
            return self.df_var

    def get_var_results(self):
        """
            Ф-я возвращает DataFrame c результатами расчета VaR различными моделями.
        """
        return self.df_var    