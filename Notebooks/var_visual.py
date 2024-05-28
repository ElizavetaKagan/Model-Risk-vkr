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

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

class VAR_visualisation():
    def __init__(self, name, df_var, confidence_level = 99, models=None):
        self.name = name
        self.df_var = df_var
        #self.T = self.df_var.shape[0]

        self.window_lst = pd.unique(self.df_var['Window'])
        self.freq_lst = pd.unique(self.df_var['Frequency'])

        self.conf_level = confidence_level
        self.alpha = (100 - confidence_level) / 100

        self.models_info = {
            'HS': {'name': 'Исторический метод', 'dist': None, 'c':'steelblue'},
            'norm': {'name': 'Нормальное распределение', 'dist': norm, 'c':'lightcoral'},
            'skew norm': {'name': 'Скошенное нормальное распределение', 'dist': skewnorm, 'c':'gold'},
            'GGD' : {'name': 'Обобщенное нормальное распределение', 'dist': gennorm, 'c':'darkorange'},
            't': {'name': 'Распределение Стьюдента', 'dist': t, 'c':'purple'},
            'nct': {'name': 'Нецентральное распределение Стьюдента', 'dist': nct, 'c':'saddlebrown'},
            'GPD': {'name': 'Обобщенное Парето распределение', 'dist': genpareto, 'c':'red'},
            'GEV': {'name': 'Обобщенное распределение экстремальных значений', 'dist': genextreme, 'c':'seagreen'},
            'GHYP': {'name': 'Обобщенное гиперболическое распределение', 'dist': genhyperbolic, 'c':'orchid'},
            'LSTM': {'name':'LSTM', 'dist':None, 'c':'darkblue'}
        }

        # список моделей, для которых проводится бэктестированиеe
        self.models_backtest = list(self.models_info.keys()) if models is None else models

        self.color_dict = pd.Series(['honeydew', 'white'], index=[1, 0]).to_dict()

    def var_wf(self, window, freq):
        return self.df_var[(self.df_var['Window'] == window) & (self.df_var['Frequency'] == freq)]

    def calculate_breakdown(self, window, freq):
        """
            Ф-я возращает таблицу, в столбцах которой количество и доля пробоев, в строках - модели.
            window - окно обучения;
            freq - частота переобучения параметров.
        """
        df_wf = self.var_wf(window, freq)
        T = df_wf.shape[0]

        df_res = pd.DataFrame(index=self.models_backtest, columns = ['Кол-во пробоев'])
        for method in self.models_backtest:
             df_res.loc[method, 'Кол-во пробоев'] = np.sum(df_wf[self.name] < df_wf[method])
        df_res['Доля пробоев (%)'] = (df_res['Кол-во пробоев'] / T * 100).astype(float)
        df_res['Доля пробоев (%)'] = np.round(df_res['Доля пробоев (%)'], 2)
        return df_res

    def get_backtest_results(self):
        """
            Ф-я возращает таблицу, где для каждой пары (окно, частота перекалибровки параметров)
                отображается доля пробитий для различных моделей.
        """
        d = pd.DataFrame(index = pd.MultiIndex.from_product([self.window_lst, self.freq_lst], names=['Window','Freq']))
        for w in self.window_lst:
            for f in self.freq_lst:
                res = self.calculate_breakdown(w, f)
                for method in self.models_backtest:
                    d.loc[(w, f), self.models_info[method]['name']] = res.loc[method]['Доля пробоев (%)']
        print('Доля отклонений (%) \n')
        return d

    def plot_backtest(self, window, freq,
                      models=None,
                      start_date=None, end_date=None,
                      fig_size = (18, 6)):
        """
            Ф-я отображает результаты бэктеста с заданными параметрами окна (window)
                    и частоты переобучения параметров(freq).
            models - список моделей, которые выводятся. По умолчанию все модели, для которых производился расчет.
            Чтобы вывести определенный временной интервал, необходимо указать аргументы start_date и/или end_date.
        """
        models = self.models_backtest if models is None else models

        df_wf = self.var_wf(window, freq)
        start_date = min(df_wf.index) if start_date is None else start_date
        end_date = max(df_wf.index) if end_date is None else end_date
        df_wf_1 = df_wf[(df_wf.index >= start_date) & (df_wf.index <= end_date)]

        fig, axs = plt.subplots(1, 1, figsize=fig_size)
        axs.plot(df_wf_1[self.name], color='silver', linewidth = 1, label = 'Returns')
        for m in models:
            axs.plot(df_wf_1[m], linewidth = 0.8, label = m, color=self.models_info[m]['c'])
        axs.legend()
        axs.set_title(f'{self.name}: VaR {self.conf_level}% \n Window = {window}, frequency = {freq}')
        plt.show()

    def plot_parametr_window(self, freq,
                        models=None,
                        fig_size = (16, 4)):
        """
            Зависимость доли отклонений от окна для заданной частоты перекалибровки параметров (freq) для различных моделей (models).
        """
        models = self.models_backtest if models is None else models

        fig, axs = plt.subplots(1, 1, figsize=fig_size)
        for model in models:
            ar = []
            for w in self.window_lst:
                ar.append(self.calculate_breakdown(w, freq).loc[model]['Доля пробоев (%)'])
            axs.plot(self.window_lst, ar, linewidth = 0.8, label = model, color=self.models_info[model]['c'])
        axs.set_xticks(self.window_lst)
        axs.set_xlabel('Размер окна')
        axs.set_ylabel('Доля отклонений (%)')
        axs.set_title(f'{self.name}: VaR {self.conf_level}% (frequency = {freq})')
        axs.legend(loc='upper right')
        plt.show()

    def plot_parametr_frequency(self, window,
                        models=None,
                        fig_size = (16, 4)):
        """
            Зависимость доли отклонений от частоты перекалибровки параметров для заданного окна (window) для различных моделей (models).
        """
        models = self.models_backtest if models is None else models

        fig, axs = plt.subplots(1, 1, figsize=fig_size)
        for model in models:
            ar = []
            for f in self.freq_lst:
                ar.append(self.calculate_breakdown(window, f).loc[model]['Доля пробоев (%)'])
            axs.plot(self.freq_lst, ar, linewidth = 0.8, label = model, color=self.models_info[model]['c'])
        axs.set_xticks(self.freq_lst)
        axs.set_xlabel('Частота перекалибровки параметров')
        axs.set_ylabel('Доля отклонений (%)')
        axs.set_title(f'{self.name}: VaR {self.conf_level}% (window = {window})')
        axs.legend(loc='upper right')
        plt.show()

    def plot_parametr_backtest(self, model, parametr, window=None, freq=None, fig_size=(16, 4)):
        """
            Ф-я отображает линию VaR для выбранной модели model и фиксированного одного из параметров.
            parametr - параметр, по которому изучается зависимость.
            Если parametr = 'window', тогда фиксируется аргумент freq.
            Если parametr = 'frequency', тогда фиксируется аргумент window.
        """
        fig, axs = plt.subplots(1, 1, figsize=fig_size)
        df_wf0 = self.var_wf(250, 25)
        axs.plot(df_wf0[self.name], color='silver', linewidth = 1, label = 'Returns')
        if parametr == 'window':
            freq = 25 if freq is None else freq
            for w in self.window_lst:
                df_wf = self.var_wf(w, freq)
                axs.plot(df_wf[model], linewidth = 0.8, label = w)
            axs.set_title(f'{self.name}: VaR {self.conf_level}% (frequency = {freq})')
        elif parametr == 'frequency':
            window = 250 if window is None else window
            for f in self.freq_lst:
                df_wf = self.var_wf(window, f)
                axs.plot(df_wf[model], linewidth = 0.8, label = f)
            axs.set_title(f'{self.name}: VaR {self.conf_level}% (window = {window})')
        else:
            print('Неверно указан параметр.')
        axs.legend()
        plt.show()

    def test_kupiec(self, window, freq, lev=0.05):
        d = pd.DataFrame()
        df_wf = self.var_wf(window, freq)
        T = df_wf.shape[0]
        a = self.alpha
        for method in self.models_backtest:
            I = (df_wf[self.name] < df_wf[method]).astype(int)
            k = np.sum(I)
            L_pof = -2 * np.log(((1 - a)**(T - k) * (a**k)) / ((1 - k/T)**(T-k) * ((k/T)**k)))
            p_value = chi2.sf(L_pof, df=1)
            d.loc[self.models_info[method]['name'], 'Kupiec POF'] = L_pof
            d.loc[self.models_info[method]['name'], 'p value'] = np.round(p_value, 5)
            d.loc[self.models_info[method]['name'], 'H0 принимается'] = 1 if p_value > lev else 0
        return d

    def test_kupiec_all(self, lev=0.05):
        """
            Ф-я возращает таблицу, где для каждой пары (окно, частота перекалибровки параметров)
                отображается результат теста Kupiec.
            1 - H0 не отвергается, модель принимается
            0 - модель не принимается
        """
        d = pd.DataFrame(index = pd.MultiIndex.from_product([self.window_lst, self.freq_lst], names=['Window','Freq']))
        for w in self.window_lst:
            for f in self.freq_lst:
                res = self.test_kupiec(w, f, lev)
                for method in res.index:
                    d.loc[(w, f), method] = res.loc[method]['H0 принимается'].astype(int)
        d = d.style.applymap(lambda v: f"background-color: {self.color_dict.get(v, 'None')}").format('{:.0f}', na_rep='-')
        return d

    def test_christoffersen(self, window, freq, lev=0.05):
        d = pd.DataFrame()
        df_wf = self.var_wf(window, freq)
        T = df_wf.shape[0]
        for method in self.models_backtest:
            I = (df_wf[self.name] < df_wf[method]).astype(int)
            n00, n01, n10, n11 = 0,0,0,0
            for t in range(1, len(I)):
                if I[t-1] == 0 and I[t] == 0:
                    n00 += 1
                elif I[t-1] == 0 and I[t] == 1:
                    n01 += 1
                elif I[t-1] == 1 and I[t] == 0:
                    n10 += 1
                else:
                    n11 += 1
            p0 = n01/(n00 + n01)
            p1 = n11/(n10 + n11)
            p = (n01 + n11)/(n00 + n01 + n10 + n11)
            L_ind = -2 * np.log((1 - p)**(n00 + n10) * p**(n01 + n11)) + 2 * np.log((1 - p0)**n00 * p0**n01 * (1 - p1)**n10 * p1**n11)
            p_value = chi2.sf(L_ind, df=1)
            d.loc[self.models_info[method]['name'], 'LR_ind'] = L_ind
            d.loc[self.models_info[method]['name'], 'p value'] = np.round(p_value, 5)
            d.loc[self.models_info[method]['name'], 'H0 принимается'] = 1 if p_value > lev else 0
        return d

    def test_christoffersen_all(self, lev=0.05):
        """
            Ф-я возращает таблицу, где для каждой пары (окно, частота перекалибровки параметров)
                отображается результат теста Christoffersen.
            1 - H0 не отвергается, модель принимается
            0 - модель не принимается
        """
        d = pd.DataFrame(index = pd.MultiIndex.from_product([self.window_lst, self.freq_lst], names=['Window','Freq']))
        for w in self.window_lst:
            for f in self.freq_lst:
                res = self.test_christoffersen(w, f, lev)
                for method in res.index:
                    d.loc[(w, f), method] = res.loc[method]['H0 принимается'].astype(int)
        d = d.style.applymap(lambda v: f"background-color: {self.color_dict.get(v, 'None')}").format('{:.0f}', na_rep='-')
        return d