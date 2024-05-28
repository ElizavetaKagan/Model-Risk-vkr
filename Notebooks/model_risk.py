import pandas as pd
import numpy as np

import datetime
from datetime import datetime

from copy import copy, deepcopy

import sys
import warnings

import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import norm, skewnorm, gennorm, t, nct, genpareto, genextreme, genhyperbolic, chi2, ncx2

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

class Model_risk_calculation():
    def __init__(self, df_var, name, confidence_level = 99, models=None):
        self.df_var = deepcopy(df_var)
        self.T = self.df_var.shape[0]
        self.name = name

        self.conf_level = confidence_level
        self.alpha = (100 - confidence_level) / 100

        self.models_info = {
            'HS': {'name': 'Исторический метод', 'c':'steelblue'},
            'norm': {'name': 'Нормальное распределение', 'c':'lightcoral'},
            'skew norm': {'name': 'Скошенное нормальное распределение', 'c':'gold'},
            'GGD' : {'name': 'Обобщенное нормальное распределение', 'c':'darkorange'},
            't': {'name': 'Распределение Стьюдента', 'c':'purple'},
            'nct': {'name': 'Нецентральное распределение Стьюдента', 'c':'saddlebrown'},
            'GPD': {'name': 'Обобщенное Парето распределение', 'c':'red'},
            'GEV': {'name': 'Обобщенное распределение экстремальных значений', 'c':'seagreen'},
            'GHYP': {'name': 'Обобщенное гиперболическое распределение', 'c':'orchid'},
            'LSTM': {'name':'LSTM', 'c':'darkblue'}
        }

        # Получение списка моделей, использованных для вычисления VaR
        self.models = list(self.df_var.columns.drop(self.name)) if models is None else models
        self.N = len(self.models)

        self.df_res = pd.DataFrame(index = self.models)

    def risk_ratio(self, fig_size=(16, 4)):
        # чем выше значение показателя, тем выше степень модельного риска в конкретный период
        df_rr = pd.DataFrame(index = self.df_var.index)
        for d in self.df_var.index:
            val_lst = np.abs(self.df_var.loc[d][self.models].values)
            df_rr.loc[d, 'Risk Ratio'] = np.max(val_lst) / np.min(val_lst)

        fig, axs = plt.subplots(1, 1, figsize=fig_size)
        axs.plot(df_rr['Risk Ratio'], color='black')
        axs.set_title("Risk Ratio")
        axs.axhline(y=1, color='salmon', linestyle = '--', linewidth=0.8)
        plt.show()

    def legal_robustness(self, fig_size=(16, 4)):
        df_lr = pd.DataFrame(index = self.df_var.index)
        for d in self.df_var.index:
            val_lst = self.df_var.loc[d][self.models].values
            var_mean = np.mean(val_lst)
            s1 = np.abs(val_lst - var_mean)
            df_lr.loc[d, 'Legal Robustness'] = np.mean(s1) / var_mean

        fig, axs = plt.subplots(1, 1, figsize=fig_size)
        axs.plot(df_lr['Legal Robustness'], color='steelblue')
        axs.set_title("Legal Robustness")
        plt.show()

    # Консервативность

    def absolute_measure(self, plot_=1, return_=1, fig_size=(16, 6)):
        df_am = pd.DataFrame(index = self.df_var.index, columns=self.models)
        for d in self.df_var.index:
            var_lst = np.abs(self.df_var.loc[d][self.models].values)
            var_max = np.max(var_lst)
            df_am.loc[d] = var_max / var_lst - 1

        if plot_:
            fig, axs = plt.subplots(1, 1, figsize=fig_size)
            for method in df_am.columns:
                axs.plot(df_am[method], label=method, color=self.models_info[method]['c'])
            axs.set_title("Absolute measure")
            axs.legend()
            plt.show()

        df_mean = pd.DataFrame(columns=["AM среднее"], data=df_am.mean(axis=0))
        self.df_res = pd.merge(self.df_res, df_mean, how='left', left_index=True, right_index=True)
        
        if return_:
            df_mean['Метод'] = [self.models_info[i]['name'] for i in df_mean.index]
            df_mean.set_index('Метод', inplace=True)
            df_mean.sort_values(by="AM среднее", inplace=True)
            print('Чем меньше показатель, тем ниже значение модельного риска (лучше).\n')
            return df_mean

    def relative_measure(self, plot_=1, return_=1, fig_size=(16, 6)):
        df_rm = pd.DataFrame(index = self.df_var.index, columns=self.models)
        for d in self.df_var.index:
            var_lst = np.abs(self.df_var.loc[d][self.models].values)
            var_max = np.max(var_lst)
            var_min = np.min(var_lst)
            df_rm.loc[d] = (var_max - var_lst ) / (var_max - var_min)

        if plot_:
            fig, axs = plt.subplots(1, 1, figsize=fig_size)
            for method in df_rm.columns:
                axs.plot(df_rm[method], label=method, color=self.models_info[method]['c'])
            axs.set_title("Relative measure")
            axs.legend()
            plt.show()

        df_mean = pd.DataFrame(columns=["RM среднее"], data=df_rm.mean(axis=0))
        self.df_res = pd.merge(self.df_res, df_mean, how='left', left_index=True, right_index=True)

        if return_:
            df_mean['Метод'] = [self.models_info[i]['name'] for i in df_mean.index]
            df_mean.set_index('Метод', inplace=True)
            df_mean.sort_values(by="RM среднее", inplace=True)
            print('Чем меньше показатель, тем ниже значение модельного риска (лучше).\n')
            return df_mean

    def mean_relative_deviation(self, plot_=1, return_=1, fig_size=(16, 6)):
        df_mrd = pd.DataFrame(index = self.df_var.index, columns=self.models)
        for d in self.df_var.index:
            var_lst = np.abs(self.df_var.loc[d][self.models].values)
            var_mean = np.mean(var_lst)
            df_mrd.loc[d] = var_lst / var_mean - 1

        if plot_:
            fig, axs = plt.subplots(1, 1, figsize=fig_size)
            for method in df_mrd.columns:
                axs.plot(df_mrd[method], label=method, color=self.models_info[method]['c'])
            axs.axhline(y=0, color='salmon', linestyle = '--', linewidth=0.8)
            axs.set_title("Mean Relative Deviation")
            axs.legend()
            plt.show()

        df_mean = pd.DataFrame(columns=["MRD среднее"], data=df_mrd.mean(axis=0))
        self.df_res = pd.merge(self.df_res, df_mean, how='left', left_index=True, right_index=True)

        if return_:
            df_mean['Метод'] = [self.models_info[i]['name'] for i in df_mean.index]
            df_mean.set_index('Метод', inplace=True)
            df_mean.sort_values(by="MRD среднее", ascending=False, inplace=True)
            print('При отрицательном значении показатель больше подвержена модельному риску, при положительном — меньше.\n')
            return df_mean

    # Точность

    def modified_binary_loss_function(self, return_=1):
        df_mblf = pd.DataFrame(index = self.models, columns=['MBLF'])
        for method in self.models:
             I = (self.df_var[self.name] < self.df_var[method]).astype(int)
             #df_mblf.loc[method]['BLF'] = np.sum(I)/self.T - self.alpha
             df_mblf.loc[method]['MBLF'] = np.abs(np.sum(I)/self.T - self.alpha)
        self.df_res = pd.merge(self.df_res, df_mblf, how='left', left_index=True, right_index=True)
        if return_:
            df_mblf['Метод'] = [self.models_info[i]['name'] for i in df_mblf.index]
            df_mblf.set_index('Метод', inplace=True)
            df_mblf.sort_values(by='MBLF', inplace=True)
            print('Чем ниже значение показателя, тем ближе к заданному уровню доверия (лучше).\n')
            return df_mblf

    def moc_method(self, method, grid= None, plot_=0):
        grid = list(np.arange(0, 2.01, 0.01)) if grid is None else grid
        res = []
        for m in grid:
            I = (self.df_var[self.name] < m * self.df_var[method]).astype(int)
            moc_cur = np.round(self.alpha * self.T) - np.sum(I)
            res.append(moc_cur) 
        
        close_to_0 = res[0]
        moc_val = 0
        for i in range(0, len(res)):
            if np.abs(res[i]) < abs(close_to_0):
                moc_val, close_to_0 = grid[i], res[i]
        
        if plot_:
            fig, axs = plt.subplots(1, 1, figsize=(8, 4))
            axs.plot(grid, res, color='steelblue')
            axs.axhline(y=0, color='salmon', linestyle = '--', linewidth=0.8)
            axs.axvline(x=moc_val, color='goldenrod', linewidth=0.8)
            plt.show()

        return moc_val

    def moc(self, return_=1):
        df_moc = pd.DataFrame(index = self.models, columns=['MOC'])
        for method in self.models:
            df_moc.loc[method]['MOC'] = self.moc_method(method)
        self.df_res = pd.merge(self.df_res, df_moc, how='left', left_index=True, right_index=True)
        
        if return_:
            df_moc['Метод'] = [self.models_info[i]['name'] for i in df_moc.index]
            df_moc.set_index('Метод', inplace=True)
            df_moc.sort_values(by='MOC', inplace=True)
            print('Чем ближе индикатор к единице, тем точнее модель.\n')
            return df_moc

    # Эффективность

    def scaled_var(self):
        new_df = pd.DataFrame(index = self.df_var.index, columns=self.models)
        for t in self.df_var.index:
            for m in self.models:
                new_df.loc[t][m] = self.df_var.loc[t][m] * self.df_res.loc[m]['MOC']
        return new_df

    def efficiency_ratio(self, plot_=1, return_=1, fig_size=(16, 6)):
        if 'MOC' not in self.df_res.columns:
            self.moc()
        df_scaled = self.scaled_var()

        df_er = pd.DataFrame(index = self.df_var.index, columns=self.models)
        for d in self.df_var.index:
            var_lst = np.abs(df_scaled.loc[d][self.models].values)
            var_max = np.max(var_lst)
            var_min = np.min(var_lst)
            df_er.loc[d] = (var_max - var_lst ) / (var_max - var_min)

        if plot_:
            fig, axs = plt.subplots(1, 1, figsize=fig_size)
            for method in df_er.columns:
                axs.plot(df_er[method], label=method, color=self.models_info[method]['c'])
            axs.set_title("Efficiency Ratio")
            axs.legend()
            plt.show()

        df_mean = pd.DataFrame(columns=["ER"], data=df_er.mean(axis=0))
        self.df_res = pd.merge(self.df_res, df_mean, how='left', left_index=True, right_index=True)

        if return_:
            df_mean['Метод'] = [self.models_info[i]['name'] for i in df_mean.index]
            df_mean.set_index('Метод', inplace=True)
            df_mean.sort_values(by="ER",  ascending=False, inplace=True)
            print('Чем ближе коэффициент к единице, тем эффективнее модель. \n')
            return df_mean

    def mean_relative_scaled_bias(self, plot_=1, return_=1, fig_size=(16, 6)):
        if 'MOC' not in self.df_res.columns:
            self.moc()
        df_scaled = self.scaled_var()

        df_mrsb = pd.DataFrame(index = self.df_var.index, columns=self.models)
        for d in self.df_var.index:
            var_lst = np.abs(df_scaled.loc[d][self.models].values)
            var_mean = np.mean(var_lst)
            df_mrsb.loc[d] = var_lst / var_mean - 1

        if plot_:
            fig, axs = plt.subplots(1, 1, figsize=fig_size)
            for method in df_mrsb.columns:
                axs.plot(df_mrsb[method], label=method, color=self.models_info[method]['c'])
            axs.set_title("Mean Relative Scaled Bias")
            axs.legend()
            plt.show()

        df_mean = pd.DataFrame(columns=["MRSB"], data=df_mrsb.mean(axis=0))
        self.df_res = pd.merge(self.df_res, df_mean, how='left', left_index=True, right_index=True)

        if return_:
            df_mean['Метод'] = [self.models_info[i]['name'] for i in df_mean.index]
            df_mean.set_index('Метод', inplace=True)
            #df_mean.sort_values(by="MRSB", ascending=False, inplace=True)
            return df_mean

    def magnitude_loss_function(self, return_=1):
        df_lm = pd.DataFrame(index = self.models, columns=['LMLF', 'MLF'])
        for method in self.models:
            I = (self.df_var[self.name] < self.df_var[method]).astype(int)
            s1 = (self.df_var[self.name] - self.df_var[method])**2 + 1 
            s2 = np.exp(self.df_var[self.name] - self.df_var[method])
            df_lm.loc[method]['LMLF'] = np.sum(s1 * I) / self.T
            df_lm.loc[method]['MLF'] = np.sum(s2 * I) / self.T
        self.df_res = pd.merge(self.df_res, df_lm, how='left', left_index=True, right_index=True)

        if return_:
            df_lm['Метод'] = [self.models_info[i]['name'] for i in df_lm.index]
            df_lm.set_index('Метод', inplace=True)
            df_lm.sort_values(by='MLF', inplace=True)
            print('Чем меньше значение показателя, тем менее глубокое пробитие. \n')
            return df_lm

    def get_result(self):
        self.df_res['Метод'] = [self.models_info[i]['name'] for i in self.df_res.index]
        self.df_res = self.df_res.set_index('Метод')
        return self.df_res

    def model_risk_final(self):
        self.absolute_measure(plot_=0, return_=0)
        self.relative_measure(plot_=0, return_=0)
        self.mean_relative_deviation(plot_=0, return_=0)
        self.modified_binary_loss_function(return_=0)
        self.moc(return_=0)
        self.efficiency_ratio(plot_=0, return_=0)
        self.mean_relative_scaled_bias(plot_=0, return_=0)
        self.magnitude_loss_function(return_=0)

        return self.get_result()