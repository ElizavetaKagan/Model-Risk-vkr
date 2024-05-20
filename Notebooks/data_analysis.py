import pandas as pd
import numpy as np
import math
import random
import datetime
from datetime import datetime
from copy import copy, deepcopy

import yfinance as yf

import sys
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
import scipy.stats as stats
from scipy.stats import norm, skewnorm, gennorm, t, nct, genpareto, genextreme, genhyperbolic, exponpow, chi2, ncx2
from scipy.stats import skew, kurtosis, goodness_of_fit

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

class FinancialInstrument:
    def __init__(self, df, name):
        """
            Входные данные:
            df - DataFrame с ценами
            name - наименование фин. инструмента (акция, курс валют и т.д.)
        """
        self.name = name
        self.data_price = df[[name]] # ряд цен
        self.data = self.calculate_returns() # ряд дохоностей
        self.data_price['date'] = df.index
        self.data['date'] = self.data.index
        self.T = self.data.shape[0]

        # перечень распределений
        self.dist_dict = {'norm': [norm, 'Нормальное распределение'],
                    'skew norm': [skewnorm, 'Скошенное нормальное распределение'],
                    'GGD' : [gennorm, 'Обобщенное нормальное распределение'],
                    't': [t, 'Распределение Стьюдента'],
                    'nct': [nct, 'Нецентральное распределение Стьюдента'],
                    'GPD': [genpareto, 'Обобщенное Парето распределение'],
                    'GEV': [genextreme, 'Обобщенное распределение экстремальных значений'],
                    'GHYP': [genhyperbolic, 'Обобщенное гиперболическое распределение']
                    }

    def calculate_returns(self):
        """ Вычисление ряда доходностей. """
        return self.data_price.pct_change().dropna()

    def available_distributions(self):
        """ Ф-я выводит список доступных распределений. """
        for i in self.dist_dict:
            print(self.dist_dict[i][1])

    def price_dynamics(self, fig_size=(14, 3)):
        """ Ф-я отображает ряд цен. """
        fig, axs = plt.subplots(1, 1, figsize = fig_size)
        g1 = sns.lineplot(data = self.data_price, x = 'date', y=self.name, ax = axs, color = 'steelblue')
        g1.set(title=f'{self.name}: prices')
        g1.set(xlabel = None, ylabel = None)
        plt.show()

    def returns_dynamics(self, fig_size=(14, 3)):
        """ Ф-я отображает ряд доходностей. """
        fig, axs = plt.subplots(1, 1, figsize = fig_size)
        g1 = sns.lineplot(data = self.data, x = 'date', y=self.name, ax = axs, color = 'steelblue', linewidth=0.5)
        g1.set(title=f'{self.name}: returns')
        g1.set(xlabel = None, ylabel = None)
        plt.show()

    def histogram(self, bins=100, fig_size=(14, 5)):
        """
            Ф-я отображает гистограмму распределения значений доходностей.
            По умолчанию bins = 100.
        """

        coef_skew, txt1 = self.skewness(print_info=0, return_info=1)
        coef_kurtosis, txt2 = self.kurtosis(print_info=0, return_info=1)

        fig, axs = plt.subplots(1, 1, figsize = fig_size)
        g1 = sns.histplot(data = self.data, x = self.name, bins = bins,
                            stat = 'density', kde = True,
                            ax = axs, color = 'steelblue')
        g1.set(title=f'{self.name}: histogram of daily returns \n Skewness = {coef_skew} ({txt1}) \n Kurtosis = {coef_kurtosis} ({txt2})')
        g1.set(xlabel = None, ylabel = None)
        axs.axvline(x=0, color='indianred', linewidth=0.75)
        plt.show()

    def skewness(self, print_info=0, return_info=1):
        """
            Ф-я вычисления коэффициента асимметрии.
            Если print_info=1, выводится значение коэффициента и комментарий.
            Если return_info=1, возвращается значение коэффициента.
        """
        coef_skew = stats.skew(self.data[self.name])
        if coef_skew == 0:
            full_txt, short_txt = "Symmetrical.", "symmetrical"
        elif coef_skew < 0:
            full_txt, short_txt = "The left tail of the distribution is longer or stretched out.", "left skewed"
        else:
            full_txt, short_txt = "The right tail of the distribution is longer or stretched out.", "right skewed"

        if print_info:
            print('Skewness: ', np.round(coef_skew, 3))
            print(full_txt)
        if return_info:
            return np.round(coef_skew, 2), short_txt

    def kurtosis(self, print_info=0, return_info=1):
        """
            Ф-я вычисления коэффициента эксцесса.
            Если print_info=1, выводится значение коэффициента и комментарий.
            Если return_info=1, возвращается значение коэффициента.
        """
        coef_kurt = stats.kurtosis(self.data[self.name])
        if coef_kurt == 0:
            full_txt, short_txt = "Mesokurtic. This distribution has a kurtosis similar to that of the normal distribution.", "mesokurtic"
        elif coef_kurt > 0:
            full_txt, short_txt = "Leptokurtic. This distribution appears as a curve one with long tails.", "leptokurtic"
        else:
            full_txt, short_txt = "Platykurtic. These types of distributions have short tails.", "platykurtic"

        if print_info:
            print('Kurtosis: ', np.round(coef_kurt, 3))
            print(full_txt)
        if return_info:
            return np.round(coef_kurt, 2), short_txt

    def get_moving_dynamic(self, window_size=250, fig_size=(14, 10)):
        """
            Ф-я отображает среднее значение, стандартное отклонение, коэффициент асимметрии и коэффициент эксцесса в динамике.
            window_size - размер окна, на котором вычисляются значения.
        """
        mean_lst, std_lst, skew_lst, kurt_lst = [], [], [], []
        for i in range(window_size, self.T):
            ret = self.data[self.name][i-window_size : i]
            mean_lst.append(np.mean(ret))
            std_lst.append(np.std(ret))
            skew_lst.append(stats.skew(ret))
            kurt_lst.append(stats.kurtosis(ret))

        fig, axs = plt.subplots(4, 1, figsize = fig_size)
        axs[0].plot(self.data.index[window_size:], mean_lst, color='steelblue', linewidth=0.8)
        axs[0].set_ylabel('Mean')
        axs[1].plot(self.data.index[window_size:], std_lst, color='steelblue', linewidth=0.8)
        axs[1].set_ylabel('Standard deviation')
        axs[2].plot(self.data.index[window_size:], skew_lst, color='steelblue', linewidth=0.8)
        axs[2].axhline(y=0, color='indianred', linewidth=0.6)
        axs[2].set_ylabel('Skewness')
        axs[3].plot(self.data.index[window_size:], kurt_lst, color='steelblue', linewidth=0.8)
        axs[3].axhline(y=0, color='indianred', linewidth=0.6)
        axs[3].set_ylabel('Kurtosis')
        plt.show()

    def stationarity_adf(self, lev=0.05):
        adf_test = sm.tsa.adfuller(self.data[self.name])
        if (adf_test[1] < lev):
            res = 'the series is stationary.'
        else:
            res = 'the series is not stationary.'
        print(f'ADF test: {res}')

    def stationarity_kpss(self, lev=0.05):
        kpss_test = sm.tsa.kpss(self.data[self.name])
        if (kpss_test[1] < lev) :
            res = 'the series is not stationary.'
        else:
            res = 'the series is stationary.'
        print(f'KPSS test: {res}')

    def stationarity_test(self, lev=0.05):
        """
            Тест на стационарность.
            Ф-я выводит результаты ADF и KPSS тестов.
            lev -
        """
        print('Stationarity test')
        self.stationarity_adf(lev=lev)
        self.stationarity_kpss(lev=lev)

    def hill_estimator(self, m=25):
        """
            Оценка Хилла для последних m наблюдений.
            Выводится значение показателя для левого и правого хвоста.
        """
        print('Hill estimator')
        # left tail
        t1 = self.data[self.name].sort_values()[:m+1].values
        t1_m = t1[-1]
        left_tail_index = np.mean(np.log(t1[1:]/t1_m))
        print(f'Left tail index: {np.round(left_tail_index, 3)}')
        #right tail
        t2 = self.data[self.name].sort_values()[-m-1:].values
        t2_m = t2[-m-1]
        right_tail_index = np.mean(np.log(t2[1:]/t2_m))
        print(f'Right tail index: {np.round(right_tail_index, 3)}')

    def ljungbox_test(self, lags=[1, 5, 25, 50, 250]):
            """
                 Тест Ljung-Box на наличие автокорреляции.
                 Ф-я выводит DataFrame с результатми теста для различных лагов.
                 Если lags - целое число, результат теста сообщается для всех длин лага меньшего размера. 
                 Если lags - список, результат сообщается для лагов в списке.
            """
            lb_test = sm.stats.diagnostic.acorr_ljungbox(x=self.data[self.name], lags=lags, return_df=True)
            lb_test['comment'] = lb_test['lb_pvalue'].apply(
                lambda x: 'p_value > 0.05' if x > 0.05 else '')
            print('Ljung-Box test \n', lb_test)

    def goodness_of_fit_test(self, dist_list=None, test_type='ad', lev=0.05):
            """
                Тест Goodness-of-Fit на соотвествие фактичеких данных известному распределению.
                dist_list - список проверяемых распределений. По умолчанию ['norm', 'skew norm', 'GGD', 't', 'nct', 'GPD', 'GEV', 'GHYP']
                test_type может принимать одно из следующих значений:
                    'ad' - критерий Андерсона-Дарлинга,
                    'ks' - критерий Колмогорова-Смирнова,
                    'cvm' - критерий Крамера — Мизеса,
                    'filliben' - критерий Филлибина.
            """
            print('Goodness-of-Fit test')
            dist_list = list(self.dist_dict.keys()) if dist_list is None else dist_list
            for d in dist_list:
                if d in self.dist_dict:
                    try:
                        gf_test = goodness_of_fit(dist=self.dist_dict[d][0], data=self.data[self.name],
                                                  statistic=test_type)
                        print(f"{self.dist_dict[d][1]}: p_value = {gf_test[2]}")
                        if gf_test[2] > lev:
                            print("$H_{0} не отклоняется!$")
                        print("\n")
                    except:
                        continue

    def qq_plot(self, dist_list=None, fig_size=(4, 4)):
            """
                Ф-я отображает QQ-plot.
                dist_list - список проверяемых распределений. По умолчанию ['norm', 'skew norm', 'GGD', 't', 'nct', 'GPD', 'GEV', 'GHYP']
            """
            dist_list = list(self.dist_dict.keys()) if dist_list is None else dist_list
            for dist in dist_list:
                if dist in self.dist_dict:
                    fig, axs = plt.subplots(1, 1, figsize=fig_size)
                    # sm.qqplot(self.data[self.name], fit=True, line = 'q')
                    params = self.dist_dict[dist][0].fit(self.data[self.name])
                    stats.probplot(self.data[self.name], fit=True, dist=self.dist_dict[dist][0], sparams=params, plot=plt)
                    axs.set_title(f'Probability Plot: \n {self.dist_dict[dist][1]}', {'fontsize': 10})
                    plt.show()
                else:
                    print('Данного распределения нет в списке дступных.')

    def full_info(self,
                      bins=100, window_size=250, tail_m=25,
                      lags=[1, 5, 25, 50, 250],
                      dist_list=None):
            """
                Построение графиков цен и доходностей, гистограммы распределения значений доходностей (bins).
                Ф-я отображает среднее значение, стандартное отклонение, коэффициент асимметрии и коэффициент эксцесса в динамике (window_size).
                Оценка Хилла (m наблюдений в хвостах рассматривается.)
                Тест на стационарность.
                Тест Ljung-Box (lags).
                QQ-plot для распределений из dist_list. По умолчанию ['norm', 'skew norm', 'GGD', 't', 'nct', 'GPD', 'GEV', 'GHYP'].
            """
            self.price_dynamics()
            self.returns_dynamics()
            self.histogram(bins=bins)
            self.get_moving_dynamic(window_size=window_size)
            self.hill_estimator(m=tail_m)
            print('\n')
            self.stationarity_test()
            print('\n')
            self.ljungbox_test(lags=lags)
            print('\n')
            self.qq_plot(dist_list=dist_list)