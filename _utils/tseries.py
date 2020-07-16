#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns



########################################################################
# generating random series
########################################################################
def pure_random_walk(num_periods):
    out_process = [np.random.randn(), ]  # custom to start at 0, so X(1) = 0 + Z(1)
    for each_period in range(num_periods):
        this_z = np.random.randn()
        # Xt = Xt-1 + Zt
        out_process.append(out_process[-1] + this_z)
    return out_process


def pure_ma_process(betas, num_periods, start_val=0, mean=0):
    current_zs = [start_val, ]
    out_process = []
    for each_period in range(num_periods):
        this_z = [np.random.randn(), ]
        if len(current_zs) < len(betas):
            current_zs += this_z

        else:
            current_zs = current_zs[1:] + this_z

        # Xt = β0Zt + β1Zt-1 + ... + βqZt-q
        out_process.append(sum([beta * zt for beta, zt in zip(betas, current_zs)]) + mean)
    return out_process


def pure_ar_process(alphas, start_val, num_periods, mean=0):
    out_process = [start_val, ]
    for each_period in range(num_periods):
        this_z = np.random.randn()
        out_process.append(sum([alpha * xt for alpha, xt in zip(alphas, out_process[-len(alphas):])]) + this_z)
    return np.array(out_process) + mean


########################################################################
# plotting and showing
########################################################################
def plot_series(x, y, title="", **kwargs):
    sns.lineplot(x=x, y=y, **kwargs)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()


def plot_series_acf(x, y, lags, title="Autocorrelation Function to {} lags"):
    plot_acf(y, lags=lags)
    plt.title(title.format(lags))
    plt.show()


def plot_series_pacf(x, y, lags, title="Partial Autocorrelation Function to {} lags"):
    plot_pacf(y, lags=lags)
    plt.title(title.format(lags))
    plt.show()


def plot_series_acf_pacf():
    # combine both on a plot
    raise NotImplementedError()
