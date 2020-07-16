#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime as dt
import functools as ft
import itertools as it
import numpy as np
import pandas as pd
import logging
import dask.dataframe as dd

import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict

import scipy as sp
import scipy.stats as sp_stats
import scipy.signal as sp_signal

from pmdarima.arima.stationarity import ADFTest
from pmdarima.arima.utils import ndiffs
import pmdarima.model_selection

import sklearn as skl
import sklearn.metrics as skl_metrics
import sklearn.model_selection as skl_model_select
import statsmodels.api as sm
import linearmodels as lm

# before could install fbprophet, had to manually resolve dependencies which clash in own requirements.txt
# pip install holidays==0.9.8
import fbprophet
import fbprophet.diagnostics as prophet_diagnostics

import tbats

from copy import copy

import _utils
# import eda
from backtest_results import RegionBacktestResults

_log = logging.getLogger()
_log.setLevel("INFO")

# date format 2015-12-27
df = _utils.kaggle_readzip("tseries - avocado-prices.zip", file_dir="data", parse_dates=["Date"])

_log.warning("DEV ONLY - SUB SAMPLE OF REGIONS")
dev_only_region_sample = [
    "Albany",
    # "Columbus",
    # "LosAngeles",
    # "NorthernNewEngland",
    # "PhoenixTucson",
    # "Roanoke",
    "TotalUS",
]
df = df.loc[df["region"].isin(dev_only_region_sample)]

_utils.clean_prep_data(df)


############################################################################################
# forecasting and prediction
############################################################################################
backtest_region = "TotalUS"  # "PhoenixTucson"
backtest_type = "organic"  # "conventional"
backtest_end = pd.to_datetime("2018-03-25")  # the start of last period in sample
backtest_num_periods = 52  # the entire last annual seasonal cycle

sample_weekly_frequency = "W-SUN"

df["backtest_period"] = 0  # initialise new column to indicate if row is in the backtest period
df.loc[df["Date"] > backtest_end - dt.timedelta(weeks=backtest_num_periods), "backtest_period"] = 1

g = sns.FacetGrid(df.loc[df["region"] == backtest_region], col="type", hue="backtest_period")
g.map(plt.scatter, "Date", "AveragePrice", alpha=0.7)
plt.show()

# downsample the weekly frequency data to monthly
sample_monthly_frequency = "M"
monthly_downsampled_mean = df.groupby(["region", "type"]).resample(sample_monthly_frequency, on="Date").mean()
monthly_downsampled_mean.reset_index(level=["region", "type"], inplace=True)
monthly_downsampled_mean['Date'] = monthly_downsampled_mean.index


def backtest_groupby_wrapper(_groupby_apply_func, inp_df, model_key, apply_kwargs):
    # TODO - could maybe be a decorator (need to check arg handling)
    _log.info(f"started backtest for model '{model_key}'")

    # groupby apply
    backtest_results = []
    for (region, avo_type), grp_df in inp_df.groupby(["region", "type"]):
        _log.info(f"started backtest for region {region}, type {avo_type}")
        this_backtest_res = _groupby_apply_func(region, avo_type, grp_df, model_key, **apply_kwargs)
        # exception handling, drop returned None
        if this_backtest_res:
            this_backtest_res.calculate_model_selection_criterion()
            backtest_results.append(this_backtest_res)

    return backtest_results

# daskifying the groupby apply
# ddf = dd.from_pandas(df, npartitions=4)
def backtest_dask_groupby_wrapper(_groupby_apply_func, inp_ddf, model_key, apply_kwargs):
    # dask-ified, currently NO DIFFERENCE in the syntax
    _log.info(f"started backtest for model '{model_key}'")

    # groupby apply
    backtest_results = []
    for (region, avo_type), grp_df in inp_ddf.groupby(["region", "type"]):
        _log.info(f"started backtest for region {region}, type {avo_type}")
        backtest_results.append(_groupby_apply_func(region, avo_type, grp_df, model_key, **apply_kwargs))

    return backtest_results


############################################################################################
# fb prophet
# wraps STAN bayesian forecasting language. as result can handle multiple seasonalities
############################################################################################
prophet_model_key = "weekly prophet, US holidays"
def prophet_groupby_apply(region, avo_type, grp_df, model_key, n_forecasts):
    prophet_temp_col_names = {"Date": "ds", "AveragePrice": "y", }
    grp_df.rename(columns=prophet_temp_col_names, inplace=True)

    # fits bayesian model per region
    # TODO - try with weekly_seasonality=True
    # https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html
    prophet_model = fbprophet.Prophet()
    prophet_model.add_country_holidays(country_name="US")

    # Prophet docs say this adds predictions row-by-row for dates that are in-sample prediction
    # so is equivalent to backtesting for other models
    # question if is totally equivalent because will have different lengths of training period supplied
    # i.e model has more information to fit
    prophet_model.fit(grp_df)
    # backtest_df = prophet_model.make_future_dataframe(periods=0, freq=sample_freq, include_history=True)
    # in_sample_forecast = prophet_model.predict(backtest_df)
    in_sample_forecast = prophet_model.predict().tail(n_forecasts)  # take last n_forecasts in_sample predictions

    # [_ for _ in dir(prophet_model) if not _.startswith("__")]
    # prophet_model.params

    # TODO - cross validate model training period, can then vary the horizon
    """
    prophet_diagnostics.cross_validation, determining optimal training period for rolling in-sample backtest
    Computes forecasts from historical cutoff points, which user can input.
    If not provided beginning from (end - horizon), works backwards making
    cutoffs with a spacing of period until initial is reached.
    """
    # df_cross_validated = prophet_diagnostics.cross_validation(
    #     prophet_model, horizon='1 W', period='1 M', initial="3 M",
    # )
    # TODO - interrogating and plotting the components: trend
    # prophet_model.plot_components(in_sample_forecast)  # trend, yearly, weekly

    # TODO - check what these metrics are like
    # test_metrics = prophet_diagnostics.performance_metrics(in_sample_forecast)
    # nice blog on different params in search
    # https://towardsdatascience.com/implementing-facebook-prophet-efficiently-c241305405a3

    merged = in_sample_forecast.merge(grp_df, how="left", on="ds", sort=False)
    n_params = np.nan
    # n_params = prophet_model.params  # TODO - extract/count number of nonzero params

    return RegionBacktestResults(model_key,
                                 region,
                                 avo_type,
                                 dates=merged["ds"],
                                 # TODO - perform cross validation to get optimal train length
                                 forecast_horizon=1,
                                 y=merged["y"],
                                 yhat=merged["yhat"],
                                 # TODO - log likelihood and info criteria
                                 # https://github.com/facebook/prophet/issues/549#issuecomment-435482584
                                 # calculating model log-likelihood
                                 n_params=n_params,
                                 others=prophet_model,
                                 )

######################################################################################
# linear regression - panel data, so use PanelOLS
######################################################################################
def backtest_panel_linear_weekly_fixed_window(inp_df, train_num_periods, forecast_horizon, sample_freq=sample_weekly_frequency):
    """
    n_prds - number of days that we want to get prediction and calculate metrics

    choices:
    1. Minimum Number of Observations. First, we must select the minimum number of observations required to train the
    model. This may be thought of as the window width if a sliding window is used (see next point).
    2. Sliding or Expanding Window. Next, we need to decide whether the model will be trained on all data it has
    available or only on the most recent observations.
    This determines whether a sliding or expanding window will be used.
    """
    model_key = f"panel OLS, weekly, forecast horizon {forecast_horizon} and train len {train_num_periods}"

    # multi-index on region and Date for PanelOLS,
    mi_inp_df = inp_df.set_index(["region", "Date"]).sort_index()
    dependent_var = "AveragePrice"

    # TODO - add lagged AveragePrice? but its then basically AR(1). would like some more exog vbls!
    # exogenous variable, cannot use "region" because already in index and PanelOLS uses as a grouping
    exog = ["type", ]  # 'Small Bags', 'Large Bags', 'XLarge Bags',
    mi_inp_df = sm.add_constant(mi_inp_df)
    exog += ["const", ]

    predictions = []
    backtest_eval_range = inp_df.loc[(inp_df["region"] == "TotalUS") &
                                     (inp_df["type"] == "organic") &
                                     (inp_df["backtest_period"] == 1), "Date"]

    n_params = 2  # 2 model params, on type==organic and a constant

    # TODO - tseries cross validation, basically a growing index set for training set rows
    # skl_model_select.TimeSeriesSplit(n_splits=)

    yhat = defaultdict(list)  # region, type = []

    # rolling backtest per period, add the predictions for each region to a dict
    for test_date in backtest_eval_range:

        # rolling backtest slice the train and test sets
        train = mi_inp_df.loc[(slice(None), pd.date_range(test_date - dt.timedelta(weeks=train_num_periods),
                                                          periods=train_num_periods, freq=sample_freq)), :]
        test = mi_inp_df.loc[(slice(None), pd.date_range(test_date,
                                                         # periods=forecast_horizon, freq=sample_freq)), :]
                                                         periods=1, freq=sample_freq)), :].copy()

        # see linearmodels docs for arg definition https://bashtage.github.io/linearmodels/doc/panel/models.html
        mod = lm.PanelOLS(train[dependent_var], train[exog], entity_effects=False, time_effects=False)
        res = mod.fit(use_lsdv=False, cov_type="heteroskedastic", auto_df=True, count_effects=True)
        test["yhat"] = res.predict(test[exog])["predictions"]

        # unpacking the predictions into dict so whole backtest prediction array can be added to RegionBacktestResults
        for (region, avo_type), regional_predictions in test.groupby(["region", "type"]):
            yhat[(region, avo_type)].append(regional_predictions["yhat"].values[0])

    for (region, avo_type), preds in yhat.items():
        region_type_results = RegionBacktestResults(model_key,
                                                    region=region,
                                                    avo_type=avo_type,
                                                    dates=backtest_eval_range,
                                                    forecast_horizon=forecast_horizon,
                                                    y=inp_df.loc[(inp_df["region"] == "TotalUS") &
                                                                 (inp_df["type"] == "organic") &
                                                                 (inp_df["backtest_period"] == 1), "AveragePrice"].values,
                                                    yhat=preds,
                                                    n_params=n_params,
                                                    )
        region_type_results.calculate_model_selection_criterion()
        predictions.append(region_type_results)
    return predictions


######################################################################################
# ARIMA
# hardcoded order across the regions (reasonable to assume follow same process)
######################################################################################
# optimising with below just considers all possible combos of P, D, Q spec of ARIMA
# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/
"""
# https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method
1) Model identification and model selection: 
making sure that the variables are stationary, identifying seasonality in the dependent series 
(seasonally differencing it if necessary), and using plots of the
autocorrelation (ACF) and partial autocorrelation (PACF) functions of the dependent time series 
to decide which (if any) autoregressive or moving average component should be used in the model.

2) Parameter estimation using computation algorithms to arrive at coefficients that best fit the selected ARIMA model. 
The most common methods use maximum likelihood estimation or non-linear least-squares estimation.

3) Statistical model checking by testing whether the estimated model conforms to the specifications of a stationary 
univariate process. In particular, the residuals should be independent of each other and 
constant in mean and variance over time. (Plotting the mean and variance of residuals over time 
and performing a Ljung–Box test or plotting autocorrelation and partial autocorrelation of the residuals 
are helpful to identify misspecification.) 
If the estimation is inadequate, we have to return to step one and attempt to build a better model.


evaluating the ACF and PACFs
what is "longest number of periods" that "make sense" as a process predictor?
    in weekly data, 52 would be massive...
    in monthly data


ACF notes from Janert Ch4
ACF = "memory" the series
if correlation drops quickly then signal quickly loses memory of recent past
if drops off slowly then the process is relatively stead over longer periods of time

ACF drops off and then rises again: there is periodicity

- ACF intended for tseries with no trend and zero mean!!
2 ways of removing a trend: 
subtract the trend: fit an exponential, linear or other such trend. or apply smoothing and subtract 
or difference the series

*warning comparing results from different sources or different software packages
there are different ways of normalising the numerator and denominator of formula 

"""
# BOX JENKINS
# assess number of lags and suitability of ARIMA. wouldn't expect a seasonal component bc weekly sampling frequency
# arima_dev_series = df.loc[(df["region"] == backtest_region) & (df["type"] == backtest_type), ["Date", "AveragePrice"]]
# sns.lineplot(x="Date", y="AveragePrice", data=arima_dev_series).set_title(f"Average Price for region {backtest_region} and type {backtest_type}")
# plt.show()

"""naive plot shows:
- not stationary, variance and levels change over time
- seasonality with annual period. makes sense: annual supply and growing period.
- trend: no noticeable obvious trend present, maybe small upward trend over period. upward would make sense: inflation.
"""

# because of long period to cycle, and that weeks dont match up 1-to-1 with years
# first_differenced = np.log(arima_dev_series["AveragePrice"]).diff(1)
# # trim off empty data after differencing
# first_differenced = first_differenced[1:]
# sns.lineplot(x="Date", y="AveragePrice", data=first_differenced.reset_index())
#
# # seem like annual cycles, annual seasonal difference
# seasonal_differenced = arima_dev_series.diff(52)
# # trim off the first year of empty data
# seasonal_differenced = seasonal_differenced[52:]
# seasonal_differenced.drop("Date", axis=1, inplace=True)
# sns.lineplot(x="Date", y="AveragePrice", data=seasonal_differenced.reset_index())

# for series, nlags, title in [(arima_dev_series, 60, "original series"),
#                              (monthly_downsampled_mean, 30, "monthly downsampled series"),
#                              (first_differenced, 60, "first diff weekly"),
#                              (seasonal_differenced, 60, "seasonal diff weekly"),
#                              # (, ),
#                              ]:
#     _utils.plot_series_acf(x=range(len(arima_dev_series)), y=series["AveragePrice"], lags=nlags, title=title+f" acf, {nlags}")
#     _utils.plot_series_pacf(x=range(len(arima_dev_series)), y=series["AveragePrice"], lags=nlags, title=title+f" pacf, {nlags}")

# for region backetst_region = TotalUS:
# monthly suggests AR(2) with some MA components
# weekly pacf suggests AR(2) as well

# using pmdarima package test methods to decide if should difference
# Test whether we should difference at the alpha=0.05
# significance level
# adf_test = ADFTest(alpha=0.05)
# p_val, should_diff = adf_test.should_diff(arima_dev_series["AveragePrice"].values)
#
# # Estimate the number of differences using an ADF test:
# n_adf = ndiffs(arima_dev_series["AveragePrice"].values, test='adf')
# # Or a KPSS test (auto_arima default):
# n_kpss = ndiffs(arima_dev_series["AveragePrice"].values, test='kpss')
# # Or a PP test:
# n_pp = ndiffs(arima_dev_series["AveragePrice"].values, test='pp')
#
# # de-trending with scipy signal
# arima_dev_series["linear_detrended_price"] = sp_signal.detrend(arima_dev_series["AveragePrice"])
# sns.lineplot(x="Date", y="linear_detrended_price", data=arima_dev_series)
#
# # de-trending with first difference
# arima_dev_series.set_index("Date", inplace=True)
# fd_arima_dev_series = arima_dev_series.diff()
# fd_arima_dev_series = fd_arima_dev_series[1:]  # drop the first NULL obsv
#
# # checking first diff has no trend
# sns.lineplot(x="Date", y="AveragePrice", data=fd_arima_dev_series.reset_index())
# plt.show()
# # checking has 0 mean
# arima_dev_series.mean()
#
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(fd_arima_dev_series["AveragePrice"], lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(fd_arima_dev_series["AveragePrice"], lags=40, ax=ax2)
# plt.show()


# evaluate an ARIMA model for a given order (p,d,q)
arima_static_order_model_key = ("ARIMA, weekly, hardcoded order, order {arima_order}, forecast horizon "
                                "{forecast_horizon} and train len {train_num_periods}")
def arima_static_order_groupby_apply(region, avo_type, grp_df, model_key,
                                     arima_order, train_num_periods,
                                     forecast_horizon, sample_freq, ):
    # adapted from https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

    # fitting new model per region and per avocado type
    backtest_eval_range = grp_df.loc[grp_df["backtest_period"] == 1, "Date"]
    try:
        yhat = []
        # incremental model fits and predicts "as-of time n"
        for test_date in backtest_eval_range:
            # rolling backtest, slice the train and test sets
            train = grp_df.loc[grp_df["Date"].isin(pd.date_range(test_date - dt.timedelta(weeks=train_num_periods),
                                                                 periods=train_num_periods, freq=sample_freq)), :]

            model = sm.tsa.ARIMA(train[["AveragePrice"]], order=arima_order, freq=sample_freq)
            model_fit = model.fit(disp=False)

            # tuple format: forecast, fcerr, conf_int
            nstep_ahead = model_fit.forecast(steps=1)[0][0]
            yhat.append(nstep_ahead)

        return RegionBacktestResults(model_key.format(**{"arima_order": arima_order,
                                                         "forecast_horizon": forecast_horizon,
                                                         "train_num_periods": train_num_periods}),
                                     region=region,
                                     avo_type=avo_type,
                                     dates=backtest_eval_range,
                                     forecast_horizon=forecast_horizon,
                                     y=grp_df.loc[grp_df["backtest_period"] == 1, "AveragePrice"].values,
                                     yhat=yhat,
                                     n_params=sum(arima_order),
                                     )
    except ValueError as e:
        if "coefficients are not stationary" in str(e):
            _log.warning(f"coefficients are not stationary for order {arima_order}, region {region}, avo_type {avo_type}")
            return None

        else:
            raise e

    # except HessianInversionWarning as e:
    except np.linalg.LinAlgError as e:
        if "SVD did not converge" in str(e):
            _log.warning(f"SVD did not converge for order {arima_order}, region {region}, avo_type {avo_type}")
            return None
        else:
            raise e

# Plot residual errors to ensure they're white noise
# if residuals are correlated then likely some additional information left on the table
# residuals = pd.DataFrame(model_fit.resid)
# fig, ax = plt.subplots(1,2)
# residuals.plot(title="Residuals", ax=ax[0])
# residuals.plot(kind='kde', title='Density', ax=ax[1])
# plt.show()



######################################################################################
# TBATS - "Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components" - TODO
######################################################################################
# https://medium.com/intive-developers/forecasting-time-series-with-multiple-seasonalities-using-tbats-in-python-398a00ac0e8a
# https://github.com/intive-DataScience/tbats
# TODO - formally assess the length of periods

tbats_model_key = ("weekly tbats, "
                   "forecast horizon {forecast_horizon}, train len {train_num_periods} and "
                   "seasonal_periods {seasonal_periods}")
def tbats_groupby_apply(region, avo_type, grp_df, model_key, train_num_periods, forecast_horizon,
                        sample_freq,
                        seasonal_periods,
                        n_jobs, ):
    # fitting new model per region and per avocado type
    backtest_eval_range = grp_df.loc[grp_df["backtest_period"] == 1, "Date"]

    yhat = []
    model_params = len(seasonal_periods) + 4  # TODO - improve extraction of model params model.params.summary()
    # incremental model fits and predicts "as-of time n"
    for test_date in backtest_eval_range:
        # rolling backtest, slice the train and test sets
        train = grp_df.loc[grp_df["Date"].isin(pd.date_range(test_date - dt.timedelta(weeks=train_num_periods),
                                                             periods=train_num_periods, freq=sample_freq)), :]
        # Fit the model
        model = tbats.TBATS(seasonal_periods=seasonal_periods,
                            n_jobs=n_jobs,)  # monthly, quarterly and annual periodicity
        model = model.fit(train["AveragePrice"])
        nstep_ahead = model.forecast(steps=1)
        yhat.append(nstep_ahead)

    return RegionBacktestResults(model_key,
                                 region,
                                 avo_type,
                                 dates=backtest_eval_range,
                                 forecast_horizon=forecast_horizon,
                                 y=grp_df.loc[grp_df["backtest_period"] == 1, "AveragePrice"].values,
                                 yhat=yhat,
                                 n_params=model_params,
                                 )


######################################################################################
# seasonal arima - TODO
######################################################################################
# Box-Jenkins Seasonal ARIMA modelling: seasonal differencing, need to specify period of seasonality
# order = (1, 0, 0)
# seasonal_order = (1, 0, 0, 52)  # P, D, Q, s where s = length of seasonal period to difference
# mod = sm.tsa.statespace.SARIMAX(arima_dev_series["AveragePrice"],
#                                 order=order,
#                                 seasonal_order=seasonal_order,
#                                 enforce_stationarity=False,
#                                 enforce_invertibility=False,
#                                 )
# results = mod.fit()
# results.aic  # trying to minimise for given combinations of seasonal differencing
# results.plot_diagnostics(figsize=(18, 8))
# plt.show()




######################################################################################
# autoarima
######################################################################################
auto_arima_model_key = ("ARIMA, weekly, auto optimised order, "
                        "forecast horizon {forecast_horizon}, train len {train_num_periods} and "
                        "seasonal_differencing_period {seasonal_differencing_period} ")
def arima_auto_groupby_apply(region, avo_type, grp_df,
                             model_key,
                             train_num_periods,
                             forecast_horizon,
                             seasonal_differencing_period,
                             sample_freq,
                             ):
    # fitting new model per region and per avocado type
    backtest_eval_range = grp_df.loc[grp_df["backtest_period"] == 1, "Date"]

    yhat = []
    model_orders = []
    for test_date in backtest_eval_range:
        # rolling backtest, slice the train and test sets
        train = grp_df.loc[grp_df["Date"].isin(pd.date_range(test_date - dt.timedelta(weeks=train_num_periods),
                                                             periods=train_num_periods, freq=sample_freq)), :]

        # TODO - can perform seasonal differencing, this param "counts differently" towards Degrees of Freedom
        # e.g. ARIMA(order=(0, 1, 1), seasonal_order=(0, 0, 0, 52))
        model = pmdarima.auto_arima(train[["AveragePrice"]], m=52)
        yhat.append(model.predict(1)[0])

        # checking that each auto-arima model has same (p, d, q) order for whole backtest for each region.
        # affects the model aggregated BIC, comparing apples with apples (a single specified model)
        model_orders.append(model.order)

    unique_model_orders = np.unique(model_orders)
    _log.warning(f"multiple model orders found over backtest: {unique_model_orders}")

    return RegionBacktestResults(model_key.format(**{"forecast_horizon": forecast_horizon,
                                                     "train_num_periods": train_num_periods,
                                                     "seasonal_differencing_period": seasonal_differencing_period,
                                                     }),
                                 region=region,
                                 avo_type=avo_type,
                                 dates=backtest_eval_range,
                                 forecast_horizon=forecast_horizon,
                                 y=grp_df.loc[grp_df["backtest_period"] == 1, "AveragePrice"].values,
                                 yhat=yhat,
                                 n_params=np.mean([sum(_) for _ in model_orders]),
                                 )


######################################################################################
# model fitting and forecasting - just producing the different yhats
######################################################################################
# weekly arima, different (p, d, q) specifications
# https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/The_Box-Jenkins_Method.pdf
# advised to try (1, d, 0), (2, d, 1), (4, 3),
# altho unsure for what frequency
weekly_arima_results = []
for arima_order in [
    (1, 0, 0),
    (1, 1, 1),
    (2, 0, 1),
    (4, 0, 3),
]:
    weekly_arima_results.extend(backtest_groupby_wrapper(arima_static_order_groupby_apply,
                                                         df,
                                                         arima_static_order_model_key,
                                                         {"arima_order": arima_order,
                                                          "forecast_horizon": 1,
                                                          "train_num_periods": 104,
                                                          "sample_freq": sample_weekly_frequency,
                                                          },
                                                         ))


# monthly arima
# fitting box-jenkins SARIMA model - doesnt fit well to weekly because very large number of lags (13 or 52) to capture
# a seasonal cycle
# https://otexts.com/fpp2/weekly.html
# Weekly data is difficult to work with because the seasonal period
# (the number of weeks in a year, 52.18) is both large and non-integer.
arima_monthly_static_order_model_key = ("ARIMA, monthly, hardcoded order, order {arima_order}, forecast horizon "
                                        "{forecast_horizon} and train len {train_num_periods}")
monthly_arima_results = backtest_groupby_wrapper(arima_static_order_groupby_apply,
                                                 monthly_downsampled_mean,
                                                 arima_monthly_static_order_model_key,
                                                 {"arima_order": (1, 0, 0),
                                                  "forecast_horizon": 1,
                                                  "train_num_periods": 24,
                                                  "sample_freq": sample_monthly_frequency,
                                                  },
                                                 )

# auto_arima_results = backtest_groupby_wrapper(arima_auto_groupby_apply,
#                                               df,
#                                               auto_arima_model_key,
#                                               {"forecast_horizon": 1,
#                                                "train_num_periods": 104,
#                                                "seasonal_differencing_period": 52,
#                                                "sample_freq": sample_weekly_frequency,
#                                                },
#                                               )

# weekly prophet
prophet_results = backtest_groupby_wrapper(prophet_groupby_apply,
                                           df,
                                           prophet_model_key,
                                           {"n_forecasts": backtest_num_periods, },
                                           )

# panel OLS
panel_linear_results = backtest_panel_linear_weekly_fixed_window(df, train_num_periods=104, forecast_horizon=1)

# tbats - taking AGES
tbats_results = backtest_groupby_wrapper(tbats_groupby_apply,
                                         df,
                                         tbats_model_key,
                                         {"forecast_horizon": 1,
                                          "train_num_periods": 104,
                                          "sample_freq": sample_weekly_frequency,
                                          "seasonal_periods": (4.3333, 13, 52.178),
                                          "n_jobs": 1,
                                          },
                                         )


######################################################################################
# model forecast plotting
######################################################################################
def plot_model_predictions_vs_actuals(backtests_list, model_name, region, avo_type):
    # TODO - including training period observations for prior context?

    # plotting predicted yhat versus actual y in backtest period
    # https://seaborn.pydata.org/generated/seaborn.lineplot.html#seaborn.lineplot
    records = [{"date": _.dates, "actual": _.y, "predicted": _.yhat} for _ in backtests_list if _.region == region and _.avo_type == avo_type]
    plot_data = pd.DataFrame(records).set_index("date")
    plot = sns.lineplot(data=plot_data)
    plt.title(f"Actuals vs Predicted for model {model_name}, region {region}, and avo type {avo_type}", )
    # plt.legend(loc="lower right")
    plt.xticks(rotation=45, )
    plt.show()

# plot_model_predictions_vs_actuals(weekly_arima_results, "weekly arima", "TotalUS", "conventional")
# plot_model_predictions_vs_actuals(tbats_results, "weekly tbats", "TotalUS", "conventional")
plot_model_predictions_vs_actuals(monthly_arima_results, "monthly arima", "TotalUS", "conventional")
# plot_model_predictions_vs_actuals(auto_arima_results, "auto arima", "TotalUS", "conventional")
# plot_model_predictions_vs_actuals(panel_linear_results, "Panel Linear", "LosAngeles", "conventional")
# plot_model_predictions_vs_actuals(panel_linear_results, "Panel Linear", "TotalUS", "organic")
# plot_model_predictions_vs_actuals(prophet_results, "Prophet", "TotalUS", "conventional")


######################################################################################
# model forecast evaluation
######################################################################################
"""
answering question of "best" model
    - per forecast horizon
    (if forecasting 1-step ahead or 4+ periods. 
    Implies that weight each forecasted period, e.g. forecast for n+1 and n+4 weighted same in eval.
    
    - per region
    (if one model particularly good across regions)
    
    - per type (organic, conventional)
    
    - per series aggregation: weekly versus monthly

"best" is smallest info criteria across other categories
mean, median, smallest variance.
min/max other dimension (e.g. if consider organic, then Tampa worst predicted region, Pittsburgh best) 
t-test (paired, unequal variance) that is significantly the better performing model
https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm

pivot eval results into a DF? or tabular format. so can sort and groupby models per region or type

# todo - examine residuals? if white noise (good) or conains a systematic component (bad, could use to predict)
statsmodels results have plot_diagnostics, which show graphical output of the residuals. Eg if are white noise and fall on Q-Q

**RISK** - because of barrage of different models, risk that "one of them" captures the best
similar to holm-bonferroni where just submit a barrage of tests and hypotheses, one of them bound to be statistically significant
https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method 

"""

# TODO - see if can use this, looks useful
# model_selection.RollingForecastCV()

# flatten the model results
all_model_results = [backtest_res for model_res in
                     [weekly_arima_results,
                      monthly_arima_results,
                      # auto_arima_results,
                      panel_linear_results,
                      prophet_results,
                      # tbats_results,
                      ]
                     for backtest_res in model_res
                     ]

# sort models by their RMSE, (AIC ?)
# just pivoting all regional models
eval_df = pd.DataFrame({
    "model_name": [_.model_name for _ in all_model_results],
    "region": [_.region for _ in all_model_results],
    "avo_type": [_.avo_type for _ in all_model_results],
    "rmse": [_.rmse for _ in all_model_results],
    "aic": [_.aic for _ in all_model_results],
    "bic": [_.bic for _ in all_model_results],
    "hqic": [_.hqic for _ in all_model_results],
})

eval_df.sort_values(by=["region", "avo_type", "rmse", ], inplace=True)


# eval_df.groupby("region").mean() # eval_df.groupby("model_name").mean()
eval_df.to_csv("data/eval_run_out.csv", index=False)

_log.info("completed run!")




