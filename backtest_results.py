#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""

import numpy as np
import sklearn.metrics as skl_metrics

# model descriptor: algorithm, frequency, seasonality, params, train period

class ModelBacktestResults:
    """
    for the whole backtest period
    per model, what are the aggregated evaluation statistics

    for the combination_areas (e.g. TotalUS) vs cities (e.g. PhoenixTuscon)
    """
    pass


class RegionBacktestResults:
    """
    calculating in-sample error in backtest period, but for range of statistics
    standardising the model backtest output, so can standardise model fit evaluation
    """
    def __init__(self, model_name, region, avo_type, dates, forecast_horizon, y, yhat, n_params,
                 others=None, rmse=None, mape=None, aic=None, bic=None, hqic=None, ):
        self.model_name = model_name  # e.g. ARIMA(p, d, q)
        self.region = region
        self.avo_type = avo_type
        self.dates = dates
        self.forecast_horizon = forecast_horizon

        self.yhat = yhat
        self.y = y

        self.n_params = n_params
        self.others = others  # others: prophet_model object itself so can extract components

        # many models offer these out of box, so just use them!
        self.rmse = rmse
        self.mape = mape
        self.aic = aic
        self.bic = bic
        self.hqic = hqic

    def __repr__(self):
        return f"model '{self.model_name}', region '{self.region}', avo_type '{self.avo_type}'"

    def calculate_model_selection_criterion(self):
        # convenience wrapper to calculate the other model evaluation methods
        mse = self.calculate_mse(self.yhat, self.y)
        self.rmse = mse**0.5
        self.aic = self.calculate_aic(len(self.y), mse, self.n_params)
        self.bic = self.calculate_bic(len(self.y), mse, self.n_params)

    @staticmethod
    def calculate_mse(predicted, actuals):
        """
        calculate the mean square error of
        :return:
        """
        # mse = predicted.sub(actual).mean()**2
        # mse = inp_df["yhat"].sub(inp_df["y"]).mean()**2
        return skl_metrics.mean_squared_error(actuals, predicted)


    @staticmethod
    def calculate_aic(n, mse, num_params):
        """
        calculate aic for regression, borrowed from machinelearningmastery.com/
        example:
        num_params = len(model.coef_) + 1
        aic = calculate_aic(len(y), mse, num_params)
        """
        aic = n * np.log(mse) + 2 * num_params
        return aic


    @staticmethod
    def calculate_bic(n, mse, num_params):
        """
        calculate bic for regression, borrowed from machinelearningmastery.com/
        example:
        num_params = len(model.coef_) + 1
        aic = calculate_bic(len(y), mse, num_params)
        """
        bic = n * np.log(mse) + num_params * np.log(n)
        return bic

