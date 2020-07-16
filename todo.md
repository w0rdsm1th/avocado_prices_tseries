# todo
3) auto arima
4) tbats
5) plotting predictions vs actual on line plot per model
multiprocessing for fit on gaming PC or cloud?

6) moving the backtest_eval_range into the top level groupby so can pass standardised range to all model functions
7) n-step ahead forecasting, e.g. not just forecasting for next period but forecasting for next 3 periods and then evaluate

conda update -n base -c defaults conda



##############################################
# motivation

project using kaggle's US Avocado prices dataset
https://www.kaggle.com/neuromusic/avocado-prices

Objectives:
    - learning about different t-series algorithms
    - comparison of different forecasting methods and how they fit this data in rolling backtests
    - plan to transplant findings into a notebook and host on kaggle

RISK: number of tests conducted erodes their power. so many different regions and time periods tested
helped by broad number of regions and long backtest period (52 weeks)

- intro and motivation for analysis
https://business.financialpost.com/financial-times/avocado-crime-soars-ahead-of-super-bowl-as-mexican-gangs-hijack-truckloads-of-green-gold-heading-north
TSeries tradition of running a barage of different predictive methods and seeing which do best with rolling benchmarks.
e.g. Makridakis competitions, https://mofc.unic.ac.cy/the-dataset/

using barage of different models to see which performs best
https://link.springer.com/content/pdf/10.1007/s10618-016-0483-9.pdf


- New Qs:
    how do quantity sold and average price covary over time?
    how does the type impact

- standardised backtest criteria: e.g. ensuring all models share same data (not trained on different windows), and pre-agreed criteria
https://stats.stackexchange.com/questions/219747/what-selection-criteria-to-use-and-why-aic-rmse-mape-all-possible-model-s
RMSE: no penalty from more terms, prone to overfit
https://machinelearningmastery.com/probabilistic-model-selection-measures/

time series cross-validation as here
https://robjhyndman.com/hyndsight/tscv/


- comparison of different price forecasting methods
in-sample backtesting metric: MSE over next month, per city
    exponential-smoothing AKA Holt-Winters
        play around with: how choose initial value, "backcasting" as suggested here https://online.stat.psu.edu/stat501/node/1001
    exponential smoothing
    basically predictions are weighted combination of previous obsvs
        single exponential smoothing: works for tseries with no overall trend or seasonality
        double exponential smoothing: for tseries with trend BUT no seasonality
        triple exponential smoothing (holt-winters m,ethod): for tseries with trend AND seasonality (either additive or multiplicative seasonality)
*case for trend (gradual inflation) and multiplicative seasonality as the seasonal variation increases over time)

    arima, and box-jenkins, some competing libraries:
        statsmodels
        pmdarima, wraps R’s beloved auto.arima

        questions:
            1) see if Box-Jenkins transformation improves the forecasting performance (AIC,...)
            Granger & Newbold (1986) found that it didnt

            2) if seasonal differencing
            nice Box-Jenkins step-by-step
            https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/The_Box-Jenkins_Method.pdf
    
    prophet: is bayesian, uses stan under covers
        # https://facebook.github.io/prophet/docs/quick_start.html#python-api
        # https://statmodeling.stat.columbia.edu/2017/03/01/facebooks-prophet-uses-stan/

    pystan - https://pystan.readthedocs.io/en/latest/
        https://pystan.readthedocs.io/en/latest/getting_started.html
        requires JSON-like model specification as string...

    panel OLS (multi-variable, what are the exogenous variables? lagged values of quantities sold?)
    compare performance with time_effects=True (autocorrelation or heteroskedasticity thru time?)

    error correction model? GLS

 todo
    State Space (or Hidden Markov) Models
    State-space models: Dynamic linear models and the Kalman filter.
    e.g. stochastic alpha and beta in
    Xt = α + βt + εt
    TBATS - was recommended by Hyndman in his post
    https://medium.com/intive-developers/forecasting-time-series-with-multiple-seasonalities-using-tbats-in-python-398a00ac0e8a
    https://github.com/intive-DataScience/tbats

# Hyndman suggests fourier transforming the process
# https://robjhyndman.com/hyndsight/forecasting-weekly-data/
# "So ARIMA and ETS models do not tend to give good results, even with a period of 52 as an approximation."
# discussion about arbitrary Fourier here https://stats.stackexchange.com/q/122270
# irishStat criticism: You are tacitly ignoring level shifts, local time trends and one-time unusual values.


list of other algos
http://www.timeseriesclassification.com/algorithm.php

- compare keeping it weekly (which harms the seasonality adjustment according to Rob Hyndman because its not a
straightforward comparison to last year, e.g. -12months)
https://robjhyndman.com/hyndsight/forecasting-weekly-data/
https://otexts.com/fpp2/weekly.html

->> Seasonality in Weekly Forecasting, from https://www.forecastsolutions.co.uk/forecasting-seasonality.htm
Seasonal analysis of weekly data is often more difficult than with monthly data.  It becomes less likely that annual events will take place in the same calendar period, so may necessitate cleansing those instances, such as bank holidays, from the sales history and adding future instances to the forecast as planned events.

The volatility of weekly information is inevitably greater than with months.  The result is often that a set of weekly seasonal indices may display a ragged effect due to the increased volatility.  A number of approaches are possible to smooth out the weekly indices, such as the following:
- group seasonal indices (calculate the seasonal pattern at a group level, then assign the indices to each member of the group)
- seasonal simplification (aggregate the history to 4 wk periods, calculate the 4 wk indices and assign to each week of the period)
- seasonal smoothing (apply a 5 week centred moving average to the seasonal indices to smooth them out)
- use sophisticated forecasting methods such as TBATS that incorporate fourier analysis in the seasonal calculation (using trigonometric sine and cosine waves)


- compare OLS region coefficients to find the "strongest" de/increase in prices over the period
R**2, slope and intercept

- are these real prices? adjust by CPI, and then grocery food index?

- suspected data quality issues
1) price for region == "TotalUS" and type == "organic" crashing to 1 for 6 weeks near start of measurement period
Date	AveragePrice
21/06/2015	1.66
28/06/2015	1.64
05/07/2015	1
12/07/2015	1
19/07/2015	1
26/07/2015	1
02/08/2015	1
09/08/2015	1
16/08/2015	1.75
23/08/2015	1.72
30/08/2015	1.66

2) some missing periods

convert to a kaggle kernel
    - markdown
    - ipython
    - nice graphics



