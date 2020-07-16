

## STATS AND T-SERIES
- sklearn tseries cross validation, basically a growing index set for training set rows
skl_model_select.TimeSeriesSplit(n_splits=)

- TBATS - allows for multiple seasonal periods, high frequency seasonality, non-integer seasonality and dual-calendar effects. 
https://robjhyndman.com/publications/complex-seasonality/
BATS: restricts each seasonality to deterministic process st = st-m + γdt
**so BATS can only model integer periodicity**

- open question: how to handle and record n-params for models that auto-select/optimise parameters during a backtest 

## PYTHON AND LIB METHODS1
# plotting

Lesson learned: slicker plotting and easier colouring by categorical to supply whole DF to lmplot
sns.scatterplot(df.loc[df["region"] == "Albany", "Date"], df.loc[df["region"] == "Albany", "AveragePrice"])

# Lesson learned: multi-index slicing with dates
# mi.loc[(slice("Albany"), slice("conventional"), slice(None), ), :]  # dont work on multi-index, resorting to query
# mi.loc[(slice("Albany"), slice("organic"), slice(None), ), :]
# mi.query('region == "Albany" & type == "organic"')


# attrs on fitted ARIMA returned
model = sm.tsa.ARIMA(train[["AveragePrice"]], order=arima_order, freq=sample_freq)
[_ for _ in dir(model_fit) if not _.startswith("__")]
['_cache',
 '_data_attr',
 '_forecast_conf_int',
 '_forecast_error',
 '_get_robustcov_results',
 '_ic_df_model',
 '_use_t',
 'aic',
 'arfreq',
 'arparams',
 'arroots',
 'bic',
 'bse',
 'conf_int',
 'cov_params',
 'cov_params_default',
 'data',
 'df_model',
 'df_resid',
 'f_test',
 'fittedvalues',
 'forecast',
 'hqic',
 'initialize',
 'k_ar',
 'k_constant',
 'k_exog',
 'k_ma',
 'k_trend',
 'llf',
 'load',
 'mafreq',
 'maparams',
 'maroots',
 'mle_retvals',
 'mle_settings',
 'model',
 'n_totobs',
 'nobs',
 'normalized_cov_params',
 'params',
 'plot_predict',
 'predict',
 'pvalues',
 'remove_data',
 'resid',
 'save',
 'scale',
 'sigma2',
 'summary',
 'summary2',
 't_test',
 't_test_pairwise',
 'tvalues',
 'use_t',
 'wald_test',
 'wald_test_terms']