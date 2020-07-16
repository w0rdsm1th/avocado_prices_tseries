#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import logging

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

import scipy as sp
import scipy.stats as sp_stats
import scipy.signal as sp_signal

from pmdarima.arima.stationarity import ADFTest
from pmdarima.arima.utils import ndiffs

import sklearn as skl
import sklearn.metrics as skl_metrics
import sklearn.model_selection as skl_model_select
import statsmodels.api as sm
import linearmodels as lm

# before could install fbprophet, had to manually resolve dependencies which clash in own requirements.txt
# pip install holidays==0.9.8
import fbprophet
import fbprophet.diagnostics as prophet_diagnostics

import pmdarima.model_selection

import _utils
from backtest_results import RegionBacktestResults

_log = logging.getLogger()
_log.setLevel("INFO")

# date format 2015-12-27
df = _utils.kaggle_readzip("tseries - avocado-prices.zip", file_dir="data", parse_dates=["Date"])


###########################################################################################
# first look: columns, variable types (categorical, date, continuous), categorical values
###########################################################################################

df.shape
df.columns
df.dtypes

# the "#" column counts up to 52 for most of period, except for 2018 where counts "down" to start of year
# 0,2018-03-25,1.57,149396.5,16361.69,109045.03,65.45,23924.33,19273.8,4270.53,380.0,conventional,2018,Albany
# 1,2018-03-18,1.35,105304.65,13234.86,61037.58,55.0,30977.21,26755.9,3721.31,500.0,conventional,2018,Albany
# 2,2018-03-11,1.12,144648.75,15823.35,110950.68,70.0,17804.72,14480.52,3033.09,291.11,conventional,2018,Albany
# 3,2018-03-04,1.08,139520.6,12002.12,105069.57,95.62,22353.29,16128.51,5941.45,283.33,conventional,2018,Albany
# 4,2018-02-25,1.28,104278.89,10368.77,59723.32,48.0,34138.8,30126.31,3702.49,310.0,conventional,2018,Albany
# 5,2018-02-18,1.43,85630.24,5499.73,61661.76,75.0,18393.75,15677.67,2336.08,380.0,conventional,2018,Albany
# 6,2018-02-11,1.45,121804.36,8183.48,95548.47,61.0,18011.41,13264.91,4295.39,451.11,conventional,2018,Albany
# 7,2018-02-04,1.03,216738.47,7625.65,195725.06,143.53,13244.23,10571.6,2422.63,250.0,conventional,2018,Albany
# 8,2018-01-28,1.57,93625.03,3101.17,74627.23,55.59,15841.04,11614.79,4159.58,66.67,conventional,2018,Albany
# 9,2018-01-21,1.69,135196.35,3133.37,116520.88,88.78,15453.32,10023.79,5429.53,0.0,conventional,2018,Albany
# 10,2018-01-14,1.42,95246.38,2897.41,76570.67,44.0,15734.3,10012.8,5721.5,0.0,conventional,2018,Albany
# 11,2018-01-07,1.13,98540.22,2940.63,76192.61,42.63,19364.35,8633.09,10707.93,23.33,conventional,2018,Albany


# Date format is ISO (no worries about US dates and locality)

# Numerical column names are  avocado lookup codes and contain number of avocados sold
lookup_codes = {
    "4046": "small_vol_sold",
    "4225": "large_vol_sold",
    "4770": "extra_large_vol_sold",
}

############################################################################################
# what is the relationship between price and total volume sold? in previous period?
############################################################################################


############################################################################################
# EDA of the data
############################################################################################

# types

# quantities sold

# regions with average most/least expensive. most: HartfordSpringfield and SanFrancisco, Least: PhoenixTucson and Houston
df.groupby(["region", "type"])["AveragePrice"].mean()

# most/least variance. most: Seattle, Least: Pittsburgh
df.groupby(["region", "type"])["AveragePrice"].std()

############################################################################################
# cleaning ahead of modelling
# data quality?
############################################################################################
df.type.value_counts()
len(df.region.unique())

df.region.value_counts()  # count of rows per region - most are equal to 338, but one not quite 338
df.region.unique()
len(df.region.unique())

# Date col

def _min_max_num_dates(inp):
    return inp["Date"].min(), inp["Date"].max(), len(inp["Date"])

# applying min-max per region and type
sample_continuity_check = df.groupby(["region", "type"]).apply(_min_max_num_dates)
sample_continuity_check.value_counts()
sample_continuity_check.loc[sample_continuity_check == (pd.to_datetime("2015-01-04 00:00:00"),
                                                        pd.to_datetime("2018-03-25 00:00:00"),
                                                        166)]
# data quality issue, WestTexNewMexico, Organic: missing 3x samples!
sample_continuity_check[slice("WestTexNewMexico")]

# consistent sampling frequency: are all the dates at equivalent times?
df["Date"].dt.dayofweek.value_counts()

# wider point about non-equivalence of even the lowest-level areas e.g. California and Scranton
df["region"].unique()





# df.groupby(["region", "type"])["AveragePrice"].describe().to_clipboard()

# price for region == "TotalUS" and type == "organic" crashing to 1 for 6 weeks near start of measurement period
# then restoring to "normal" levels
# stands out in line plot of AveragePrice, suspicious values

############################################################################################
# Visualise
############################################################################################
# sns.scatterplot(df.loc[df["region"] == "Albany", "Date"], df.loc[df["region"] == "Albany", "AveragePrice"])

region_to_plot = "Philadelphia"  # "TotalUS"  # "Portland"
price_scatt = sns.lmplot("Date", "AveragePrice", data=df.loc[df["region"] == region_to_plot],
                         fit_reg=False, hue="type", legend=False)
plt.title(f"Average USD Price of Single Avocado \n for Region '{region_to_plot}'", )
plt.legend(loc="lower right")
plt.xticks(rotation=45, )
plt.show()

# one big row of regions
g = sns.FacetGrid(df, col="region", hue="type")
g.map(plt.scatter, "Date", "AveragePrice", alpha=0.7)
plt.show()


"""takeaways
- conventional vs organic: organic almost always more expensive than conventional.
in some places conventional _briefly_ beats organic in second seasonal peak. which is interesting! 
e.g. Chicago and Pittsburgh in 2018-06

- Strong seasonality to the prices.
prices for conventional and organic are most expensive in August and least expensive in January 
interesting how early the spikes start and long they last for and if they have any persevering effects.

exceptions to this rule: Pittsburgh, virtually totally flat entire period! 

- across regions nationally the second season is the "higher" spike

- in many places there isn't a strong 3 year trend. prices dont appear noticeably higher at end of period to start
"""

# TODO - violin plot of price for upper and lower regions?
#   or for the least correlated 2 sub-regions?
# sns.violinplot(x="year", y="AveragePrice", hue="type", data=df)
# plt.show()

# todo heatmap of relative price through time by region

# TODO - test for trend. test if prices are statistically higher at end of period (average 2 comparable months)
#  nb repeated measures of same population!

############################################################################################
# summary stats
############################################################################################

"""Chatfield Ch2
- if seasonality or trend present then usual stats of mean/variance can be misleading
simple: successive yearly averages

- moving average of series as a descriptive stat
Spencer and Henderson's MAs which specify rate of weight decay

- filters:
depends on time series and what frequency of variation we are interested in
e.g. if want smoothed values, need to remove local fluctuations (or high freq variations) >> need a LOW PASS filter
but if want the residuals after seasonality then want a HIGH PASS filter
CAUTION: Slutsky showing that on a completely random series applying both averaging and differencing could induce 
sinusoidal variation in the data. 

- multiple filters (e.g. filter I, then filter II).
Spencer 15-point MA is actually a convolution of four (quite logical looking) filters

- differencing: esp useful to remove trend
difference until stationary (integral to Box-Jenkins procedure)
non-seasonal: difference once
seasonal differencing: 

- seasonal variation
difficulty where error terms not exactly additive or multiplicative: εt
Additive: to eliminate we subtract the previous year period (e.g. last Jan)
   Xt = mt + St + εt
multiplicative: to eliminate divide by previous year period (e.g. last Jan)
   Xt = mt*St + εt
   Xt = mt*St*εt
Question: measure the seasonality or eliminate the seasonality from forecasting?

- seasonal variation AND substantial trend: have to just take each period 
"""

# simple: successive yearly averages
# per region, per type, per year (or 3 months?) mean/median
yearly_describe = df.groupby(["region", "type", "year"])["AveragePrice"].describe()

for each_type in df["type"].unique():
    print(f"type == {each_type}-----------------")
    for successive_yrs in zip(df["year"].unique()[:-1], df["year"].unique()[1:]):
        first_year = yearly_describe.query(f'year == {successive_yrs[0]} & type == "{each_type}"').reset_index()
        mask = first_year["mean"] < yearly_describe.query(f'year == {successive_yrs[1]} & type == "{each_type}"').reset_index()["mean"]

        print(f"num regions where {successive_yrs[1]} mean price was *greater* than {successive_yrs[0]} price {sum(mask)}")
        # print(f"regions {first_year['region'][mask]}")

# moving average of series as a descriptive stat

# correlation of regions with each other versus comparing types
# Q what region is most strongly correlated with TotalUS?
each_type = "organic"  # "conventional"
totalUS = "TotalUS"
s1 = df.query(f'region == "{totalUS}" & type == "{each_type}"')["AveragePrice"]
min_corr, max_corr = 1, 0
min_corr_region, max_corr_region = "", ""
for other_region in df["region"].unique():
    s2 = df.query(f'region == "{other_region}" & type == "{each_type}"')["AveragePrice"]
    if other_region == totalUS or len(s2) != len(s1):
        continue
    this_combo_corr = abs(np.corrcoef(s1, s2)[1][0])
    # print(f'correlation coefficient: {s1.corr(s2)}')  # pandas are nan for all
    if this_combo_corr < min_corr:
        min_corr = this_combo_corr
        min_corr_region = other_region
    if this_combo_corr > max_corr:
        max_corr = this_combo_corr
        max_corr_region = other_region


########################################################
# tests
########################################################

# statistically different mean and spread of average price per city?

# t-test, independent because samples are from different cities (NOT same city before & after a treatment)
# sp_stats.ttest_ind()

# chi-square

# anova, analysis of variance


# groupby and pivot to feed data into oneway ANOVA
# reshaped = []
# for region in df.region.unique():
#     reshaped.append(df[df["region"] == region].sort_values('Date')['AveragePrice'].values)
#
# sp_stats.f_oneway(*reshaped)



############### statistically different mean and spread per type of avocado sold?
# t-test, related because single city OR calculate OWN test stat by differencing exactly comparable dates and do single ttest

def _type_price_difference_test_stat(df, region):

    organic = df[(df["region"] == region) & (df["type"] == "organic")][['Date', 'AveragePrice']]
    conventional = df[(df["region"] == region) & (df["type"] == "conventional")][['Date', 'AveragePrice']]

    return organic - conventional

_type_price_difference_test_stat(df, region="LasVegas")


# size = [small, large, extra large]

# type = [organic, conventional]

############################################################################################
