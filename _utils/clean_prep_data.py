#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""

import numpy as np

# filter out overlap in areas? or combination_areas
# without a data spec, assume that these regions are their own catchment areas,
# but also cover combinations of other areas: "West" and "California", "California" and "Sacramento" or "LosAngeles"...
combination_areas = (
    "Plains",
    "Northeast",
    "Southeast",
    "Midsouth",
    "TotalUS",
    "GreatLakes",
    "SouthCentral",
    "West",
)


def clean_prep_data(inp_df):
    """
    standardising the clean and prep of data based on EDA
    acts in place on inp_df
    :param inp_df:
    :return:
    """

    # drop regions with missing observations
    inp_df.drop(inp_df.loc[(inp_df["region"] == "WestTexNewMexico") & (inp_df["type"] == "organic")].index,
                axis=0, inplace=True)

    inp_df.sort_values(by=["region", "type", "Date"], inplace=True)

    # single index on Date
    inp_df.set_index("Date", inplace=True, drop=False)

    # the "#" column counts up to 52 for most of period, except for 2018 where counts "down" to start of year
    inp_df.drop("Unnamed: 0", axis=1, inplace=True)

    # TODO interpolate or drop rows where price is "suspicious", e.g. TotalUS and Organic

    # assign new columns
    inp_df["combination_areas"] = np.where(inp_df["region"].isin(combination_areas), 1, 0)
