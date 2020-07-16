#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""

import pandas as pd
import pathlib
import typing as tp
from zipfile import ZipFile


def kaggle_readzip(file: str, file_dir: str = "data", **read_csv_args) -> tp.Union[pd.DataFrame, tp.Dict[str, pd.DataFrame]]:
    """
    helper method to read .zip files downloaded from kaggle
    add functionality as download more files and find them to be incompatible with current method
    (e.g. because zip actually contains 2 different zips)

    :param file:
    :param file_dir:
    :param read_csv_args:
    :return:
    """
    data_dir_file_path = pathlib.Path(file_dir, file)
    try:
        return pd.read_csv(data_dir_file_path, compression='zip', header=0, sep=',', quotechar='"', **read_csv_args)

    except ValueError as e:
        if "multiple files found" in str(e).lower():
            # unzip all, adapted from https://stackoverflow.com/a/44575940

            zip_file = ZipFile(data_dir_file_path)
            dfs = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename))
                   for text_file in zip_file.infolist()
                   if text_file.filename.endswith('.csv')}

            return dfs

        # else, dont recgonise Exception so raise
