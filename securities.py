# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from collections import OrderedDict as ODict

from support.filtering import Filter
from support.files import Files

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StockFile", "OptionFile", "SecurityFilter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


stocks_index = {"instrument": str, "position": str, "ticker": str, "date": np.datetime64}
stocks_columns = {"price": np.float32, "size": np.float32, "volume": np.float32}
stocks_header = Header(pd.DataFrame, index=list(stocks_index.keys()), columns=list(stocks_columns.keys()))
options_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
options_columns = {"price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
options_header = Header(pd.DataFrame, index=list(options_index.keys()), columns=list(options_columns.keys()))
securities_headers = dict(stocks=stocks_header, options=options_header)


# class StockFile(Files.Dataframe, contents=["securities", "stocks"], variable="stocks", index=stocks_index, columns=stocks_columns): pass
# class OptionFile(Files.Dataframe, contents=["securities", "options"], variables="options", index=options_index, columns=options_columns): pass


class SecurityFilter(Filter):
    @query("contract", "securities", securities=securities_headers)
    def execute(self, contract, securities, *args, **kwargs):
        securities = ODict(list(self.filtering(securities, *args, contract=contract, **kwargs)))
        yield dict(securities=securities)

    def filtering(self, securities, *args, contract, **kwargs):
        for security, dataframe in securities.items():
            prior = self.size(dataframe)
            dataframe = dataframe.reset_index(drop=False, inplace=False)
            dataframe = self.filter(dataframe, *args, **kwargs)
            post = self.size(dataframe)
            __logger__.info(f"Filter: {repr(self)}|{str(contract)}[{prior:.0f}|{post:.0f}]")
            yield security, dataframe

    def filter(self, dataframe, *args, **kwargs):
        mask = self.mask(dataframe)
        dataframe = self.where(dataframe, mask)
        return dataframe


