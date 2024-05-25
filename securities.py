# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
from collections import OrderedDict as ODict

from finance.variables import Contract
from support.files import FileDirectory, FileQuery, FileData
from support.pipelines import Header, Query
from support.filtering import Filter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StockFile", "OptionFile", "SecurityFilter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


stocks_index = {"instrument": str, "position": str, "ticker": str, "date": np.datetime64}
stocks_columns = {"price": np.float32, "size": np.float32, "volume": np.float32}
stocks_header = Header.Dataframe(index=list(stocks_index.keys()), columns=list(stocks_columns.keys()))
options_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
options_columns = {"price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
options_header = Header.Dataframe(index=list(options_index.keys()), columns=list(options_columns.keys()))
securities_headers = dict(stocks=stocks_header, options=options_header)
stocks_data = FileData.Dataframe(index=stocks_index, columns=stocks_columns)
options_data = FileData.Dataframe(index=options_index, columns=options_columns)
contract_query = FileQuery("Contract", Contract.fromstring, Contract.tostring)


class StockFile(FileDirectory, variable="stocks", query=contract_query, data=stocks_data, duplicates=False): pass
class OptionFile(FileDirectory, variables="options", query=contract_query, data=options_data, duplicates=False): pass


class SecurityFilter(Filter):
    @Query(arguments=["contract"], parameters={"securities": ["stocks", "options"]}, headers=securities_headers)
    def execute(self, *args, contract, securities, **kwargs):
        securities = ODict(list(self.calculate(securities, *args, contract=contract, **kwargs)))
        yield dict(securities)

    def calculate(self, securities, *args, contract, **kwargs):
        for security, dataframe in securities.items():
            variable = str(security.name).lower()
            prior = self.size(dataframe)
            dataframe = dataframe.reset_index(drop=False, inplace=False)
            dataframe = self.filter(dataframe)
            post = self.size(dataframe)
            __logger__.info(f"Filter: {repr(self)}|{str(contract)}|{variable}[{prior:.0f}|{post:.0f}]")
            yield variable, dataframe

    def filter(self, dataframe):
        mask = self.mask(dataframe)
        dataframe = self.where(dataframe, mask)
        return dataframe


