# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
from enum import Enum
from collections import OrderedDict as ODict

from finance.variables import Contract
from support.files import FileDirectory, FileQuery, FileData
from support.filtering import Filter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionFile", "SecurityFilter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


stocks_index = {"instrument": str, "position": str, "ticker": str, "date": np.datetime64}
stocks_columns = {"price": np.float32, "size": np.float32, "volume": np.float32}
options_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
options_columns = {"price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
options_data = FileData.Dataframe(index=options_index, columns=options_columns, duplicates=False)
contract_query = FileQuery("contract", Contract.tostring, Contract.fromstring)


class OptionFile(FileDirectory, variable="options", query=contract_query, data=options_data):
    pass


class SecurityFilter(Filter):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__columns = dict(stocks=list(stocks_columns.keys()), options=list(options_columns.keys()))
        self.__indexes = dict(stocks=list(stocks_index.keys()), options=list(options_index.keys()))
        self.__securities = ("stocks", "options")

    def execute(self, contents, *args, **kwargs):
        contract, securities = str(contents["contract"]), {security: contents[security] for security in self.securities if security in contents.keys()}
        securities = ODict(list(self.calculate(securities, *args, contract=contract, **kwargs)))
        securities = ODict(list(self.parse(securities, *args, contract=contract, **kwargs)))
        yield contents | dict(securities)

    def calculate(self, securities, *args, contract, **kwargs):
        for security, dataframe in securities.items():
            prior = self.size(dataframe)
            dataframe = dataframe.reset_index(drop=False, inplace=False)
            dataframe = self.filter(dataframe)
            post = self.size(dataframe)
            __logger__.info(f"Filter: {repr(self)}|{contract}|{security}[{prior:.0f}|{post:.0f}]")
            yield security, dataframe

    def parse(self, securities, *args, **kwargs):
        for security, dataframe in securities.itmes():
            index, columns = self.indexes[security], self.columns[security]
            dataframe = dataframe.set_index(index, drop=True, inplace=False)
            dataframe = dataframe[columns]
            yield security, dataframe

    def filter(self, dataframe):
        mask = self.mask(dataframe)
        dataframe = self.where(dataframe, mask)
        return dataframe

    @property
    def securities(self): return self.__securities
    @property
    def columns(self): return self.__columns
    @property
    def indexes(self): return self.__index



