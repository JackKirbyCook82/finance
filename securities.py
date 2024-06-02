# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from finance.variables import Contract
from support.files import FileDirectory, FileQuery, FileData
from support.pipelines import Processor
from support.filtering import Filter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StockFile", "OptionFile", "SecurityHeader", "SecurityFilter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


Header = ntuple("Header", "index columns")
stocks_index = {"instrument": str, "position": str, "ticker": str, "date": np.datetime64}
stocks_columns = {"price": np.float32, "size": np.float32, "volume": np.float32}
options_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
options_columns = {"price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
security_headers = {"stocks": Header(stocks_index, stocks_columns), "options": Header(options_index, options_columns)}
stocks_data = FileData.Dataframe(header=stocks_index | stocks_columns)
options_data = FileData.Dataframe(header=options_index | options_columns)
contract_query = FileQuery("contract", Contract.tostring, Contract.fromstring)


class StockFile(FileDirectory, variable="stocks", query=contract_query, data=stocks_data): pass
class OptionFile(FileDirectory, variable="options", query=contract_query, data=options_data): pass


class SecurityFilter(Filter):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__securities = list(security_headers.keys())

    def execute(self, contents, *args, **kwargs):
        contract, securities = str(contents["contract"]), {security: contents[security] for security in self.securities if security in contents.keys()}
        securities = ODict(list(self.calculate(securities, *args, contract=contract, **kwargs)))
        yield contents | dict(securities)

    def calculate(self, securities, *args, contract, **kwargs):
        for security, dataframe in securities.items():
            prior = self.size(dataframe)
            dataframe = self.filter(dataframe)
            post = self.size(dataframe)
            __logger__.info(f"Filter: {repr(self)}|{contract}|{security}[{prior:.0f}|{post:.0f}]")
            yield security, dataframe

    def filter(self, dataframe):
        mask = self.mask(dataframe)
        dataframe = self.where(dataframe, mask)
        return dataframe

    @property
    def securities(self): return self.__securities


class SecurityHeader(Processor):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__securities = dict(security_headers)

    def execute(self, contents, *args, **kwargs):
        securities = {security: contents[security] for security in self.securities if security in contents.keys()}
        securities = ODict(list(self.calculate(securities, *args, **kwargs)))
        yield contents | dict(securities)

    def calculate(self, securities, *args, **kwargs):
        for security, dataframe in securities.items():
            header = self.securities[security]
            dataframe = dataframe.set_index(header.index, drop=True, inplace=False)
            dataframe = dataframe[header.columns]
            yield security, dataframe

    @property
    def securities(self): return self.__securities



