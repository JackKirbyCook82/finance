# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import os
import logging
import numpy as np
import xarray as xr
from itertools import chain
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.dispatchers import kwargsdispatcher
from support.pipelines import Producer, Processor, Consumer
from support.files import DataframeFile

from finance.variables import Contract, Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityLoader", "SecuritySaver", "SecurityFilter", "SecurityParser", "SecurityFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class SecurityQuery(ntuple("Query", "inquiry contract stocks options")):
    def __str__(self):
        strings = {str(security.title): str(len(dataframe.index)) for security, dataframe in self.stocks.items()}
        strings.update({str(security.title): str(len(dataframe.index)) for security, dataframe in self.options.items()})
        arguments = f"{self.contract.ticker}|{self.contract.expire.strftime('%Y-%m-%d')}"
        parameters = ", ".join(["=".join([key, value]) for key, value in strings.items()])
        return ", ".join([arguments, parameters]) if bool(parameters) else str(arguments)


class SecurityFilter(Processor, title="Filtered"):
    def execute(self, query, *args, **kwargs):
        stocks = {security: dataframe for security, dataframe in query.stocks.items() if not bool(dataframe.empty)}
        options = {security: dataframe for security, dataframe in query.options.items() if not bool(dataframe.empty)}
        stocks = {security: self.stocks(dataframe, *args, **kwargs) for security, dataframe in stocks.items()}
        options = {security: self.options(dataframe, *args, **kwargs) for security, dataframe in options.items()}
        query = SecurityQuery(query.inquiry, query.contract, stocks, options)
        LOGGER.info(f"Filter: {repr(self)}[{str(query)}]")
        yield query

    @staticmethod
    def stocks(dataframe, *args, size=None, volume=None, **kwargs):
        dataframe = dataframe.where(dataframe["size"] >= size) if bool(size) else dataframe
        dataframe = dataframe.where(dataframe["volume"] >= volume) if bool(volume) else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @staticmethod
    def options(dataframe, *args, size=None, interest=None, volume=None, **kwargs):
        dataframe = dataframe.where(dataframe["size"] >= size) if bool(size) else dataframe
        dataframe = dataframe.where(dataframe["interest"] >= interest) if bool(interest) else dataframe
        dataframe = dataframe.where(dataframe["volume"] >= volume) if bool(volume) else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe


class SecurityParser(Processor):
    def execute(self, query, *args, **kwargs):
        stocks = {security: self.stocks(dataframe, *args, security=security, **kwargs) for security, dataframe in query.stocks.items()}
        options = {security: self.options(dataframe, *args, security=security, **kwargs) for security, dataframe in query.options.items()}
        yield SecurityQuery(query.inquiry, query.contract, stocks, options)

    @staticmethod
    def stocks(dataframe, *args, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["date", "ticker"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["date", "ticker"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe[["price", "size"]])
        return dataset

    @staticmethod
    def options(dataframe, *args, security, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["date", "ticker", "expire", "strike"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["date", "ticker", "expire", "strike"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe[["price", "size"]])
        return dataset


class SecuritySaver(Consumer, title="Saved"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, SecurityFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, query, *args, **kwargs):
        stocks, options = query.stocks, query.options
        if not bool(stocks) or all([dataframe.empty for dataframe in stocks.values()]):
            return
        if not bool(options) or all([dataframe.empty for dataframe in options.values()]):
            return
        inquiry_name = str(query.inquiry.strftime("%Y%m%d_%H%M%S"))
        inquiry_folder = self.file.path(inquiry_name)
        if not os.path.isdir(inquiry_folder):
            os.mkdir(inquiry_folder)
        contract_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        contract_folder = self.file.path(inquiry_name, contract_name)
        if not os.path.isdir(contract_folder):
            os.mkdir(contract_folder)
        for security, dataframe in chain(stocks.items(), options.items()):
            security_name = str(security).replace("|", "_") + ".csv"
            security_file = self.file.path(inquiry_name, contract_name, security_name)
            self.file.write(dataframe, file=security_file, data=security, mode="w")
            LOGGER.info("Saved: {}[{}]".format(repr(self), str(security_file)))


class SecurityLoader(Producer, title="Loaded"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, SecurityFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, *args, tickers=None, expires=None, dates=None, **kwargs):
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        inquiry_folders = list(self.file.directory())
        for inquiry_name in sorted(inquiry_folders, key=function, reverse=False):
            inquiry = function(inquiry_name)
            if dates is not None and inquiry.date() not in dates:
                continue
            contract_folders = list(self.file.directory(inquiry_name))
            for contract_name in contract_folders:
                contract = Contract(*str(contract_name).split("_"))
                ticker = str(contract.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(contract.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                filenames = {security: str(security).replace("|", "_") + ".csv" for security in list(Securities.Stocks)}
                files = {security: self.file.path(inquiry_name, contract_name, filename) for security, filename in filenames.items()}
                stocks = {security: self.file.read(file=file, data=security) for security, file in files.items()}
                filenames = {security: str(security).replace("|", "_") + ".csv" for security in list(Securities.Options)}
                files = {security: self.file.path(inquiry_name, contract_name, filename) for security, filename in filenames.items()}
                options = {security: self.file.read(file=file, data=security) for security, file in files.items()}
                contract = Contract(ticker, expire)
                yield SecurityQuery(inquiry, contract, stocks, options)


class SecurityFile(DataframeFile):
    @kwargsdispatcher("data")
    def dataheader(self, *args, data, **kwargs): raise KeyError(str(data))
    @kwargsdispatcher("data")
    def datatypes(self, *args, data, **kwargs): raise KeyError(str(data))
    @kwargsdispatcher("data")
    def datetypes(self, *args, data, **kwargs): raise KeyError(str(data))

    @dataheader.register.value(*list(Securities.Options))
    def dataheader_options(self, *args, **kwargs): return ["date", "ticker", "expire", "price", "strike", "size", "volume", "interest"]
    @dataheader.register.value(*list(Securities.Stocks))
    def dataheader_stocks(self, *args, **kwargs): return ["date", "ticker", "price", "size", "volume"]
    @datatypes.register.value(*list(Securities.Options))
    def datatypes_options(self, *args, **kwargs): return {"price": np.float32, "strike": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32}
    @datatypes.register.value(*list(Securities.Stocks))
    def datatypes_stocks(self, *args, **kwargs): return {"price": np.float32, "size": np.int32, "volume": np.int64}
    @datetypes.register.value(*list(Securities.Options))
    def datetypes_options(self, *args, **kwargs): return ["date", "expire"]
    @datetypes.register.value(*list(Securities.Stocks))
    def datetypes_stocks(self, *args, **kwargs): return ["date"]



