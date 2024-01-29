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
from abc import ABC
from itertools import chain
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.dispatchers import kwargsdispatcher
from support.calculations import Calculation, equation, source
from support.pipelines import Producer, Processor, Consumer
from support.files import DataframeFile

from finance.variables import Securities, Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityLoader", "SecuritySaver", "SecurityFilter", "SecurityCalculator", "SecurityFile"]
__copyright__ = "Copyright 2023,SE Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class PositionCalculation(Calculation, ABC): pass
class InstrumentCalculation(Calculation, ABC): pass
class LongCalculation(PositionCalculation, ABC): pass
class ShortCalculation(PositionCalculation, ABC): pass

class StockCalculation(InstrumentCalculation):
    Λ = source("Λ", "stock", position=0, variables={"to": "date", "w": "price", "x": "size", "q": "volume"})

    def execute(self, feed, *args, **kwargs):
        yield self["Λ"].w(feed)
        yield self["Λ"].x(feed)

class OptionCalculation(InstrumentCalculation):
    τ = equation("τ", "tau", np.int32, domain=("Λ.to", "Λ.tτ"), function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    Λ = source("Λ", "option", position=0, variables={"to": "date", "tτ": "expire", "w": "price", "x": "size", "q": "volume", "i": "interest"})

    def execute(self, feed, *args, **kwargs):
        yield self["Λ"].w(feed)
        yield self["Λ"].x(feed)
        yield self.τ(feed)

class PutCalculation(OptionCalculation): pass
class CallCalculation(OptionCalculation): pass
class StockLongCalculation(StockCalculation, LongCalculation): pass
class StockShortCalculation(StockCalculation, ShortCalculation): pass
class PutLongCalculation(PutCalculation, LongCalculation): pass
class PutShortCalculation(PutCalculation, ShortCalculation): pass
class CallLongCalculation(CallCalculation, LongCalculation): pass
class CallShortCalculation(CallCalculation, ShortCalculation): pass


class CalculationsMeta(type):
    def __iter__(cls):
        contents = {Securities.StockLong: StockLongCalculation, Securities.StockShort: StockShortCalculation}
        contents.update({Securities.PutLong: PutLongCalculation, Securities.PutShort: PutShortCalculation})
        contents.update({Securities.CallLong: CallLongCalculation, Securities.CallShort: CallShortCalculation})
        return ((key, value) for key, value in contents.items())

    class Stock:
        Long = StockLongCalculation
        Short = StockShortCalculation
    class Option:
        class Put:
            Long = PutLongCalculation
            Short = PutShortCalculation
        class Call:
            Long = CallLongCalculation
            Short = CallShortCalculation

    @property
    def Stocks(cls): return iter({Securities.StockLong: StockLongCalculation, Securities.StockShort: StockShortCalculation}.items())
    @property
    def Options(cls): return iter({Securities.PutLong: PutLongCalculation, Securities.PutShort: PutShortCalculation, Securities.allLong: CallLongCalculation, Securities.CallShort: CallShortCalculation}.items())
    @property
    def Puts(cls): return iter({Securities.PutLong: PutLongCalculation, Securities.PutShort: PutShortCalculation}.items())
    @property
    def Calls(cls): return iter({Securities.CallLong: CallLongCalculation, Securities.CallShort: CallShortCalculation}.items())

class Calculations(object, metaclass=CalculationsMeta):
    pass


class SecurityQuery(ntuple("Query", "current contract stocks options")):
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
        query = SecurityQuery(query.current, query.contract, stocks, options)
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


class SecurityCalculator(Processor, title="Calculated"):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {strategy: calculation(*args, **kwargs) for (strategy, calculation) in iter(Calculations)}
        self.calculations = calculations

    def execute(self, query, *args, **kwargs):
        stocks = {security: self.stocks(dataframe, *args, **kwargs) for security, dataframe in query.stocks.items()}
        options = {security: self.options(dataframe, *args, **kwargs) for security, dataframe in query.options.items()}
        stocks = {security: self.calculations[security](dataframe, *args, **kwargs) for security, dataframe in stocks.items()}
        options = {security: self.calculations[security](dataframe, *args, **kwargs) for security, dataframe in options.items()}
        yield SecurityQuery(query.current, query.contract, stocks, options)

    @staticmethod
    def stocks(dataframe, *args, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["ticker", "date"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["ticker", "date"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe[["price", "size"]])
        return dataset

    @staticmethod
    def options(dataframe, *args, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["ticker", "date", "expire", "strike"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["ticker", "date", "expire", "strike"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe[["price", "size"]])
        return dataset


class SecurityFile(DataframeFile):
    @kwargsdispatcher("data")
    def dataheader(self, *args, data, **kwargs): raise KeyError(str(data))
    @kwargsdispatcher("data")
    def datatypes(self, *args, data, **kwargs): raise KeyError(str(data))
    @kwargsdispatcher("data")
    def datetypes(self, *args, data, **kwargs): raise KeyError(str(data))

    @dataheader.register.value(*list(Securities.Options))
    def dataheader_options(self, *args, **kwargs): return ["ticker", "date", "expire", "price", "strike", "size", "volume", "interest"]
    @dataheader.register.value(*list(Securities.Stocks))
    def dataheader_stocks(self, *args, **kwargs): return ["ticker", "date", "price", "size", "volume"]
    @datatypes.register.value(*list(Securities.Options))
    def datatypes_options(self, *args, **kwargs): return {"price": np.float32, "strike": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32}
    @datatypes.register.value(*list(Securities.Stocks))
    def datatypes_stocks(self, *args, **kwargs): return {"price": np.float32, "size": np.int32, "volume": np.int64}
    @datetypes.register.value(*list(Securities.Options))
    def datetypes_options(self, *args, **kwargs): return ["date", "expire"]
    @datetypes.register.value(*list(Securities.Stocks))
    def datetypes_stocks(self, *args, **kwargs): return ["date"]


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
        current_name = str(query.current.strftime("%Y%m%d_%H%M%S"))
        current_folder = self.file.path(current_name)
        if not os.path.isdir(current_folder):
            os.mkdir(current_folder)
        ticker_expire_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        ticker_expire_folder = self.file.path(current_name, ticker_expire_name)
        if not os.path.isdir(ticker_expire_folder):
            os.mkdir(ticker_expire_folder)
        for security, dataframe in chain(stocks.items(), options.items()):
            security_name = str(security).replace("|", "_") + ".csv"
            security_file = self.file.path(current_name, ticker_expire_name, security_name)
            self.file.write(dataframe, file=security_file, data=security, mode="w")


class SecurityLoader(Producer, title="Loaded"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, SecurityFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, *args, tickers=None, expires=None, dates=None, **kwargs):
        TickerExpire = ntuple("TickerExpire", "ticker expire")
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        current_folders = list(self.file.directory())
        for current_name in sorted(current_folders, key=function, reverse=False):
            current = function(current_name)
            if dates is not None and current.date() not in dates:
                continue
            ticker_expire_folders = list(self.file.directory(current_name))
            for ticker_expire_name in ticker_expire_folders:
                ticker_expire = TickerExpire(*str(ticker_expire_name).split("_"))
                ticker = str(ticker_expire.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(ticker_expire.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                filenames = {security: str(security).replace("|", "_") + ".csv" for security in list(Securities.Stocks)}
                files = {security: self.file.path(current_name, ticker_expire_name, filename) for security, filename in filenames.items()}
                stocks = {security: self.file.read(file=file, data=security) for security, file in files.items()}
                filenames = {security: str(security).replace("|", "_") + ".csv" for security in list(Securities.Options)}
                files = {security: self.file.path(current_name, ticker_expire_name, filename) for security, filename in filenames.items()}
                options = {security: self.file.read(file=file, data=security) for security, file in files.items()}
                contract = Contract(ticker, expire)
                yield SecurityQuery(current, contract, stocks, options)



