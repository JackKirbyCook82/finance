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
from enum import IntEnum
from itertools import chain
from datetime import date as Date
from datetime import datetime as Datetime
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple

from support.dispatchers import typedispatcher, kwargsdispatcher
from support.calculations import Calculation, equation, source
from support.pipelines import Processor, Reader, Writer
from support.files import DataframeFile

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "Instruments", "Positions", "Security", "Securities", "Calculations", "SecurityFile", "SecurityReader", "SecurityWriter", "SecurityFilter", "SecurityParser", "SecurityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
Instruments = IntEnum("Instruments", ["PUT", "CALL", "STOCK"], start=1)
Positions = IntEnum("Positions", ["LONG", "SHORT"], start=1)


class DateRange(ntuple("DateRange", "minimum maximum")):
    def __new__(cls, dates):
        assert isinstance(dates, list)
        assert all([isinstance(date, Date) for date in dates])
        dates = [date if isinstance(date, Date) else date.date() for date in dates]
        return super().__new__(cls, min(dates), max(dates)) if dates else None

    def __contains__(self, date): return self.minimum <= date <= self.maximum
    def __repr__(self): return f"{self.__class__.__name__}({repr(self.minimum)}, {repr(self.maximum)})"
    def __str__(self): return f"{str(self.minimum)}|{str(self.maximum)}"
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return (self.maximum - self.minimum).days


class Security(ntuple("Security", "instrument position")):
    def __new__(cls, instrument, position, *args, **kwargs): return super().__new__(cls, instrument, position)
    def __init__(self, *args, payoff, **kwargs): self.__payoff = payoff
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])
    def __int__(self): return int(self.instrument) * 10 + int(self.position) * 1

    @property
    def title(self): return "|".join([str(string).title() for string in str(self).split("|")])
    @property
    def payoff(self): return self.__payoff

StockLong = Security(Instruments.STOCK, Positions.LONG, payoff=lambda x: np.copy(x))
StockShort = Security(Instruments.STOCK, Positions.SHORT, payoff=lambda x: - np.copy(x))
PutLong = Security(Instruments.PUT, Positions.LONG, payoff=lambda x, k: np.maximum(k - x, 0))
PutShort = Security(Instruments.PUT, Positions.SHORT, payoff=lambda x, k: - np.maximum(k - x, 0))
CallLong = Security(Instruments.CALL, Positions.LONG, payoff=lambda x, k: np.maximum(x - k, 0))
CallShort = Security(Instruments.CALL, Positions.SHORT, payoff=lambda x, k: - np.maximum(x - k, 0))


class SecuritiesMeta(type):
    def __iter__(cls): return iter([StockLong, StockShort, PutLong, PutShort, CallLong, CallShort])
    def __getitem__(cls, indexkey): return cls.retrieve(indexkey)

    @typedispatcher
    def retrieve(cls, indexkey): raise TypeError(type(indexkey).__name__)
    @retrieve.register(int)
    def integer(cls, index): return {int(security): security for security in iter(cls)}[index]
    @retrieve.register(str)
    def string(cls, string): return {str(security): security for security in iter(cls)}[str(string).lower()]
    @retrieve.register(tuple)
    def value(cls, value): return {(security.instrument, security.postion): security for security in iter(cls)}[value]

    @property
    def Stocks(cls): return iter([StockLong, StockShort])
    @property
    def Options(cls): return iter([PutLong, PutShort, CallLong, CallShort])
    @property
    def Puts(cls): return iter([PutLong, PutShort])
    @property
    def Calls(cls): return iter([CallLong, CallShort])

    class Stock:
        Long = StockLong
        Short = StockShort
    class Option:
        class Put:
            Long = PutLong
            Short = PutShort
        class Call:
            Long = CallLong
            Short = CallShort

class Securities(object, metaclass=SecuritiesMeta):
    pass


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
        contents = {Securities.Stock.Long: StockLongCalculation, Securities.Stock.Short: StockShortCalculation}
        contents.update({Securities.Option.Put.Long: PutLongCalculation, Securities.Option.Put.Short: PutShortCalculation})
        contents.update({Securities.Option.Call.Long: CallLongCalculation, Securities.Option.Call.Short: CallShortCalculation})
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

class Calculations(object, metaclass=CalculationsMeta):
    pass


class SecurityQuery(ntuple("Query", "current ticker expire stocks options")):
    def __str__(self):
        strings = {str(security.title): str(len(dataframe.index)) for security, dataframe in self.stocks.items()}
        strings.update({str(security.title): str(len(dataframe.index)) for security, dataframe in self.options.items()})
        arguments = f"{self.ticker}|{self.expire.strftime('%Y-%m-%d')}"
        parameters = ", ".join(["=".join([key, value]) for key, value in strings.items()])
        return ", ".join([arguments, parameters]) if bool(parameters) else str(arguments)


class SecurityFilter(Processor):
    def execute(self, query, *args, **kwargs):
        stocks = {security: dataframe for security, dataframe in query.stocks.items() if not bool(dataframe.empty)}
        options = {security: dataframe for security, dataframe in query.options.items() if not bool(dataframe.empty)}
        stocks = {security: self.stock(dataframe, *args, security=security, **kwargs) for security, dataframe in stocks.items()}
        options = {security: self.option(dataframe, *args, security=security, **kwargs) for security, dataframe in options.items()}
        query = SecurityQuery(query.current, query.ticker, query.expire, stocks, options)
        LOGGER.info(f"Filter: {repr(self)}[{str(query)}]")
        yield query

    @staticmethod
    def stock(dataframe, *args, size=None, volume=None, **kwargs):
        dataframe = dataframe.where(dataframe["size"] >= size) if bool(size) else dataframe
        dataframe = dataframe.where(dataframe["volume"] >= volume) if bool(volume) else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @staticmethod
    def option(dataframe, *args, size=None, interest=None, volume=None, **kwargs):
        dataframe = dataframe.where(dataframe["size"] >= size) if bool(size) else dataframe
        dataframe = dataframe.where(dataframe["interest"] >= interest) if bool(interest) else dataframe
        dataframe = dataframe.where(dataframe["volume"] >= volume) if bool(volume) else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe


class SecurityParser(Processor):
    def execute(self, query, *args, **kwargs):
        stocks = {security: dataframe for security, dataframe in query.stocks.items() if not bool(dataframe.empty)}
        options = {security: dataframe for security, dataframe in query.options.items() if not bool(dataframe.empty)}
        stocks = {security: self.stock(dataframe, *args, security=security, **kwargs) for security, dataframe in stocks.items()}
        options = {security: self.option(dataframe, *args, security=security, **kwargs) for security, dataframe in options.items()}
        query = SecurityQuery(query.current, query.ticker, query.expire, stocks, options)
        yield query

    @staticmethod
    def stock(dataframe, *args, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["ticker", "date"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["ticker", "date"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe[["price", "size"]])
        return dataset

    @staticmethod
    def option(dataframe, *args, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["ticker", "date", "expire", "strike"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["ticker", "date", "expire", "strike"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe[["price", "size"]])
        return dataset


class SecurityCalculator(Processor):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        securities = kwargs.get("calculations", ODict(list(Calculations)).keys())
        calculations = ODict([(security, calculation(*args, **kwargs)) for security, calculation in iter(Calculations) if security in securities])
        self.__calculations = calculations

    def execute(self, query, *args, **kwargs):
        stocks = {security: dataframe for security, dataframe in query.stocks.items()}
        options = {security: dataframe for security, dataframe in query.options.items()}
        stocks = {security: self.calculations[security](dataframe, *args, **kwargs) for security, dataframe in stocks.items()}
        options = {security: self.calculations[security](dataframe, *args, **kwargs) for security, dataframe in options.items()}
        yield SecurityQuery(query.current, query.ticker, query.expire, stocks, options)

    @property
    def calculations(self): return self.__calculations


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


class SecurityWriter(Writer):
    def execute(self, query, *args, **kwargs):
        stocks, options = query.stocks, query.options
        if not bool(stocks) or all([dataframe.empty for dataframe in stocks.values()]):
            return
        if not bool(options) or all([dataframe.empty for dataframe in options.values()]):
            return
        current_name = str(query.current.strftime("%Y%m%d_%H%M%S"))
        current_folder = self.destination.path(current_name)
        if not os.path.isdir(current_folder):
            os.mkdir(current_folder)
        ticker_expire_name = "_".join([str(query.ticker), str(query.expire.strftime("%Y%m%d"))])
        ticker_expire_folder = self.destination.path(current_name, ticker_expire_name)
        if not os.path.isdir(ticker_expire_folder):
            os.mkdir(ticker_expire_folder)
        for security, dataframe in chain(stocks.items(), options.items()):
            security_name = str(security).replace("|", "_") + ".csv"
            security_file = self.destination.path(current_name, ticker_expire_name, security_name)
            self.destination.write(dataframe, file=security_file, data=security, mode="w")
            LOGGER.info("Saved: {}[{}]".format(repr(self), str(security_file)))


class SecurityReader(Reader):
    def execute(self, *args, tickers=None, expires=None, dates=None, **kwargs):
        TickerExpire = ntuple("TickerExpire", "ticker expire")
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        current_folders = list(self.source.directory())
        for current_name in sorted(current_folders, key=function, reverse=False):
            current = function(current_name)
            if dates is not None and current.date() not in dates:
                continue
            ticker_expire_folders = list(self.source.directory(current_name))
            for ticker_expire_name in ticker_expire_folders:
                ticker_expire = TickerExpire(*str(ticker_expire_name).split("_"))
                ticker = str(ticker_expire.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(ticker_expire.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                filenames = {security: str(security).replace("|", "_") + ".csv" for security in list(Securities.Stocks)}
                files = {security: self.source.path(current_name, ticker_expire_name, filename) for security, filename in filenames.items()}
                stocks = {security: self.source.read(file=file, data=security) for security, file in files.items()}
                filenames = {security: str(security).replace("|", "_") + ".csv" for security in list(Securities.Options)}
                files = {security: self.source.path(current_name, ticker_expire_name, filename) for security, filename in filenames.items()}
                options = {security: self.source.read(file=file, data=security) for security, file in files.items()}
                yield SecurityQuery(current, ticker, expire, stocks, options)



