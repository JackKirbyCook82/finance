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
import pandas as pd
from abc import ABC
from enum import IntEnum
from datetime import date as Date
from datetime import datetime as Datetime
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.pipelines import Processor, Calculator, Saver, Loader
from support.calculations import Calculation, equation, source
from support.dispatchers import typedispatcher, kwargsdispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "Instruments", "Positions", "Security", "Securities", "Calculations"]
__all__ += ["SecuritySaver", "SecurityLoader", "SecurityFilter", "SecurityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class DateRange(ntuple("DateRange", "minimum maximum")):
    def __new__(cls, dates):
        assert isinstance(dates, list)
        assert all([isinstance(date, Date) for date in dates])
        dates = [date if isinstance(date, Date) else date.date() for date in dates]
        return super().__new__(cls, min(dates), max(dates)) if dates else None

    def __contains__(self, date): return self.minimum <= date <= self.maximum
    def __repr__(self): return "{}({}, {})".format(self.__class__.__name__, repr(self.minimum), repr(self.maximum))
    def __str__(self): return "{}|{}".format(str(self.minimum), str(self.maximum))
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return (self.maximum - self.minimum).days


Instruments = IntEnum("Instrument", ["PUT", "CALL", "STOCK"], start=1)
Positions = IntEnum("Position", ["LONG", "SHORT"], start=1)
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
    τ = equation("τ", "tau", np.int16, domain=("Λ.to", "Λ.tτ"), function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    Λ = source("Λ", "option", position=0, variables={"to": "date", "w": "price", "x": "size", "q": "volume", "tτ": "expire", "i": "interest"})

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


class SecurityQuery(ntuple("Query", "current ticker expire securities")): pass
class SecurityCalculator(Calculator, calculations=ODict(list(Calculations))):
    def execute(self, query, *args, **kwargs):
        securities = {security: dataframe for security, dataframe in query.securities.items() if not dataframe.empty}
        if not bool(securities):
            return
        parser = lambda security, dataframe: self.parser(dataframe, *args, security=security, **kwargs)
        calculations = {security: calculation for security, calculation in self.calculations.items() if security in securities.keys()}
        securities = {security: parser(security, dataframe) for security, dataframe in securities.items()}
        securities = {security: calculation(securities[security], *args, **kwargs) for security, calculation in calculations.items()}
        if not bool(securities):
            return
        yield SecurityQuery(query.current, query.ticker, query.expire, securities)

    @kwargsdispatcher("security")
    def parser(self, dataframe, *args, security, **kwargs): raise ValueError(str(security))

    @parser.register.value(*list(Securities.Stocks))
    def stock(self, dataframe, *args, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["ticker", "date"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["ticker", "date"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe[["price", "size"]])
        return dataset

    @parser.register.value(*list(Securities.Options))
    def option(self, dataframe, *args, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["ticker", "date", "expire", "strike"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["ticker", "date", "expire", "strike"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe[["price", "size"]])
        return dataset


class SecurityFilter(Processor):
    def execute(self, query, *args, **kwargs):
        securities = {security: self.filter(dataframe, *args, security=security, **kwargs) for security, dataframe in query.securities.items()}
        strings = {str(valuation.title): str(len(dataframe.index)) for valuation, dataframe in securities.items()}
        string = ", ".join(["=".join([key, value]) for key, value in strings.items()])
        LOGGER.info("Filtered: {}[{}]".format(repr(self), string))
        yield SecurityQuery(query.current, query.ticker, query.expire, securities)

    @kwargsdispatcher("security")
    def filter(self, dataframe, *args, security, **kwargs): raise ValueError(str(security))

    @filter.register.value(*list(Securities.Stocks))
    def stock(self, dataframe, *args, size=None, volume=None, **kwargs):
        dataframe = dataframe.where(dataframe["size"] >= size) if bool(size) else dataframe
        dataframe = dataframe.where(dataframe["volume"] >= volume) if bool(volume) else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        return dataframe

    @filter.register.value(*list(Securities.Options))
    def option(self, dataframe, *args, size=None, interest=None, volume=None, **kwargs):
        dataframe = dataframe.where(dataframe["size"] >= size) if bool(size) else dataframe
        dataframe = dataframe.where(dataframe["interest"] >= interest) if bool(interest) else dataframe
        dataframe = dataframe.where(dataframe["volume"] >= volume) if bool(volume) else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        return dataframe


class SecuritySaver(Saver):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        if not bool(securities) or not bool([dataframe.empty for dataframe in securities.values()]):
            return
        current_folder = os.path.join(self.repository, str(query.current.strftime("%Y%m%d_%H%M%S")))
        with self.locks[current_folder]:
            if not os.path.isdir(current_folder):
                os.mkdir(current_folder)
        ticker_expire_name = "_".join([str(query.ticker), str(query.expire.strftime("%Y%m%d"))])
        ticker_expire_folder = os.path.join(current_folder, ticker_expire_name)
        with self.locks[ticker_expire_folder]:
            if not os.path.isdir(ticker_expire_folder):
                os.mkdir(ticker_expire_folder)
            for security, dataframe in securities.items():
                security_filename = str(security).replace("|", "_") + ".csv"
                security_file = os.path.join(ticker_expire_folder, security_filename)
                self.write(dataframe, file=security_file, mode="w")


class SecurityLoader(Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Columns = ntuple("Columns", "datetypes datatypes")
        stock = Columns(["date"], {"ticker": str, "price": np.float32, "volume": np.int64, "size": np.int64})
        option = Columns(["date", "expire"], {"ticker": str, "strike": np.float32, "price": np.float32, "volume": np.int64, "interest": np.int64, "size": np.int64})
        columns = {str(Securities.Stock.Long): stock, str(Securities.Stock.Short): stock}
        columns.update({str(Securities.Option.Put.Long): option, str(Securities.Option.Put.Short): option})
        columns.update({str(Securities.Option.Call.Long): option, str(Securities.Option.Call.Short): option})
        self.columns = columns

    def execute(self, *args, tickers=None, expires=None, dates=None, **kwargs):
        TickerExpire = ntuple("TickerExpire", "ticker expire")
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        datatypes = lambda security: self.columns[str(security)].datatypes
        datetypes = lambda security: self.columns[str(security)].datetypes
        reader = lambda security, file: self.read(file=file, filetype=pd.DataFrame, datatypes=datatypes(security), datetypes=datetypes(security))
        for current_name in sorted(os.listdir(self.repository), key=function, reverse=False):
            current = function(current_name)
            if dates is not None and current.date() not in dates:
                continue
            current_folder = os.path.join(self.repository, current_name)
            for ticker_expire_name in os.listdir(current_folder):
                ticker_expire = TickerExpire(*str(ticker_expire_name).split("_"))
                ticker = str(ticker_expire.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(ticker_expire.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                ticker_expire_folder = os.path.join(current_folder, ticker_expire_name)
                with self.locks[ticker_expire_folder]:
                    filenames = {security: str(security).replace("|", "_") + ".csv" for security in list(Securities)}
                    files = {security: os.path.join(ticker_expire_folder, filename) for security, filename in filenames.items()}
                    securities = {security: reader(security, file) for security, file in files.items()}
                    yield SecurityQuery(current, ticker, expire, securities)



