# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import os
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
from support.dispatchers import kwargsdispatcher, typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "Instruments", "Positions", "Security", "Securities", "Calculations", "SecurityProcessor", "SecurityCalculator", "SecuritySaver", "SecurityLoader"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


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
    Λ = source("Λ", "stock", position=0, variables={"to": "date", "w": "price", "q": "size"})

    def execute(self, dataset, *args, **kwargs):
        yield self["Λ"].w(dataset)

class OptionCalculation(InstrumentCalculation):
    Λ = source("Λ", "option", position=0, variables={"to": "date", "w": "price", "q": "size", "tτ": "expire", "k": "strike", "i": "interest"})
    τ = equation("τ", "tau", np.int16, domain=("Λ.to", "Λ.tτ"), function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))

    def execute(self, dataset, *args, **kwargs):
        yield self["Λ"].w(dataset)
        yield self["Λ"].k(dataset)
        yield self.τ(dataset)

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


class SecurityCalculator(Calculator, calculations=ODict(list(iter(Calculations)))):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, datasets = contents
        assert isinstance(datasets, dict)
        assert all([isinstance(security, xr.Dataset) for security in datasets.values()])
        feeds = {str(security): dataset for security, dataset in datasets.items()}
        results = {security: calculation(feeds[str(security)], *args, **kwargs) for security, calculation in self.calculations.items()}
        yield current, ticker, expire, results


class SecurityProcessor(Processor):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, dataframes = contents
        assert isinstance(dataframes, dict)
        assert all([isinstance(security, pd.DataFrame) for security in dataframes.values()])
        dataframes = {security: self.filter(dataframe, *args, security=security, **kwargs) for security, dataframe in dataframes.items()}
        datasets = {security: self.parser(dataframe, *args, security=security, **kwargs) for security, dataframe in dataframes.items()}
        yield current, ticker, expire, datasets

    @staticmethod
    def filter(dataframe, *args, size=None, interest=None, **kwargs):
        dataframe = dataframe.where(dataframe["size"] >= size) if bool(size) else dataframe
        dataframe = dataframe.where(dataframe["interest"] >= interest) if bool(interest) else dataframe
        return dataframe

    @kwargsdispatcher("security")
    def parser(self, *args, security, **kwargs): raise ValueError(str(security))

    @parser.register.value(Securities.Stock.Long, Securities.Stock.Short)
    def stock(self, dataframe, *args, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["ticker", "date"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["ticker", "date"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe[["price", "size", "volume"]])
        return dataset

    @parser.register.value(Securities.Option.Put.Long, Securities.Option.Put.Short, Securities.Option.Call.Long, Securities.Option.Call.Short)
    def option(self, dataframe, *args, security, **kwargs):
        index = str(security) + "|strike"
        dataframe = dataframe.drop_duplicates(subset=["ticker", "date", "expire", "strike"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["ticker", "date", "expire", "strike"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe[["price", "size", "volume", "interest"]])
        dataset = dataset.rename({"strike": index})
        dataset["strike"] = dataset[index].expand_dims(["ticker", "date", "expire"])
        return dataset


class SecuritySaver(Saver):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, dataframes = contents
        assert isinstance(dataframes, dict)
        assert all([isinstance(security, pd.DataFrame) for security in dataframes.values()])
        current_folder = os.path.join(self.repository, str(current.strftime("%Y%m%d_%H%M%S")))
        if not os.path.isdir(current_folder):
            os.mkdir(current_folder)
        ticker_expire_name = "_".join([str(ticker), str(expire.strftime("%Y%m%d"))])
        ticker_expire_folder = os.path.join(current_folder, ticker_expire_name)
        assert not os.path.isdir(ticker_expire_folder)
        if not os.path.isdir(ticker_expire_folder):
            os.mkdir(ticker_expire_folder)
        for security, dataframe in dataframes.items():
            filename = str(security).replace("|", "_") + ".csv"
            file = os.path.join(ticker_expire_folder, filename)
            self.write(dataframe, file=file, mode="w")


class SecurityLoader(Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Columns = ntuple("Columns", "datetypes datatypes")
        stocks = Columns(["date"], {"ticker": str, "security": np.int32, "price": np.float32, "size": np.float32})
        options = Columns(["date", "expire"], {"ticker": str, "security": np.int32, "price": np.float32, "size": np.float32, "strike": np.float32, "interest": np.int32})
        self.columns = {Securities.Stock.Long: stocks, Securities.Stock.Short: stocks}
        self.columns.update({Securities.Option.Put.Long: options, Securities.Option.Put.Short: options})
        self.columns.update({Securities.Option.Call.Long: options, Securities.Option.Call.Short: options})

    def execute(self, ticker, *args, expires, **kwargs):
        TickerExpire = ntuple("TickerExpire", "ticker expire")
        for current_name in os.listdir(self.repository):
            current = Datetime.strptime(os.path.splitext(current_name)[0], "%Y%m%d_%H%M%S")
            current_folder = os.path.join(self.repository, current_name)
            for ticker_expire_name in os.listdir(current_folder):
                ticker_expire = TickerExpire(*str(ticker_expire_name).split("_"))
                if ticker != str(ticker_expire.ticker).upper():
                    continue
                expire = Datetime.strptime(os.path.splitext(ticker_expire.expire)[0], "%Y%m%d").date()
                if expire not in expires:
                    continue
                folder = os.path.join(current_folder, ticker_expire_name)
                security = lambda filename: Securities[str(filename).split(".")[0].replace("_", "|")]
                datatypes = lambda filename: self.columns[security(filename)].datatypes
                datetypes = lambda filename: self.columns[security(filename)].datetypes
                file = lambda filename: os.path.join(folder, filename)
                reader = lambda filename: self.read(file=file(filename), datatypes=datatypes(filename), datetypes=datetypes(filename))
                dataframes = {security(filename): reader(filename) for filename in os.listdir(folder)}
                yield current, ticker, expire, dataframes



