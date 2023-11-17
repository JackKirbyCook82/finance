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

class Securities(type):
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


class PositionCalculation(Calculation, ABC): pass
class InstrumentCalculation(Calculation, ABC): pass
class LongCalculation(PositionCalculation, ABC): pass
class ShortCalculation(PositionCalculation, ABC): pass

class StockCalculation(InstrumentCalculation):
    Λ = source("Λ", "stock", position=0, variables={"to": "date", "w": "price", "q": "size"})

    def execute(self, *args, **kwargs):
        yield self["Λ"].w(*args, **kwargs)

class OptionCalculation(InstrumentCalculation):
    Λ = source("Λ", "option", position=0, variables={"to": "date", "w": "price", "q": "size", "tτ": "expire", "k": "strike", "i": "interest"})
    τ = equation("τ", "tau", np.int16, domain=("o.to", "o.tτ"), function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))

    def execute(self, dataset, *args, **kwargs):
        yield self["Λ"].w(*args, **kwargs)
        yield self.τ(*args, **kwargs)

class PutCalculation(OptionCalculation): pass
class CallCalculation(OptionCalculation): pass
class StockLongCalculation(StockCalculation, LongCalculation): pass
class StockShortCalculation(StockCalculation, ShortCalculation): pass
class PutLongCalculation(PutCalculation, LongCalculation): pass
class PutShortCalculation(PutCalculation, ShortCalculation): pass
class CallLongCalculation(CallCalculation, LongCalculation): pass
class CallShortCalculation(CallCalculation, ShortCalculation): pass

class Calculations:
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


calculations = {Securities.Stock.Long: Calculations.Stock.Long, Securities.Stock.Short: Calculations.Stock.Short}
calculations.update({Securities.Option.Put.Long: Calculations.Option.Put.Long, Securities.Option.Put.Short: Calculations.Option.Put.Short})
calculations.update({Securities.Option.Call.Long: Calculations.Option.Call.Long, Securities.Option.Call.Short: Calculations.Option.Call.Short})
class SecurityCalculator(Calculator, calculations=calculations):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, datasets = contents
        assert isinstance(datasets, dict)
        assert all([isinstance(security, xr.Dataset) for security in datasets.values()])
        results = {security: self.calculations[security](dataset, *args, **kwargs) for security, dataset in datasets.items()}
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

    @parser.register(Securities.Stock.Long, Securities.Stock.Short)
    def stock(self, dataframe, *args, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["ticker", "date"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["ticker", "date"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe)
        return dataset

    @parser.register(Securities.Option.Put.Long, Securities.Option.Put.Short, Securities.Option.Call.Long, Securities.Option.Call.Short)
    def option(self, dataframe, *args, security, partition=None, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["ticker", "date", "expire", "strike"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["ticker", "date", "expire", "strike"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe)
        dataset = dataset.rename({"strike": str(security)})
        dataset["strike"] = dataset[str(security)]
        dataset = dataset.chunk({str(security): partition}) if bool(partition) else dataset
        return dataset


class SecuritySaver(Saver):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, dataframes = contents
        assert isinstance(dataframes, dict)
        assert all([isinstance(security, pd.DataFrame) for security in dataframes.values()])
        current_folder = os.path.join(self.repository, str(expire.strftime("%Y%m%d_%H%M%S")))
        assert not os.path.isdir(current_folder)
        ticker_expire_name = "_".join([str(ticker), str(expire.strftime("%Y%m%d"))])
        ticker_expire_folder = os.path.join(current_folder, ticker_expire_name)
        if not os.path.isdir(ticker_expire_folder):
            os.mkdir(ticker_expire_folder)
        for security, dataframe in dataframes.items():
            filename = str(security).replace("|", "_") + ".csv"
            file = os.path.join(ticker_expire_folder, filename)
            dataframe = dataframe.reset_index(drop=False, inplace=False)
            self.write(dataframe, file=file, mode="w")


class SecurityLoader(Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Columns = ntuple("Columns", "datetypes datatypes")
        stock = Columns(["date", "time"], {"ticker": str, "security": np.int32, "price": np.float32, "size": np.float32})
        option = Columns[stock.datetypes + ["expire"], stock.datatypes | {"strike": np.float32, "interest": np.int32}]
        self.columns = {Securities.Stock.Long: stock, Securities.Stock.Short: stock}
        self.columns.update({Securities.Option.Put.Long: option, Securities.Option.Put.Short: option})
        self.columns.update({Securities.Option.Call.Long: option, Securities.Option.Call.Short: option})

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



