# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import os.path
import numpy as np
import xarray as xr
import pandas as pd
from enum import IntEnum
from datetime import date as Date
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.pipelines import Processor, Calculator, Saver, Loader
from support.calculations import Calculation, equation
from support.dispatchers import kwargsdispatcher

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

    def __repr__(self): return "{}({}, {})".format(self.__class__.__name__, repr(self.minimum), repr(self.maximum))
    def __str__(self): return "{}|{}".format(str(self.minimum), str(self.maximum))
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return (self.maximum - self.minimum).days
    def __contains__(self, date): return self.minimum <= date <= self.maximum


Instruments = IntEnum("Instrument", ["PUT", "CALL", "STOCK"], start=1)
Positions = IntEnum("Position", ["LONG", "SHORT"], start=1)
class Security(ntuple("Security", "instrument position")):
    def __new__(cls, instrument, position, *args, **kwargs): return super().__new__(cls, instrument, position)
    def __init__(self, *args, payoff, **kwargs): self.__payoff = payoff
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])

    @property
    def payoff(self): return self.__payoff

class Securities:
    class Stock:
        Long = Security(Instruments.STOCK, Positions.LONG, payoff=lambda x: np.copy(x))
        Short = Security(Instruments.STOCK, Positions.SHORT, payoff=lambda x: -np.copy(x))
    class Option:
        class Put:
            Long = Security(Instruments.PUT, Positions.LONG, payoff=lambda x, k: np.maximum(k - x, 0))
            Short = Security(Instruments.PUT, Positions.SHORT, payoff=lambda x, k: - np.maximum(k - x, 0))
        class Call:
            Long = Security(Instruments.CALL, Positions.LONG, payoff=lambda x, k: np.maximum(x - k, 0))
            Short = Security(Instruments.CALL, Positions.SHORT, payoff=lambda x, k: - np.maximum(x - k, 0))


class PositionCalculation(Calculation): pass
class LongCalculation(PositionCalculation): pass
class ShortCalculation(PositionCalculation): pass

class InstrumentCalculation(Calculation): pass
class StockCalculation(InstrumentCalculation, vars={"to": "date", "w": "price", "x": "time", "q": "size"}): pass
class OptionCalculation(InstrumentCalculation, vars={"to": "date", "w": "price", "x": "time", "q": "size", "tτ": "expire", "k": "strike", "i": "interest"}):
    τ = equation("τ", "tau", np.int16, domain=("0.to", "0.tτ"), function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))

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


class SecurityProcessor(Processor):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, dataframes = contents
        assert isinstance(dataframes, dict)
        assert all([isinstance(security, pd.DataFrame) for security in dataframes.values()])
        dataframes = {security: self.filter(dataframe, *args, security=security, **kwargs) for security, dataframe in dataframes.items()}
        datasets = {security: self.parser(dataframe, *args, security=security, **kwargs) for security, dataframe in dataframes.items()}
        yield ticker, expire, datasets

    @staticmethod
    def filter(dataframe, *args, size=None, interest=None, **kwargs):
        dataframe = dataframe.where(dataframe["size"] >= size) if bool(size) else dataframe
        dataframe = dataframe.where(dataframe["interest"] >= interest) if bool(interest) else dataframe
        return dataframe

    @kwargsdispatcher("security")
    def parser(self, *args, security, **kwargs): raise ValueError(str(security))

    @parser.register(Securities.Stock.Long, Securities.Stock.Short)
    def stock(self, dataframe, *args, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["ticker"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["date", "ticker"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe).squeeze("ticker")
        return dataset

    @parser.register(Securities.Option.Put.Long, Securities.Option.Put.Short, Securities.Option.Call.Long, Securities.Option.Call.Short)
    def option(self, dataframe, *args, security, partition=None, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["ticker", "expire", "strike"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["date", "ticker", "expire", "strike"], inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe).squeeze("ticker")
        dataset = dataset.rename({"strike": str(security)})
        dataset["strike"] = dataset[str(security)]
        dataset = dataset.chunk({str(security): partition}) if bool(partition) else dataset
        return dataset


calculations = {Securities.Stock.Long: Calculations.Stock.Long, Securities.Stock.Short: Calculations.Stock.Short}
calculations.update({Securities.Option.Put.Long: Calculations.Option.Put.Long, Securities.Option.Put.Short: Calculations.Option.Put.Short})
calculations.update({Securities.Option.Call.Long: Calculations.Option.Call.Long, Securities.Option.Call.Short: Calculations.Option.Call.Short})
class SecurityCalculator(Calculator, calculations=calculations):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, datasets = contents
        assert isinstance(datasets, dict)
        assert all([isinstance(security, xr.Dataset) for security in datasets.values()])
        securities = {security: self.calculations[security](dataset, *args, **kwargs) for security, dataset in datasets.items()}
        yield ticker, expire, securities


class SecuritySaver(Saver):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, dataframes = contents
        assert isinstance(dataframes, dict)
        assert all([isinstance(security, pd.DataFrame) for security in dataframes.values()])
        folder = os.path.join(self.repository, str(ticker))
        if not os.path.isdir(folder):
            os.mkdir(folder)
        folder = os.path.join(folder, str(expire.strftime("%Y%m%d")))
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for security, dataframe in dataframes.items():
            file = os.path.join(folder, str(security).replace("|", "_") + ".csv")
            dataframe = dataframe.reset_index(drop=False, inplace=False)
            self.write(dataframe, file=file, mode="w")


class SecurityLoader(Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Columns = ntuple("Columns", "datetypes datatypes")
        stock = Columns(["date", "time"], {"ticker": str, "price": np.float32, "size": np.float32})
        option = Columns[stock.datetypes + ["expire"], stock.datatypes | {"strike": np.float32, "interest": np.int32}]
        self.columns = {Securities.Stock.Long: stock, Securities.Stock.Short: stock}
        self.columns.update({Securities.Option.Put.Long: option, Securities.Option.Put.Short: option})
        self.columns.update({Securities.Option.Call.Long: option, Securities.Option.Call.Short: option})

    def execute(self, ticker, *args, expires, **kwargs):
        folder = os.path.join(self.repository, str(ticker))
        for foldername in os.listdir(folder):
            expire = Datetime.strptime(os.path.splitext(foldername)[0], "%Y%m%d").date()
            if expire not in expires:
                continue
            dataframes = {key: value for key, value in self.securities(ticker, expire)}
            yield ticker, expire, dataframes

    def securities(self, ticker, expire):
        folder = os.path.join(self.repository, str(ticker), str(expire.strftime("%Y%m%d")))
        for filename in os.listdir(folder):
            security = Securities[str(filename).split(".")[0].replace("_", "|")]
            file = os.path.join(folder, filename)
            dataframes = self.read(file=file, datatypes=self.columns[security].datatypes, datetypes=self.columns["datetypes"][security])
            yield security, dataframes



