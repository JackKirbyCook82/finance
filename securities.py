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
from datetime import date as Date
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.pipelines import Saver, Loader

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "HistorySaver", "HistoryLoader", "OptionSaver", "OptionLoader"]
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


class HistorySaver(Saver):
    def execute(self, contents, *args, **kwargs):
        ticker, dataframe = contents
        assert isinstance(dataframe, pd.DataFrame)
        file = str(ticker).upper()
        file = os.path.join(self.repository, str(file) + ".csv")
        self.write(dataframe, file=file, mode="w")


class HistoryLoader(Loader):
    def execute(self, ticker, *args, **kwargs):
        file = str(ticker).upper()
        file = os.path.join(self.repository, str(file) + ".csv")
        datatypes = {"ticker": str, "open": np.float32, "close": np.float32, "high": np.float32, "low": np.float32, "price": np.float32, "volume": np.int64}
        dataframe = self.read(pd.DataFrame, file=file, datatypes=datatypes, datetypes=["date"])
        yield ticker, dataframe


class OptionSaver(Saver):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, dataset = contents
        assert isinstance(dataset, xr.Dataset)
        folder = os.path.join(self.repository, str(ticker).upper())
        if not os.path.isdir(folder):
            os.mkdir(folder)
        file = str(expire.strftime("%Y%m%d"))
        file = os.path.join(folder, str(file) + ".nc")
        self.write(dataset, file=file, mode="w")


class OptionLoader(Loader):
    def execute(self, ticker, *args, **kwargs):
        folder = str(ticker).upper()
        directory = os.path.join(self.repository, folder)
        for file in os.listdir(directory):
            file = os.path.join(directory, file)
            expire = str(os.path.split(file)[1]).split(".")[0]
            expire = Datetime.strptime(expire, "%Y%m%d").date()
            dataset = self.read(xr.Dataset, file=file)
            yield ticker, expire, dataset



