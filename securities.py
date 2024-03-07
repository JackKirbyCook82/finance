# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime as Datetime
from collections import OrderedDict as ODict

from support.files import DataframeFile, Header
from support.processes import Saver, Loader, Filter, Parser, Axes

from finance.variables import Query, Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFilter", "SecurityParser", "SecurityLoader", "SecuritySaver", "SecurityFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


INDEX = {"strike": np.float32}
COLUMNS = {"instrument": str, "position": str}
SCOPE = {"ticker": str, "expire": np.datetime64, "date": np.datetime64}
VALUES = {"price": np.float32, "underlying": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32}
HEADER = Header(INDEX | COLUMNS, VALUES | SCOPE)
AXES = Axes(INDEX, COLUMNS, VALUES, SCOPE)


class SecurityFilter(Filter):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        assert isinstance(securities, pd.DataFrame)
        prior = self.size(securities["size"])
        mask = self.mask(securities, *args, **kwargs)
        securities = self.filter(securities, *args, mask=mask, **kwargs)
        post = self.size(securities["size"])
        query = query(securities=securities)
        __logger__.info(f"Filter: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]")
        yield query


class SecurityParser(Parser, axes=AXES):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        assert isinstance(securities, pd.DataFrame)
        securities = self.unflatten(securities, *args, **kwargs)
        securities = self.parse(securities, *args, **kwargs)
        securities = ODict(list(securities))
        query = query(securities=securities)
        yield query

    @staticmethod
    def parse(datasets, *args, **kwargs):
        datasets = datasets.squeeze("date").squeeze("ticker").squeeze("expire")
        for instrument, position in product(datasets["instrument"].values, datasets["position"].values):
            key = f"{instrument}|{position}"
            dataset = datasets.sel({"instrument": instrument, "position": position})
            dataset = dataset.rename({"strike": key})
            dataset["strike"] = dataset[key]
            yield key, dataset


class SecurityFile(DataframeFile, header=HEADER): pass
class SecuritySaver(Saver):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        assert isinstance(securities, pd.DataFrame)
        if bool(securities.empty):
            return
        ticker = str(query.contract.ticker)
        expire = str(query.contract.expire.strftime("%Y%m%d"))
        securities = self.parse(securities, *args, **kwargs)
        folder = "_".join([ticker, expire])
        files = {"securities.csv": securities}
        self.write(folder=folder, files=files, mode="w")


class SecurityLoader(Loader):
    def execute(self, *args, **kwargs):
        folders = self.contents(folder=None)
        files = ["securities.csv"]
        for folder, contents in self.reader(folders=folders, files=files):
            ticker, expire = str(folder).split("_")
            ticker = str(ticker).upper()
            expire = Datetime.strptime(expire, "%Y%m%d")
            contract = Contract(ticker, expire)
            yield Query(contract, **contents)



