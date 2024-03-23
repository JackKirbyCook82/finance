# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime as Datetime
from collections import OrderedDict as ODict

from support.processes import Scheduler, Directory, Filter, Saver, Loader
from support.pipelines import Producer, Processor, Consumer
from support.files import DataframeFile

from finance.variables import Contract, Query

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityDirectory", "SecurityScheduler", "SecurityFilter", "SecurityLoader", "SecuritySaver", "SecurityFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


INDEX = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
VALUES = {"price": np.float32, "underlying": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32, "quantity": np.int32}


class SecurityFile(DataframeFile, variables=INDEX | VALUES): pass
class SecurityDirectory(Directory, Producer, title="Scheduled"):
    def execute(self, *args, **kwargs):
        for folder in self.directory:
            ticker, expire = folder.split("_")
            ticker, expire = str(ticker).upper(), Datetime.strptime(expire, "%Y%m%d")
            contract = Contract(ticker, expire)
            yield Query(contract, ["securities", "holdings"])


class SecurityScheduler(Scheduler, Producer, title="Scheduled"):
    def execute(self, *args, **kwargs):
        schedule = self.schedule(*args, **kwargs)
        for contents in schedule:
            contents = [contents.get(field, None) for field in ("ticker", "expire")]
            contract = Contract(*contents)
            yield Query(contract, ["securities", "holdings"])


class SecurityFilter(Filter, Processor, title="Filtered"):
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


class SecurityLoader(Loader, Processor, title="Loaded"):
    def execute(self, query, *args, **kwargs):
        ticker = str(query.contract.ticker)
        expire = str(query.contract.expire.strftime("%Y%m%d"))
        folder = "_".join([ticker, expire])
        files = {name: f"{name}.csv" for name in list(query.keys())}
        contents = {name: self.read(folder=folder, file=file) for name, file in files.items()}
        yield Query(contents)


class SecuritySaver(Saver, Consumer, title="Saved"):
    def execute(self, query, *args, **kwargs):
        ticker = str(query.contract.ticker)
        expire = str(query.contract.expire.strftime("%Y%m%d"))
        folder = "_".join([ticker, expire])
        contents = ODict(list(query.items()))
        assert all([isinstance(content, (pd.DataFrame, type(None))) for content in contents.values()])
        contents = ODict([(name, content) for name, content in contents.items() if content is not None])
        contents = ODict([(name, content) for name, content in contents.items() if not content.empty])
        if not bool(contents):
            return
        for name, content in contents.items():
            file = ".".join([name, "csv"])
            self.write(content, folder=folder, file=file, mode="w")



