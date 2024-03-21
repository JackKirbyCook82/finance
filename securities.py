# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from support.processes import Scheduler, Filter, Saver, Loader
from support.pipelines import Processor, Consumer
from support.files import DataframeFile

from finance.variables import Query, Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityScheduler", "SecurityFilter", "SecurityLoader", "SecuritySaver", "SecurityFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


INDEX = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
VALUES = {"price": np.float32, "underlying": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32, "quantity": np.int32}


class SecurityFile(DataframeFile, variables=INDEX | VALUES): pass
class SecurityScheduler(Scheduler, Processor, variables=["ticker", "expire"]):
    def execute(self, *args, **kwargs):
        for contents in self.schedule(*args, **kwargs):
            contract = Contract(contents)
            yield Query(contract)


class SecurityFilter(Filter, Processor):
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


class SecurityLoader(Loader, Processor):
    def execute(self, query, *args, **kwargs):
        pass


class SecuritySaver(Saver, Consumer):
    def execute(self, query, *args, **kwargs):
        pass


# class SecurityLoader(Loader, Processor):
#     def execute(self, query, *args, **kwargs):
#         folders = self.contents(folder=None)
#         files = ["securities.csv"]
#         reader = self.reader(folders=folders, files=files)
#         for folder, contents in iter(reader):
#             ticker, expire = str(folder).split("_")
#             ticker = str(ticker).upper()
#             expire = Datetime.strptime(expire, "%Y%m%d")
#             contract = Contract(ticker, expire)
#             yield Query(contract, **contents)


# class SecuritySaver(Saver, Consumer):
#     def execute(self, query, *args, **kwargs):
#         securities = query.securities
#         assert isinstance(securities, pd.DataFrame)
#         if bool(securities.empty):
#             return
#         ticker = str(query.contract.ticker)
#         expire = str(query.contract.expire.strftime("%Y%m%d"))
#         folder = "_".join([ticker, expire])
#         files = {"securities.csv": securities}
#         self.write(folder=folder, files=files, mode="w")



