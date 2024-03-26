# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import os.path
import logging
import numpy as np
import pandas as pd
from datetime import datetime as Datetime
from collections import OrderedDict as ODict

from support.processes import Reader, Writer, Filter
from support.pipelines import Producer, Processor, Consumer
from support.files import DataframeFile
from support.queues import FIFOQueue

from finance.variables import Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFilter", "SecurityDequeue", "SecurityQueue", "SecurityLoader", "SecuritySaver", "SecurityFile", "SecuritySchedule"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


INDEX = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
VALUES = {"price": np.float32, "underlying": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32, "quantity": np.int32}


class SecurityFile(DataframeFile, variables=INDEX | VALUES): pass
class SecuritySchedule(FIFOQueue): pass


class SecurityFilter(Filter, Processor, title="Filtered"):
    def execute(self, query, *args, **kwargs):
        contract = query["contract"]
        securities = query["securities"]
        assert isinstance(securities, pd.DataFrame)
        prior = self.size(securities["size"])
        mask = self.mask(securities, *args, **kwargs)
        securities = self.filter(securities, *args, mask=mask, **kwargs)
        post = self.size(securities["size"])
        __logger__.info(f"Filter: {repr(self)}|{str(contract)}[{prior:.0f}|{post:.0f}]")
        yield query | dict(securities=securities)


class SecurityDequeue(Reader, Producer, title="Dequeued"):
    def execute(self, *args, **kwargs):
        contract = self.read(*args, **kwargs)
        assert isinstance(contract, Contract)
        yield dict(contract=contract)


class SecurityQueue(Writer, Consumer, title="Queued"):
    def execute(self, contract, *args, **kwargs):
        assert isinstance(contract, Contract)
        query = dict(contract=contract)
        self.write(query, *args, **kwargs)


class SecurityLoader(Reader, Producer, title="Loaded"):
    def execute(self, *args, **kwargs):
        for folder in self.directory:
            ticker, expire = str(folder).split("_")
            ticker = str(ticker).upper()
            expire = Datetime.strptime(expire, "%Y%m%d")
            contract = Contract(ticker, expire)
            files = {field: os.path.join(folder, str(field) + ".csv") for field in ("securities", "holdings")}
            contents = ODict([(name, self.read(file=file)) for name, file in files.items()])
            assert all([isinstance(content, (pd.DataFrame, type(None))) for content in contents.values()])
            contents = {name: content for name, content in contents.items() if content is not None}
            contents = {name: content for name, content in contents.items() if not content.empty}
            if not bool(contents):
                return
            yield dict(contract=contract, **contents)


class SecuritySaver(Writer, Consumer, title="Saved"):
    def execute(self, query, *args, **kwargs):
        ticker = str(query["contract"].ticker)
        expire = str(query["contract"].expire.strftime("%Y%m%d"))
        folder = "_".join([ticker, expire])
        contents = {field: query.get(field, None) for field in ("securities", "holdings")}
        assert all([isinstance(content, (pd.DataFrame, type(None))) for content in contents.values()])
        contents = {name: content for name, content in contents.items() if content is not None}
        contents = {name: content for name, content in contents.items() if not content.empty}
        if not bool(contents):
            return
        for name, content in contents.items():
            file = os.path.join(folder, str(name) + ".csv")
            self.write(content, file=file, mode="w")





