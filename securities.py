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

from support.processes import Loader, Saver, Filter
from support.pipelines import Producer, Processor, Consumer
from support.files import DataframeFile

from finance.variables import Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFilter", "SecurityLoader", "SecuritySaver", "SecurityFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


INDEX = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
SECURITIES = {"price": np.float32, "underlying": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32}
HOLDINGS = {"quantity": np.int32}


class SecurityFile(DataframeFile, header=INDEX | SECURITIES, filename="security.csv"): pass
class HoldingFile(DataframeFile, header=INDEX | HOLDINGS, filename="holding.csv"): pass


class SecurityFilter(Filter, Processor, title="Filtered"):
    def execute(self, query, *args, **kwargs):
        assert isinstance(query, dict)
        contract = query["contract"]
        securities = query["security"]
        assert isinstance(securities, pd.DataFrame)
        prior = self.size(securities["size"])
        mask = self.mask(securities, *args, **kwargs)
        securities = self.filter(securities, *args, mask=mask, **kwargs)
        post = self.size(securities["size"])
        __logger__.info(f"Filter: {repr(self)}|{str(contract)}[{prior:.0f}|{post:.0f}]")
        yield query | dict(security=securities)


class SecurityLoader(Loader, Producer, title="Loaded"):
    def execute(self, *args, **kwargs):
        for folder, contents in self.reader(*args, **kwargs):
            assert all([isinstance(value, pd.DataFrame) for value in contents.values()])
            contents = {key: value for key, value in contents.items() if not value.empty}
            if not bool(contents):
                continue
            ticker, expire = str(folder).split("_")
            ticker = str(ticker).upper()
            expire = Datetime.strptime(expire, "%Y%m%d")
            contract = Contract(ticker, expire)
            yield dict(contract=contract) | contents


class SecuritySaver(Saver, Consumer, title="Saved"):
    def execute(self, query, *args, **kwargs):
        assert isinstance(query, dict)
        contents = {key: value for key, value in query.items() if key != "contract"}
        assert all([isinstance(value, pd.DataFrame) for value in contents.values()])
        if not bool(contents):
            return
        ticker = str(query["contract"].ticker)
        expire = str(query["contract"].expire.strftime("%Y%m%d"))
        folder = "_".join([ticker, expire])
        self.write(contents, *args, folder=folder, mode="w", **kwargs)



