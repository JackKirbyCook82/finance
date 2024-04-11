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

from support.pipelines import Producer, Processor, Consumer
from support.processes import Loader, Saver, Filter
from support.files import Files

from finance.variables import Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFilter", "SecurityLoader", "SecuritySaver", "StockFile", "OptionFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


StockIndex = {"instrument": str, "position": str, "ticker": str, "date": np.datetime64}
StockColumns = {"price": np.float32, "size": np.float32, "volume": np.float32}
OptionIndex = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
OptionColumns = {"price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}

class StockFile(Files.Dataframe, variable="stocks", index=StockIndex, columns=StockColumns): pass
class OptionFile(Files.Dataframe, variable="options", index=OptionIndex, columns=OptionColumns): pass


class SecurityFilter(Filter, Processor, title="Filtered"):
    def execute(self, query, *args, **kwargs):
        assert isinstance(query, dict)
        contract, options = query["contract"], query["options"]
        assert isinstance(options, pd.DataFrame)
        prior = self.size(options["size"])
        options = self.filter(options, *args, **kwargs)
        post = self.size(options["size"])
        __logger__.info(f"Filter: {repr(self)}|{str(contract)}[{prior:.0f}|{post:.0f}]")
        yield query | dict(options=options)


class SecurityLoader(Loader, Producer, files=["stocks", "options"], title="Loaded"):
    def execute(self, *args, **kwargs):
        for folder in self.directory:
            ticker, expire = str(folder).split("_")
            ticker = str(ticker).upper()
            expire = Datetime.strptime(expire, "%Y%m%d")
            contract = Contract(ticker, expire)
            query = dict(contract=contract)
            contents = {file: self.read(*args, folder=folder, file=file, **kwargs) for file in self.files}
            yield query | contents


class SecuritySaver(Saver, Consumer, files=["stocks", "options"], title="Saved"):
    def execute(self, query, *args, **kwargs):
        assert isinstance(query, dict)
        ticker = str(query["contract"].ticker)
        expire = str(query["contract"].expire.strftime("%Y%m%d"))
        folder = "_".join([ticker, expire])
        for file in self.files:
            self.write(query[file], *args, folder=folder, file=file, **kwargs)



