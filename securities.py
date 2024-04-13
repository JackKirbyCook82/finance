# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from support.pipelines import Producer, Processor, Consumer
from support.processes import Loader, Saver, Filter
from support.files import Files

from finance.variables import Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityLoader", "SecuritySaver", "SecurityFilter", "StockFile", "OptionFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


stock_index = {"instrument": str, "position": str, "ticker": str, "date": np.datetime64}
stock_columns = {"price": np.float32, "size": np.float32, "volume": np.float32}
option_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
option_columns = {"price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
query_function = lambda folder: {"contract": Contract.fromstring(folder)}
folder_function = lambda query: query["contract"].tostring()


class StockFile(Files.Dataframe, variable="stocks", index=stock_index, columns=stock_columns): pass
class OptionFile(Files.Dataframe, variable="options", index=option_index, columns=option_columns): pass
class SecurityLoader(Loader, Producer, query=query_function, title="Loaded"): pass
class SecuritySaver(Saver, Consumer, folder=folder_function, title="Saved"): pass


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



