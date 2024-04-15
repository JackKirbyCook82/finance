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
__all__ = ["SecurityLoader", "SecuritySaver", "SecurityFilter", "SecurityFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


INDEX = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
COLUMNS = {"price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
QUERY = lambda folder: {"contract": Contract.fromstring(folder)}
FOLDER = lambda query: query["contract"].tostring()


class SecurityFile(Files.Dataframe, variable="securities", index=INDEX, columns=COLUMNS): pass
class SecurityLoader(Loader, Producer, query=QUERY, title="Loaded"): pass
class SecuritySaver(Saver, Consumer, folder=FOLDER, title="Saved"): pass


class SecurityFilter(Filter, Processor, title="Filtered"):
    def execute(self, query, *args, **kwargs):
        assert isinstance(query, dict)
        contract, securities = query["contract"], query["securities"]
        assert isinstance(securities, pd.DataFrame)
        prior = self.size(securities["size"])
        securities = self.filter(securities, *args, **kwargs)
        post = self.size(securities["size"])
        __logger__.info(f"Filter: {repr(self)}|{str(contract)}[{prior:.0f}|{post:.0f}]")
        yield query | dict(options=securities)



