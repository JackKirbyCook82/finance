# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np

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


security_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
security_columns = {"price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
query_function = lambda folder: {"contract": Contract.fromstring(folder)}
folder_function = lambda query: query["contract"].tostring()


class SecurityFile(Files.Dataframe, variable="securities", index=security_index, columns=security_columns): pass
class SecuritySaver(Saver, Consumer, folder=folder_function, title="Saved"): pass
class SecurityLoader(Loader, Producer, query=query_function, title="Loaded"): pass


class SecurityFilter(Filter, Processor, title="Filtered"):
    def execute(self, query, *args, **kwargs):
        assert isinstance(query, dict)
        contract, securities = query["contract"], query["securities"]
        if self.empty(securities["size"]):
            return
        prior = self.size(securities["size"])
        mask = self.mask(securities)
        securities = self.where(securities, mask)
        post = self.size(securities["size"])
        __logger__.info(f"Filter: {repr(self)}|{str(contract)}[{prior:.0f}|{post:.0f}]")
        yield query | dict(options=securities)



