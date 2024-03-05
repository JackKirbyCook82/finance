# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime as Datetime
from collections import OrderedDict as ODict

from support.files import DataframeFile, Header
from support.processes import Saver, Loader, Filter, Parser, Pivoter, Axes

from finance.variables import Query, Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFilter", "SecurityParser", "SecurityPivoter", "SecurityLoader", "SecuritySaver", "SecurityFile"]
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
        securities = self.parse(securities, *args, **kwargs)
        query = query(securities=securities)
        yield query


class SecurityPivoter(Pivoter, axes=AXES):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        securities = self.pivot(securities, *args, **kwargs)
        query = query(securities=securities)
        return query


class SecurityFile(DataframeFile, header=HEADER): pass
class SecuritySaver(Saver):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        assert isinstance(securities, pd.DataFrame)
        if bool(securities.empty):
            return
        ticker = str(query.contract.ticker)
        expire = str(query.contract.expire.strftime("%Y%m%d"))
        content = self.parse(securities, *args, **kwargs)
        folder = "_".join([ticker, expire])
        file = "security.csv"
        self.write(content, folder=folder, file=file, mode="w")


class SecurityLoader(Loader):
    def execute(self, *args, **kwargs):
        pass

#    def execute(self, *args, **kwargs):
#        contract_names = self.directory()
#        contracts = list(map(Contract, contract_names))
#        contracts = list(sorted(contracts))
#        for contract in contracts:
#            contract_name = "_".join([str(contract.ticker), str(contract.expire.strftime("%Y%m%d"))])
#            security_file = self.path(contract_name, "security.csv")
#            securities = self.read(file=security_file)
#            securities = self.parse(securities, *args, **kwargs)
#            yield Query(contract, securities=securities)



