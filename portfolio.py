# -*- coding: utf-8 -*-
"""
Created on Thurs Fed 1 202
@name:   Portfolio Objects
@author: Jack Kirby Cook

"""

import os
import logging
import numpy as np
from collections import namedtuple as ntuple

from support.pipelines import Producer, Processor, Consumer
from support.files import DataframeFile

from finance.variables import Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["PortfolioLoader", "PortfolioSaver", "PortfolioCalculator", "PortfolioFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class PortfolioQuery(ntuple("Query", "inquiry balances portfolio")):
    def __str__(self):
        pass


class PortfolioCalculator(Processor):
    def execute(self, query, *args, **kwargs):
        portfolio = {Contract(ticker, expire): dataframe for (ticker, expire), dataframe in iter(query.portfolio.groupby(["ticker", "expire"]))}


class PortfolioSaver(Consumer, title="Saved"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, PortfolioFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, query, *args, **kwargs):
        portfolio = query.portfolio
        if bool(portfolio.empty):
            return
        file = self.path("portfolio.csv")
        self.file.write(portfolio, file=file, mode="w")
        LOGGER.info("Saved: {}[{}]".format(repr(self), str(file)))


class PortfolioLoader(Producer, title="Loaded"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, PortfolioFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, *args, **kwargs):
        file = self.path("portfolio.csv")
        if not os.path.isfile(file):
            return
        portfolio = self.read(file=file)
        yield portfolio


class PortfolioFile(DataframeFile):
    def dataheader(self, *args, **kwargs): return ["security", "date", "ticker", "expire", "strike", "price", "size", "volume", "interest", "acquired", "quantity", "paid", "cost"]
    def datatypes(self, *args, **kwargs): return {"strike": np.float32, "price": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32, "quantity": np.int16, "paid": np.float32, "cost": np.float32}
    def datetypes(self, *args, **kwargs): return ["date", "expire", "acquired"]



