# -*- coding: utf-8 -*-
"""
Created on Thurs Fed 1 202
@name:   Portfolio Objects
@author: Jack Kirby Cook

"""

import os
import logging
import numpy as np
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.pipelines import Producer, Processor, Consumer
from support.files import DataframeFile

from finance.variables import Contract, Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["PortfolioLoader", "PortfolioSaver", "PortfolioCalculator", "PortfolioFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class PortfolioQuery(ntuple("Query", "inquiry contract options")):
    def __str__(self):
        pass


class PortfolioCalculator(Processor):
    def execute(self, query, *args, **kwargs):
        pass

#    def groupings(self, indexes):
#        if not indexes:
#            yield []
#            return
#        for group in (((indexes[0],) + pairs) for pairs in combinations(indexes[1:], 1)):
#            for groups in self.groupings([pair for pair in indexes if pair not in group]):
#                yield [group] + groups


class PortfolioSaver(Consumer, title="Saved"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, PortfolioFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, query, *args, **kwargs):
        options = query.options
        if not bool(options) or all([dataframe.empty for dataframe in options.values()]):
            return
        inquiry_name = str(query.inquiry.strftime("%Y%m%d_%H%M%S"))
        inquiry_folder = self.file.path(inquiry_name)
        if not os.path.isdir(inquiry_folder):
            os.mkdir(inquiry_folder)
        contract_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        contract_folder = self.file.path(inquiry_name, contract_name)
        if not os.path.isdir(contract_folder):
            os.mkdir(contract_folder)
        for security, dataframe in options.items():
            security_name = str(security).replace("|", "_") + ".csv"
            security_file = self.file.path(inquiry_name, contract_name, security_name)
            self.file.write(dataframe, file=security_file, data=security, mode="w")
            LOGGER.info("Saved: {}[{}]".format(repr(self), str(security_file)))


class PortfolioLoader(Producer, title="Loaded"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, PortfolioFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, *args, **kwargs):
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        inquiry_folders = list(self.file.directory())
        for inquiry_name in sorted(inquiry_folders, key=function, reverse=False):
            inquiry = function(inquiry_name)
            contract_folders = list(self.file.directory(inquiry_name))
            for contract_name in contract_folders:
                contract = Contract(*str(contract_name).split("_"))
                ticker = str(contract.ticker).upper()
                expire = Datetime.strptime(os.path.splitext(contract.expire)[0], "%Y%m%d").date()
                filenames = {security: str(security).replace("|", "_") + ".csv" for security in list(Securities.Options)}
                files = {security: self.file.path(inquiry_name, contract_name, filename) for security, filename in filenames.items()}
                options = {security: self.file.read(file=file) for security, file in files.items()}
                contract = Contract(ticker, expire)
                yield PortfolioQuery(inquiry, contract, options)


class PortfolioFile(DataframeFile):
    @staticmethod
    def dataheader(*args, **kwargs): return ["date", "ticker", "expire", "strike", "price", "size", "volume", "interest", "quantity", "paid", "cost"]
    @staticmethod
    def datatypes(*args, **kwargs): return {"strike": np.float32, "price": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32, "quantity": np.int16, "paid": np.float32, "cost": np.float32}
    @staticmethod
    def datetypes(*args, **kwargs): return ["date", "expire"]



