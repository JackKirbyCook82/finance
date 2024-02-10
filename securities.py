# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import os
import logging
import numpy as np
import xarray as xr
from datetime import datetime as Datetime

from support.files import DataframeFile
from support.pipelines import Producer, Processor, Consumer

from finance.variables import Query, Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityQuery", "SecurityLoader", "SecuritySaver", "SecurityFilter", "SecurityParser", "SecurityFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class SecurityQuery(Query): pass
class SecurityFilter(Processor, title="Filtered"):
    def execute(self, query, *args, size=0, interest=0, volume=0, **kwargs):
        options = query.contents
        options = options.where(options["size"] >= size) if bool(size) else options
        options = options.where(options["interest"] >= interest) if bool(interest) else options
        options = options.where(options["volume"] >= volume) if bool(volume) else options
        options = options.dropna(axis=0, how="all")
        options = options.reset_index(drop=True, inplace=False)
        query = SecurityQuery(query.inquiry, query.contract, options)
        LOGGER.info(f"Filter: {repr(self)}[{str(query)}]")
        yield query


class SecurityParser(Processor, title="Parsed"):
    def execute(self, query, *args, **kwargs):
        options = query.contents
        index = ["date", "ticker", "expire", "strike"]
        options = options.drop_duplicates(subset=["security", "date", "ticker", "expire", "strike"], keep="last", inplace=False)
        options = {security: dataframe.drop("security", axis=1, inplace=False) for security, dataframe in iter(options.groupby("security"))}
        options = {security: dataframe.set_index(index, inplace=False, drop=True) for security, dataframe in options.items()}
        options = {security: xr.Dataset.from_dataframe(dataframe) for security, dataframe in options.items()}
        yield SecurityQuery(query.inquiry, query.contract, options)


class SecuritySaver(Consumer, title="Saved"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, SecurityFile)
        super().__init__(*args, **kwargs)
        assert isinstance(file, SecurityFile)
        self.file = file

    def execute(self, query, *args, **kwargs):
        options = query.contents
        inquiry_name = str(query.inquiry.strftime("%Y%m%d_%H%M%S"))
        inquiry_folder = self.file.path(inquiry_name)
        if not os.path.isdir(inquiry_folder):
            os.mkdir(inquiry_folder)
        contract_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        contract_file = self.file.path(inquiry_name, contract_name + ".csv")
        self.file.write(options, file=contract_file, mode="w")
        LOGGER.info("Saved: {}[{}]".format(repr(self), str(contract_file)))


class SecurityLoader(Producer, title="Loaded"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, SecurityFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, *args, tickers=None, expires=None, **kwargs):
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        inquiry_folders = list(self.file.directory())
        for inquiry_name in sorted(inquiry_folders, key=function, reverse=False):
            inquiry = function(inquiry_name)
            contract_filenames = list(self.file.directory(inquiry_name))
            for contract_filename in contract_filenames:
                contract_name = str(contract_filename).split(".")[0]
                contract = Contract(*str(contract_name).split("_"))
                ticker = str(contract.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(contract.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                contract_file = self.file.path(inquiry_name, contract_name + ".csv")
                options = self.file.read(file=contract_file)
                yield SecurityQuery(inquiry, contract, options)


class SecurityFile(DataframeFile):
    def dataheader(self, *args, data, **kwargs): return ["security", "ticker", "expire", "strike", "date", "quantity", "price", "underlying", "entry", "size", "volume", "interest"]
    def datatypes(self, *args, data, **kwargs): return {"strike": np.float322, "quantity": np.int32, "price": np.float3, "underlying": np.float32, "entry": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32}
    def datetypes(self, *args, data, **kwargs): return ["expire", "date"]



