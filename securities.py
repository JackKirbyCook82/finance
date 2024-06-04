# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Contract
from support.filtering import Filter
from support.pipelines import Header
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFiles", "SecurityHeaders", "SecurityFilter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


stocks_index = {"instrument": str, "position": str, "ticker": str, "date": np.datetime64}
stocks_columns = {"price": np.float32, "size": np.float32, "volume": np.float32}
options_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
options_columns = {"price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}


class StockFile(File, variable="stocks", query=("contract", Contract), datatype=pd.DataFrame, header=stocks_index | stocks_columns): pass
class OptionFile(File, variable="options", query=("contract", Contract), datatype=pd.DataFrame, header=options_index | options_columns): pass
class StockHeader(Header, variable="stocks", axes={"index": stocks_index, "columns": stocks_columns}): pass
class OptionHeader(Header, variable="options", axes={"index": options_index, "columns": options_columns}): pass
class SecurityFilter(Filter, variables=["stocks", "options"], query="contract"): pass


class SecurityFiles(object):
    STOCKS = StockFile
    OPTIONS = OptionFile

class SecurityHeaders(object):
    STOCKS = StockHeader
    OPTIONS = OptionHeader



