# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Historical Objects
@author: Jack Kirby Cook

"""

import numpy as np

from support.pipelines import Producer, Consumer
from support.processes import Loader, Saver
from support.files import Files

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HistoryLoader", "HistorySaver", "HistoryFile"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


history_index = {"ticker": str, "date": np.datetime64}
history_columns = {"price": np.float32, "volume": np.float32}
query_function = lambda folder: {"ticker": str(folder).upper()}
folder_function = lambda query: str(query["ticker"]).upper()


class HistoryFile(Files.Dataframe, variable="historicals", index=history_index, columns=history_columns): pass
class HistorySaver(Saver, Consumer, folder=folder_function, title="Saved"): pass
class HistoryLoader(Loader, Producer, query=query_function, title="Loaded"): pass



