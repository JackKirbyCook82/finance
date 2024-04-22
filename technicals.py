# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np

from support.queues import Queues
from support.files import Files

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HistoryFile", "HistoryQueue"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


history_index = {"ticker": str, "date": np.datetime64}
history_columns = {"high": np.float32, "low": np.float32, "open": np.float32, "close": np.float32, "price": np.float32, "price": np.float32, "volume": np.float32}


class HistoryFile(Files.Dataframe, variable="historicals", index=history_index, columns=history_columns): pass
class HistoryQueue(Queues.FIFO, variable="contract"): pass



