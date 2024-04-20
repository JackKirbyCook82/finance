# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np

from support.pipelines import Producer, Consumer
from support.processes import Loader, Saver
from support.files import Files

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalLoader", "TechnicalSaver", "TechnicalFile"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


technical_index = {"ticker": str, "date": np.datetime64}
technical_columns = {"price": np.float32, "volume": np.float32}
query_function = lambda folder: {"ticker": str(folder).upper()}
folder_function = lambda query: str(query["ticker"]).upper()


class TechnicalFile(Files.Dataframe, variable="technicals", index=technical_index, columns=technical_columns): pass
class TechnicalSaver(Saver, Consumer, folder=folder_function, title="Saved"): pass
class TechnicalLoader(Loader, Producer, query=query_function, title="Loaded"): pass



