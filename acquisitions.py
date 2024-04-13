# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Acquisition Objects
@author: Jack Kirby Cook

"""

import logging
import pandas as pd
from itertools import product

from support.tables import Tables, Options

from finance.holdings import HoldingReader, HoldingWriter, HoldingStatus
from finance.variables import Scenarios

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionReader", "AcquisitionWriter", "AcquisitionTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


acquisition_formats = {(lead, lag): lambda column: f"{column:.02f}" for lead, lag in product(["npv", "cost"], list(map(lambda scenario: str(scenario.name).lower(), Scenarios)))}
acquisition_formats.update({(lead, lag): lambda column: f"{column * 100:.02f}%" for lead, lag in product(["apy"], list(map(lambda scenario: str(scenario.name).lower(), Scenarios)))})
acquisition_formats.update({("priority", ""): lambda column: f"{column * 100:.02f}"})
acquisition_formats.update({("status", ""): lambda column: str(HoldingStatus(int(column)).name).lower()})
acquisition_options = Options.Dataframe(rows=20, columns=25, width=1000, formats=acquisition_formats, numbers=lambda column: f"{column:.02f}")


class AcquisitionTable(Tables.Dataframe, variable="acquisitions", options=acquisition_options): pass
class AcquisitionReader(HoldingReader, variable="acquisitions"): pass
class AcquisitionWriter(HoldingWriter, variable="acquisitions"):
    def execute(self, query, *args, **kwargs):
        valuations = query["valuations"]
        assert isinstance(valuations, pd.DataFrame)
        valuations = self.market(valuations, *args, **kwargs)
        valuations = self.prioritize(valuations, *args, **kwargs)
        if bool(valuations.empty):
            return
        self.write(valuations, *args, **kwargs)



