# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Divestiture Objects
@author: Jack Kirby Cook

"""

import logging
from itertools import product

from support.tables import Tables, Options

from finance.holdings import HoldingReader, HoldingWriter, HoldingStatus
from finance.variables import Scenarios

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureReader", "DivestitureWriter", "DivestitureTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


divestiture_formats = {(lead, lag): lambda column: f"{column:.02f}" for lead, lag in product(["npv", "cost"], list(map(lambda scenario: str(scenario.name).lower(), Scenarios)))}
divestiture_formats.update({(lead, lag): lambda column: f"{column * 100:.02f}%" for lead, lag in product(["apy"], list(map(lambda scenario: str(scenario.name).lower(), Scenarios)))})
divestiture_formats.update({("priority", ""): lambda column: f"{column * 100:.02f}"})
divestiture_formats.update({("status", ""): lambda column: str(HoldingStatus(int(column)).name).lower()})
divestiture_options = Options.Dataframe(rows=20, columns=25, width=1000, formats=divestiture_formats, numbers=lambda column: f"{column:.02f}")


class DivestitureTable(Tables.Dataframe, variable="divestitures", options=divestiture_options): pass
class DivestitureReader(HoldingReader, variable="divestitures"): pass
class DivestitureWriter(HoldingWriter, variable="divestitures"):
    def execute(self, *args, **kwargs):
        pass



