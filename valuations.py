# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np

from support.calculations import Calculation
from support.processes import Calculator
from support.pipelines import Processor
from support.filtering import Filter
from support.files import Files

from finance.variables import Securities, Valuations


__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationFile", "ValuationFilter", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


valuations_index = {security: str for security in list(map(str, Securities))} | {"strategy": str, "valuation": str, "scenario": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
valuations_columns = {"spot": np.float32, "minimum": np.float32, "maximum": np.float32, "size": np.float32, "underlying": np.float32}


class ValuationFile(Files.Dataframe, variable="valuations", index=valuations_index, columns=valuations_columns):
    pass


class ValuationFilter(Filter, Processor, title="Filtered"):
    def __init__(self, *args, scenario, **kwargs):
        super().__init__(*args, **kwargs)
        self.columns = ["apy", "npv", "cost", "size"]
        self.scenario = scenario

    def execute(self, contents, *args, **kwargs):
        contract, valuations = contents["contract"], contents["valuations"]
        scenario = str(self.scenario.name).lower()
        if self.empty(valuations["size"]):
            return
        prior = self.size(valuations["size"])
        index = set(valuations.columns) - ({"scenario"} | set(self.columns))
        valuations = valuations.pivot(columns="scenario", index=index)
        mask = self.mask(valuations, variable=scenario)
        valuations = self.where(valuations, mask)
        valuations = valuations.stack("scenario")
        valuations = valuations.reset_index(drop=False, inplace=False)
        post = self.size(valuations["size"])
        __logger__.info(f"Filter: {repr(self)}|{str(contract)}[{prior:.0f}|{post:.0f}]")
        yield contents | dict(valuations=valuations)


class ValuationCalculation(Calculation):
    pass


class ValuationCalculator(Calculator, Processor):
    def execute(self, contents, *args, **kwargs):
        pass



