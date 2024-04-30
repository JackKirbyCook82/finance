# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from collections import OrderedDict as ODict

from support.calculations import Equation, Calculation, Calculator
from support.pipelines import Processor
from support.filtering import Filter
from support.files import Files

from finance.variables import Securities, Valuations, Scenarios

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationFile", "ValuationFilter", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


valuations_index = {security: str for security in list(map(str, Securities))} | {"strategy": str, "valuation": str, "scenario": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
valuations_columns = {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.float32}


class ValuationFile(Files.Dataframe, variable="valuations", index=valuations_index, columns=valuations_columns):
    pass


class ValuationFilter(Filter, Processor, title="Filtered"):
    def __init__(self, *args, scenario, **kwargs):
        super().__init__(*args, **kwargs)
        self.__scenario = scenario

    def execute(self, contents, *args, **kwargs):
        contract, valuations = contents["contract"], contents["valuations"]
        scenario = str(self.scenario.name).lower()
        if self.empty(valuations):
            return
        prior = self.size(valuations)
        valuations = valuations.reset_index(drop=False, inplace=False)
        index = set(valuations.columns) - ({"scenario"} | set(valuations_columns.keys()))
        valuations = valuations.pivot(columns="scenario", index=index)
        mask = self.mask(valuations, variable=scenario)
        valuations = self.where(valuations, mask)
        valuations = valuations.stack("scenario")
        valuations = valuations.set_index(index, drop=False, inplace=False)
        post = self.size(valuations)
        __logger__.info(f"Filter: {repr(self)}|{str(contract)}[{prior:.0f}|{post:.0f}]")
        yield contents | dict(valuations=valuations)

    @property
    def scenario(self): return self.__scenario


class ArbitrageEquation(Equation):
    tau = lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D")
    inc = lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0)
    exp = lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0)
    npv = lambda inc, exp, tau, ρ: np.divide(inc, np.power(1 + ρ, tau / 365)) - exp
    irr = lambda inc, exp, tau: np.power(np.divide(inc, exp), np.power(tau, -1)) - 1
    apy = lambda irr, tau: np.power(irr + 1, np.power(tau / 365, -1)) - 1


class ValuationCalculation(Calculation, ABC, fields=["valuation", "scenario"]): pass
class ArbitrageCalculation(ValuationCalculation, valuation=Valuations.ARBITRAGE):
    def execute(self, strategies, *args, discount, **kwargs):
        pass

#        equation = ArbitrageEquation(domain=["to", "tτ", "vo", "vτ", "ρ"])
#        domain = self.domain(strategies, *args, **kwargs)
#        yield xr.apply_ufunc(equation.npv, *domain, output_dtypes=[np.float32], vectorize=True).to_dataset(name="npv")
#        yield xr.apply_ufunc(equation.apy, *domain, output_dtypes=[np.float32], vectorize=True).to_dataset(name="apy")
#        yield xr.apply_ufunc(equation.exp, *domain, output_dtypes=[np.float32], vectorize=True).to_dataset(name="cost")
#        yield strategies["underlying"]
#        yield strategies["size"]

#    @staticmethod
#    @abstractmethod
#    def domain(strategies, *args, discount, **kwargs): pass


class MinimumArbitrageCalculation(ArbitrageCalculation, scenario=Scenarios.MINIMUM): pass
class MaximumArbitrageCalculation(ArbitrageCalculation, scenario=Scenarios.MAXIMUM): pass


class ValuationCalculator(Calculator, Processor, calculations=ODict(list(ValuationCalculation)), title="Calculated"):
    def __init__(self, *args, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        self.__valuation = valuation

    def execute(self, contents, *args, **kwargs):
        strategies = contents["strategies"]
        assert isinstance(strategies, xr.Dataset)
        if self.empty(strategies["size"]):
            return
        valuations = self.calculate(strategies, *args, **kwargs)
        valuations = valuations.to_dataframe().dropna(how="all", inplace=False)
        valuations = valuations.set_index(list(valuations_index.keys()), drop=False, inplace=False)
        yield contents | dict(valuations=valuations)

    def calculate(self, strategies, *args, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        variable = str(self.valuation.name).lower()
        calculations = {scenario: calculation for (valuation, scenario), calculation in self.calculations.items() if valuation is self.valuation}
        valuations = {scenario: calculation(strategies, *args, **kwargs) for scenario, calculation in calculations.items()}
        valuations = [dataset.assign_coords({"scenario": str(scenario.name).lower()}).expand_dims("scenario") for scenario, dataset in valuations.items()]
        valuations = xr.concat(valuations, dim="scenario").assign_coords({"valuation": variable})
        valuations = valuations.drop_vars(["instrument", "position"], errors="ignore")
        valuations = valuations.expand_dims(list(set(iter(valuations.coords)) - set(iter(valuations.dims))))
        return valuations

    @property
    def valuation(self): return self.__valuation



