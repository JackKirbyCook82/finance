# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import xarray as xr
from abc import ABC

from support.calculations import Variable, Equation, Calculation, Calculator
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


valuation_index = {security: str for security in list(map(str, Securities))} | {"strategy": str, "valuation": str, "scenario": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
valuation_columns = {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.float32}


class ValuationFile(Files.Dataframe, variable="valuations", index=valuation_index, columns=valuation_columns):
    pass


class ValuationFilter(Filter, Processor):
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
        index = set(valuations.columns) - ({"scenario"} | set(valuation_columns.keys()))
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


class ValuationEquation(Equation): pass
class ArbitrageEquation(ValuationEquation):
    tau = Variable("tau", "tau", np.int32, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    inc = Variable("inc", "income", np.float32, function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = Variable("exp", "cost", np.float32, function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    npv = Variable("npv", "npv", np.float32, function=lambda inc, exp, tau, ρ: np.divide(inc, np.power(1 + ρ, tau / 365)) - exp)
    irr = Variable("irr", "irr", np.float32, function=lambda inc, exp, tau: np.power(np.divide(inc, exp), np.power(tau, -1)) - 1)
    apy = Variable("apy", "apy", np.float32, function=lambda irr, tau: np.power(irr + 1, np.power(tau / 365, -1)) - 1)
    tτ = Variable("tτ", "expire", np.datetime64, position=0, locator="expire")
    to = Variable("to", "date", np.datetime64, position=0, locator="date")
    vo = Variable("vo", "spot", np.float32, position=0, locator="spot")

class MinimumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "minimum", np.float32, position=0, locator="minimum")

class MaximumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "maximum", np.float32, position=0, locator="maximum")


class ValuationCalculation(Calculation, ABC, fields=["valuation", "scenario"]): pass
class ArbitrageCalculation(ValuationCalculation, ABC, valuation=Valuations.ARBITRAGE):
    def execute(self, strategies, *args, discount, **kwargs):
        equation = self.equation(*args, **kwargs)
        yield equation.npv(strategies, discount=discount)
        yield equation.apy(strategies, discount=discount)
        yield equation.exp(strategies, discount=discount)
        yield strategies["underlying"]
        yield strategies["size"]

class MinimumArbitrageCalculation(ArbitrageCalculation, scenario=Scenarios.MINIMUM, equation=MinimumArbitrageEquation): pass
class MaximumArbitrageCalculation(ArbitrageCalculation, scenario=Scenarios.MAXIMUM, equation=MaximumArbitrageEquation): pass


class ValuationCalculator(Calculator, Processor, calculation=ValuationCalculation):
    def __init__(self, *args, valuation, **kwargs):
        assert "valuation" not in kwargs.keys() and "scenario" not in kwargs.keys()
        assert valuation in Valuations
        super().__init__(*args, **kwargs)
        self.__valuation = valuation

    def execute(self, contents, *args, **kwargs):
        strategies = contents["strategies"]
        assert isinstance(strategies, xr.Dataset)
        if self.empty(strategies["size"]):
            return
        valuations = self.calculate(strategies, *args, **kwargs)
        valuations = valuations.to_dataframe().dropna(how="all", inplace=False)
        valuations = valuations.set_index(list(valuation_index.keys()), drop=False, inplace=False)
        yield contents | dict(valuations=valuations)

    def calculate(self, strategies, *args, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        variable = str(self.valuation.name).lower()
        calculations = {fields["scenario"]: calculation for fields, calculation in self.calculations.items() if fields["valuation"] is self.valuation}
        valuations = {scenario: calculation(strategies, *args, **kwargs) for scenario, calculation in calculations.items()}
        valuations = [dataset.assign_coords({"scenario": str(scenario.name).lower()}).expand_dims("scenario") for scenario, dataset in valuations.items()]
        valuations = xr.concat(valuations, dim="scenario").assign_coords({"valuation": variable})
        valuations = valuations.drop_vars(["instrument", "position"], errors="ignore")
        valuations = valuations.expand_dims(list(set(iter(valuations.coords)) - set(iter(valuations.dims))))
        return valuations

    @property
    def valuation(self): return self.__valuation



