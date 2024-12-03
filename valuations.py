# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC
from collections import namedtuple as ntuple

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging, Sourcing, Pivoting
from support.calculations import Calculation, Equation, Variable
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ValuationEquation(Equation, ABC):
    tau = Variable("tau", "tau", np.int32, xr.DataArray, vectorize=True, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    inc = Variable("inc", "income", np.float32, xr.DataArray, vectorize=True, function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = Variable("exp", "cost", np.float32, xr.DataArray, vectorize=True, function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))

    xo = Variable("xo", "underlying", np.float32, xr.DataArray, locator="underlying")
    tτ = Variable("tτ", "expire", np.datetime64, xr.DataArray, locator="expire")
    to = Variable("to", "current", np.datetime64, xr.DataArray, locator="current")
    vo = Variable("vo", "spot", np.float32, xr.DataArray, locator="spot")
    ρ = Variable("ρ", "discount", np.float32, types.NoneType, locator="discount")

class ArbitrageEquation(ValuationEquation, ABC):
    irr = Variable("irr", "irr", np.float32, xr.DataArray, vectorize=True, function=lambda inc, exp, tau: np.power(np.divide(inc, exp), np.power(tau, -1)) - 1)
    npv = Variable("npv", "npv", np.float32, xr.DataArray, vectorize=True, function=lambda inc, exp, tau, ρ: np.divide(inc, np.power(1 + ρ, tau / 365)) - exp)
    apy = Variable("apy", "apy", np.float32, xr.DataArray, vectorize=True, function=lambda irr, tau: np.power(irr + 1, np.power(tau / 365, -1)) - 1)

class MinimumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "minimum", np.float32, xr.DataArray, locator="minimum")

class MaximumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "maximum", np.float32, xr.DataArray, locator="maximum")


class ValuationCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class ArbitrageCalculation(ValuationCalculation, ABC):
    def execute(self, strategies, *args, discount, **kwargs):
        with self.equation(strategies, discount=discount) as equation:
            yield strategies["underlying"]
            yield strategies["current"]
            yield strategies["size"]
            yield equation.exp()
            yield equation.npv()
            yield equation.apy()

class MinimumArbitrageCalculation(ArbitrageCalculation, equation=MinimumArbitrageEquation, register=(Variables.Valuations.ARBITRAGE, Variables.Scenarios.MINIMUM)): pass
class MaximumArbitrageCalculation(ArbitrageCalculation, equation=MaximumArbitrageEquation, register=(Variables.Valuations.ARBITRAGE, Variables.Scenarios.MAXIMUM)): pass


class ValuationCalculator(Logging, Sizing, Emptying, Sourcing, Pivoting):
    def __init__(self, *args, header, **kwargs):
        super().__init__(*args, **kwargs)
        Identity = ntuple("Identity", "valuation scenario")
        calculations = {Identity(*identity): calculation for identity, calculation in dict(ValuationCalculation).items()}
        calculations = {identity.scenario: calculation for identity, calculation in calculations.items() if identity.valuation == header.valuation}
        self.calculations = {scenario: calculation(*args, **kwargs) for scenario, calculation in calculations.items()}
        self.header = header

    def execute(self, strategies, *args, **kwargs):
        if self.empty(strategies, "size"): return
        for contract, dataset in self.source(strategies, *args, query=Querys.Contract, **kwargs):
            if self.empty(dataset, "size"): continue
            valuations = self.calculate(dataset, *args, **kwargs)
            valuations = self.pivot(valuations, stacking=self.header.variants, by="scenario")
            size = self.size(valuations)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(valuations): continue
            yield valuations

    def calculate(self, strategies, *args, **kwargs):
        scenarios = dict(self.scenarios(strategies, *args, **kwargs))
        valuations = dict(self.valuations(scenarios, *args, **kwargs))
        valuations = pd.concat(list(valuations.values()), axis=0)
        return valuations

    def scenarios(self, strategies, *args, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        function = lambda mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}
        for scenario, calculation in self.calculations.items():
            valuations = calculation(strategies, *args, **kwargs)
            assert isinstance(valuations, xr.Dataset)
            coordinates = dict(valuation=self.header.valuation, scenario=scenario)
            coordinates = function(coordinates)
            valuations = valuations.assign_coords(coordinates).expand_dims("scenario")
            yield scenario, valuations

    @staticmethod
    def source(strategies, *args, **kwargs):
        assert isinstance(strategies, (list, xr.Dataset))
        assert all([isinstance(dataset, xr.Dataset) for dataset in strategies]) if isinstance(strategies, list) else True
        strategies = [strategies] if isinstance(strategies, xr.Dataset) else strategies
        strategies = [datasets.expand_dims("ticker").expand_dims("expire").stack(contract=["ticker", "expire"]) for datasets in strategies]
        strategies = ([contract, dataset] for datasets in strategies for contract, dataset in datasets.groupby("contract"))
        for contract, dataset in strategies:
            contract = Querys.Contract(list(contract))
            dataset = dataset.unstack().drop_vars("contract")
            yield contract, dataset

    @staticmethod
    def valuations(scenarios, *args, **kwargs):
        assert isinstance(scenarios, dict)
        for scenario, dataset in scenarios.items():
            dataset = dataset.drop_vars(list(Variables.Security), errors="ignore")
            dataset = dataset.expand_dims(list(set(iter(dataset.coords)) - set(iter(dataset.dims))))
            dataframe = dataset.to_dataframe().dropna(how="all", inplace=False)
            dataframe = dataframe.reset_index(drop=False, inplace=False)
            yield scenario, dataframe






