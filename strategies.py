# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC
from itertools import product
from collections import OrderedDict as ODict

from support.calculations import Variable, Domain, Equation, Calculation, Calculator
from support.pipelines import Processor
from support.files import Files

from finance.variables import Securities, Strategies

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyFile", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


strategy_index = {security: str for security in list(map(str, Securities))} | {"strategy": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
strategy_columns = {"spot": np.float32, "minimum": np.float32, "maximum": np.float32, "size": np.float32, "underlying": np.float32}


class StrategyFile(Files.Dataframe, variable="strategies", index=strategy_index, columns=strategy_columns):
    pass


class StrategyDomain(Domain, feed={"ε": "discount"}): pass
class LongOptionDomain(StrategyDomain, feed={"qα": "size", "xα": "underlying", "yα": "price", "kα": "strike"}): pass
class ShortOptionDomain(StrategyDomain, feed={"qβ": "size", "xβ": "underlying", "yβ": "price", "kβ": "strike"}): pass


class StrategyEquation(Equation):
    qo = Variable("size", np.int32, lambda qα, qβ: np.minimum(qα, qβ))
    xo = Variable("underlying", np.float32, lambda xα, xβ: (xα + xβ) / 2)
    wo = Variable("spot", np.float32, lambda yo, ε: yo * 100 - ε)
    whτ = Variable("maximum", np.float32, lambda yhτ, ε: yhτ * 100 - ε)
    wlτ = Variable("minimum", np.float32, lambda ylτ, ε: ylτ * 100 - ε)

class VerticalEquation(StrategyEquation):
    yo = Variable("spot", np.float32, lambda yα, yβ: - yα + yβ)

class CollarEquation(StrategyEquation):
    yo = Variable("spot", np.float32, lambda yα, yβ, xo: - yα + yβ - xo)

class VerticalPutEquation(StrategyEquation):
    yhτ = Variable("maximum", np.float32, lambda kα, kβ: np.maximum(kα - kβ, 0))
    ylτ = Variable("minimum", np.float32, lambda kα, kβ: np.minimum(kα - kβ, 0))

class VerticalCallEquation(StrategyEquation):
    yhτ = Variable("maximum", np.float32, lambda kα, kβ: np.maximum(-kα + kβ, 0))
    ylτ = Variable("minimum", np.float32, lambda kα, kβ: np.minimum(-kα + kβ, 0))

class CollarLongEquation(StrategyEquation):
    yhτ = Variable("maximum", np.float32, lambda kα, kβ: np.maximum(kα, kβ))
    ylτ = Variable("minimum", np.float32, lambda kα, kβ: np.minimum(kα, kβ))

class CollarShortEquation(StrategyEquation):
    yhτ = Variable("maximum", np.float32, lambda kα, kβ: np.maximum(-kα, -kβ))
    ylτ = Variable("minimum", np.float32, lambda kα, kβ: np.minimum(-kα, -kβ))


class StrategyCalculation(Calculation, ABC, fields=["strategy"], domain=[LongOptionDomain, ShortOptionDomain]):
    def execute(self, long, short, *args, fees, **kwargs):
        domain = self.domain(long, short, fees=fees)
        equation = self.equation(domain)
        yield equation.yhτ
        yield equation.ylτ
        yield equation.yo
        yield equation.xo
        yield equation.qo

class VerticalPutCalculation(StrategyCalculation, strategy=Strategies.Vertical.Put, equation=VerticalPutEquation): pass
class VerticalCallCalculation(StrategyCalculation, strategy=Strategies.Vertical.Call, equation=VerticalCallEquation): pass
class CollarLongCalculation(StrategyCalculation, strategy=Strategies.Collar.Long, equation=CollarLongEquation): pass
class CollarShortCalculation(StrategyCalculation, strategy=Strategies.Collar.Short, equation=CollarShortEquation): pass


class StrategyCalculator(Calculator, Processor, calculation=StrategyCalculation):
    def execute(self, contents, *args, **kwargs):
        options = contents["options"]
        assert isinstance(options, pd.DataFrame)
        if self.empty(options):
            return
        options = ODict([(security, dataset) for security, dataset in self.options(options) if not self.empty(dataset["size"])])
        if not bool(options):
            return
        strategies = [strategy for strategy in self.calculate(options, *args, **kwargs)]
        strategies = xr.concat(strategies, dim="scenario")
        yield contents | dict(strategies=strategies)

    def calculate(self, options, *args, **kwargs):
        assert isinstance(options, dict)
        assert all([isinstance(option, xr.Dataset) for option in options.values()])
        for fields, calculation in self.calculations.items():
            if not all([security in options.keys() for security in list(fields["strategy"].options)]):
                continue
            variable = str(fields["strategy"]).lower()
            options = [options[security] for security in list(fields["strategy"].options)]
            dataset = calculation(*options, *args, **kwargs)
            variables = {"strategy": xr.Variable("strategy", [variable]).squeeze("strategy")}
            dataset = dataset.assign_coords(variables)
            if not self.size(dataset["size"]):
                continue
            yield dataset

    @staticmethod
    def options(dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        if bool(dataframe.empty):
            return
        datasets = xr.Dataset.from_dataframe(dataframe)
        datasets = datasets.squeeze("ticker").squeeze("expire").squeeze("date")
        for instrument, position in product(datasets["instrument"].values, datasets["position"].values):
            option = Securities[f"{instrument}|{position}"]
            dataset = datasets.sel({"instrument": instrument, "position": position})
            dataset = dataset.rename({"strike": str(option)})
            dataset["strike"] = dataset[str(option)]
            yield option, dataset



