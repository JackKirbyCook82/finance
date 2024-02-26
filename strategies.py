# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import xarray as xr
from abc import ABC
from itertools import product
from collections import OrderedDict as ODict

from support.processes import Calculator
from support.calculations import Calculation, equation, source, constant

from finance.variables import Securities, Strategies, Actions

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculation", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


OPEN_VARIABLES = {"k": "strike", "yo": "price", "xo": "underlying", "qo": "size"}
CLOSE_VARIABLES = {"k": "strike", "yo": "price", "xo": "underlying", "qo": "size", "Δs": "quantity"}


class StrategyCalculation(Calculation, ABC, fields=["action", "strategy"]):
    wτn = equation("wτn", "minimum", np.float32, domain=("yτn", "ε"), function=lambda yτn, ε: yτn * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("yτx", "ε"), function=lambda yτx, ε: yτx * 100 - ε)
    wo = equation("wo", "spot", np.float32, domain=("yo", "ε"), function=lambda yo, ε: yo * 100 - ε)
    ε = constant("ε", "fees", position="fees")


class OpenStrategyCalculation(StrategyCalculation, action=Actions.OPEN):
    pα = source("pα", str(Securities.Option.Put.Long), position=str(Securities.Option.Put.Long), variables=OPEN_VARIABLES, destination=True)
    pβ = source("pβ", str(Securities.Option.Put.Short), position=str(Securities.Option.Put.Short), variables=OPEN_VARIABLES, destination=True)
    cα = source("cα", str(Securities.Option.Call.Long), position=str(Securities.Option.Call.Long), variables=OPEN_VARIABLES, destination=True)
    cβ = source("cβ", str(Securities.Option.Call.Short), position=str(Securities.Option.Call.Short), variables=OPEN_VARIABLES, destination=True)

    def execute(self, feeds, *args, fees, **kwargs):
        yield self.wτn(**feeds, fees=fees)
        yield self.wτx(**feeds, fees=fees)
        yield self.wo(**feeds, fees=fees)
        yield self.qo(**feeds)


class CloseStrategyCalculation(StrategyCalculation, action=Actions.CLOSE):
    pα = source("pα", str(Securities.Option.Put.Long), position=str(Securities.Option.Put.Long), variables=CLOSE_VARIABLES, destination=True)
    pβ = source("pβ", str(Securities.Option.Put.Short), position=str(Securities.Option.Put.Short), variables=CLOSE_VARIABLES, destination=True)
    cα = source("cα", str(Securities.Option.Call.Long), position=str(Securities.Option.Call.Long), variables=CLOSE_VARIABLES, destination=True)
    cβ = source("cβ", str(Securities.Option.Call.Short), position=str(Securities.Option.Call.Short), variables=CLOSE_VARIABLES, destination=True)

    def execute(self, feeds, *args, fees, **kwargs):
        yield self.wτn(**feeds, fees=fees)
        yield self.wτx(**feeds, fees=fees)
        yield self.wo(**feeds, fees=fees)
        yield self.qo(**feeds)
        yield self.Δs(**feeds)


class VerticalPutCalculation(StrategyCalculation, ABC, strategy=Strategies.Vertical.Put):
    yτn = equation("yτn", "minimum", np.float32, domain=("pα.k", "pβ.k"), function=lambda kpα, kpβ: np.minimum(kpα - kpβ, 0))
    yτx = equation("yτx", "maximum", np.float32, domain=("pα.k", "pβ.k"), function=lambda kpα, kpβ: np.maximum(kpα - kpβ, 0))
    yo = equation("yo", "spot", np.float32, domain=("pα.yo", "pβ.yo"), function=lambda ypα, ypβ: - ypα + ypβ)
    qo = equation("qo", "size", np.float32, domain=("pα.qo", "pβ.qo"), function=lambda qpα, qpβ: np.minimum(qpα, qpβ))

class VerticalCallCalculation(StrategyCalculation, ABC, strategy=Strategies.Vertical.Call):
    yτn = equation("yτn", "minimum", np.float32, domain=("cα.k", "cβ.k"), function=lambda kcα, kcβ: np.minimum(-kcα + kcβ, 0))
    yτx = equation("yτx", "maximum", np.float32, domain=("cα.k", "cβ.k"), function=lambda kcα, kcβ: np.maximum(-kcα + kcβ, 0))
    yo = equation("yo", "spot", np.float32, domain=("cα.yo", "cβ.yo"), function=lambda ycα, ycβ: - ycα + ycβ)
    qo = equation("qo", "size", np.float32, domain=("cα.qo", "cβ.qo"), function=lambda qcα, qcβ: np.minimum(qcα, qcβ))

class CollarLongCalculation(StrategyCalculation, ABC, strategy=Strategies.Collar.Long):
    xoμ = equation("xoμ", "underlying", np.float32, domain=("pα.xo", "cβ.xo"), function=lambda xpα, xcβ: np.average([xpα, xcβ]))
    yτn = equation("yτn", "minimum", np.float32, domain=("pα.k", "cβ.k"), function=lambda kpα, kcβ: np.minimum(kpα, kcβ))
    yτx = equation("yτx", "maximum", np.float32, domain=("pα.k", "cβ.k"), function=lambda kpα, kcβ: np.maximum(kpα, kcβ))
    yo = equation("yo", "spot", np.float32, domain=("pα.yo", "cβ.yo", "xoμ"), function=lambda ypα, ycβ, xoμ: - ypα + ycβ - xoμ)
    qo = equation("qo", "size", np.float32, domain=("pα.qo", "cβ.qo"), function=lambda qpα, qcβ: np.minimum(qpα, qcβ))

class CollarShortCalculation(StrategyCalculation, ABC, strategy=Strategies.Collar.Short):
    xoμ = equation("xoμ", "underlying", np.float32, domain=("cα.xo", "pβ.xo"), function=lambda xcα, xpβ: np.average([xcα, xpβ]))
    yτn = equation("yτn", "minimum", np.float32, domain=("cα.k", "pβ.k"), function=lambda kcα, kpβ: np.minimum(-kcα, -kpβ))
    yτx = equation("yτx", "maximum", np.float32, domain=("cα.k", "pβ.k"), function=lambda kcα, kpβ: np.maximum(-kcα, -kpβ))
    yo = equation("yo", "spot", np.float32, domain=("cα.yo", "pβ.yo", "xoμ"), function=lambda ycα, ypβ, xoμ: - ycα + ypβ + xoμ)
    qo = equation("qo", "size", np.float32, domain=("cα.qo", "pβ.qo"), function=lambda qcα, qpβ: np.minimum(qcα, qpβ))


class OpenVerticalPutCalculation(OpenStrategyCalculation, VerticalPutCalculation): pass
class CloseVerticalPutCalculation(CloseStrategyCalculation, VerticalPutCalculation):
    Δs = equation("Δs", "quantity", np.float32, domain=("pα.Δs", "pβ.Δs"), function=lambda Δpα, Δpβ: np.minimum(Δpα, Δpβ))


class OpenVerticalCallCalculation(OpenStrategyCalculation, VerticalCallCalculation): pass
class CloseVerticalCallCalculation(CloseStrategyCalculation, VerticalCallCalculation):
    Δs = equation("Δs", "quantity", np.int32, domain=("cα.Δs", "cβ.Δs"), function=lambda Δcα, Δcβ: np.minimum(Δcα, Δcβ))


class OpenCollarLongCalculation(OpenStrategyCalculation, CollarLongCalculation): pass
class CloseCollarLongCalculation(CloseStrategyCalculation, CollarLongCalculation):
    Δs = equation("Δs", "quantity", np.int32, domain=("pα.Δs", "cβ.Δs"), function=lambda Δpα, Δcβ: np.minimum(Δpα, Δcβ))


class OpenCollarShortCalculation(OpenStrategyCalculation, CollarShortCalculation): pass
class CloseCollarShortCalculation(CloseStrategyCalculation, CollarShortCalculation):
    Δs = equation("Δs", "quantity", np.int32, domain=("cα.Δs", "pβ.Δs"), function=lambda Δcα, Δpβ: np.minimum(Δcα, Δpβ))


class StrategyCalculator(Calculator, calculations=ODict(list(StrategyCalculation))):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        assert isinstance(securities, xr.Dataset)
        options = {option: dataset for option, dataset in self.options(securities, *args, **kwargs)}
        sizes = {option: np.count_nonzero(~np.isnan(dataset["size"].values)) for option, dataset in options.items()}
        calculations = {variable.strategy: calculation for variable, calculation in self.calculations.items()}
        for strategy, calculation in calculations.items():
            if not all([str(security) in sizes.keys() and sizes[str(security)] > 0 for security in strategy.securities]):
                return
            strategies = calculation(options, *args, **kwargs)
            strategies = strategies.assign_coords({"strategy": str(strategy)})
            size = np.count_nonzero(~np.isnan(strategies["size"].values))
            if not bool(size):
                continue
            yield query(strategies=strategies)

    @staticmethod
    def options(datasets, *args, **kwargs):
        datasets = datasets.squeeze("date").squeeze("ticker").squeeze("expire")
        for instrument, position in product(datasets["instrument"].values, datasets["position"].values):
            key = f"{instrument}|{position}"
            dataset = datasets.sel({"instrument": instrument, "position": position})
            dataset = dataset.rename({"strike": key})
            dataset["strike"] = dataset[key]
            yield key, dataset








