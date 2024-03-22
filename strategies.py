# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from itertools import product
from collections import OrderedDict as ODict

from support.calculations import Calculation, equation, source, constant
from support.processes import Calculator
from support.pipelines import Processor

from finance.variables import Securities, Strategies

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculation", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


INDEX = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
VALUES = {"price": np.float32, "underlying": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32}


class StrategyCalculation(Calculation, fields=["strategy"]):
    pα = source("pα", str(Securities.Option.Put.Long), position=str(Securities.Option.Put.Long), variables={"k": "strike", "yo": "price", "xo": "underlying", "qo": "size"}, destination=True)
    pβ = source("pβ", str(Securities.Option.Put.Short), position=str(Securities.Option.Put.Short), variables={"k": "strike", "yo": "price", "xo": "underlying", "qo": "size"}, destination=True)
    cα = source("cα", str(Securities.Option.Call.Long), position=str(Securities.Option.Call.Long), variables={"k": "strike", "yo": "price", "xo": "underlying", "qo": "size"}, destination=True)
    cβ = source("cβ", str(Securities.Option.Call.Short), position=str(Securities.Option.Call.Short), variables={"k": "strike", "yo": "price", "xo": "underlying", "qo": "size"}, destination=True)

    wτn = equation("wτn", "minimum", np.float32, domain=("yτn", "ε"), function=lambda yτn, ε: yτn * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("yτx", "ε"), function=lambda yτx, ε: yτx * 100 - ε)
    wo = equation("wo", "spot", np.float32, domain=("yo", "ε"), function=lambda yo, ε: yo * 100 - ε)
    ε = constant("ε", "fees", position="fees")

    def execute(self, feeds, *args, fees, **kwargs):
        yield self.wτn(**feeds, fees=fees)
        yield self.wτx(**feeds, fees=fees)
        yield self.wo(**feeds, fees=fees)
        yield self.qo(**feeds)


class VerticalPutCalculation(StrategyCalculation, strategy=Strategies.Vertical.Put):
    qo = equation("qo", "size", np.float32, domain=("pα.qo", "pβ.qo"), function=lambda qpα, qpβ: np.minimum(qpα, qpβ))
    yτn = equation("yτn", "minimum", np.float32, domain=("pα.k", "pβ.k"), function=lambda kpα, kpβ: np.minimum(kpα - kpβ, 0))
    yτx = equation("yτx", "maximum", np.float32, domain=("pα.k", "pβ.k"), function=lambda kpα, kpβ: np.maximum(kpα - kpβ, 0))
    yo = equation("yo", "spot", np.float32, domain=("pα.yo", "pβ.yo"), function=lambda ypα, ypβ: - ypα + ypβ)

class VerticalCallCalculation(StrategyCalculation, strategy=Strategies.Vertical.Call):
    qo = equation("qo", "size", np.float32, domain=("cα.qo", "cβ.qo"), function=lambda qcα, qcβ: np.minimum(qcα, qcβ))
    yτn = equation("yτn", "minimum", np.float32, domain=("cα.k", "cβ.k"), function=lambda kcα, kcβ: np.minimum(-kcα + kcβ, 0))
    yτx = equation("yτx", "maximum", np.float32, domain=("cα.k", "cβ.k"), function=lambda kcα, kcβ: np.maximum(-kcα + kcβ, 0))
    yo = equation("yo", "spot", np.float32, domain=("cα.yo", "cβ.yo"), function=lambda ycα, ycβ: - ycα + ycβ)

class CollarLongCalculation(StrategyCalculation, strategy=Strategies.Collar.Long):
    qo = equation("qo", "size", np.float32, domain=("pα.qo", "cβ.qo"), function=lambda qpα, qcβ: np.minimum(qpα, qcβ))
    yτn = equation("yτn", "minimum", np.float32, domain=("pα.k", "cβ.k"), function=lambda kpα, kcβ: np.minimum(kpα, kcβ))
    yτx = equation("yτx", "maximum", np.float32, domain=("pα.k", "cβ.k"), function=lambda kpα, kcβ: np.maximum(kpα, kcβ))
    xoμ = equation("xoμ", "underlying", np.float32, domain=("pα.xo", "cβ.xo"), function=lambda xpα, xcβ: np.average([xpα, xcβ]))
    yo = equation("yo", "spot", np.float32, domain=("pα.yo", "cβ.yo", "xoμ"), function=lambda ypα, ycβ, xoμ: - ypα + ycβ - xoμ)

class CollarShortCalculation(StrategyCalculation, strategy=Strategies.Collar.Short):
    qo = equation("qo", "size", np.float32, domain=("cα.qo", "pβ.qo"), function=lambda qcα, qpβ: np.minimum(qcα, qpβ))
    yτn = equation("yτn", "minimum", np.float32, domain=("cα.k", "pβ.k"), function=lambda kcα, kpβ: np.minimum(-kcα, -kpβ))
    yτx = equation("yτx", "maximum", np.float32, domain=("cα.k", "pβ.k"), function=lambda kcα, kpβ: np.maximum(-kcα, -kpβ))
    xoμ = equation("xoμ", "underlying", np.float32, domain=("cα.xo", "pβ.xo"), function=lambda xcα, xpβ: np.average([xcα, xpβ]))
    yo = equation("yo", "spot", np.float32, domain=("cα.yo", "pβ.yo", "xoμ"), function=lambda ycα, ypβ, xoμ: - ycα + ypβ + xoμ)


class StrategyCalculator(Calculator, Processor, calculations=ODict(list(StrategyCalculation))):
    def execute(self, query, *args, **kwargs):
        unflatten = dict(index=list(INDEX.keys()), columns=list(VALUES.keys()))
        securities = query.securities
        assert isinstance(securities, pd.DataFrame)
        securities = self.unflatten(securities, *args, **unflatten, **kwargs)
        securities = self.parse(securities, *args, **kwargs)
        securities = ODict(list(securities))
        sizes = {security: np.count_nonzero(~np.isnan(dataset["size"].values)) for security, dataset in securities.items()}
        calculations = {variable.strategy: calculation for variable, calculation in self.calculations.items()}
        for strategy, calculation in calculations.items():
            if not all([str(security) in sizes.keys() and sizes[str(security)] > 0 for security in strategy.securities]):
                return
            strategies = calculation(securities, *args, **kwargs)
            strategies = strategies.assign_coords({"strategy": str(strategy)})
            size = np.count_nonzero(~np.isnan(strategies["size"].values))
            if not bool(size):
                continue
            yield query(strategies=strategies)

    @staticmethod
    def parse(datasets, *args, **kwargs):
        datasets = datasets.squeeze("date").squeeze("ticker").squeeze("expire")
        for instrument, position in product(datasets["instrument"].values, datasets["position"].values):
            key = f"{instrument}|{position}"
            dataset = datasets.sel({"instrument": instrument, "position": position})
            dataset = dataset.rename({"strike": key})
            dataset["strike"] = dataset[key]
            yield key, dataset



