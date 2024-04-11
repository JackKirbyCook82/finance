# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
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


class StrategyCalculation(Calculation, fields=["strategy"]):
    pα = source("pα", str(Securities.Option.Put.Long), position=str(Securities.Option.Put.Long), variables={"k": "strike", "yo": "price", "xo": "underlying", "qo": "size"}, destination=True)
    pβ = source("pβ", str(Securities.Option.Put.Short), position=str(Securities.Option.Put.Short), variables={"k": "strike", "yo": "price", "xo": "underlying", "qo": "size"}, destination=True)
    cα = source("cα", str(Securities.Option.Call.Long), position=str(Securities.Option.Call.Long), variables={"k": "strike", "yo": "price", "xo": "underlying", "qo": "size"}, destination=True)
    cβ = source("cβ", str(Securities.Option.Call.Short), position=str(Securities.Option.Call.Short), variables={"k": "strike", "yo": "price", "xo": "underlying", "qo": "size"}, destination=True)

    wo = equation("wo", "spot", np.float32, domain=("yo", "ε"), function=lambda yo, ε: yo * 100 - ε)
    wτn = equation("wτn", "minimum", np.float32, domain=("yτn", "ε"), function=lambda yτn, ε: yτn * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("yτx", "ε"), function=lambda yτx, ε: yτx * 100 - ε)
    ε = constant("ε", "fees", position="fees")

    def execute(self, feeds, *args, fees, **kwargs):
        feeds = {str(key): value for key, value in feeds.items()}
        yield self.wτn(**feeds, fees=fees)
        yield self.wτx(**feeds, fees=fees)
        yield self.wo(**feeds, fees=fees)
        yield self.qo(**feeds)


class VerticalPutCalculation(StrategyCalculation, strategy=Strategies.Vertical.Put):
    qo = equation("qo", "size", np.float32, domain=("pα.qo", "pβ.qo"), function=lambda qpα, qpβ: np.minimum(qpα, qpβ))
    yo = equation("yo", "spot", np.float32, domain=("pα.yo", "pβ.yo"), function=lambda ypα, ypβ: - ypα + ypβ)
    yτn = equation("yτn", "minimum", np.float32, domain=("pα.k", "pβ.k"), function=lambda kpα, kpβ: np.minimum(kpα - kpβ, 0))
    yτx = equation("yτx", "maximum", np.float32, domain=("pα.k", "pβ.k"), function=lambda kpα, kpβ: np.maximum(kpα - kpβ, 0))

class VerticalCallCalculation(StrategyCalculation, strategy=Strategies.Vertical.Call):
    qo = equation("qo", "size", np.float32, domain=("cα.qo", "cβ.qo"), function=lambda qcα, qcβ: np.minimum(qcα, qcβ))
    yo = equation("yo", "spot", np.float32, domain=("cα.yo", "cβ.yo"), function=lambda ycα, ycβ: - ycα + ycβ)
    yτn = equation("yτn", "minimum", np.float32, domain=("cα.k", "cβ.k"), function=lambda kcα, kcβ: np.minimum(-kcα + kcβ, 0))
    yτx = equation("yτx", "maximum", np.float32, domain=("cα.k", "cβ.k"), function=lambda kcα, kcβ: np.maximum(-kcα + kcβ, 0))

class CollarLongCalculation(StrategyCalculation, strategy=Strategies.Collar.Long):
    qo = equation("qo", "size", np.float32, domain=("pα.qo", "cβ.qo"), function=lambda qpα, qcβ: np.minimum(qpα, qcβ))
    yo = equation("yo", "spot", np.float32, domain=("pα.yo", "cβ.yo", "xoμ"), function=lambda ypα, ycβ, xoμ: - ypα + ycβ - xoμ)
    yτn = equation("yτn", "minimum", np.float32, domain=("pα.k", "cβ.k"), function=lambda kpα, kcβ: np.minimum(kpα, kcβ))
    yτx = equation("yτx", "maximum", np.float32, domain=("pα.k", "cβ.k"), function=lambda kpα, kcβ: np.maximum(kpα, kcβ))
    xoμ = equation("xoμ", "underlying", np.float32, domain=("pα.xo", "cβ.xo"), function=lambda xpα, xcβ: np.mean([xpα, xcβ]))

class CollarShortCalculation(StrategyCalculation, strategy=Strategies.Collar.Short):
    qo = equation("qo", "size", np.float32, domain=("cα.qo", "pβ.qo"), function=lambda qcα, qpβ: np.minimum(qcα, qpβ))
    yo = equation("yo", "spot", np.float32, domain=("cα.yo", "pβ.yo", "xoμ"), function=lambda ycα, ypβ, xoμ: - ycα + ypβ + xoμ)
    yτn = equation("yτn", "minimum", np.float32, domain=("cα.k", "pβ.k"), function=lambda kcα, kpβ: np.minimum(-kcα, -kpβ))
    yτx = equation("yτx", "maximum", np.float32, domain=("cα.k", "pβ.k"), function=lambda kcα, kpβ: np.maximum(-kcα, -kpβ))
    xoμ = equation("xoμ", "underlying", np.float32, domain=("cα.xo", "pβ.xo"), function=lambda xcα, xpβ: np.mean([xcα, xpβ]))


class StrategyCalculator(Calculator, Processor, calculations=ODict(list(StrategyCalculation)), title="Calculated"):
    def execute(self, query, *args, **kwargs):
        stocks, options = query["stocks"], query["options"]
        function = lambda record: (Securities.security(record["instrument"], record["position"]), np.float32(np.round(record["price"], decimals=2)))
        stocks = ODict([function(record) for record in stocks.reset_index(drop=False, inplace=False).to_dict("records")])
        options = ODict([(option, dataset) for option, dataset in self.separate(options) if self.size(dataset["size"])])
        calculations = ODict([(variable.strategy, calculation) for variable, calculation in self.calculations.items()])
        for strategy, calculation in calculations.items():
            if not all([option in options.keys() for option in list(strategy.options)]):
                continue
            variables = {"strategy": str(strategy)} | {str(stock): stocks[stock] for stock in list(strategy.stocks)}
            variables = {key: xr.Variable(key, [value]).squeeze(key) for key, value in variables.items()}
            strategies = calculation(options, *args, **kwargs)
            strategies = strategies.assign_coords(variables)
            if not self.size(strategies["size"]):
                continue
            yield query | dict(strategies=strategies)

    @staticmethod
    def separate(dataframes):
        assert isinstance(dataframes, pd.DataFrame)
        if dataframes.empty:
            return
        datasets = xr.Dataset.from_dataframe(dataframes)
        datasets = datasets.squeeze("date").squeeze("ticker").squeeze("expire")
        for instrument, position in product(datasets["instrument"].values, datasets["position"].values):
            option = Securities[f"{instrument}|{position}"]
            dataset = datasets.sel({"instrument": instrument, "position": position})
            dataset = dataset.rename({"strike": str(option)})
            dataset["strike"] = dataset[str(option)]
            yield option, dataset



