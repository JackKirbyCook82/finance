# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np

from support.pipelines import Processor
from support.calculations import Calculation, equation, source, constant

from finance.variables import Query, Securities, Strategies

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculation", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


strategy_variables = {"qo": "size", "yo": "price", "xo": "underlying", "k": "strike", "ws": "entry", "Δo": "quantity"}
class StrategyCalculation(Calculation):
    pα = source("pα", str(Securities.Option.Put.Long), position=str(Securities.Option.Put.Long), variables=strategy_variables, destination=True)
    pβ = source("pβ", str(Securities.Option.Put.Short), position=str(Securities.Option.Put.Short), variables=strategy_variables, destination=True)
    cα = source("cα", str(Securities.Option.Call.Long), position=str(Securities.Option.Call.Long), variables=strategy_variables, destination=True)
    cβ = source("cβ", str(Securities.Option.Call.Short), position=str(Securities.Option.Call.Short), variables=strategy_variables, destination=True)

    wo = equation("wo", "spot", np.float32, domain=("yo", "ε"), function=lambda yo, ε: yo * 100 - ε)
    wτn = equation("wτn", "minimum", np.float32, domain=("yτn", "ε"), function=lambda yτn, ε: yτn * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("yτx", "ε"), function=lambda yτx, ε: yτx * 100 - ε)
    wτi = equation("wτi", "martingale", np.float32, domain=("yτi", "ε"), function=lambda yτi, ε: yτi * 100 - ε)
    ε = constant("ε", "fees", position="fees")

    def execute(self, feeds, *args, fees, **kwargs):
        yield self.qo(**feeds, fees=fees)
        yield self.ws(**feeds, fees=fees)
        yield self.wo(**feeds, fees=fees)
        yield self.wτn(**feeds, fees=fees)
        yield self.wτx(**feeds, fees=fees)
        yield self.wτi(**feeds, fees=fees)
        yield self.Δo(**feeds, fees=fees)


class VerticalCalculation(StrategyCalculation): pass
class CollarCalculation(StrategyCalculation): pass

class VerticalPutCalculation(VerticalCalculation):
    ws = equation("ws", "entry", np.float32, domain=("pα.ws", "pβ.ws"), function=lambda wpα, wpβ: -(wpα + wpβ))
    Δo = equation("Δo", "quantity", np.int32, domain=("pα.Δo", "pβ.Δo"), function=lambda Δpα, Δpβ: np.minimum(Δpα, Δpβ))
    qo = equation("qo", "size", np.int64, domain=("pα.qo", "pβ.qo"), function=lambda qpα, qpβ: np.minimum(qpα, qpβ))
    yo = equation("yo", "spot", np.float32, domain=("pα.yo", "pβ.yo"), function=lambda ypα, ypβ: ypβ - ypα)
    yτn = equation("yτn", "minimum", np.float32, domain=("pα.k", "pβ.k"), function=lambda kpα, kpβ: np.minimum(kpα - kpβ, 0))
    yτx = equation("yτx", "maximum", np.float32, domain=("pα.k", "pβ.k"), function=lambda kpα, kpβ: np.maximum(kpα - kpβ, 0))
    yτi = equation("yτi", "martingale", np.float32, domain=("pα.k", "pβ.k", "pα.xo", "pβ.xo"), function=lambda kpα, kpβ, xpα, xpβ: (np.maximum(kpα - xpα, 0) - np.maximum(kpβ - xpβ, 0)))

class VerticalCallCalculation(VerticalCalculation):
    ws = equation("ws", "entry", np.float32, domain=("cα.ws", "cβ.ws"), function=lambda wcα, wcβ: -(wcα + wcβ))
    Δo = equation("Δo", "quantity", np.int32, domain=("cα.Δo", "cβ.Δo"), function=lambda Δcα, Δcβ: np.minimum(Δcα, Δcβ))
    qo = equation("qo", "size", np.int64, domain=("cα.qo", "cβ.qo"), function=lambda qcα, qcβ: np.minimum(qcα, qcβ))
    yo = equation("yo", "spot", np.float32, domain=("cα.yo", "cβ.yo"), function=lambda ycα, ycβ: ycβ - ycα)
    yτn = equation("yτn", "minimum", np.float32, domain=("cα.k", "cβ.k"), function=lambda kcα, kcβ: np.minimum(kcβ - kcα, 0))
    yτx = equation("yτx", "maximum", np.float32, domain=("cα.k", "cβ.k"), function=lambda kcα, kcβ: np.maximum(kcβ - kcα, 0))
    yτi = equation("yτi", "martingale", np.float32, domain=("cα.k", "cβ.k", "cα.xo", "cβ.xo"), function=lambda kcα, kcβ, xcα, xcβ: (np.maximum(xcα - kcα, 0) - np.maximum(xcβ - kcβ, 0)))

class CollarLongCalculation(CollarCalculation):
    ws = equation("ws", "entry", np.float32, domain=("pα.ws", "cβ.ws"), function=lambda wpα, wcβ: -(wpα + wcβ))
    Δo = equation("Δo", "quantity", np.int32, domain=("pα.Δo", "cβ.Δo"), function=lambda Δpα, Δcβ: np.minimum(Δpα, Δcβ))
    qo = equation("q", "size", np.int64, domain=("pα.qo", "cβ.qo"), function=lambda qpα, qcβ: np.minimum(qpα, qcβ))
    yo = equation("yo", "spot", np.float32, domain=("pα.yo", "cβ.yo", "pα.xo", "cβ.xo"), function=lambda ypα, ycβ, xpα, xcβ: ycβ - ypα - np.mean(xpα, xcβ))
    yτn = equation("yτn", "minimum", np.float32, domain=("pα.k", "cβ.k"), function=lambda kpα, kcβ: np.minimum(kpα, kcβ))
    yτx = equation("yτx", "maximum", np.float32, domain=("pα.k", "cβ.k"), function=lambda kpα, kcβ: np.maximum(kpα, kcβ))
    yτi = equation("yτi", "martingale", np.float32, domain=("pα.k", "cβ.k", "pα.xo", "cβ.xo"), function=lambda kpα, kcβ, xpα, xcβ: (np.maximum(kpα - xpα, 0) - np.maximum(xcβ - kcβ, 0) + np.mean(xpα, xcβ)))

class CollarShortCalculation(CollarCalculation):
    ws = equation("ws", "entry", np.float32, domain=("cα.ws", "pβ.ws"), function=lambda wcα, wpβ: -(wcα + wpβ))
    Δo = equation("Δo", "quantity", np.int32, domain=("cα.Δo", "pβ.Δo"), function=lambda Δcα, Δpβ: np.minimum(Δcα, Δpβ))
    qo = equation("qo", "size", np.int64, domain=("cα.qo", "pβ.qo"), function=lambda qcα, qpβ: np.minimum(qcα, qpβ))
    yo = equation("yo", "spot", np.float32, domain=("cα.yo", "pβ.yo", "cα.xo", "pβ.xo"), function=lambda wcα, wpβ, xcα, xpβ: wpβ - wcα + np.mean(xcα, xpβ))
    yτn = equation("yτn", "minimum", np.float32, domain=("cα.k", "pβ.k"), function=lambda kcα, kpβ: np.minimum(-kcα, -kpβ))
    yτx = equation("yτx", "maximum", np.float32, domain=("cα.k", "pβ.k"), function=lambda kcα, kpβ: np.maximum(-kcα, -kpβ))
    yτi = equation("yτi", "martingale", np.float32, domain=("cα.k", "pβ.k", "cα.xo", "pβ.xo"), function=lambda kcα, kpβ, xcα, xpβ: (np.maximum(xcα - kcα, 0) - np.maximum(kpβ - xpβ, 0) - np.mean(xcα, xpβ)))


class StrategyQuery(Query, fields=["strategy", "strategies"]): pass
class StrategyCalculator(Processor, title="Calculated"):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {Strategies.Collar.Long: CollarLongCalculation, Strategies.Collar.Short: CollarShortCalculation}
        calculations.update({Strategies.Vertical.Put: VerticalPutCalculation, Strategies.Vertical.Call: VerticalCallCalculation})
        calculations = {strategy: calculation(*args, **kwargs) for strategy, calculation in calculations.items()}
        self.calculations = calculations

    def execute(self, query, *args, **kwargs):
        securities = query.markets
        securities = self.parse(securities, *args, **kwargs)
        for strategy, calculation in self.calculations.items():
            if any([security not in securities.keys() for security in strategy.securities]):
                continue
            strategies = calculation(securities, *args, **kwargs)
            if not bool(strategies["spot"].size):
                continue
            strategies.expand_dims({"strategy": str(strategy)})
            yield StrategyQuery(query.inquiry, query.contract, strategy=strategy, strategies=strategies)

    @staticmethod
    def parse(datasets, *args, **kwargs):
        datasets = {security: dataset.rename({"strike": str(security)}) for security, dataset in datasets.items()}
        for security, dataset in datasets.items():
            dataset["strike"] = dataset[str(security)].expand_dims(["date", "ticker", "expire"])
        return datasets








