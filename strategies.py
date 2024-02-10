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
__all__ = ["StrategyQuery", "StrategyCalculation", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class StrategyCalculation(Calculation):
    pα = source("pα", str(Securities.Option.Put.Long), position=str(Securities.Option.Put.Long), variables={"qo": "size", "yo": "price", "xo": "underlying", "k": "strike"}, destination=True)
    pβ = source("pβ", str(Securities.Option.Put.Short), position=str(Securities.Option.Put.Short), variables={"qo": "size", "yo": "price", "xo": "underlying", "k": "strike"}, destination=True)
    cα = source("cα", str(Securities.Option.Call.Long), position=str(Securities.Option.Call.Long), variables={"qo": "size", "yo": "price", "xo": "underlying", "k": "strike"}, destination=True)
    cβ = source("cβ", str(Securities.Option.Call.Short), position=str(Securities.Option.Call.Short), variables={"qo": "size", "yo": "price", "xo": "underlying", "k": "strike"}, destination=True)
    wo = equation("wo", "spot", np.float32, domain=("yo", "ε"), function=lambda yo, ε: yo * 100 - ε)
    wτn = equation("wτn", "minimum", np.float32, domain=("yτn", "ε"), function=lambda yτn, ε: yτn * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("yτx", "ε"), function=lambda yτx, ε: yτx * 100 - ε)
    wτi = equation("wτi", "martingale", np.float32, domain=("yτi", "ε"), function=lambda yτi, ε: yτi * 100 - ε)
    ε = constant("ε", "fees", position="fees")

    def execute(self, feeds, *args, fees, **kwargs):
        yield self.qo(**feeds, fees=fees)
        yield self.wo(**feeds, fees=fees)
        yield self.wτn(**feeds, fees=fees)
        yield self.wτx(**feeds, fees=fees)
        yield self.wτi(**feeds, fees=fees)


class VerticalCalculation(StrategyCalculation, register="Vertical"): pass
class CollarCalculation(StrategyCalculation, register="Collar"): pass

class VerticalPutCalculation(VerticalCalculation, register="Put"):
    qo = equation("qo", "size", np.int64, domain=("pα.qo", "pβ.qo"), function=lambda qpα, qpβ: np.minimum(qpα, qpβ))
    yo = equation("yo", "spot", np.float32, domain=("pα.yo", "pβ.yo"), function=lambda ypα, ypβ: ypβ - ypα)
    yτn = equation("yτn", "minimum", np.float32, domain=("pα.k", "pβ.k"), function=lambda kpα, kpβ: np.minimum(kpα - kpβ, 0))
    yτx = equation("yτx", "maximum", np.float32, domain=("pα.k", "pβ.k"), function=lambda kpα, kpβ: np.maximum(kpα - kpβ, 0))
    yτi = equation("yτi", "martingale", np.float32, domain=("pα.k", "pβ.k", "pα.xo", "pβ.xo"), function=lambda kpα, kpβ, xpα, xpβ: (np.maximum(kpα - xpα, 0) - np.maximum(kpβ - xpβ, 0)))

class VerticalCallCalculation(VerticalCalculation, register="Call"):
    qo = equation("qo", "size", np.int64, domain=("cα.qo", "cβ.qo"), function=lambda qcα, qcβ: np.minimum(qcα, qcβ))
    yo = equation("yo", "spot", np.float32, domain=("cα.yo", "cβ.yo"), function=lambda ycα, ycβ: ycβ - ycα)
    yτn = equation("yτn", "minimum", np.float32, domain=("cα.k", "cβ.k"), function=lambda kcα, kcβ: np.minimum(kcβ - kcα, 0))
    yτx = equation("yτx", "maximum", np.float32, domain=("cα.k", "cβ.k"), function=lambda kcα, kcβ: np.maximum(kcβ - kcα, 0))
    yτi = equation("yτi", "martingale", np.float32, domain=("cα.k", "cβ.k", "cα.xo", "cβ.xo"), function=lambda kcα, kcβ, xcα, xcβ: (np.maximum(xcα - kcα, 0) - np.maximum(xcβ - kcβ, 0)))

class CollarLongCalculation(CollarCalculation, register="Long"):
    qo = equation("q", "size", np.int64, domain=("pα.qo", "cβ.qo"), function=lambda qpα, qcβ: np.minimum(qpα, qcβ))
    yo = equation("yo", "spot", np.float32, domain=("pα.yo", "cβ.yo", "pα.xo", "cβ.xo"), function=lambda ypα, ycβ, xpα, xcβ: ycβ - ypα - np.mean(xpα, xcβ))
    yτn = equation("yτn", "minimum", np.float32, domain=("pα.k", "cβ.k"), function=lambda kpα, kcβ: np.minimum(kpα, kcβ))
    yτx = equation("yτx", "maximum", np.float32, domain=("pα.k", "cβ.k"), function=lambda kpα, kcβ: np.maximum(kpα, kcβ))
    yτi = equation("yτi", "martingale", np.float32, domain=("pα.k", "cβ.k", "pα.xo", "cβ.xo"), function=lambda kpα, kcβ, xpα, xcβ: (np.maximum(kpα - xpα, 0) - np.maximum(xcβ - kcβ, 0) + np.mean(xpα, xcβ)))

class CollarShortCalculation(CollarCalculation, register="Short"):
    qo = equation("qo", "size", np.int64, domain=("cα.qo", "pβ.qo"), function=lambda qcα, qpβ: np.minimum(qcα, qpβ))
    yo = equation("yo", "spot", np.float32, domain=("cα.yo", "pβ.yo", "cα.xo", "pβ.xo"), function=lambda wcα, wpβ, xcα, xpβ: wpβ - wcα + np.mean(xcα, xpβ))
    yτn = equation("yτn", "minimum", np.float32, domain=("cα.k", "pβ.k"), function=lambda kcα, kpβ: np.minimum(-kcα, -kpβ))
    yτx = equation("yτx", "maximum", np.float32, domain=("cα.k", "pβ.k"), function=lambda kcα, kpβ: np.maximum(-kcα, -kpβ))
    yτi = equation("yτi", "martingale", np.float32, domain=("cα.k", "pβ.k", "cα.xo", "pβ.xo"), function=lambda kcα, kpβ, xcα, xpβ: (np.maximum(xcα - kcα, 0) - np.maximum(kpβ - xpβ, 0) - np.mean(xcα, xpβ)))


class StrategyQuery(Query): pass
class StrategyCalculator(Processor, title="Calculated"):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {Strategies.Collar.Long: CollarLongCalculation, Strategies.Collar.Short: CollarShortCalculation}
        calculations.update({Strategies.Vertical.Put: VerticalPutCalculation, Strategies.Vertical.Call: VerticalCallCalculation})
        calculations = {strategy: calculation(*args, **kwargs) for strategy, calculation in calculations.items()}
        self.calculations = calculations

    def execute(self, query, *args, **kwargs):
        options = query.contents
        options = {security: dataset.rename({"strike": str(security)}) for security, dataset in options.items()}
        for security, dataset in options.items():
            dataset["strike"] = dataset[str(security)].expand_dims(["date", "ticker", "expire"])
        for strategy, calculation in self.calculations.items():
            if any([security not in options.keys() for security in strategy.securities]):
                continue
            strategies = calculation(options, *args, **kwargs)
            if not bool(strategies["spot"].size):
                continue
            strategies.expand_dims({"strategy": str(strategy)})
            yield StrategyQuery(query.inquiry, query.contract, strategies)




