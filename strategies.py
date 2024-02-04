# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
from collections import namedtuple as ntuple

from support.calculations import Calculation, equation, source, constant
from support.pipelines import Processor

from finance.variables import Securities, Strategies

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class StrategyCalculation(Calculation):
    sμ = equation("sμ", "underlying", np.float32, domain=("sα.w", "sβ.w"), function=lambda wsα, wsβ: np.add(wsα, wsβ) / 2)
    pμ = equation("pμ", "underlying", np.float32, domain=("pα.w", "pβ.w"), function=lambda wpα, wpβ: np.add(wpα, wpβ) / 2)
    cμ = equation("cμ", "underlying", np.float32, domain=("cα.w", "cβ.w"), function=lambda wcα, wcβ: np.add(wcα, wcβ) / 2)
    ε = constant("ε", "fees", position="fees")

    pα = source("pα", str(Securities.Option.Put.Long), position=str(Securities.Option.Put.Long), variables={"w": "price", "k": "strike", "x": "size"}, destination=True)
    pβ = source("pβ", str(Securities.Option.Put.Short), position=str(Securities.Option.Put.Short), variables={"w": "price", "k": "strike", "x": "size"}, destination=True)
    cα = source("cα", str(Securities.Option.Call.Long), position=str(Securities.Option.Call.Long), variables={"w": "price", "k": "strike", "x": "size"}, destination=True)
    cβ = source("cβ", str(Securities.Option.Call.Short), position=str(Securities.Option.Call.Short), variables={"w": "price", "k": "strike", "x": "size"}, destination=True)
    sα = source("sα", str(Securities.Stock.Long), position=str(Securities.Stock.Long), variables={"w": "price", "x": "size"}, destination=True)
    sβ = source("sβ", str(Securities.Stock.Short), position=str(Securities.Stock.Short), variables={"w": "price", "x": "size"}, destination=True)

    def execute(self, feeds, *args, fees, **kwargs):
        feeds = {str(security): dataset for security, dataset in feeds.items()}
        yield self.x(**feeds, fees=fees)
        yield self.wo(**feeds, fees=fees)
        yield self.wτn(**feeds, fees=fees)
        yield self.wτx(**feeds, fees=fees)
        yield self.wτo(**feeds, fees=fees)

class VerticalCalculation(StrategyCalculation): pass
class CollarCalculation(StrategyCalculation): pass

class VerticalPutCalculation(VerticalCalculation):
    x = equation("x", "size", np.int64, domain=("pα.x", "pβ.x"), function=lambda xpα, xpβ: np.minimum(xpα, xpβ))
    wo = equation("wo", "spot", np.float32, domain=("pα.w", "pβ.w", "ε"), function=lambda wpα, wpβ, ε: (wpβ - wpα) * 100 - ε)
    wτn = equation("wτn", "minimum", np.float32, domain=("pα.k", "pβ.k", "ε"), function=lambda kpα, kpβ, ε: np.minimum(kpα - kpβ, 0) * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("pα.k", "pβ.k", "ε"), function=lambda kpα, kpβ, ε: np.maximum(kpα - kpβ, 0) * 100 - ε)
    wτo = equation("wτo", "current", np.float32, domain=("pα.k", "pβ.k", "sμ", "ε"), function=lambda kpα, kpβ, sμ, ε: (np.maximum(kpα - sμ, 0) - np.maximum(kpβ - sμ, 0)) * 100 - ε)

class VerticalCallCalculation(VerticalCalculation):
    x = equation("x", "size", np.int64, domain=("cα.x", "cβ.x"), function=lambda xcα, xcβ: np.minimum(xcα, xcβ))
    wo = equation("wo", "spot", np.float32, domain=("cα.w", "cβ.w", "ε"), function=lambda wcα, wcβ, ε: (wcβ - wcα) * 100 - ε)
    wτn = equation("wτn", "minimum", np.float32, domain=("cα.k", "cβ.k", "ε"), function=lambda kcα, kcβ, ε: np.minimum(kcβ - kcα, 0) * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("cα.k", "cβ.k", "ε"), function=lambda kcα, kcβ, ε: np.maximum(kcβ - kcα, 0) * 100 - ε)
    wτo = equation("wτo", "current", np.float32, domain=("cα.k", "cβ.k", "sμ", "ε"), function=lambda kcα, kcβ, sμ, ε: (np.maximum(sμ - kcα, 0) - np.maximum(sμ - kcβ, 0)) * 100 - ε)

class CollarLongCalculation(CollarCalculation):
    x = equation("x", "size", np.int64, domain=("pα.x", "cβ.x"), function=lambda xpα, xcβ: np.minimum(xpα, xcβ))
    wo = equation("wo", "spot", np.float32, domain=("pα.w", "cβ.w", "sα.w", "ε"), function=lambda wpα, wcβ, wsα, ε: (wcβ - wpα - wsα) * 100 - ε)
    wτn = equation("wτn", "minimum", np.float32, domain=("pα.k", "cβ.k", "ε"), function=lambda kpα, kcβ, ε: np.minimum(kpα, kcβ) * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("pα.k", "cβ.k", "ε"), function=lambda kpα, kcβ, ε: np.maximum(kpα, kcβ) * 100 - ε)
    wτo = equation("wτo", "current", np.float32, domain=("pα.k", "cβ.k", "sμ", "ε"), function=lambda kpα, kcβ, sμ, ε: (np.maximum(kpα - sμ, 0) - np.maximum(sμ - kcβ, 0) + sμ) * 100 - ε)

class CollarShortCalculation(CollarCalculation):
    x = equation("x", "size", np.int64, domain=("pβ.x", "cα.x"), function=lambda xpβ, xcα: np.minimum(xpβ, xcα))
    wo = equation("wo", "spot", np.float32, domain=("pβ.w", "cα.w", "sβ.w", "ε"), function=lambda wpβ, wcα, wsβ, ε: (wpβ - wcα + wsβ) * 100 - ε)
    wτn = equation("wτn", "minimum", np.float32, domain=("pβ.k", "cα.k", "ε"), function=lambda kpβ, kcα, ε: np.minimum(-kpβ, -kcα) * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("pβ.k", "cα.k", "ε"), function=lambda kpβ, kcα, ε: np.maximum(-kpβ, -kcα) * 100 - ε)
    wτo = equation("wτo", "current", np.float32, domain=("pβ.k", "cα.k", "sμ", "ε"), function=lambda kpβ, kcα, sμ, ε: (np.maximum(sμ - kcα, 0) - np.maximum(kpβ - sμ, 0) - sμ) * 100 - ε)


class StrategyQuery(ntuple("Query", "inquiry contract strategies")):
    def __str__(self): return f"{self.contract.ticker}|{self.contract.expire.strftime('%Y-%m-%d')}"


class StrategyCalculator(Processor, title="Calculated"):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        strategies = {Strategies.Collar.Long: [Securities.Option.Put.Long, Securities.Option.Call.Short, Securities.Stock.Long]}
        strategies.update({Strategies.Collar.Short: [Securities.Option.Call.Long, Securities.Option.Put.Short, Securities.Stock.Short]})
        strategies.update({Strategies.Vertical.Put: [Securities.Option.Put.Long, Securities.Option.Put.Short]})
        strategies.update({Strategies.Vertical.Call: [Securities.Option.Call.Long, Securities.Option.Call.Short]})
        calculations = {Strategies.Collar.Long: CollarLongCalculation, Strategies.Collar.Short: CollarShortCalculation}
        calculations.update({Strategies.Vertical.Put: VerticalPutCalculation, Strategies.Vertical.Call: VerticalCallCalculation})
        calculations = {security: calculation(*args, **kwargs) for security, calculation in calculations.items()}
        self.strategies = strategies
        self.calculations = calculations

    def execute(self, query, *args, **kwargs):
        stocks = {security: dataset for security, dataset in query.stocks.items() if dataset["price"].size > 0}
        options = {security: dataset for security, dataset in query.options.items() if dataset["price"].size > 0}
        if not bool(stocks) or not bool(options) or len(stocks) != 2:
            return
        stocks = {security: self.stocks(dataset, *args, security=security, **kwargs) for security, dataset in stocks.items()}
        options = {security: self.options(dataset, *args, security=security, **kwargs) for security, dataset in options.items()}
        function = lambda strategy: all([security in list(stocks.keys()) + list(options.keys()) for security in self.strategies[strategy]])
        calculations = {strategy: calculation for strategy, calculation in self.calculations.items() if function(strategy)}
        strategies = {strategy: calculation(stocks | options, *args, **kwargs) for strategy, calculation in calculations.items()}
        if not bool(strategies):
            return
        yield StrategyQuery(query.inquiry, query.contract, strategies)

    @staticmethod
    def stocks(dataset, *args, **kwargs): return dataset
    @staticmethod
    def options(dataset, *args, security, **kwargs):
        dataset = dataset.rename({"strike": str(security)})
        dataset["strike"] = dataset[str(security)].expand_dims(["date", "ticker", "expire"])
        return dataset




