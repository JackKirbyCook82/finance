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
from scipy.stats import norm
from functools import reduce
from collections import namedtuple as ntuple

from finance.variables import Querys, Variables, Strategies, Securities
from support.mixins import Emptying, Sizing, Partition, Logging
from support.calculations import Calculation, Equation, Variable

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class StrategyLocator(ntuple("Locator", "axis security")): pass
class StrategyEquation(Equation, ABC, datatype=xr.DataArray, vectorize=True):
    ε = Variable.Constant("ε", "fees", np.float32, locator="fees")

    kpα = Variable.Independent("kpα", "strike", np.float32, locator=StrategyLocator("strike", Securities.Options.Puts.Long))
    kpβ = Variable.Independent("kpβ", "strike", np.float32, locator=StrategyLocator("strike", Securities.Options.Puts.Short))
    kcα = Variable.Independent("kcα", "strike", np.float32, locator=StrategyLocator("strike", Securities.Options.Calls.Long))
    kcβ = Variable.Independent("kcβ", "strike", np.float32, locator=StrategyLocator("strike", Securities.Options.Calls.Short))

    ypα = Variable.Independent("ypα", "price", np.float32, locator=StrategyLocator("price", Securities.Options.Puts.Long))
    ypβ = Variable.Independent("ypβ", "price", np.float32, locator=StrategyLocator("price", Securities.Options.Puts.Short))
    ycα = Variable.Independent("ycα", "price", np.float32, locator=StrategyLocator("price", Securities.Options.Calls.Long))
    ycβ = Variable.Independent("ycβ", "price", np.float32, locator=StrategyLocator("price", Securities.Options.Calls.Short))

    xpα = Variable.Independent("xpα", "underlying", np.float32, locator=StrategyLocator("underlying", Securities.Options.Puts.Long))
    xpβ = Variable.Independent("xpβ", "underlying", np.float32, locator=StrategyLocator("underlying", Securities.Options.Puts.Short))
    xcα = Variable.Independent("xcα", "underlying", np.float32, locator=StrategyLocator("underlying", Securities.Options.Calls.Long))
    xcβ = Variable.Independent("xcβ", "underlying", np.float32, locator=StrategyLocator("underlying", Securities.Options.Calls.Short))

    μpα = Variable.Independent("μpα", "trend", np.float32, locator=StrategyLocator("trend", Securities.Options.Puts.Long))
    μpβ = Variable.Independent("μpβ", "trend", np.float32, locator=StrategyLocator("trend", Securities.Options.Puts.Short))
    μcα = Variable.Independent("μcα", "trend", np.float32, locator=StrategyLocator("trend", Securities.Options.Calls.Long))
    μcβ = Variable.Independent("μcβ", "trend", np.float32, locator=StrategyLocator("trend", Securities.Options.Calls.Short))

    σpα = Variable.Independent("σpα", "volatility", np.float32, locator=StrategyLocator("volatility", Securities.Options.Puts.Long))
    σpβ = Variable.Independent("σpβ", "volatility", np.float32, locator=StrategyLocator("volatility", Securities.Options.Puts.Short))
    σcα = Variable.Independent("σcα", "volatility", np.float32, locator=StrategyLocator("volatility", Securities.Options.Calls.Long))
    σcβ = Variable.Independent("σcβ", "volatility", np.float32, locator=StrategyLocator("volatility", Securities.Options.Calls.Short))

    qpα = Variable.Independent("qpα", "size", np.int32, locator=StrategyLocator("size", Securities.Options.Puts.Long))
    qpβ = Variable.Independent("qpβ", "size", np.int32, locator=StrategyLocator("size", Securities.Options.Puts.Short))
    qcα = Variable.Independent("qcα", "size", np.int32, locator=StrategyLocator("size", Securities.Options.Calls.Long))
    qcβ = Variable.Independent("qcβ", "size", np.int32, locator=StrategyLocator("size", Securities.Options.Calls.Short))

    whτ = Variable.Dependent("whτ", "maximum", np.float32, function=lambda yhτ, *, ε: yhτ * 100 - ε)
    weτ = Variable.Dependent("weτ", "expected", np.float32, function=lambda yeτ, *, ε: yeτ * 100 - ε)
    wlτ = Variable.Dependent("wlτ", "minimum", np.float32, function=lambda ylτ, *, ε: ylτ * 100 - ε)
    wo = Variable.Dependent("wo", "spot", np.float32, function=lambda yo, *, ε: yo * 100 - ε)

    dpα = Variable.Dependent("zpα", "zscore", np.float32, function=lambda kpα, xo: kpα - xo)
    dpβ = Variable.Dependent("zpβ", "zscore", np.float32, function=lambda kpβ, xo: kpβ - xo)
    dcα = Variable.Dependent("zcα", "zscore", np.float32, function=lambda kcα, xo: kcα - xo)
    dcβ = Variable.Dependent("zcβ", "zscore", np.float32, function=lambda kcβ, xo: kcβ - xo)

    zpα = Variable.Dependent("zpα", "zscore", np.float32, function=lambda dpα, σo: dpα / σo)
    zpβ = Variable.Dependent("zpβ", "zscore", np.float32, function=lambda dpβ, σo: dpβ / σo)
    zcα = Variable.Dependent("zcα", "zscore", np.float32, function=lambda dcα, σo: dcα / σo)
    zcβ = Variable.Dependent("zcβ", "zscore", np.float32, function=lambda dcβ, σo: dcβ / σo)

    def execute(self, *args, **kwargs):
        yield self.wlτ()
        yield self.weτ()
        yield self.whτ()
        yield self.wo()
        yield self.xo()
        yield self.qo()


class GreeksEquation(StrategyEquation, axes=["value", "delta", "gamma", "theta", "vega", "rho"]):
    vpα = Variable.Independent("vpα", "value", np.float32, locator=StrategyLocator("value", Securities.Options.Puts.Long))
    vpβ = Variable.Independent("vpβ", "value", np.float32, locator=StrategyLocator("value", Securities.Options.Puts.Short))
    vcα = Variable.Independent("vcα", "value", np.float32, locator=StrategyLocator("value", Securities.Options.Calls.Long))
    vcβ = Variable.Independent("vcβ", "value", np.float32, locator=StrategyLocator("value", Securities.Options.Calls.Short))

    Δpα = Variable.Independent("Δpα", "delta", np.float32, locator=StrategyLocator("delta", Securities.Options.Puts.Long))
    Δpβ = Variable.Independent("Δpβ", "delta", np.float32, locator=StrategyLocator("delta", Securities.Options.Puts.Short))
    Δcα = Variable.Independent("Δcα", "delta", np.float32, locator=StrategyLocator("delta", Securities.Options.Calls.Long))
    Δcβ = Variable.Independent("Δcβ", "delta", np.float32, locator=StrategyLocator("delta", Securities.Options.Calls.Short))

    Γpα = Variable.Independent("Γpα", "gamma", np.float32, locator=StrategyLocator("gamma", Securities.Options.Puts.Long))
    Γpβ = Variable.Independent("Γpβ", "gamma", np.float32, locator=StrategyLocator("gamma", Securities.Options.Puts.Short))
    Γcα = Variable.Independent("Γcα", "gamma", np.float32, locator=StrategyLocator("gamma", Securities.Options.Calls.Long))
    Γcβ = Variable.Independent("Γcβ", "gamma", np.float32, locator=StrategyLocator("gamma", Securities.Options.Calls.Short))

    Θpα = Variable.Independent("Θpα", "theta", np.float32, locator=StrategyLocator("theta", Securities.Options.Puts.Long))
    Θpβ = Variable.Independent("Θpβ", "theta", np.float32, locator=StrategyLocator("theta", Securities.Options.Puts.Short))
    Θcα = Variable.Independent("Θcα", "theta", np.float32, locator=StrategyLocator("theta", Securities.Options.Calls.Long))
    Θcβ = Variable.Independent("Θcβ", "theta", np.float32, locator=StrategyLocator("theta", Securities.Options.Calls.Short))

    Vpα = Variable.Independent("Vpα", "vega", np.float32, locator=StrategyLocator("vega", Securities.Options.Puts.Long))
    Vpβ = Variable.Independent("Vpβ", "vega", np.float32, locator=StrategyLocator("vega", Securities.Options.Puts.Short))
    Vcα = Variable.Independent("Vcα", "vega", np.float32, locator=StrategyLocator("vega", Securities.Options.Calls.Long))
    Vcβ = Variable.Independent("Vcβ", "vega", np.float32, locator=StrategyLocator("vega", Securities.Options.Calls.Short))

    Ppα = Variable.Independent("Ppα", "rho", np.float32, locator=StrategyLocator("rho", Securities.Options.Puts.Long))
    Ppβ = Variable.Independent("Ppβ", "rho", np.float32, locator=StrategyLocator("rho", Securities.Options.Puts.Short))
    Pcα = Variable.Independent("Pcα", "rho", np.float32, locator=StrategyLocator("rho", Securities.Options.Calls.Long))
    Pcβ = Variable.Independent("Pcβ", "rho", np.float32, locator=StrategyLocator("rho", Securities.Options.Calls.Short))


class VerticalPutGreekEquation(GreeksEquation):
    vo = Variable.Dependent("vo", "value", np.float32, function=lambda vpα, vpβ: vpα - vpβ)
    Δo = Variable.Dependent("Δo", "delta", np.float32, function=lambda Δpα, Δpβ: Δpα - Δpβ)
    Γo = Variable.Dependent("Γo", "gamma", np.float32, function=lambda Γpα, Γpβ: Γpα - Γpβ)
    Θo = Variable.Dependent("Θo", "theta", np.float32, function=lambda Θpα, Θpβ: Θpα - Θpβ)
    Vo = Variable.Dependent("Vo", "vega", np.float32, function=lambda Vpα, Vpβ: Vpα - Vpβ)
    Po = Variable.Dependent("Po", "rho", np.float32, function=lambda Ppα, Ppβ: Ppα - Ppβ)

class VerticalCallGreekEquation(GreeksEquation):
    vo = Variable.Dependent("vo", "value", np.float32, function=lambda vcα, vcβ: vcα - vcβ)
    Δo = Variable.Dependent("Δo", "delta", np.float32, function=lambda Δcα, Δcβ: Δcα - Δcβ)
    Γo = Variable.Dependent("Γo", "gamma", np.float32, function=lambda Γcα, Γcβ: Γcα - Γcβ)
    Θo = Variable.Dependent("Θo", "theta", np.float32, function=lambda Θcα, Θcβ: Θcα - Θcβ)
    Vo = Variable.Dependent("Vo", "vega", np.float32, function=lambda Vcα, Vcβ: Vcα - Vcβ)
    Po = Variable.Dependent("Po", "rho", np.float32, function=lambda Pcα, Pcβ: Pcα - Pcβ)

class CollarLongGreekEquation(GreeksEquation):
    vo = Variable.Dependent("vo", "value", np.float32, function=lambda vpα, vcβ: vpα - vcβ)
    Δo = Variable.Dependent("Δo", "delta", np.float32, function=lambda Δpα, Δcβ: Δpα - Δcβ)
    Γo = Variable.Dependent("Γo", "gamma", np.float32, function=lambda Γpα, Γcβ: Γpα - Γcβ)
    Θo = Variable.Dependent("Θo", "theta", np.float32, function=lambda Θpα, Θcβ: Θpα - Θcβ)
    Vo = Variable.Dependent("Vo", "vega", np.float32, function=lambda Vpα, Vcβ: Vpα - Vcβ)
    Po = Variable.Dependent("Po", "rho", np.float32, function=lambda Ppα, Pcβ: Ppα - Pcβ)

class CollarShortGreekEquation(GreeksEquation):
    vo = Variable.Dependent("vo", "value", np.float32, function=lambda vcα, vpβ: vcα - vpβ)
    Δo = Variable.Dependent("Δo", "delta", np.float32, function=lambda Δcα, Δpβ: Δcα - Δpβ)
    Γo = Variable.Dependent("Γo", "gamma", np.float32, function=lambda Γcα, Γpβ: Γcα - Γpβ)
    Θo = Variable.Dependent("Θo", "theta", np.float32, function=lambda Θcα, Θpβ: Θcα - Θpβ)
    Vo = Variable.Dependent("Vo", "vega", np.float32, function=lambda Vcα, Vpβ: Vcα - Vpβ)
    Po = Variable.Dependent("Po", "rho", np.float32, function=lambda Pcα, Ppβ: Pcα - Ppβ)


class VerticalPutEquation(VerticalPutGreekEquation, StrategyEquation, register=Strategies.Verticals.Put):
    yhτ = Variable.Dependent("yhτ", "maximum", np.float32, function=lambda kpα, kpβ: np.maximum(kpα - kpβ, 0))
    yeτ = Variable.Dependent("yeτ", "expected", np.float32, function=lambda fpα, fpβ: fpα - fpβ)
    ylτ = Variable.Dependent("ylτ", "minimum", np.float32, function=lambda kpα, kpβ: np.minimum(kpα - kpβ, 0))
    yo = Variable.Dependent("yo", "spot", np.float32, function=lambda ypα, ypβ: ypβ - ypα)

    xo = Variable.Dependent("xo", "underlying", np.float32, function=lambda xpα, xpβ: np.divide(xpα + xpβ, 2))
    μo = Variable.Dependent("μo", "trend", np.float32, function=lambda μpα, μpβ: np.divide(μpα + μpβ, 2))
    σo = Variable.Dependent("σo", "volatility", np.float32, function=lambda σpα, σpβ: np.divide(σpα + σpβ, 2))
    qo = Variable.Dependent("qo", "size", np.int32, function=lambda qpα, qpβ: np.minimum(qpα, qpβ))

    fpα = Variable.Dependent("fpα", "function", np.float32, function=lambda dpα, zpα, σo: + dpα * norm.cdf(+zpα) + σo * norm.pdf(+zpα))
    fpβ = Variable.Dependent("fcβ", "function", np.float32, function=lambda dpβ, zpβ, σo: + dpβ * norm.cdf(+zpβ) + σo * norm.pdf(+zpβ))


class VerticalCallEquation(VerticalCallGreekEquation, StrategyEquation, register=Strategies.Verticals.Call):
    yhτ = Variable.Dependent("yhτ", "maximum", np.float32, function=lambda kcα, kcβ: np.maximum(kcβ - kcα, 0))
    yeτ = Variable.Dependent("yeτ", "expected", np.float32, function=lambda fcα, fcβ: fcα - fcβ)
    ylτ = Variable.Dependent("ylτ", "minimum", np.float32, function=lambda kcα, kcβ: np.minimum(kcβ - kcα, 0))
    yo = Variable.Dependent("yo", "spot", np.float32, function=lambda ycα, ycβ: ycβ - ycα)

    xo = Variable.Dependent("xo", "underlying", np.float32, function=lambda xcα, xcβ: np.divide(xcα + xcβ, 2))
    μo = Variable.Dependent("μo", "trend", np.float32, function=lambda μcα, μcβ: np.divide(μcα + μcβ, 2))
    σo = Variable.Dependent("σo", "volatility", np.float32, function=lambda σcα, σcβ: np.divide(σcα + σcβ, 2))
    qo = Variable.Dependent("qo", "size", np.int32, function=lambda qcα, qcβ: np.minimum(qcα, qcβ))

    fcα = Variable.Dependent("fpα", "function", np.float32, function=lambda dcα, zcα, σo: - dcα * norm.cdf(-zcα) + σo * norm.pdf(-zcα))
    fcβ = Variable.Dependent("fcβ", "function", np.float32, function=lambda dcβ, zcβ, σo: - dcβ * norm.cdf(-zcβ) + σo * norm.pdf(-zcβ))


class CollarLongEquation(CollarLongGreekEquation, StrategyEquation, register=Strategies.Collars.Long):
    yhτ = Variable.Dependent("yhτ", "maximum", np.float32, function=lambda kpα, kcβ: + np.maximum(kpα, kcβ))
    yeτ = Variable.Dependent("yeτ", "expected", np.float32, function=lambda xo, fpα, fcβ: fpα - fcβ + xo)
    ylτ = Variable.Dependent("ylτ", "minimum", np.float32, function=lambda kpα, kcβ: + np.minimum(kpα, kcβ))
    yo = Variable.Dependent("yo", "spot", np.float32, function=lambda ypα, ycβ, xo: ycβ - ypα - xo)

    xo = Variable.Dependent("xo", "underlying", np.float32, function=lambda xpα, xcβ: np.divide(xpα + xcβ, 2))
    μo = Variable.Dependent("μo", "trend", np.float32, function=lambda μpα, μcβ: np.divide(μpα + μcβ, 2))
    σo = Variable.Dependent("σo", "volatility", np.float32, function=lambda σpα, σcβ: np.divide(σpα + σcβ, 2))
    qo = Variable.Dependent("qo", "size", np.int32, function=lambda qpα, qcβ: np.minimum(qpα, qcβ))

    fpα = Variable.Dependent("fpα", "function", np.float32, function=lambda dpα, zpα, σo: + dpα * norm.cdf(+zpα) + σo * norm.pdf(+zpα))
    fcβ = Variable.Dependent("fcβ", "function", np.float32, function=lambda dcβ, zcβ, σo: - dcβ * norm.cdf(-zcβ) + σo * norm.pdf(-zcβ))


class CollarShortEquation(CollarShortGreekEquation, StrategyEquation, register=Strategies.Collars.Short):
    yhτ = Variable.Dependent("yhτ", "maximum", np.float32, function=lambda kcα, kpβ: - np.minimum(kcα, kpβ))
    yeτ = Variable.Dependent("yeτ", "expected", np.float32, function=lambda xo, fcα, fpβ: fcα - fpβ - xo)
    ylτ = Variable.Dependent("ylτ", "minimum", np.float32, function=lambda kcα, kpβ: - np.maximum(kcα, kpβ))
    yo = Variable.Dependent("yo", "spot", np.float32, function=lambda ycα, ypβ, xo: ypβ - ycα + xo)

    xo = Variable.Dependent("xo", "underlying", np.float32, function=lambda xcα, xpβ: np.divide(xcα + xpβ, 2))
    μo = Variable.Dependent("μo", "trend", np.float32, function=lambda μcα, μpβ: np.divide(μcα + μpβ, 2))
    σo = Variable.Dependent("σo", "volatility", np.float32, function=lambda σcα, σpβ: np.divide(σcα + σpβ, 2))
    qo = Variable.Dependent("qo", "size", np.int32, function=lambda qcα, qpβ: np.minimum(qcα, qpβ))

    fcα = Variable.Dependent("fcα", "function", np.float32, function=lambda dcα, zcα, σo: - dcα * norm.cdf(-zcα) + σo * norm.pdf(-zcα))
    fpβ = Variable.Dependent("fpβ", "function", np.float32, function=lambda dpβ, zpβ, σo: + dpβ * norm.cdf(+zpβ) + σo * norm.pdf(+zpβ))


class StrategyCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, strategies, **kwargs):
        assert isinstance(strategies, list) and all([value in list(Strategies) for value in list(strategies)])
        super().__init__(*args, **kwargs)
        equations = {strategy: StrategyEquation[strategy] for strategy in strategies}
        calculations = {strategy: Calculation[xr.DataArray](*args, equation=equation, **kwargs) for strategy, equation in equations.items()}
        self.__calculations = calculations

    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return
        settlements = self.keys(options, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        strategies = self.calculate(options, *args, **kwargs)
        size = self.size(strategies, "size")
        self.console(f"{str(settlements)}[{int(size):.0f}]")
        if self.empty(strategies, "size"): return
        yield strategies

    def calculate(self, options, *args, **kwargs):
        print(options)
        raise Exception()

        options = list(self.separator(options, *args, **kwargs))

    def separator(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for dataframe in self.values(options, by=Querys.Settlement):
            yield dict(self.separate(dataframe, *args, **kwargs))

    @staticmethod
    def separate(options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        for security, dataframe in options.groupby(list(Variables.Securities.Security), sort=False):
            if dataframe.empty: continue
            security = Securities(security)
            dataframe = dataframe.drop(columns=list(Variables.Securities.Security))
            dataframe = dataframe.set_index(list(Querys.Settlement) + ["strike"], drop=True, inplace=False)
            dataset = xr.Dataset.from_dataframe(dataframe)
            dataset = reduce(lambda content, axis: content.squeeze(axis), list(Querys.Settlement), dataset)
            dataset = dataset.rename({"strike": str(security)})
            dataset["strike"] = dataset[str(security)]
            yield security, dataset

    @property
    def calculations(self): return self.__calculations


