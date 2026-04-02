# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Greek Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from numba import njit
from types import SimpleNamespace
from collections import OrderedDict as ODict

from finance.equations import Equations
from finance.concepts import Concepts
from support.mixins import Logging
from support.concepts import DateRange

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["GreekCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


value = Equations.Greeks.Value
delta = Equations.Greeks.Delta
gamma = Equations.Greeks.Gamma
theta = Equations.Greeks.Theta
rho = Equations.Greeks.Rho
vega = Equations.Greeks.Vega
vomma = Equations.Greeks.Vomma
vanna = Equations.Greeks.Vanna
charm = Equations.Greeks.Charm


@njit(cache=True)
def calculation(x, k, τ, σ, i, r, n):
    y = np.empty(n, dtype=np.float64)
    Δ = np.empty(n, dtype=np.float64)  # Delta, Δ = dy/dx
    Γ = np.empty(n, dtype=np.float64)  # Gamma, Γ = d²y/dx²
    Θ = np.empty(n, dtype=np.float64)  # Theta, Θ = dy/dτ
    Ρ = np.empty(n, dtype=np.float64)  # Rho, Ρ = dy/dr
    V = np.empty(n, dtype=np.float64)  # Vega, V = dy/dσ
    Φ = np.empty(n, dtype=np.float64)  # Vomma, Φ = d²y/dσ²
    Ψ = np.empty(n, dtype=np.float64)  # Vanna, Ψ = d²y/dx*dσ
    Χ = np.empty(n, dtype=np.float64)  # Charm, Χ = d²y/dx*dτ

    for idx in range(n):
        y[idx] = value(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Δ[idx] = delta(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Γ[idx] = gamma(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Θ[idx] = theta(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Ρ[idx] = rho(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        V[idx] = vega(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Φ[idx] = vomma(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Ψ[idx] = vanna(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
        Χ[idx] = charm(x[idx], k[idx], τ[idx], σ[idx], i[idx], r)
    return y, Δ, Γ, Θ, Ρ, V, Φ, Ψ, Χ


class GreekCalculator(Logging):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inlet = ODict(x="underlying", k="strike", τ="tau", σ="volatility", i="option", r="interest")
        outlet = ODict(y="value", Δ="delta", Γ="gamma", Θ="theta", Ρ="rho", V="vega", Φ="vomma", Ψ="vanna", Χ="charm")
        variables = SimpleNamespace(inlet=inlet, outlet=outlet)
        self.__variables = variables

    def __call__(self, options, *args, interest, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if bool(options.empty): return options
        x = options["underlying"].to_numpy(np.float64)
        k = options["strike"].to_numpy(np.float64)
        τ = options["tau"].to_numpy(np.float64)
        σ = options["volatility"].to_numpy(np.float64)
        i = options["option"].apply(int).to_numpy(np.float64)
        greeks = list(calculation(x, k, τ, σ, i, float(interest), len(options)))
        greeks = dict(zip(self.variables.outlet.values(), greeks))
        options = pd.concat([options,  pd.DataFrame(greeks)], axis=1)
        self.alert(options)
        return options

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.OPTION).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        self.console("Filtered", f"{str(instrument)}[{str(tickers)}, {str(expires)}, {len(dataframe):.0f}]")

    @property
    def variables(self): return self.__variables



