# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Divestiture Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from collections import OrderedDict as ODict

from finance.variables import Strategies
from finance.targets import TargetHoldings, TargetReader, TargetWriter, TargetTable

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DivestitureWriter", "DivestitureReader", "DivestitureTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class DivestitureTable(TargetTable): pass
class DivestitureReader(TargetReader): pass
class DivestitureWriter(TargetWriter):
    def execute(self, query, *args, **kwargs):
        valuations, holdings = query.valuations, query.holdings
        assert isinstance(valuations, pd.DataFrame) and isinstance(holdings, pd.DataFrame)
        market = self.market(valuations, *args, **kwargs)
        market = self.prioritize(market, *args, **kwargs)
        if bool(market.empty):
            return
        portfolio = self.portfolio(holdings)
        market = self.closures(market, portfolio)
        if bool(market.empty):
            return
        self.write(market, *args, **kwargs)

    def closures(self, market, portfolio):
        divesting = self.divesting(portfolio)
        function = lambda series: bool(divesting - self.acquiring(series))
        mask = market.apply(function)
        market = market.where(mask).dropna(axis=0, how="all")
        market = market.reset_index(drop=True, inplace=False)
        return market

    @staticmethod
    def acquiring(market):
        assert isinstance(market, pd.Series)
        acquisitions = market.to_dict(into=ODict)
        contract = dict(ticker=acquisitions["ticker"], expire=acquisitions["expire"])
        securities = Strategies[acquisitions["strategy"]].securities
        options = {security: market.get(str(security), np.NaN) for security in securities}
        options = {security: strike for security, strike in options.items() if not np.isnan(strike)}
        options = [dict(instrument=security.instrument, position=security.position, strike=strike) for security, strike in options.items()]
        acquisitions = [contract | option for option in options]
        return TargetHoldings(acquisitions)

    @staticmethod
    def divesting(portfolio):
        assert isinstance(portfolio, pd.DataFrame)
        divestitures = list(portfolio.to_dict("records"))
        contracts = [dict(ticker=divestiture["ticker"], expire=divestiture["expire"]) for divestiture in divestitures]
        options = [dict(security=divestiture["security"], position=divestiture["position"], strike=divestiture["strike"]) for divestiture in divestitures]
        divestitures = [contract | option for contract, option in zip(contracts, options)]
        return TargetHoldings(divestitures)

    @staticmethod
    def portfolio(holdings):
        index = ["security", "position", "strike", "ticker", "expire", "date"]
        holdings = holdings.set_index(index, drop=True, inplace=False)
        holdings = holdings.loc[holdings.index.repeat(holdings["quantity"])]
        holdings = holdings.reset_index(drop=True, inplace=False)[index]
        return holdings



