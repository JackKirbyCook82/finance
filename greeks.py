# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Greek Objects
@author: Jack Kirby Cook

"""

import pandas as pd

from equations import Equations
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["GreekCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class GreekEquations(object):
    pass


class GreekCalculator(Logging):
    def __call__(self, *args, options, **kwargs):
        assert isinstance(options, pd.DataFrame)



