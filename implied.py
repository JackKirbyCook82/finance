# -*- coding: utf-8 -*-
"""
Created on Tues Mar 24 2026
@name:   Implied Objects
@author: Jack Kirby Cook

"""

import pandas as pd

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ImpliedCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class ImpliedCalculator(object):
    def __call__(self, *args, options, **kwargs):
        assert isinstance(options, pd.DataFrame)


