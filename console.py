# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 12 2026
@name:   Finance Logging Objects
@author: Jack Kirby Cook
@file:   finance/logging.py

"""

import logging
from enum import Enum, auto

from support.decorators import Decorator
from support.mixins import Mixin

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["LoggingMixin", "LoggingDecorator", "LoggingSizing"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class LoggingSizing(Enum):
    INLET = auto()
    OUTLET = auto()
    BOTH = auto()


class LoggingMixin(Mixin):
    def __init__(self, *args, categories=None, quantities=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.__categories = categories
        self.__quantities = quantities

    def console(self, content, *args, title, **kwargs):
        pass

    @property
    def categories(self): return self.__categories
    @property
    def quantities(self): return self.__quantities


class LoggingDecorator(Decorator):
    def __new__(cls, *args, title, categories=None, quantities=None, sizing=None, percentage=False, **kwargs):
        pass




