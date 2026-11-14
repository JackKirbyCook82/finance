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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        name = self.__class__.__name__
        self.__name = kwargs.get("name", name)
        self.__logger = __logger__

    def console(self, *strings, title, **kwargs):
        string = ", ".join(strings)
        string = f"{str(title)}: {str(self.name)}: {str(string)}"
        self.logger.info(string)

    @property
    def logger(self): return self.__logger
    @property
    def name(self): return self.__name


class LoggingDecorator(Decorator):
    def __init__(self, *collections, title, sizing=None, percentage=False, **kwargs):
        super().__init__(**kwargs)
        self.__collections = collections
        self.__percentage = percentage
        self.__sizing = sizing
        self.__title = title
        self.__logger = __logger__

    def decorator(self, inlet, *args, **kwargs):
        size = lambda content: len(content) if content is not None else 0
        outlet = super().decorator(inlet, *args, **kwargs)
        strings = list()
        for collection in self.collections:
            try: strings.append(str(collection(outlet)))
            except (KeyError, AttributeError): pass
        try: name = str(self.instance.name)
        except AttributeError: name = str(self.instance.__class__.__name__)
        string = f"{str(self.title)}: {str(name)}: {', '.join(strings)}"
        if self.sizing is None: pass
        elif self.sizing == LoggingSizing.BOTH: string += f" {size(inlet):.0f}->{size(outlet):.0f}"
        elif self.sizing == LoggingSizing.INLET: string += f" {size(inlet):.0f}"
        elif self.sizing == LoggingSizing.OUTLET: string += f" {size(outlet):.0f}"
        else: raise ValueError(self.sizing)
        if self.percentage and inlet is not None: string += f" {(size(inlet) / size(outlet)) * 100:.0f}%"
        self.logger.info(string)
        return outlet

    @property
    def collections(self): return self.__collections
    @property
    def percentage(self): return self.__percentage
    @property
    def logger(self): return self.__logger
    @property
    def sizing(self): return self.__sizing
    @property
    def title(self): return self.__title



