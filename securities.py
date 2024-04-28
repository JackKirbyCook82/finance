# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np

from support.pipelines import Processor
from support.filtering import Filter
from support.files import Files

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFilter", "OptionFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


options_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
options_columns = {"price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}


class OptionFile(Files.Dataframe, variable="options", index=options_index, columns=options_columns): pass
class SecurityFilter(Filter, Processor, title="Filtered"):
    def execute(self, contents, *args, **kwargs):
        assert isinstance(contents, dict)
        contract, options = contents["contract"], contents["options"]
        if self.empty(options["size"]):
            return
        prior = self.size(options["size"])
        mask = self.mask(options)
        options = self.where(options, mask)
        post = self.size(options["size"])
        __logger__.info(f"Filter: {repr(self)}|{str(contract)}[{prior:.0f}|{post:.0f}]")
        yield contents | dict(options=options)



