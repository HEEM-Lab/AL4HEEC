# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:35:39 2023

@author: ZHANG Jun
"""

__version__ = '0.0.1'

from packaging import version
import agat

if version.parse(agat.__version__) < version.parse('9.0.0'):
    raise RuntimeError('Please use agat version >= 9.0.0')

del version, agat
