# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 23:44:21 2025

@author: Arthur
"""

import ast

def read_params(param_file):
    params = {}
    with open(param_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                try:
                    params[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    params[key] = value
    return params