"""Predicts restoration scenarios"""
import copy
import os
import sys
import time
import numpy as np
import pandas as pd
import gambit
import nashpy as nash

g = gambit.Game.new_table([2,2])
print(gambit.__file__)
# A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
# rps = nash.Game(A)
# print(rps)
# eqs = rps.support_enumeration()
# print(list(eqs))
