import os, sys, fnmatch
import ROOT
from ROOT import *
import re

bins = [0, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 25000]
tpts = [
    247678, 229501, 204182, 182705, 169174, 156493, 157174, 166272, 191051,
    145908
]
geceffs = [
    0.049, 0.110, 0.136, 0.148, 0.142, 0.136, 0.107, 0.056, 0.047, 0.067
]

for i, j in enumerate(tpts):
    print tpts[i] * geceffs[i]
