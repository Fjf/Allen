#!/usr/bin/python

# Draw PV resolution as a function of number of tracks in PV
# author: Vladimir Gligorov (vladimir.gligorov@cern.ch)
# date:   02/2019
#

import os, sys
import argparse
import ROOT
from ROOT import gStyle
from ROOT import gROOT
from ROOT import TStyle
from ROOT import gPad

sys.path.append('../')
from common.LHCbStyle import *
from common.Legend import *
from common.ConfigHistos import *

f = ROOT.TFile.Open("../../../output/GPU_PVChecker.root", "read")
t = f.Get("PV_tree")

setLHCbStyle()

ranges = {"x": 100., "y": 100., "z": 800.}
pvcanv = {}

reshist = {}
for coord in ["x", "y"]:
    reshist[coord] = ROOT.TH2F("reshist" + coord, "reshist" + coord, 15, 0,
                               150, 20, -1. * ranges[coord], ranges[coord])

coord = "z"
reshist[coord] = ROOT.TH2F("reshist" + coord, "reshist" + coord, 15, 0,
                               150, 40, -1. * ranges[coord], ranges[coord])

for entry in range(t.GetEntries()):
    t.GetEntry(entry)
    for coord in ["x", "y", "z"]:
        reshist[coord].Fill(t.ntrinmcpv,
                            1000. * t.__getattr__("diff_" + coord))

if not os.path.isdir("../../../plotsfornote"):
  os.mkdir("../../../plotsfornote")

arr = ROOT.TObjArray()
for coord in ["x","y","z"]:
    pvcanv[coord] = ROOT.TCanvas("pvcanv" + coord, "pvcanv" + coord, 900, 800)
    pvcanv[coord].cd(1)
    gauss = ROOT.TF1("gausx", "gaus(0)", -1. * ranges[coord], ranges[coord])
    gauss.SetParameter(0,10)
    gauss.SetParameter(1,0.5)
    gauss.SetParameter(1,0.9)
    reshist[coord].FitSlicesY(gauss, 0,-1, 0, "R", arr)

    arr[2].GetXaxis().SetTitle("Number of tracks in MC PV")
    arr[2].GetYaxis().SetTitle("Resolution (#mum)")
    arr[2].GetYaxis().SetTitleOffset(0.95)
    maxY = 27.
    if (coord=="z"): maxY = 220.
    arr[2].GetYaxis().SetRangeUser(0, maxY)
    arr[2].DrawCopy()
    pvcanv[coord].SaveAs("../../../plotsfornote/PVRes_" + coord + "_VsNTr.pdf")
