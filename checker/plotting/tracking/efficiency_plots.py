#!/usr/bin/python

# Script for accessing histograms of reconstructible and
# reconstructed tracks for different tracking categories
# created by PrChecker2
#
# The efficency is calculated usig TGraphAsymmErrors
# and Bayesian error bars
#
# author: Dorothea vom Bruch (dorothea.vom.bruch@cern.ch)
# date:   10/2018
#

import os, sys
import argparse
import ROOT
from ROOT import gStyle
from ROOT import gROOT
from ROOT import TStyle
from ROOT import TLegend
from ROOT import gPad
from ROOT import TMultiGraph
from ROOT import THStack

sys.path.append('../')
from common.LHCbStyle import *
from common.Legend import *

from common.ConfigHistos import *


def getEfficiencyHistoNames():
    return ["eta", "p", "pt", "phi", "nPV"]
    #return ["p"]


def getTrackers():
    return ["Velo", "Upstream", "Forward"]


def getGhostHistoNames():
    # return ["eta", "nPV"] # currently no eta information available from track
    return ["nPV"]


# f = [ROOT.TFile.Open("../../../output/checkerplots/KstEE/PrCheckerPlots.root", "read"),
#      ROOT.TFile.Open("../../../output/checkerplots/KstMuMu/PrCheckerPlots.root", "read"),
#      ROOT.TFile.Open("../../../output/checkerplots/Ds2KKPi/PrCheckerPlots.root", "read"),
#      ROOT.TFile.Open("../../../output/checkerplots/minbias/PrCheckerPlots.root", "read"),
#      ROOT.TFile.Open("../../../output/checkerplots/Bs2PhiPhi/PrCheckerPlots.root", "read")]
f = [ROOT.TFile.Open("../../../output/PrCheckerPlots.root", "read")]
outputfile = ROOT.TFile("../../../plotsfornote_root/efficiency_plots.root",
                        "recreate")

setLHCbStyle()

efficiencyHistoDict = efficiencyHistoDict()
efficiencyHistos = getEfficiencyHistoNames()
ghostHistos = getGhostHistoNames()
ghostHistoDict = ghostHistoDict()
categories = categoriesDict()
cuts = getCuts()
trackers = getTrackers()

for tracker in trackers:
    outputfile.cd()
    trackerDir = outputfile.mkdir(tracker)
    trackerDir.cd()

    for cut in cuts[tracker]:
        cutDir = trackerDir.mkdir(cut)
        cutDir.cd()
        histoBaseName = tracker + "/" + cut + "_"

        # calculate efficiency
        for histo in efficiencyHistos:
            title = "efficiency vs. " + histo + ", " + categories[tracker][
                cut]["title"]
            name = "efficiency vs. " + histo
            canvas = ROOT.TCanvas(name, title)
            ROOT.gPad.SetTicks()
            # get efficiency for not electrons category
            histoName = histoBaseName + "notElectrons_" + efficiencyHistoDict[
                histo]["variable"]
            print("not electrons: " + histoName)
            numeratorName = histoName + "_reconstructed"
            numerator = f[0].Get(numeratorName)
            for infile in f:
                numerator.Add(infile.Get(numeratorName))
            denominatorName = histoName + "_reconstructible"
            denominator = f[0].Get(denominatorName)
            for infile in f:
                denominator.Add(infile.Get(denominatorName))
            print(numerator.GetEntries())
            print(denominator.GetEntries())
            if numerator.GetEntries() == 0 or denominator.GetEntries() == 0:
                continue
            numerator.Sumw2()
            denominator.Sumw2()

            g_efficiency_notElectrons = ROOT.TGraphAsymmErrors()
            g_efficiency_notElectrons.Divide(numerator, denominator,
                                             "cl=0.683 b(1,1) mode")
            if categories[tracker][cut]["plotElectrons"]:
                g_efficiency_notElectrons.SetTitle("efficiency, not electrons")
            else:
                g_efficiency_notElectrons.SetTitle("efficiency")

            # get efficiency for electrons category
            if categories[tracker][cut]["plotElectrons"]:
                histoName = histoBaseName + "electrons_" + efficiencyHistoDict[
                    histo]["variable"]
                print("electrons: " + histoName)
                numeratorName = histoName + "_reconstructed"
                numerator = f[0].Get(numeratorName)
                for infile in f:
                    numerator.Add(infile.Get(numeratorName))
                denominatorName = histoName + "_reconstructible"
                denominator = f[0].Get(denominatorName)
                for infile in f:
                    denominator.Add(infile.Get(denominatorName))
                if numerator.GetEntries() == 0 or denominator.GetEntries(
                ) == 0:
                    continue
                numerator.Sumw2()
                denominator.Sumw2()

                g_efficiency_electrons = ROOT.TGraphAsymmErrors()
                g_efficiency_electrons.Divide(numerator, denominator,
                                              "cl=0.683 b(1,1) mode")
                g_efficiency_electrons.SetTitle("efficiency, electrons")
                g_efficiency_electrons.SetMarkerColor(ROOT.kAzure - 3)
                g_efficiency_electrons.SetLineColor(ROOT.kAzure - 3)

            # draw them both
            mg = TMultiGraph()
            mg.Add(g_efficiency_notElectrons)
            if categories[tracker][cut]["plotElectrons"]:
                mg.Add(g_efficiency_electrons)

            mg.Draw("ap")

            xtitle = efficiencyHistoDict[histo]["xTitle"]
            mg.GetXaxis().SetTitle(xtitle)
            mg.GetYaxis().SetTitle("efficiency")
            mg.GetYaxis().SetRangeUser(0, 1)

            # draw variable distribution in same canvas
            histoName = histoBaseName + "notElectrons_" + efficiencyHistoDict[
                histo]["variable"]
            variableHistoName = histoName + "_reconstructed"
            variable = f[0].Get(variableHistoName)
            for infile in f:
                variable.Add(infile.Get(variableHistoName))
            norm = 0.9 / variable.GetMaximum()
            variable.Scale(norm)
            if categories[tracker][cut]["plotElectrons"]:
                variable.SetTitle(efficiencyHistoDict[histo]["title"] +
                                  " distribution, not electrons")
            else:
                variable.SetTitle(efficiencyHistoDict[histo]["title"] +
                                  " distribution")
            variable.SetLineColor(ROOT.kBlack)
            variable.SetFillColorAlpha(ROOT.kBlack, 0.2)
            variable.Draw("hist bar same")

            if categories[tracker][cut]["plotElectrons"]:
                histoName = histoBaseName + "electrons_" + efficiencyHistoDict[
                    histo]["variable"]
                variableHistoName = histoName + "_reconstructed"
                variable_electrons = f[0].Get(variableHistoName)
                for infile in f :
                    variable_electrons.Add(infile.Get(variableHistoName))
                norm = 0.9 / variable_electrons.GetMaximum()
                variable_electrons.Scale(norm)
                variable_electrons.SetTitle(efficiencyHistoDict[histo]["title"]
                                            + " distribution, electrons")
                variable_electrons.SetLineColor(ROOT.kAzure - 3)
                variable_electrons.SetFillColorAlpha(ROOT.kAzure - 3, 0.2)
                variable_electrons.Draw("hist bar same")

            place = find_place(canvas)
            legend = TLegend(place[0], place[1], place[2], place[3])
            if categories[tracker][cut]["plotElectrons"]:
                legend.AddEntry(g_efficiency_notElectrons,
                                "efficiency, not electrons", "ep")
            else:
                legend.AddEntry(g_efficiency_notElectrons, "efficiency", "ep")
            if categories[tracker][cut]["plotElectrons"]:
                legend.AddEntry(g_efficiency_electrons,
                                "efficiency, electrons", "ep")
                legend.AddEntry(
                    variable, efficiencyHistoDict[histo]["title"] +
                    " distribution, not electrons", "f")
            else:
                legend.AddEntry(
                    variable,
                    efficiencyHistoDict[histo]["title"] + " distribution", "f")
            if categories[tracker][cut]["plotElectrons"]:
                legend.AddEntry(
                    variable_electrons, efficiencyHistoDict[histo]["title"] +
                    " distribution, electrons", "f")
            legend.SetFillColorAlpha(ROOT.kWhite, 0.)
            legend.Draw("same")
            #canvas.PlaceLegend("f")
            #place_legend(canvas)

            canvas.Write()
            cleantitle = categories[tracker][cut]["title"].replace(
                " ", "").replace(",", "_").replace("<", "_")
            canvas.SaveAs("../../../plotsfornote/" + tracker + "Eff" + histo +
                          cleantitle + ".pdf")
            #canvas.Print("../../../output/checkerplots/forreviewdoc/"+histoBaseName.replace("/","_")+efficiencyHistoDict[histo]["variable"]+"_eff.pdf")

    # calculate ghost rate
    histoBaseName = tracker + "/"
    for histo in ghostHistos:
        trackerDir.cd()
        title = "ghost rate vs " + histo
        canvas = ROOT.TCanvas(title, title)
        ROOT.gPad.SetTicks()
        numeratorName = histoBaseName + ghostHistoDict[histo][
            "variable"] + "_Ghosts"
        denominatorName = histoBaseName + ghostHistoDict[histo][
            "variable"] + "_Total"
        print("ghost histo: " + histoBaseName)
        numerator = f[0].Get(numeratorName)
        for infile in f:
            numerator.Add(infile.Get(numeratorName))
        denominator = f[0].Get(denominatorName)
        for infile in f:
            denominator.Add(infile.Get(denominatorName))
        numerator.Sumw2()
        denominator.Sumw2()

        g_efficiency = ROOT.TGraphAsymmErrors()
        g_efficiency.Divide(numerator, denominator, "cl=0.683 b(1,1) mode")

        xtitle = ghostHistoDict[histo]["xTitle"]
        g_efficiency.GetXaxis().SetRangeUser(1, 14)
        g_efficiency.GetXaxis().SetTitle(xtitle)
        g_efficiency.GetYaxis().SetTitle("ghost rate")
        g_efficiency.Draw("ap")

        canvas.Write()
        canvas.SaveAs("../../../plotsfornote/" + tracker + "GhostRate.pdf")
        #canvas.Print("../../../output/checkerplots/forreviewdoc/"+histoBaseName.replace("/","_")+ghostHistoDict[histo]["variable"]+"_ghost.pdf")

outputfile.Write()
outputfile.Close()
for infile in f:
    infile.Close()
