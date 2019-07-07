#!/usr/bin/env python3
#
#
#	Requirements: numpy, astropy, matplotlib
#
#	Data is expected in the output provided by OptionPriceVsB_GatherResults.sh
#
#

import numpy as np
from astropy.io import ascii
from matplotlib import pyplot as plt
import os.path
from matplotlib.pyplot import cm

inputFile = "../outputs/OptionPriceVsB/optionpricevsb_gatheredresults.dat"
outputDirectory = "../outputs/OptionPriceVsB/graphs/"
BValues = [0, 0.25, 0.33, 0.5, 0.66, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5]

if not os.path.exists(inputFile):
	print("Error: input file", inputfile, "does not exist.")
	exit(1)

if not os.path.exists(outputDirectory):
	print("Error: output directory", outputDirectory, "does not exist.")
	exit(2)


## Read the data & sort it by B (useful for graph representation)
data = ascii.read(inputFile, format="basic")
data.sort("B")

## Apply mask to isolate N = 200mln results for pricing evolution plot
mask_N200mln = (data["N"] == 200000000)
data_N200mln = data[mask_N200mln]
data_N200mln

plt.plot(data_N200mln["B"], data_N200mln["exactPrice"], color="crimson", label="Exact price", marker="x", linewidth=1)
plt.xlabel("$B$")
plt.ylabel("Estimated Monte Carlo exact price [EUR]")
plt.grid()
plt.savefig(outputDirectory + "OptionPriceVsB_PriceVsB_N200mln.pdf")
plt.close()

## Plot exact errors vs. N in log-log scale
colorlist=iter(cm.cool(np.linspace(0, 1, len(BValues))))

for selectedB in BValues:
    mask = (data["B"] == selectedB)
    maskeddata = data[mask]
    maskeddata.sort("N")
    currentcolor=next(colorlist)
    if ((selectedB > 0.25) & (selectedB < 5)):
        plt.plot(maskeddata["N"], maskeddata["exactError"], color=currentcolor, marker="", label="B = " + str(selectedB), linewidth=1)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of simulations")
plt.ylabel("Absolute error on Monte Carlo exact price [EUR]")
plt.title("Error on Monte Carlo exact price vs. $N$")
plt.legend(bbox_to_anchor=(1.01, 1.05))
plt.grid()
plt.savefig(outputDirectory + "OptionPriceVsB_ExactErrorVsN_WithAllBs.pdf")
plt.close()

## Average H(B) for all N, plot as function of B
HBs = []

for selectedB in BValues:
    mask = (data["B"] == selectedB)
    maskeddata = data[mask]
    HBs.append(np.average(maskeddata["exactError"] * np.sqrt(maskeddata["N"])))

plt.plot(BValues, HBs, color="crimson", marker="x")
plt.title("$H(B)$")
plt.xlabel("$B$")
plt.ylabel("$H(B) = \\varepsilon_{MC} \\times \sqrt{N}$")
plt.grid()
plt.savefig(outputDirectory + "OptionPriceVsB_HBVsB.pdf")
plt.close()

print("Analysis completed! You will find the graphs saved in", outputDirectory, ".")
exit(0)
