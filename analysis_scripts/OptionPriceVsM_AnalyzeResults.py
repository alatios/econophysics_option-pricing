#!/usr/bin/env python3
#
#
#	Requirements: numpy, astropy, matplotlib
#
#	Data is expected in the output provided by OptionPriceVsM_GatherResults.sh
#
#

import numpy as np
from astropy.io import ascii
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import PercentFormatter
import os.path
import scipy
from scipy.stats import norm

inputFile = "../outputs/OptionPriceVsM/OptionPriceVsM_GatheredResults.dat"
outputDirectory = "../outputs/OptionPriceVsM/graphs/"

NValues = [1000, 10000, 100000, 1000000, 10000000, 100000000]

if not os.path.exists(inputFile):
	print("Error: input file", inputfile, "does not exist.")
	exit(1)

if not os.path.exists(outputDirectory):
	print("Error: output directory", outputDirectory, "does not exist.")
	exit(2)


## Read the data & sort it by number of intervals (useful for graph representation)
data = ascii.read(inputFile, format="basic")
data.sort("m")

## Apply masks to isolate data corresponding to a defined number of simulations
mask_N100mln = (data["N"] == 100000000)
data_N100mln = data[mask_N100mln]

## Plot price vs. m for highest N
plt.plot(data_N100mln["m"], data_N100mln["exactPrice"], color="black", label="Exact price", marker="", linewidth=2, linestyle="--")
plt.plot(data_N100mln["m"], data_N100mln["eulerPrice"], color="magenta", label="Euler price", marker="", linewidth=2)
plt.xscale("log")
plt.legend()
plt.xlabel("Number of fixing dates")
plt.ylabel("Monte Carlo estimated price [EUR]")
plt.grid()
plt.savefig(outputDirectory + "OptionPriceVsM_PriceVsM_N100mln.pdf", bbox_inches='tight')
plt.close()

## Plot exact price vs. m for all N
colorlist=iter(cm.cool(np.linspace(0, 1, len(NValues))))

for selectedN in NValues:
    mask = (data["N"] == selectedN)
    maskeddata = data[mask]
    maskeddata.sort("m")
    currentcolor=next(colorlist)
    plt.plot(maskeddata["m"], maskeddata["exactPrice"], color=currentcolor, marker="", label="$N = {10}^{" + str(int(np.log10(selectedN))) + "}$", linewidth=1.5)

plt.xscale("log")
plt.legend()
plt.xlabel("Number of fixing dates")
plt.ylabel("Monte Carlo estimated exact price [EUR]")
plt.grid()
plt.savefig(outputDirectory + "OptionPriceVsM_ExactPriceVsM_WithDifferentNs.pdf", bbox_inches='tight')
plt.close()

## Plot Euler price vs. m for all N
colorlist=iter(cm.cool(np.linspace(0, 1, len(NValues))))

for selectedN in NValues:
    mask = (data["N"] == selectedN)
    maskeddata = data[mask]
    maskeddata.sort("m")
    currentcolor=next(colorlist)
    plt.plot(maskeddata["m"], maskeddata["eulerPrice"], color=currentcolor, marker="", label="$N = {10}^{" + str(int(np.log10(selectedN))) + "}$", linewidth=1.5)

plt.xscale("log")
plt.legend()
plt.xlabel("Number of fixing dates")
plt.ylabel("Monte Carlo estimated Euler price [EUR]")
plt.grid()
plt.savefig(outputDirectory + "OptionPriceVsM_EulerPriceVsM_WithDifferentNs.pdf", bbox_inches='tight')
plt.close()

## Plot discrepancies vs. m for all N + line highlighting zero value
colorlist=iter(cm.cool(np.linspace(0, 1, len(NValues))))

for selectedN in NValues:
    mask = (data["N"] == selectedN)
    maskeddata = data[mask]
    maskeddata.sort("m")
    currentcolor=next(colorlist)
    discrepancies = np.array(- maskeddata["exactPrice"] + maskeddata["eulerPrice"])
    plt.plot(maskeddata["m"], discrepancies, color=currentcolor, marker="", label="$N = {10}^{" + str(int(np.log10(selectedN))) + "}$", linewidth=1.5)

plt.grid()
plt.xscale("log")
plt.xlabel("Number of fixing dates")
plt.ylabel("$P_{Euler} - P_{Exact}$")
plt.axhline(0, color="grey", linestyle="--", linewidth=2)
plt.legend()
plt.savefig(outputDirectory + "OptionPriceVsM_DiscrepancyVsM_WithDifferentNs.pdf", bbox_inches='tight')
plt.close()

## Plot exact error vs. m for all N
colorlist=iter(cm.cool(np.linspace(0, 1, len(NValues))))

for selectedN in NValues:
    mask = (data["N"] == selectedN)
    maskeddata = data[mask]
    maskeddata.sort("m")
    currentcolor=next(colorlist)
    plt.plot(maskeddata["m"], maskeddata["exactError"]/maskeddata["exactPrice"], color=currentcolor, marker="", label="$N = {10}^{" + str(int(np.log10(selectedN))) + "}$", linewidth=1.5)

plt.xscale("log")
plt.legend()
plt.xlabel("Number of fixing dates")
plt.ylabel("$\sigma_{Exact}\,/\,P_{Exact}$")
plt.grid()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.savefig(outputDirectory + "OptionPriceVsM_ExactErrorVsM_WithDifferentNs.pdf", bbox_inches='tight')
plt.close()

## Plot Euler error vs. m for all N
colorlist=iter(cm.cool(np.linspace(0, 1, len(NValues))))

for selectedN in NValues:
    mask = (data["N"] == selectedN)
    maskeddata = data[mask]
    maskeddata.sort("m")
    currentcolor=next(colorlist)
    plt.plot(maskeddata["m"], maskeddata["eulerError"]/maskeddata["eulerPrice"], color=currentcolor, marker="", label="$N = {10}^{" + str(int(np.log10(selectedN))) + "}$", linewidth=1.5)

plt.xscale("log")
plt.legend()
plt.xlabel("Number of fixing dates")
plt.ylabel("$\sigma_{Euler}\,/\,P_{Euler}$")
plt.grid()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.savefig(outputDirectory + "OptionPriceVsM_EulerErrorVsM_WithDifferentNs.pdf", bbox_inches='tight')
plt.close()

## Comparison with Bernoulli approximation of N=10^8
B = 1
sigma = 0.25
r = 0.0001
ms = data_N100mln["m"]
deltats = 1./ms

ps = norm.cdf(B - (r-(sigma**2)/2.) * np.sqrt(deltats)/sigma) * (1 - norm.cdf( - B - (r-(sigma**2)/2.) * np.sqrt(deltats)/sigma))
teorerrs = (1/np.sqrt(ms)) * np.sqrt(ps * (1-ps))
rel_teorerrs = teorerrs / data_N100mln["exactPrice"]


plt.plot(data_N100mln["m"], data_N100mln["exactError"]/data_N100mln["exactPrice"], color="orchid", marker="x", label="Monte Carlo error ($N={10}^8$)", linewidth=2.5, markersize=8)
plt.plot(data_N100mln["m"], rel_teorerrs/np.sqrt(10**8), color="dimgrey", marker="", linestyle="--", label="Bernoulli approximation ($N={10}^8$)", linewidth=2)

plt.xscale("log")
plt.legend()
plt.xlabel("Number of fixing dates")
plt.ylabel("$\sigma_{Exact}\,/\,P_{Exact}$")
plt.grid()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.savefig(outputDirectory + "OptionPriceVsM_ExactErrorVsM_N108.pdf", bbox_inches='tight')
plt.close()

print("Analysis completed! You will find the graphs saved in", outputDirectory, ".")
exit(0)
