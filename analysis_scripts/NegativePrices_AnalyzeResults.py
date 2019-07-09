#!/usr/bin/env python3
#
#
#	Requirements: numpy, astropy, matplotlib
#
#	Data is expected in the output provided by NegativePrices_GatherResults.sh
#
#

import numpy as np
from astropy.io import ascii
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import PercentFormatter
import os.path

inputFile = "../outputs/NegativePrices/NegativePrices_GatheredResults.dat"
sigmaValues=[0., 0.25, 0.5, 0.75, 1.]
numberOfSImulations=20000000
timeToMaturity=1.
outputDirectory = "../outputs/NegativePrices/graphs/"

if not os.path.exists(inputFile):
	print("Error: input file", inputfile, "does not exist.")
	exit(1)

if not os.path.exists(outputDirectory):
	print("Error: output directory", outputDirectory, "does not exist.")
	exit(2)

data = ascii.read(inputFile, format="basic")
data.sort("sigma")

colorlist=iter(cm.cool(np.linspace(0, 1, len(sigmaValues))))

for selectedSigma in sigmaValues:
    mask = (data["sigma"] == selectedSigma)
    maskeddata = data[mask]
    maskeddata.sort("m")
    currentcolor=next(colorlist)
    plt.plot(timeToMaturity/maskeddata["m"], maskeddata["counter"]/numberOfSImulations, color=currentcolor, marker="", label="$\sigma$ = " + str(selectedSigma), linewidth=1.2)

plt.grid()
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.legend()
plt.xlabel("$\Delta t = \\frac{T}{m}$")
plt.ylabel("Paths producing at least one negative price")
plt.savefig(outputDirectory + "NegativePrices_PercentageVsM_VariousSigmas.pdf", bbox_inches='tight')
plt.close()

print("Analysis completed! You will find the graphs saved in", outputDirectory, ".")
exit(0)
