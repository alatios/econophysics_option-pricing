#!/usr/bin/env python3

import numpy as np
from astropy.io import ascii
from matplotlib import pyplot as plt
import os.path

inputFile = "../outputs/OptionPriceBimodal/OptionPriceBimodal_GatheredResults.dat"
outputDirectory = "../outputs/OptionPriceBimodal/graphs/"

if not os.path.exists(inputFile):
	print("Error: input file", inputFile, "does not exist.")
	exit(1)
	
if not os.path.exists(outputDirectory):
	print("Error: output directory", outputDirectory, "does not exist.")
	exit(2)

data = ascii.read(inputFile, format="basic")
data.sort("m")

plt.plot(data["m"], data["gauss"], color="dimgrey", linestyle="-.", label="Gaussian lognormal")
plt.plot(data["m"], data["bimodal"], color="orchid", label="Bimodal lognormal")
plt.xlabel("Number of fixing dates")
plt.ylabel("Monte Carlo estimated price [EUR]")
plt.xscale("log")
plt.legend()
plt.grid()
plt.savefig(outputDirectory + "OptionPriceBimodal_PriceVsM.pdf", bbox_inches='tight')
plt.close()

print("Analysis completed! You will find the graphs saved in", outputDirectory, ".")
exit(0)
