#!/usr/bin/env python3
#
#
#	Requirements: numpy, astropy, matplotlib
#
#	Data is expected in a file with the following format:
#
#	thread	path	payoff
#	0		0		0
#	0		1		85.3987
#	0		2		28.5695
#	0		3		14.1398
#	...		...		...
#	1		0		29.3849
#	1		1		0
#	1		2		0
#	1		3		19.1862
#	...		...		...
#

import numpy as np
from astropy.io import ascii
from matplotlib import pyplot as plt

inputFile = "test_payoffs.dat"
outputFile = "FinalPayoffDistribution.png"
numberOfBins = 300

## Read the data (this can take a long time)
print("Now beginning data extraction. This may take a long time.")
data = ascii.read(inputFile, format="basic")
print("Data extraction completed! The rest of the analysis should run smoothly.")

## Plot the graph, then save it
plt.hist(data["payoff"], bins=numberOfBins)
plt.xlabel("Underlying final spot price (USD)")
plt.grid(linestyle="--")
plt.yscale("log")
plt.savefig(outputFile)

print("Analysis completed! You will find the graph saved in", outputFile, ".")
