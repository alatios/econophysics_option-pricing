#!/usr/bin/env python3
#
#
#	Requirements: numpy, astropy, matplotlib
#
#	Data is expected in a file with the following format:
#
#	thread	path	spotprice
#	0		0		204.684
#	0		1		184.178
#	0		2		157.348
#	0		3		193.686
#	...		...		...
#	1		0		137.103
#	1		1		149.818
#	1		2		219.666
#	1		3		203.438
#	...		...		...
#

import numpy as np
from astropy.io import ascii
from matplotlib import pyplot as plt

inputFile = "test_finalspotprices.dat"
outputFile = "FinalSpotPriceDistribution.png"
numberOfBins = 300

## Read the data (this can take a long time)
print("Now beginning data extraction. This may take a long time.")
data = ascii.read(inputFile, format="basic")
print("Data extraction completed! The rest of the analysis should run smoothly.")

## Plot the graph, then save it
plt.hist(data["spotprice"], bins=numberOfBins)
plt.xlabel("Underlying final spot price (USD)")
plt.grid(linestyle="--")
plt.savefig(outputFile)

print("Analysis completed! You will find the graph saved in", outputFile, ".")
