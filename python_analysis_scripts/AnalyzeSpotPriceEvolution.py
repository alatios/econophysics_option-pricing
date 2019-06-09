#!/usr/bin/env python3
#
#
#	Requirements: numpy, astropy, matplotlib
#
#	Data is expected in a file with the following format:
#
#	thread	path	interval	spotprice
#	0		0		0			100
#	0		0		1			100.934
#	...
#	0		0		365		166.848
#	0		1		0		100
#	0		1		1		99.9513
#	...
#	0		2		365		129.476
#	1		0		0		100
#	1		0		1		98.5811
# 	...
#
#	Only paths 0-2 are required. More are allowed, but they will not be studied.
#	You can selected whatever thread you prefer and customize number of steps.
#

import numpy as np
from astropy.io import ascii
from matplotlib import pyplot as plt

inputFile = "spotpricetest.dat"
outputFile = "SpotPriceEvolution.png"
thread = 7;
numberOfIntervals = 365;

## Read the data (this can take a long time)
print("Now beginning data extraction. This may take a long time.")
cppdata = ascii.read(inputFile, format="basic")
print("Data extraction completed! The rest of the analysis should run smoothly.")

## Parameters
strikeprice = 140;
deltatime = 1. / 365;
riskfreerate = 0.5;
volatility = 0.25;

## Build and apply data masks to create three new tables
cppsinglemask = (cppdata["thread"] == thread) & (cppdata["path"] == 0)
cpptwomask = (cppdata["thread"] == thread) & (cppdata["path"] == 1)
cppthreemask = (cppdata["thread"] == thread) & (cppdata["path"] == 2)

singlerun_cppdata = cppdata[cppsinglemask]
tworun_cppdata = cppdata[cpptwomask]
threerun_cppdata = cppdata[cppthreemask]

## Evaluate risk-free evolution
eulerRiskFree = np.zeros(numberOfIntervals+1)
eulerRiskFree[0] = 100.;

for i in range(1,numberOfIntervals+1):
    eulerRiskFree[i] = eulerRiskFree[i-1] * (1 + deltatime * riskfreerate)

## Plot risk-free evolution, spot prices, strike price
plt.plot(singlerun_cppdata["interval"], eulerRiskFree, color="crimson", linestyle="--", label="Risk free evolution")
plt.plot(singlerun_cppdata["interval"], singlerun_cppdata["spotprice"], label="Run 1")
plt.plot(tworun_cppdata["interval"], tworun_cppdata["spotprice"], label="Run 2", color="green")
plt.plot(threerun_cppdata["interval"], threerun_cppdata["spotprice"], label="Run 3", color="orange")
plt.plot(threerun_cppdata["interval"], np.full(numberOfIntervals+1, strikeprice), label="Strike price",
         color="black", linewidth=4, linestyle="dotted")
plt.grid()
plt.legend()
plt.xlabel("Time [days]")
plt.ylabel("Spot price [USD]")
plt.savefig(outputFile)

print("Analysis completed! You will find the graph saved in", outputFile, ".")
