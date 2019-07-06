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
import os.path

inputFile = "../outputs/OptionPriceVsM/optionpricevsm_gatheredresults.dat"
outputDirectory = "../outputs/OptionPriceVsM/graphs/"

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
mask_N5mln = (data["N"] == 5000000)
data_N5mln = data[mask_N5mln]

mask_N10mln = (data["N"] == 10000000)
data_N10mln = data[mask_N10mln]

mask_N50mln = (data["N"] == 50000000)
data_N50mln = data[mask_N50mln]

mask_N100mln = (data["N"] == 100000000)
data_N100mln = data[mask_N100mln]

mask_N200mln = (data["N"] == 200000000)
data_N200mln = data[mask_N200mln]

## Plot price vs. m for highest N
plt.plot(data_N200mln["m"], data_N200mln["exactPrice"], color="crimson", label="Exact price", marker="x", linewidth=0)
plt.plot(data_N200mln["m"], data_N200mln["eulerPrice"], color="black", label="Euler price", marker="+", linewidth=0)
plt.xscale("log")
plt.legend()
plt.xlabel("Number of fixing dates")
plt.ylabel("Monte Carlo estimated price [EUR]")
plt.grid()
plt.title("Estimated Monte Carlo price vs. $m$, $N=200$mln")
plt.savefig(outputDirectory+"OptionPriceVsM_PriceVsM_N200mln.pdf")
plt.close()

## Plot exact price vs. m for all N
plt.plot(data_N5mln["m"], data_N5mln["exactPrice"], color="darkorange", label="N = 5mln", marker="x", linewidth=0)
plt.plot(data_N10mln["m"], data_N10mln["exactPrice"], color="gold", label="N = 10mln", marker="x", linewidth=0)
plt.plot(data_N50mln["m"], data_N50mln["exactPrice"], color="green", label="N = 50mln", marker="x", linewidth=0)
plt.plot(data_N100mln["m"], data_N100mln["exactPrice"], color="blue", label="N = 100mln", marker="x", linewidth=0)
plt.plot(data_N200mln["m"], data_N200mln["exactPrice"], color="indigo", label="N = 200mln", marker="x", linewidth=0)
plt.xscale("log")
plt.legend()
plt.xlabel("Number of fixing dates")
plt.ylabel("Monte Carlo estimated exact price [EUR]")
plt.grid()
plt.title("Monte Carlo exact price vs. $m$")
plt.savefig(outputDirectory+"OptionPriceVsM_ExactPriceVsM_WithDifferentNs.pdf")
plt.close()

## Plot Euler price vs. m for all N
plt.plot(data_N5mln["m"], data_N5mln["eulerPrice"], color="darkorange", label="N = 5mln", marker="+", linewidth=0)
plt.plot(data_N10mln["m"], data_N10mln["eulerPrice"], color="gold", label="N = 10mln", marker="+", linewidth=0)
plt.plot(data_N50mln["m"], data_N50mln["eulerPrice"], color="green", label="N = 50mln", marker="+", linewidth=0)
plt.plot(data_N100mln["m"], data_N100mln["eulerPrice"], color="blue", label="N = 100mln", marker="+", linewidth=0)
plt.plot(data_N200mln["m"], data_N200mln["eulerPrice"], color="indigo", label="N = 200mln", marker="+", linewidth=0)
plt.xscale("log")
plt.legend()
plt.xlabel("Number of fixing dates")
plt.ylabel("Monte Carlo estimated Euler price [EUR]")
plt.grid()
plt.title("Monte Carlo Euler price vs. $m$")
plt.savefig(outputDirectory+"OptionPriceVsM_EulerPriceVsM_WithDifferentNs.pdf")
plt.close()

## Evaluate discrepancies between exact and Euler prices for each N 
discrepancies_5mln = np.array(data_N5mln["exactPrice"] - data_N5mln["eulerPrice"])/np.sqrt(data_N5mln["eulerError"]**2 + data_N5mln["exactError"]**2)
discrepancies_10mln = np.array(data_N10mln["exactPrice"] - data_N10mln["eulerPrice"])/np.sqrt(data_N10mln["eulerError"]**2 + data_N10mln["exactError"]**2)
discrepancies_50mln = np.array(data_N50mln["exactPrice"] - data_N50mln["eulerPrice"])/np.sqrt(data_N50mln["eulerError"]**2 + data_N50mln["exactError"]**2)
discrepancies_100mln = np.array(data_N100mln["exactPrice"] - data_N100mln["eulerPrice"])/np.sqrt(data_N100mln["eulerError"]**2 + data_N100mln["exactError"]**2)
discrepancies_200mln = np.array(data_N200mln["exactPrice"] - data_N200mln["eulerPrice"])/np.sqrt(data_N200mln["eulerError"]**2 + data_N200mln["exactError"]**2)

## Plot discrepancies vs. m for all N + line highlighting zero value
plt.plot(data_N5mln["m"], discrepancies_5mln, color="darkorange", marker="x", label="N = 5 mln", linewidth=0.8)
plt.plot(data_N10mln["m"], discrepancies_10mln, color="gold", marker="x", label="N = 10 mln", linewidth=0.8)
plt.plot(data_N50mln["m"], discrepancies_50mln, color="green", marker="x", label="N = 50 mln", linewidth=0.8)
plt.plot(data_N100mln["m"], discrepancies_100mln, color="blue", marker="x", label="N = 100 mln", linewidth=0.8)
plt.plot(data_N200mln["m"], discrepancies_200mln, color="indigo", marker="x", label="N = 200 mln", linewidth=0.8)
plt.grid()
plt.xscale("log")
plt.xlabel("Number of fixing dates")
plt.ylabel("$\\frac{P(Exact) - P(Euler)}{\sqrt{(\sigma^2_{Euler} + \sigma^2_{Exact})}}$")
plt.title("Discrepancy between exact and Euler lognormal prices in units of sigma")
plt.axhline(0, color="crimson", linestyle="--")
plt.legend()
plt.savefig(outputDirectory+"OptionPriceVsM_DiscrepancyVsM_WithDifferentNs.pdf")
plt.close()

## Plot exact error vs. m for all N
plt.plot(data_N5mln["m"], data_N5mln["exactError"]/data_N5mln["exactPrice"] * 100, color="darkorange", label="N = 5mln", marker="x", linewidth=1)
plt.plot(data_N10mln["m"], data_N10mln["exactError"]/data_N10mln["exactPrice"] * 100, color="gold", label="N = 10mln", marker="x", linewidth=1)
plt.plot(data_N50mln["m"], data_N50mln["exactError"]/data_N50mln["exactPrice"] * 100, color="green", label="N = 50mln", marker="x", linewidth=1)
plt.plot(data_N100mln["m"], data_N100mln["exactError"]/data_N100mln["exactPrice"] * 100, color="blue", label="N = 100mln", marker="x", linewidth=1)
plt.plot(data_N200mln["m"], data_N200mln["exactError"]/data_N200mln["exactPrice"] * 100, color="indigo", label="N = 200mln", marker="x", linewidth=1)
plt.xscale("log")
plt.legend()
plt.xlabel("Number of fixing dates")
plt.ylabel("$\\frac{\sigma_{Exact}}{P(Exact)}$ [%]")
plt.grid()
plt.title("Monte Carlo error on exact formula vs. $m$")
plt.savefig(outputDirectory+"OptionPriceVsM_ExactErrorVsM_WithDifferentNs.pdf")
plt.close()

## Plot Euler error vs. m for all N
plt.plot(data_N5mln["m"], data_N5mln["eulerError"]/data_N5mln["eulerPrice"] * 100, color="darkorange", label="N = 5mln", marker="x", linewidth=0.8)
plt.plot(data_N10mln["m"], data_N10mln["eulerError"]/data_N10mln["eulerPrice"] * 100, color="gold", label="N = 10mln", marker="x", linewidth=0.8)
plt.plot(data_N50mln["m"], data_N50mln["eulerError"]/data_N50mln["eulerPrice"] * 100, color="green", label="N = 50mln", marker="x", linewidth=0.8)
plt.plot(data_N100mln["m"], data_N100mln["eulerError"]/data_N100mln["eulerPrice"] * 100, color="blue", label="N = 100mln", marker="x", linewidth=0.8)
plt.plot(data_N200mln["m"], data_N200mln["eulerError"]/data_N200mln["eulerPrice"] * 100, color="indigo", label="N = 200mln", marker="x", linewidth=0.8)
plt.xscale("log")
plt.legend()
plt.xlabel("Number of fixing dates")
plt.ylabel("$\\frac{\sigma_{Euler}}{P(Euler)}$ [%]")
plt.grid()
plt.title("Monte Carlo error on Euler formula vs. $m$")
plt.savefig(outputDirectory+"OptionPriceVsM_EulerErrorVsM_WithDifferentNs.pdf")
plt.close()

print("Analysis completed! You will find the graphs saved in", outputDirectory, ".")
exit(0)
