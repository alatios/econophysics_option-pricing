#!/usr/bin/env python3

import numpy as np
from astropy.io import ascii
from matplotlib import pyplot as plt
from matplotlib import axes
import os.path

inputFile = "CorrelationTests_MainIntraStream.dat"
inputFileIntraStreamAutocorr = "CorrelationTests_Autocorrelations.dat"
inputFileInterStreamAutocorr = "CorrelationTests_InterstreamAutocorrelations.dat"
outputDirectory = "graphs/"

if not os.path.exists(inputFile):
	print("Error: input file", inputfile, "does not exist.")
	exit(1)

if not os.path.exists(inputFileIntraStreamAutocorr):
	print("Error: input file", inputFileIntraStreamAutocorr, "does not exist.")
	exit(1)

if not os.path.exists(inputFileInterStreamAutocorr):
	print("Error: input file", inputFileInterStreamAutocorr, "does not exist.")
	exit(1)
	
if not os.path.exists(outputDirectory):
	print("Error: output directory", inputFileInterStreamAutocorr, "does not exist.")
	exit(2)

data = ascii.read(inputFile, format="basic")

plt.scatter(data["uniavg"], data["gaussavg"], c=data["thread"], cmap="cool", marker="x", s=30, label="")
plt.xlabel("Uniform numbers thread average")
plt.ylabel("Gaussian numbers thread average")
plt.axhline(0, color="dimgrey")
plt.axvline(0.5, color="dimgrey", label="Expected average")
plt.grid()
cbar = plt.colorbar()
cbar.set_label('Thread number', rotation=270, labelpad=+13)
plt.savefig(outputDirectory + "CorrelationTests_UniAvgVsGaussAvg.pdf")
plt.close()

plt.scatter(data["gaussvar"], data["gausskurt"], c=data["thread"], cmap="cool", marker="x", s=30, label="")
plt.xlabel("Gaussian numbers thread variance")
plt.ylabel("Gaussian numbers thread kurtosis")
plt.axhline(3, color="dimgrey")
plt.axvline(1, color="dimgrey", label="Expected average")
plt.grid()
cbar = plt.colorbar()
cbar.set_label('Thread number', rotation=270, labelpad=+13)
plt.savefig(outputDirectory + "CorrelationTests_KurtosisVsVariance.pdf")
plt.close()

autocorrdata = ascii.read(inputFileIntraStreamAutocorr, format="basic")

plt.scatter(autocorrdata["unicorr"], autocorrdata["gausscorr"], c=autocorrdata["offset"], cmap="cool", marker="x", s=20, label="")
plt.grid()
plt.axhline(0, color="dimgrey")
plt.axvline(0.25, color="dimgrey", label="Expected average")
plt.xlabel("$<x_i,x_{i+k}>$, uniform numbers")
plt.ylabel("$<x_i,x_{i+k}>$, gaussian numbers")
cbar = plt.colorbar()
cbar.set_label('Autocorrelation offset $k$', rotation=270, labelpad=+13)
plt.savefig(outputDirectory + "CorrelationTests_IntraStreamCorrelations.pdf")
plt.close()

intercorrdata = ascii.read(inputFileInterStreamAutocorr, format="basic")

plt.scatter(intercorrdata["unicorr"], intercorrdata["gausscorr"], c=intercorrdata["offset"], cmap="cool", marker="x", s=20, label="")
plt.grid()
plt.xlim((0.23, 0.27))
plt.ylim((-0.1,0.1))
plt.axhline(0, color="dimgrey")
plt.axvline(0.25, color="dimgrey", label="Expected average")
plt.xlabel("$<x_i,x_{i+k}>$, uniform numbers")
plt.ylabel("$<x_i,x_{i+k}>$, gaussian numbers")
cbar = plt.colorbar()
cbar.set_label('Autocorrelation offset $k$', rotation=270, labelpad=+13)
plt.savefig(outputDirectory + "CorrelationTests_InterStreamCorrelations.pdf")
plt.close()

plt.scatter(intercorrdata["offset"], intercorrdata["unicorr"], color="skyblue", marker="x", s=20)
plt.ylim((0.23, 0.27))
plt.xlabel("Autocorrelation offset $k$")
plt.ylabel("$<x_i,x_{i+k}>$, uniform numbers")
plt.axhline(0.25, color="dimgrey")
plt.grid()
plt.savefig(outputDirectory + "CorrelationTests_UniformInterStreamAutocorrelationVsOffset.pdf")
plt.close()

plt.hist(intercorrdata["unicorr"], bins=2000, color="skyblue")
plt.xlim((0.247, 0.252))
plt.xlabel("$<x_i,x_{i+k}>$, uniform numbers")
plt.axvline(0.25, color="dimgrey")
plt.grid()
plt.savefig(outputDirectory + "CorrelationTests_UniformInterStreamAutocorrelationHistogram.pdf")
plt.close()

plt.scatter(intercorrdata["offset"], intercorrdata["gausscorr"], color="orchid", marker="x", s=20)
plt.ylim((-0.10, 0.10))
plt.xlabel("Autocorrelation offset $k$")
plt.ylabel("$<x_i,x_{i+k}>$, gaussian numbers")
plt.axhline(0., color="dimgrey")
plt.grid()
plt.savefig(outputDirectory + "CorrelationTests_GaussInterStreamAutocorrelationVsOffset.pdf")
plt.close()

plt.hist(intercorrdata["gausscorr"], bins=2000, color="orchid")
plt.xlim((-0.02, 0.02))
plt.xlabel("$<x_i,x_{i+k}>$, gaussian numbers")
plt.axvline(0.0, color="dimgrey")
plt.grid()
plt.savefig(outputDirectory + "CorrelationTests_GaussInterStreamAutocorrelationHistogram.pdf")
plt.close()

print("Analysis completed! You will find the graphs saved in", outputDirectory, ".")
exit(0)
