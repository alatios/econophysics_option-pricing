#!/usr/bin/env python3
#
#
#	Requirements: numpy, astropy, matplotlib
#
#	Data is expected in the output provided by ComputationTimeStudies_Tesla_GatherResults.sh
#
#

import numpy as np
from astropy.io import ascii
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import os.path

inputFile = "../outputs/ComputationTimeStudies_Tesla/ComputationTime_Tesla_GatheredResults.dat"
mvalues = [1, 10, 100, 200, 300, 400]
outputDirectory = "../outputs/ComputationTimeStudies_Tesla/graphs/"

if not os.path.exists(inputFile):
	print("Error: input file", inputfile, "does not exist.")
	exit(1)

if not os.path.exists(outputDirectory):
	print("Error: output directory", outputDirectory, "does not exist.")
	exit(2)

data = ascii.read(inputFile, format="basic")
data.sort("blocks")

colorlist=iter(cm.cool(np.linspace(0, 1, len(mvalues))))

for selectedM in mvalues:
    mask = (data["m"] == selectedM)
    maskeddata = data[mask]
    maskeddata.sort("blocks")
    currentcolor=next(colorlist)
    plt.plot(maskeddata["blocks"], maskeddata["gputime"]/1000., color=currentcolor, marker="", label="$m = $" + str(selectedM), linewidth=1)

plt.xlabel("Number of GPU blocks")
plt.ylabel("GPU computation time [sec]")
plt.axvline(14, color="dimgrey", linestyle="--", linewidth=1.5)
plt.axvline(2*14, color="dimgrey", linestyle="--", linewidth=1.5)
plt.axvline(3*14, color="dimgrey", linestyle="--", linewidth=1.5)
plt.axvline(4*14, color="dimgrey", linestyle="--", linewidth=1.5)
plt.axvline(5*14, color="dimgrey", linestyle="--", linewidth=1.5)
plt.axvline(6*14, color="dimgrey", linestyle="--", linewidth=1.5)
plt.legend()
plt.grid()
plt.savefig(outputDirectory + "ComputationTime_Tesla_GPUTimeVsNOfBlocks_VariousM.pdf", bbox_inches='tight')
plt.close()

print("Analysis completed! You will find the graphs saved in", outputDirectory, ".")
exit(0)
