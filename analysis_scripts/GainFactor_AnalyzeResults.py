import numpy as np
from astropy.io import ascii
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import os.path

inputFile = "../outputs/GainFactor_Tesla/GainFactor_GatheredResults.dat"
outputDirectory = "../outputs/GainFactor_Tesla/graphs/"
mvalues = [1, 10, 100, 200, 300, 400]

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
    plt.plot(maskeddata["blocks"], maskeddata["cpu"]/maskeddata["gpu"], color=currentcolor, marker="", label="$m = $" + str(selectedM), linewidth=1)

plt.grid()
plt.legend()
plt.xlabel("Number of GPU blocks")
plt.ylabel("CPU-GPU gain factor")
plt.savefig(outputDirectory + "GainFactor_Tesla_GainFactorVsBlocks_VariousM.pdf", bbox_inches='tight')
plt.close()

print("Analysis completed! You will find the graphs saved in", outputDirectory, ".")
exit(0)
