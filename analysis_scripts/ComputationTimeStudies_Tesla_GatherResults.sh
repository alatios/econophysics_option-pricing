#!/bin/bash

OUTPUTFILE="../outputs/ComputationTimeStudies_Tesla/ComputationTime_Tesla_GatheredResults.dat"

echo "blocks m gputime" > ${OUTPUTFILE};

grep "Computation time" ../outputs/ComputationTimeStudies_Tesla/output__* \
	| awk '{gsub("../outputs/ComputationTimeStudies_Tesla/output__","");print}' \
	| awk '{gsub("_"," ");print}' \
	| awk '{gsub(".dat:Computation time \\[ms\\]:","");print}' \
	>> ${OUTPUTFILE};

exit 0
