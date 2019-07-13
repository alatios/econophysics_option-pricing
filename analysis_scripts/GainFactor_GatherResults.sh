#!/bin/bash

OUTPUTFILE="../outputs/GainFactor_Tesla/GainFactor_GatheredResults.dat"

awk '{print FILENAME ":" $0}' ../outputs/GainFactor_Tesla/output__* \
	| sed -n '/## HOST OUTPUT MONTE CARLO DATA ##/,/## DEVICE OUTPUT MONTE CARLO DATA ##/p' \
	| grep "Computation time" \
	| awk '{gsub("../outputs/GainFactor_Tesla/output__","");print}' \
	| awk '{gsub("_"," ");print}' \
	| awk '{gsub(".dat:Computation time \\[ms\\]:","");print}' \
	> CPU.dat;

awk '{print FILENAME ":" $0}' ../outputs/GainFactor_Tesla/output__* \
	| sed -n '/## DEVICE OUTPUT MONTE CARLO DATA ##/,/Negative price counter/p' \
	| grep "Computation time" \
	| sed -e '/^.*/s/^.*\(\[ms\]: \)//' \
	> GPU.dat

paste -d" " CPU.dat GPU.dat > ${OUTPUTFILE};
sed -i '1 i\blocks m cpu gpu' ${OUTPUTFILE};
rm GPU.dat CPU.dat;

exit 0
