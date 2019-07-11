#!/bin/bash

awk '{print FILENAME ":" $0}' ../outputs/GainFactor_Tesla/output__* \
	| sed -n '/## HOST OUTPUT MONTE CARLO DATA ##/,/## DEVICE OUTPUT MONTE CARLO DATA ##/p' \
	| grep "Computation time" \
	| awk '{gsub("../outputs/GainFactor_Tesla/output__","");print}' \
	| awk '{gsub("_"," ");print}' \
	| awk '{gsub(".dat:Computation time \\[ms\\]:","");print}' \
	> CPU.dat;

exit 0
