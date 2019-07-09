#!/bin/bash

OUTPUTFILE="../outputs/NegativePrices/NegativePrices_GatheredResults.dat"

echo "sigma m counter" > ${OUTPUTFILE};
grep Negative ../outputs/NegativePrices/output__* \
	| awk '{gsub("../outputs/NegativePrices/output__","");print}' \
	| awk '{gsub("_"," ");print}' \
	| awk '{gsub(".dat:Negative price counter:","");print}' \
	>> ${OUTPUTFILE};

exit 0
