#!/bin/bash

OUTPUTFILE="../outputs/OptionPriceBimodal/OptionPriceBimodal_GatheredResults.dat"

grep "price via exact" ../outputs/OptionPriceBimodal/output__*g.dat \
	| awk '{gsub("../outputs/OptionPriceBimodal/output__","");print}' \
	| awk '{gsub("_g.dat:Monte Carlo estimated price via exact formula \\[EUR\\]:","");print}' \
	> gaussres.dat;
	
grep "price via exact" ../outputs/OptionPriceBimodal/output__*b.dat \
	| sed -n -e 's/^.*]: //p ' \
	> bimodres.dat;
	
paste -d" " gaussres.dat bimodres.dat > ${OUTPUTFILE};
sed -i '1 i\m gauss bimodal' ${OUTPUTFILE};

rm gaussres.dat bimodres.dat;

exit 0
