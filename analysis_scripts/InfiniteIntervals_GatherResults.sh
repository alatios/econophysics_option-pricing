#!/bin/bash

OUTPUTFILE="../outputs/InfiniteIntervals/InfiniteIntervals_GatheredResults.dat"

grep "price via exact" ../outputs/InfiniteIntervals/output__* \
	| awk '{gsub("../outputs/InfiniteIntervals/output__","");print}' \
	| awk '{gsub("_"," ");print}' \
	| awk '{gsub(".dat:Monte Carlo estimated price via exact formula \\[EUR\\]:", "");print}' \
	> exactPrice.dat

grep "estimated error via exact" ../outputs/InfiniteIntervals/output__*.dat \
        | sed -n -e 's/^.*]: //p ' \
        > exactError.dat;

grep "price via Euler" ../outputs/InfiniteIntervals/output__*.dat \
        | sed -n -e 's/^.*]: //p ' \
        > eulerPrice.dat;

grep "estimated error via Euler" ../outputs/InfiniteIntervals/output__*.dat \
        | sed -n -e 's/^.*]: //p ' \
        > eulerError.dat;

paste -d" " exactPrice.dat exactError.dat eulerPrice.dat eulerError.dat > ${OUTPUTFILE};
sed -i '1 i\B K exactPrice exactError eulerPrice eulerError' ${OUTPUTFILE};

rm exactPrice.dat exactError.dat eulerPrice.dat eulerError.dat;

exit 0
