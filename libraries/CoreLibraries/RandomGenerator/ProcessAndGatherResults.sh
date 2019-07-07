#!/bin/bash
#
#	Joins completely verbose output of CorrelationTest.cu in an astropy.io-readable table.
#

INPUTFILE="outputcorrelationtest.dat"
OUTPUTFILE="outputcorrelationtest_processed.dat"

sed -n '/Uniform averages (exp. 0.5):/,/Gaussian averages (exp. 0):/p' ${INPUTFILE} \
	| head -n -2 \
	| tail -n +2 \
	| awk '{gsub("<xi>@thread","");print}' \
	| awk '{gsub(":\t"," ");print}' \
	> uniformavgs.dat;

sed -n '/Gaussian averages (exp. 0):/,/Gaussian variances (exp. 1)/p' ${INPUTFILE} \
	| head -n -2 \
	| tail -n +2 \
	| sed -e '/^.*/s/^.*\(:\t\)//' \
	> gaussavgs.dat;

sed -n '/Gaussian variances (exp. 1)/,/Gaussian kurtosises/p' ${INPUTFILE} \
	| head -n -2 \
	| tail -n +2 \
	| sed -e '/^.*/s/^.*\(:\t\)//' \
	> gaussvar.dat;

sed -n '/Gaussian kurtosises/,/Uniform autocorrelations/p' ${INPUTFILE} \
	| head -n -2 \
	| tail -n +2 \
	| sed -e '/^.*/s/^.*\(:\t\)//' \
	> gausskurt.dat;

paste -d" " uniformavgs.dat gaussavgs.dat gaussvar.dat gausskurt.dat > ${OUTPUTFILE};
sed -i '1 i\thread uniavg gaussavg gaussvar gausskurt' ${OUTPUTFILE};
rm uniformavgs.dat gaussavgs.dat gaussvar.dat gausskurt.dat;
exit 0
