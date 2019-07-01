#!/bin/bash
cd InputStructures/InputOptionData/ &&
make run &&
cd  ../InputMCData/ &&
make run &&
cd  ../InputGPUData/ &&
make run &&
cd ../../CoreLibraries/Path &&
make run &&
cd ../Statistics &&
make run &&
echo -e "\n\n\n\n\n\nCORRELATION TESTS\n\n\n\n\n" &&
cd ../RandomGenerator &&
make run &&
cd ../..
