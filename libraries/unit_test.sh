#!/bin/bash
cd InputStructures/InputOptionData/ &&
make run &&
cd ../InputMarketData/ $$
make run &&
cd  ../InputMCData/ &&
make run &&
cd  ../InputGPUData/ &&
make run &&
cd ../../OutputStructures/OutputMCData/ &&
make run &&
cd ../../CoreLibraries/Path &&
make run &&
cd ../Statistics &&
make run &&
cd ../DataStreamManager &&
make run &&
cd ../..
