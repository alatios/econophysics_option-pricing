#!/bin/bash
cd InputMarketData/ &&
make run &&
cd ../InputOptionData/ &&
make run &&
cd  ../InputMCData/ &&
make run &&
cd  ../InputGPUData/ &&
make run &&
cd  ../OutputMCData/ &&
make run &&
cd ../Path &&
make run &&
cd ../OutputMCPerThread &&
make run &&
cd ../..
