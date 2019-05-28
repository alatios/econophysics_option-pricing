#!/bin/bash
cd InputMarketData/ &&
make esegui &&
cd ../InputOptionData/ &&
make esegui &&
cd  ../InputMCData/ &&
make esegui &&
cd  ../InputGPUData/ &&
make esegui &&
cd  ../OutputMCData/ &&
make esegui &&
cd ../Path &&
make esegui &&
cd ../PathPerThread &&
make esegui &&
cd ../..
