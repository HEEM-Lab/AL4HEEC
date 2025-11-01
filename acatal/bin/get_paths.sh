#!/bin/bash
find . -name OUTCAR > paths.log;
sed -i 's/OUTCAR$//g' paths.log;
sed -i "s#^.#${PWD}#g" paths.log;
