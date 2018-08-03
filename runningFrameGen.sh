#!/bin/bash

counter=0
for fileName in *.gsd; do
	cd ..
	ovitos getAllTheLastFramesBash.py "$counter" "$fileName"
	let counter+=1
	cd testLFGSDS
done