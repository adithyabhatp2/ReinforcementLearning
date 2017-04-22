#!/usr/bin/env bash


#cd ~/private/gitRepository/Reinf2/
mkdir -p models

numEps=5000

#train

for ver in v1 v2
do
	outDir=./runs_${ver}
	mkdir -p ${outDir}

	for numLayers in 1 2
	do
		for numNodes in 20 10 30 40
		do
	    	echo numLayers $numLayers numNodes $numNodes ${outDir}/cartpole_${ver}_l${numLayers}_n${numNodes}_e${numEps}.out
	    	python cartpole_${ver}.py $numLayers $numNodes $numEps > ${outDir}/cartpole_${ver}_l${numLayers}_n${numNodes}_e${numEps}.out
		done
	done
done

ver=v3
outDir=./runs_${ver}
mkdir -p ${outDir}

for numLayers in 1
do
	for numNodes in 20 10 30 40
	do
		for epTime in 200 400 600	
    	do
			echo numLayers $numLayers numNodes $numNodes epTime $epTime ${outDir}/cartpole_${ver}_l${numLayers}_n${numNodes}_e${numEps}_t${epTime}.out
			python cartpole_${ver}.py $numLayers $numNodes $numEps $epTime > ${outDir}/cartpole_${ver}_l${numLayers}_n${numNodes}_e${numEps}_t${epTime}.out
    	done
	done
done

#test
