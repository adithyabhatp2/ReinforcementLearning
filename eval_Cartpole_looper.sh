#!/usr/bin/env bash


# cd ~/private/gitRepository/Reinf3/

modelDir=./models
trainEps=5000
testEps=1000

outDir=./evals

rm -rf ${outDir}/*
mkdir -p $outDir

#test

for ver in v1 v2
do
	for numLayers in 1
	do
		for numNodes in 20 10 30 40
		do
	    	echo Model - numLayers $numLayers numNodes $numNodes trainEps $trainEps
	    	modelPath=${modelDir}/cartpole_${ver}_l${numLayers}_n${numNodes}_e${trainEps}.h5
	    	echo Path -  ${modelPath}
	    	for rep in 1 2 3 4
	    	do
	    		echo Rep ${rep}
	    		python cartpole_loadModel.py ${modelPath} ${testEps} | tail -1 >> $outDir/cartpole_${ver}_l${numLayers}_n${numNodes}_e${trainEps}.eval
	    	done
		done
	done
done

ver=v3

for numLayers in 1
do
	for numNodes in 20 10 30 40
	do
		for epTime in 200 400 600
    	do
			echo Model - numLayers $numLayers numNodes $numNodes epTime $epTime
			modelPath=${modelDir}/cartpole_${ver}_l${numLayers}_n${numNodes}_e${trainEps}_t${epTime}.h5
			echo Path - ${modelPath}
			for rep in 1 2 3 4
			do
				echo Rep ${rep}
				python cartpole_loadModel.py ${modelPath} ${testEps} | tail -1 >> $outDir/cartpole_${ver}_l${numLayers}_n${numNodes}_e${trainEps}_t${epTime}.eval
			done
    	done
	done
done

