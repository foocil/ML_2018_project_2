#!/bin/bash

# help		->	display help
# clean		->	clean
# all		->	all pipeline

if [ "$1" == "help" ] || [ -z "$1" ]
then
	echo "Machine Learning Project 2 - Recommandation System"
	echo "Usage: ./ml.sh COMMAND"
	echo "Command list:"
	echo "  all:    Executes all pipeline"
	echo "  clean:  Clean report directors"
	echo "  help:   Duh!"
fi

if [ "$1" == "clean" ]
then
	rm -rf reports/*
fi
