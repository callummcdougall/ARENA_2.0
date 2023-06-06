#!/bin/bash

# read first argument as file and use wc to find out how many lines it has
lines=$(cat "$1" | wc -l)

# check if lines is greater than 25
if [ "$lines" -gt 25 ]; then
    exit
fi

echo $(cat "$1" | grep filename | cut -c 12-34) " " $(cat "$1" | grep '<name' | cut -c 9-17)