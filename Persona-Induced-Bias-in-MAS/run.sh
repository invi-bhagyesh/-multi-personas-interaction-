#!/bin/bash

DATASET="cps"
GROUP="1"
MODEL="gpt"
PERSONA1="0"
PERSONA2="1"
INITIAL="T"
TURN="5"

python code/accuracy.py --group $GROUP --model $MODEL
python code/cps.py --group $GROUP --model $MODEL --persona1 $PERSONA1 --persona2 $PERSONA2 --turn 1 --initial $INITIAL
python code/collaboration.py --dataset $DATASET --group $GROUP --model $MODEL --num1 2 --num2 2 --persona1 $PERSONA1 --persona2 $PERSONA2 --turn $TURN --initial $INITIAL