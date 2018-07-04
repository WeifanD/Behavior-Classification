#! /bin/sh
# $1 basepath 
# $2 input: eg /opt/rwork/works/script_001_behavioral/20180517/155918_548199/input.csv 
# $3 output: eg /opt/rwork/works/script_001_behavioral/20180517/155918_548199/out
# $4 date: 2018-05-17

tmp_input=$1/input/
mkdir -p $tmp_input
Rscript $1/plot.r $2 $tmp_input $3
/root/miniconda3/bin/python $1/test.py $1/train_model.m $tmp_input
