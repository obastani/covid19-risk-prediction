#!/bin/sh

cd Dropbox/RNN_to_improve/covid19-rnn/python/
python main.py
python process.py
python plot.py
#python plot_alt.py
python upload_github.py