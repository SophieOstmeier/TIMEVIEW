#!/bin/bash
python benchmark.py --datasets flchain_1000 --baselines TTS --n_trials 10 --n_tune 100 --seed 0 --device gpu --n_basis 9 --rnn_type lstm