#!/bin/bash
#PBS -l nodes=20:ppn=20 -q regular -l walltime=00:15:00
#PBS -N fft

cd /home/cdenobi/P2.11_seed/Exercises

python TripAdvisorGridRF_test.py
