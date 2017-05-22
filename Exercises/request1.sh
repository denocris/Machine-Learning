#!/bin/bash
#PBS -l nodes=1:ppn=20 -q regular -l walltime=03:00:00
#PBS -N rf

cd /home/cdenobi/P2.11_seed/Exercises

python TripAdvisorGridRF.py
