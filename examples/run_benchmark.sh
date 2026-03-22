#!/usr/bin/env bash
python -m fluxion.benchmarks.mixed_workload --workload mixed --requests 80 --trials 3
python -m fluxion.benchmarks.mixed_workload --workload stress_burst --requests 80 --trials 3
