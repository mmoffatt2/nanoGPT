# NSGA-II Search

Minimal utilities for multi-objective hyperparameter/architecture search using NSGA-II within Evo_GPT.

## Files

- **nsga2.py**: core NSGA-II algorithm and operators
- **search_space.py**: search space and variation (mutation/crossover)
- **remote_trainer.py**: training functions via ssh remote connection
- **test.py**: small example/smoke test

## Run Example

* In test.py, set the TODO section [hosts, user, key_filename]
* Ensure the master machine has its public key stored in the slave machines

* From this folder run `python test.py`

