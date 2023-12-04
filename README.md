# This is the code for the ICDE paper "Uncovering the Limitations of Eliminating Selection Bias for Recommendation: Missing Mechanisms, Disentanglement, and Identifiability".
## Environment Requirement

The code runs well at python 3.8.10. The required packages are as follows:

- pytorch == 1.9.1 + cu111
- numpy == 1.21.5
- scipy == 1.7.3
- pandas == 1.5.0
- cppimport == 22.8.2

## Datasets

We use three public datasets (KuaiRec, Yahoo!R3 and Coat) for real-world experiments and ML-100K dataset (which named u.data in the semi-synthetic/data folder) for semi-synthetic experiments. 

## Run the Code for semi-synthetic experiments
Step 1: Run the complete.ipynb file to recover the whole rating matrix
Step 2: Run the convert.ipynb file to generate propensities and adjust conversion rates.
Step 3: Run the synthetic_final.ipynb file to get the results for varying data sparsity.

## Run the Code for real-world experiments

- For dataset KuaiRec:

```shell
python DT-DR.py --dataset kuai
python DT-IPS.py --dataset kuai
```

- For dataset Yahoo!R3:

```shell
python DT-DR.py --dataset yahoo
python DT-IPS.py --dataset yahoo
```

- For dataset Coat:

```shell
python DT-DR.py --dataset coat
python DT-IPS.py --dataset coat
```
