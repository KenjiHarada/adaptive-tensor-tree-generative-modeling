# Adaptive Tensor Tree Generative Modeling
Sample codes of the adaptive tensor tree method to construct a generative model for data.

## Description
Based on the tensor tree network with the Born machine framework, we propose a general method for constructing a generative model by expressing the target distribution function as the quantum wave function amplitude represented by a tensor tree. The key idea is dynamically optimizing the tree structure that minimizes the bond mutual information. The proposed method offers enhanced performance and uncovers hidden relational structures in the target data.

We illustrate potential practical applications with four examples:
* random patterns
* QMNIST hand-written digits
* Bayesian networks
* The stock price fluctuation pattern in S&P500
  
In random patterns and QMNIST hand-written digits, strongly correlated variables were concentrated near the center of the network; in Bayesian networks, the causality pattern was identified; and, in the stock price fluctuation pattern, a structure corresponding to the eleven sectors emerged.

### Other information
We call the proposed method for generative modeling an adaptive tensor tree (ATT) method. This repository contains sample Python codes of the ATT method for the above applications with datasets. You can find the details of the ATT method in our preprint, [arXiv:2408.10669](https://arxiv.org/abs/2408.10669).

## Codes
* apply_att.py : Python code to apply the ATT method for generative modeling to a general dataset.
* born_machine.py : Python class for the ATT method, imported by test_general.py

These codes need some Python libraries such as numpy, [torch](https://pytorch.org), and [opt_einsum](https://github.com/dgasmith/opt_einsum).

## Applications
We prepare four folders as follows.
 * Random : artificial random patterns
 * QMNIST : images of hand-written digits
 * Bayesian_Network : artificial data with causal dependencies
 * SP500 : the stock price fluctuation pattern in S&P500

If you unzip the zip file "att_examples.zip", you can extract these folders.

These folders contain a shell script and the Data folder. Please use the shell script "run.sh" to apply the ATT method to the dataset in the Data folder. After running the shell script, the Results folder will contain the optimized tensor tree generative model as a pickle file.

The shell script "run.sh" easily creates the ATT generative models for the "Random" and "Bayesian_Network" cases. However, more computation time is required for the "SP500" and "QMNIST" cases. Specifically, the "QMNIST" case requires significant computation time. The "Random" folder also contains Jupyter notebook examples for plotting NLL values and visualizing the network structure: plot.ipynb and graph.ipynb, respectively.

#### Based dataset
Our dataset of images of hand-written digits is based on [QMNIST](https://github.com/facebookresearch/qmnist).
Our dataset about the stock price fluctuation pattern in S&P500 is based on [Kaggle dataset: Andrew Maranhão, S&P500 Stocks](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks/data).

#### Keywords
tensor tree, generative modeling, network structure optimization, mutual information

#### License
Copyright 2024 Kenji Harada  
[Licensed under the Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
