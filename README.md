# Intro

This repository contains all of the code for the master thesis "Using Deep Learning Methods for Side Channel Attacks" at Faculty of Mathematics, University of Belgrade by Aleksandar Vracarevic.

## Motivation

Cryptographical algorithms in use today are proved to be secure from the mathematical standpoint, they're still executed on the physical devices that could be succeptible to Side Channel Attacks. This category of attacks uses the phyisical characteristics of the device, such as power consumption, electromagnetic radiation, heat radiation, execution timing during the execution of the crypto algorithms. These side channels could be exploitet in order to gain information about the secret variables such as secret keys. Even though the SCA is known since the end of the last century, they have gained prominence with the development of machine learning. Most research in this area uses multi-layer perceptrons and convolutional neural networks. Development of the attention mechanisms, especially within Transformers models has shown a potential for usage in SCA.

## Goal of thesis



## Side Channel Attacks

## Deep Learning

## Prerequisites

* [Ledger's rainbow tool](https://github.com/Ledger-Donjon/rainbow) - for trace generation
  * To install it, first clone the repo `git clone https://github.com/Ledger-Donjon/rainbow.git`, then `pip install .` from the cloned directory.
* [Ledger's lascar tool](https://github.com/Ledger-Donjon/lascar/tree/master) - for trace containers
  * To install it, run `pip3 install "git+https://github.com/Ledger-Donjon/lascar.git"`
* (_Optional_) jupyter notebook for running locally.