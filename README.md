# Information Retrieval Project (Group 17)

This repository contains our code for the project: "Measuring the effect of Quantization using Locality Sensitive Hashing"

## Project Structure

The notebooks we used for developing the Locality-Sensitive Hashing (LSH) based search index are located in [Development](/Development/).

The [Measurements](/Measurements/) folder contains our finished [LSH index](/Measurements/LSH.py) and our [quantizers](/Measurements/quantizers.py). The measurements we conducted were done in [measurements](/Measurements/measurements.ipynb).

Data such as the TREC documents and vectors are excluded from this repository.

## TREC results

For our measurements we used the [TREC evaluation tools](https://github.com/usnistgov/trec_eval/tree/main), our created run files and evaluation results are stored in [Measurements/TREC](/Measurements/Trec/), the files are stored as `name_r_x`. Where `r` is the value of `r` in the hash function and `x` represents the `nbits`.