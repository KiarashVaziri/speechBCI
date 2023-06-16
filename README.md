## A high-performance speech neuroprosthesis
[![System diagram](SystemDiagram.png)](https://www.biorxiv.org/content/10.1101/2023.01.21.524489v2.abstract)

## Overview

This repo is associated with this [preprint](https://www.biorxiv.org/content/10.1101/2023.01.21.524489v2.abstract) and [dataset](https://doi.org/10.5061/dryad.x69p8czpq). The code contains the RNN decoder (NeuralDecoder) and language model decoder (LanguageModelDecoder) used in the paper, and can be used to reproduce the core offline decoding results. The jupyter notebooks in AnalysisExamples show how to prepare the data for decoder training (rnn_step1_makeTFRecords.ipynb), train the RNN decoder (rnn_step2_trainBaselineRNN.ipynb), and evaluate it using the language model (rnn_step3_baselineRNNInference.ipynb). Intermediate results from these steps (.tfrecord files for training, RNN weights from my original run of this code) and the trigram language model we used are available [here](https://doi.org/10.5061/dryad.x69p8czpq). Example neural tuning analyses (e.g., classification, PSTHs) are also included in the AnalysisExamples folder (classificationWindowSlide.ipynb, examplePSTH.ipynb, naiveBayesClassification.ipynb, tuningHeatmaps.ipynb). 

## Results

When trained on the "train" partition and evaluated on the "test" partition with a trigram language model, my original run of the code achieved an 18.% word error rate. 

## Train/Test/CompetitionHoldOut Partitions

We have partitioned the data into a "train", "test" and "competitionHoldOut" partition (the partitioned data can be downloaded [here](https://doi.org/10.5061/dryad.x69p8czpq) as competitionData.tar.gz and has been formatted for machine learning). "test" contains the last block of each day (40 sentences), "competitionHoldOut" contains the first two (80 sentences), and "train" contains the rest. The transcriptions for the "competitionHoldOut" partition are redacted (we intend to launch a speech decoding competition using this data shortly). 

## Installation

NeuralDecoder should be installed as a python package (python setup.py install). LanguageModelDecoder needs to be compiled first and then installed as a python package (see LanguageModelDecoder/README.md). 




