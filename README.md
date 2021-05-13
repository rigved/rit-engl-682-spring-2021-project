# RIT ENGL-682 Spring 2021 Project

De-biasing Transcribed Text from Automatic Speech Recognition Systems

## Introduction

This project is an attempt to correct errors in the transcribed test due to accent bias in Automatic Speech Recognition (ASR) systems. The project uses Mozilla's DeepSpeech ASR system as an example. However, these techniques can be easily extended to other ASR systems.

This project is part of my [RIT ENGL-682 course project](https://shieldofachilles.in/pages/rit-engl-682-project-rigved-rakshit.html).

## Prerequisites

The code runs faster when using a CUDA-supported GPU with compute capability >= 3.5. [Mozilla's pre-built DeepSpeech model v0.9.3](https://deepspeech.readthedocs.io/en/r0.9/USING.html) depends on CUDA v10.1 and CuDNN v7.6. This in turn requires Tensorflow v2.3.

However, the code will run just fine without any GPU. In this case, modify the 
list of requirements mentioned in the [environment.yml](https://github.com/rigved/rit-engl-682-spring-2021-project/blob/main/environment.yml) file and replace "tensorflow-gpu" with "tensorflow" and "deepspeech-gpu" with "deepspeech".

## Installation

Use Anaconda to create a new environment from the provided [environment.yml](https://github.com/rigved/rit-engl-682-spring-2021-project/blob/main/environment.yml) file.

```bash
conda env create -f environment.yml
```

## Usage

- The code has been set up such that you can split the `validated.csv` file into multiple smaller files and process them in parallel.
- This way, you can run multiple instances of the code to maximize resources. For example, you use one `validated.csv` file in a CUDA-enabled `conda` environment and another `validated.csv` file in a non-GPU `conda` environment. This will utilize the GPU and all the CPUs on the machine.
- Run the following commands in your newly created `deepspeech` Anaconda environment.

### Audio pre-processing to generate time-stretched, pitch-shifted, and volume-augmented audio files

```bash
mkdir -p modified_data
python audio_preprocessing.py data modified_data validated.csv
```

This step will perform pre-processing on all the voice recordings in the [data](https://github.com/rigved/rit-engl-682-spring-2021-project/tree/main/data) directory and store the modified clips in the `modified_data` directory.

### DeepSpeech inference

Download the pre-built Mozilla DeepSpeech v0.9.3 models:

```bash
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```

Perform speech recognition:

```bash
python deepspeech_english_inference.py data ./ validated.csv hypothesis.txt gold_standard.txt
python deepspeech_english_inference.py modified_data ./ validated.csv preprocessed_hypothesis.txt preprocessed_gold_standard.txt
```

This step will generate the DeepSpeech transcriptions from the original and the pre-processed voice recordings.

### Language Tool post-processing

```bash
python language_tool.py ./ hypothesis.txt language_tool_hypothesis.txt
```

This step will generate the corrections as proposed by Language Tool. The DeepSpeech transcriptions on the original voice recordings are used in this case.

### Evaluation

```bash
wer gold_standard.txt hypothesis.txt
```

This step calculates Word Error Rate, Word Recognition Rate, and Sentence Error Rate for the given file. Use one of the different `hypothesis` files to generate metrics for the pre-processed and post-processed outputs.

#### Plotting metrics

The [results\_plot.R](https://github.com/rigved/rit-engl-682-spring-2021-project/blob/main/results_plot.R) plots the results that are reported in [project paper](https://shieldofachilles.in/static/rit/engl_682/ENGL_682_Spring_2021_Project_by_Rigved_Rakshit_Final_Report.pdf).

## Demo

Click this button to launch an interactive demo of this code:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rigved/rit-engl-682-spring-2021-project/HEAD?filepath=index.ipynb)

Or copy this link to your browser to launch the demo:

https://mybinder.org/v2/gh/rigved/rit-engl-682-spring-2021-project/HEAD?filepath=index.ipynb

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html)
