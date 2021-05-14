# RIT ENGL-682 Spring 2021 Project

De-biasing Transcribed Text from Automatic Speech Recognition Systems

## Introduction

This project is an attempt to correct errors in the transcribed test due to accent bias in Automatic Speech Recognition (ASR) systems. The project uses Mozilla's DeepSpeech ASR system as an example. However, these techniques can be easily extended to other ASR systems.

This project is part of my [RIT ENGL-682 course project](https://shieldofachilles.in/pages/rit-engl-682-project-rigved-rakshit.html).

## Prerequisites

The code runs faster when using a CUDA-supported GPU with compute capability >= 3.5. [Mozilla's pre-built DeepSpeech model v0.9.3](https://deepspeech.readthedocs.io/en/r0.9/USING.html) depends on CUDA v10.1 and CuDNN v7.6. This in turn requires Tensorflow v2.3.

However, the code will run just fine without any GPU. In this case, modify the 
list of requirements mentioned in the [environment.yml](https://github.com/rigved/rit-engl-682-spring-2021-project/blob/main/environment.yml) file and replace "tensorflow-gpu" with "tensorflow" and "deepspeech-gpu" with "deepspeech".

The project uses Mozilla's pre-built English DeepSpeech model and scorer files. The [DeepSpeech inference](https://github.com/rigved/rit-engl-682-spring-2021-project#deepspeech-inference) section below details how to retrieve those files.

## Installation

Use Anaconda to create a new environment from the provided [environment.yml](https://github.com/rigved/rit-engl-682-spring-2021-project/blob/main/environment.yml) file. All the following sections assume that you are running the code in this Anaconda or equivalent environment.

```bash
conda env create -f environment.yml
```

In case you get a `ResolvePackageNotFound` failure while setting up the Anaconda environment, use the following alternate environment file that contains no build information.

```bash
conda env create -f environment_no_builds.yml
```

## Usage

- The code has been set up such that you can split the `validated.csv` file into multiple smaller files and process them in parallel.
- This way, you can run multiple instances of the code to maximize resources. For example, you use one `validated.csv` file in a CUDA-enabled `conda` environment and another `validated.csv` file in a non-GPU `conda` environment. This process will help utilize the GPU and all the CPUs on the machine.
- Run the following commands in your newly created `deepspeech` Anaconda environment.

### Audio pre-processing to generate time-stretched, pitch-shifted, and volume-augmented audio files

```bash
mkdir -p modified_data
python audio_preprocessing.py data modified_data validated.csv
```

This step will perform pre-processing on all the voice recordings in the [data](https://github.com/rigved/rit-engl-682-spring-2021-project/tree/main/data) directory and stores these modified clips in the `modified_data` directory.

### DeepSpeech inference

Download the pre-built Mozilla DeepSpeech v0.9.3 models that are used by the remaining sections:

```bash
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```

Perform speech recognition:

```bash
python deepspeech_english_inference.py data ./ validated.csv hypothesis.txt gold_standard.txt
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

The [results\_plot.R](https://github.com/rigved/rit-engl-682-spring-2021-project/blob/main/results_plot.R) plots the results that are reported in the [project paper](https://shieldofachilles.in/static/rit/engl_682/ENGL_682_Spring_2021_Project_by_Rigved_Rakshit_Final_Report.pdf).

## Demo

Click the following button to launch an interactive demo of this code. NOTE: The demo takes at least 30 seconds to load to the large Mozilla DeepSpeech model and data files.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rigved/rit-engl-682-spring-2021-project/HEAD?filepath=index.ipynb)

Or copy this link to your browser to launch the demo:

https://mybinder.org/v2/gh/rigved/rit-engl-682-spring-2021-project/HEAD?filepath=index.ipynb

## Example Output from Demo

Here is an example from the demo of DeepSpeech Inference on a sample original voice recording:

![DeepSpeech Inference on sample original voice recording](DeepSpeech_Inference_on_sample_original_voice_recording.jpg)

Here is an example from the demo of DeepSpeech Inference on a sample louder voice recording:

![DeepSpeech Inference on sample louder voice recording](DeepSpeech_Inference_on_sample_louder_voice_recording.jpg)

### Demo Setup

The demo uses the [Binder](https://mybinder.org/) service to convert the [index.ipynb](https://github.com/rigved/rit-engl-682-spring-2021-project/blob/main/index.ipynb) Jupyter notebook, along with [pre-build](https://github.com/rigved/rit-engl-682-spring-2021-project/blob/main/start) and [post-build](https://github.com/rigved/rit-engl-682-spring-2021-project/blob/main/postBuild) scripts, into an interactive, Docker-based Jupyter notebook hosted in the cloud. The pre-build script enables the NVIDIA CUDA environment in the Binder Docker image. The post-build script downloads the Mozilla DeepSpeech v0.9.3 pre-built model and scorer files. These files are too large to fit within GitHub's filesize restrictions and are not included in this GitHub repo. The Binder service uses the [environment.yml](https://github.com/rigved/rit-engl-682-spring-2021-project/blob/main/environment.yml) file to setup the Anaconda environment inside the Binder Docker image.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html)
