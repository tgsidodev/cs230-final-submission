# cs230-final-project

# Code Layout:
* get_started.sh: A script to install requirements, and download and preprocess the data.
* requirements.txt: Used by get_started.sh to install requirements.
* docker/: A directory containing a Dockerfile:
– Dockerfile: This is the specification for the Docker image abisee/cs224n-dfp:v4
which is available on Docker Hub at https://hub.docker.com/r/abisee/cs224n-dfp/.

* code/: A directory containing all code:
  * preprocessing/: Code to preprocess the SQuAD data, so it is ready for training:
  * download_wordvecs.py: Downloads and stores the pretrained word vectors (GloVe).
  * squad_preprocess.py: Downloads and preprocesses the official SQuAD train and
  dev sets and writes the preprocessed versions to file.
  * data_batcher.py: Reads the pre-processed data from file and processes it into batches
  for training.
  * evaluate.py: The official evaluation script from SQuAD. Your model can import  evaluation functions from this file, but you should not change this file.
  * main.py: The top-level entrypoint to the code. You can run this file to train the model, view examples from the model and evaluate the model.
  * modules.py: Contains model components, like a RNN Encoder module, a
  basic dot-product attention module, DCN attention module, BiDAF attention module, a LSTM Encoder module, and a simple softmax layer.

  * official_eval_helper.py: Contains code to read an official SQuAD JSON file, use
  your model to predict answers, and write those answers to another JSON file. This is
  required for official evaluation. (See Section 7).
  * pretty_print.py: Contains code to visualize model output.
  * qa_model.py: Contains the model definition.
  * vocab.py: Contains code to read GloVe embeddings from file and make them into an
    embedding matrix.

Adapted from CS224n

# Setup:
* If you are on your local machine, first open requirements.txt and change the line
tensorflow-gpu==1.4.1 to tensorflow==1.4.1. This will ensure that the setup script
installs the CPU version of TensorFlow on your machine.
* If you are on your GPU machine (e.g. your Azure VM), do not change requirements.txt
(it should say tensorflow-gpu==1.4.1).

Now run the following commands:
* cd cs224n-win18-squad # Ensure we’re in the project directory
* ./get_started.sh # Run the startup script
* Say yes when the script asks you to confirm that new packages will be installed.
* To activate the squad environment, run source activate squad

# Train
* source activate squad # Remember to always activate your squad environment
* cd code # Change to code directory
* python main.py --experiment_name=ensemble --mode=train # Start training

Note: this code is adapted in part from the [Neural Language Correction](https://github.com/stanfordmlgroup/nlc/) code by the Stanford Machine Learning Group.
