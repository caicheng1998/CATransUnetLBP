# CATransUnetLBP

This repository contains script which were used to build and use the CATransUnet model to predict ligand binding pocket on protein surface.

# Get Started

## Dependency

Please, install the following packages:

- numpy 1.23.3
- tensorflow-gpu 2.2.0
- openbabel 3.1.1
- scikit-image 0.19.2
- pandas 1.4.4
- h5py 2.10.0
- tqdm

## Content

- ./data_utils: utils to prepare data for training and predicting.
- ./model_dir: pretrained CATranUnet model.
- ./processed_data: prepared data to train a model.

## Usage

### step 1:

Setup Environment

<pre>
#create conda environment
conda create -n env_name python=3.8.10 
conda activate env_name
conda install tqdm
conda install numpy==1.23.3
conda install scikit-image==0.19.2
conda install -c conda-forge h5py=2.2.0
conda install -c conda-forge openbabel=3.1.1
conda install -c conda-forge tensorflow=2.2.0
</pre>

### step 2:

Download model file from [this link](https://drive.google.com/file/d/1KgFj2JBjbwyyfgCiXcCx27qpdGL9SH16/view?usp=sharing), and move and extract it into ./model_dir directory.

### step 3:

Download data file from [this link](https://drive.google.com/file/d/1Rrcfo2c_SQ4r_OnFV8gmBzsSpw6lbni_/view?usp=sharing), and move it into ./processed_data directory.

### step 4:

You can train new model by running "train.py".

You can run "python predict.py -h" to get more optional arguments details.

Example:
<pre>
python train.py --input processed_data/scpdb_dataset.hdf --output model_dir
</pre>

### step 5:

You can predict ligand binding pocket by running "predict.py".

You can run "python train.py -h" to get more optional arguments details.

Example:
<pre>
python predict.py --input test/protein_file/2ihzA.pdb --output test/result --format pdb
</pre>



