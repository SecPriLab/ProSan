# ProSan

This repository contains the official code implementation for **_ProSan: Utility-based Prompt Privacy Sanitizer_**.

## Prerequisites

Follow these steps to set up the environment and download the necessary data.

### 1. Create Conda Environment

First, create a conda virtual environment named `prosan` using Python 3.9.

```bash
conda create -n prosan python=3.9
```

### 2. Activate Environment

Activate the newly created conda environment.

```bash
conda activate prosan
```

### 3. Install Dependencies

Install all the required libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Download Datasets

Download the datasets from the following link and place them into a folder named `dataset` in the root directory of this project.

[Download Datasets Here](https://drive.google.com/drive/folders/1GvxMTERNShiJQVSOiUAZIaiXpy8SXkGu?usp=sharing)

After this step, your project structure should look like this:

```
.
├── dataset/
├── results/
├── src/
│   ├── ProSan_CodeAlpaca.py
│   ├── ProSan_MedQA.py
│   └── ProSan_SAMSum.py
└── requirements.txt
```

## Running the Code

To run the experiments for each dataset, execute the corresponding command from the project's root directory.

### CodeAlpaca Dataset

```bash
python /src/ProSan_CodeAlpaca.py
```

### MedQA Dataset

```bash
python /src/ProSan_MedQA.py
```

### SAMSum Dataset

```bash
python /src/ProSan_SAMSum.py
```

The sanitized outputs and evaluation results will be automatically saved in the `results/` directory upon completion.