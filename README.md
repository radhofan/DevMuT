# DevMuTase2024

This is the open resposity of the paper "DevMuT: Testing Deep Learning Framework via Developer Expertise-Based Mutation". Here is the structure of the result.



#### Description

In this work, we first conduct an interview with our industry partners with rich experiennce on DL framework development. Based on the interview result, we design a novel framework testing method, i.e., DevMuT, which adopts the mutation operators and constraints based on the expertise of developers and detect defects in more stages of DL models' lifecycle.

Specifically, we present the interview results in the "Interview" folder. We design an interview guideline to introduce the our purpose and the questions about their experience on development and defects detection, localization, and fix. The whole process of the interview can be found in the figure file "Overview of Interview.pdf". The details about the developers and the interview results can be found in the "InterviewResult.md" file. Based on the interview result, we conclude an table to present the details of our mutation operators and the relevant constraints which can be found in the "Mutation Operator and constraint.xlsx" file. The first columnn represennts the type of the model scripts that mutation operators are adopted on. The second column represents the specific objects in the model script that are adopted mutation operators. The third column lists the type of mutation operatos and the fourth column lists the relevant constraints. The last column adds some extra information about the mutation operaors like mutation range.

In RQ2, we list all the defects detected by DevMuT and show the number of reports, confirmations, and repairs for different types of defects in the first sheet. The other sheets shows the performance defect, accuracy defect, resource defect, crash defect, functional defect and document defect, respectively. We shows the breif symptoms of the defect, the issue tags, and the currenct state in the first column, second column and third column, respectively. Please note that the defect report marked in yellow is a high priority defect type recognized by developers. Since the url link may expose our personal informationn which violates double blind principle, we do not show such kind of information. If you are interested in one of these defects, please contact us.

In RQ1 and RQ3, we compare different testing methods based on three model diversity metrics adopted in existing work COMET. In RQ1, we preset the JSON file of the models generated by different methods (Muffin, COMET, and DevMuT), i.e., the files in "COMET_modeljson", "DevMuT_modeljson" and "Muffin_modeljson". These JSON files are converted from the TensorFlow and MindSpore model objects for the coverage calculation. Besides, we also show the coverage results of different methods. Please note that DevMuT and COMET are mutation-based testing methods that adopt one seed model and mutate multiple times. Therefore, these two methods generate result files on each seed model as shown in the folder "COMET_coverage" and folder "DevMuT_coverage". Each "CSV" file represents the coverage results on each seed model: the first column represents the name of different mutants, and the second to fourth columns represent the LIC, LSC, and LPC values, respectively. Since Muffin belongs to the generation-based testig methods and we follow its default setting, the result only stores in one csv file "Muffin_coverage.csv". Similar to RQ1, we present the coverage result generated by random strategy, MCMC strategy, and DDQN strategy, respectively.



If you have any questions, please leave a message here to contact us. 


# Run

## Installation

Ensure you are using Python 3.9 and a Linux-64 platform:

```bash
$ conda create -n DevMut python=3.9
$ conda activate DevMut
$ pip install -r requirements.txt
```

## Dataset

We provide a few simple datasets`./code/DevMut/dataset/`. Due to the large size of other datasets, we will provide a download link upon request. Please contact us to obtain the dataset.

## Usage

### Step 1: Set Environment Variables

```bash
export CONTEXT_DEVICE_TARGET=GPU
export CUDA_VISIBLE_DEVICES=0, 1
```

### Step 2: Run the master file:mutation_test.py

```bash
python mutation_test.py
```

### Step 3: Check Output

Results will be available in the `./code/DevMut/common/log/` directory.. This folder will contain two files:
- A `.log` file: Contains the log details.
- A `.txt` file: Records the results of the process.


## Parameter Settings

The parameters for running the mutation tests can be configured in `mutation_test.py`. Below are the adjustable parameters:

- `model_name`: Name of the model. Options: `resnet50`, `unet`, `unetplus`, `vgg16`, `textcnn`.
- `mutation_iterations`: Number of mutation iterations.
- `epoch`: Number of epochs for training.
- `batch_size`: Size of the batches for training.
- `mutation_type`: Type of mutation. Options: `'LD'`, `'PM'`, `'LA'`, `'RA'`, `'CM'`, `'SM'`, `'DM'`.
  - MO1 corresponds to `RA`
  - MO2 corresponds to `SM` and `DM`
  - MO3 corresponds to `LA` and `LD`
  - MO4 corresponds to `CM` and `LD`
  - MO5 corresponds to `PM`
  - MO6 and MO7 require configuration in the YAML files located in the `./code/DevMut/config/` folder.

- `mutation_strategy`: Mutation strategy. Options: `'random'`, `'ddqn'`, `'MCMC'`.
- `dataset_path`: Path to the dataset.


# Directory Structure
- **code/DevMuT**
  - The source code for the project.

- **Scott-Knott Test Results**
  - The results of the statistical analysis.
  -  These figures are based on the statistical analysis of RQ1 as requested by reviewers.In the figures, the closer a method is to the y-axis, the better its performance.Methods on opposite sides of the dashed line have significant differences in performance, while methods on the same side do not.

- **Interview**
  - Details about the interview guidelines, results, and backgrounds.

- **RQ1/quantitative analysis.xlsx**
  - We design the mutation operator based on interview insights from three perspectives。We analyze the numbers of the contents mentioned by interviewees for designing mutation operators. Based on these quantitative results, we select mutation objects, operators, and guidelines that cover over 80% of the interviewees' opinions.

