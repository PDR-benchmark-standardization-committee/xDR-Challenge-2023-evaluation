# xDR-Challenge-2023-evaluation
This is the evaluation scripts for evaluating in the scoring trials of xDR Challenge 2023 (IPIN 2023 competition track5).

This README introduces evaluation indexes evaluated by the evaluation scripts and the requirement for using the scripts. 

| **Name of Index** | **Corresponding indicators** | **Description** |
 ---       | ---                     |---
| I_ce        | CE (Circular Error)               　 | Checking the absolute positional error between trajectory and ground truth at check points.　         |
| I_ca        | CA_w (Circular Accuracy in the loacl space)        　     | Checking the deviation of the error distribution in local x-y coordinate system |
| I_eag       | EAG (Error Accumulation Gradient)  | Checking the speed of error accumulation from the correction points      |
| I_ve        | VE (Velocity Error)                 |Checking the error of velocity compared with correct velocity of ground-truth      |
| I_obstacle  | Requirement for Obstacle Avoidance | Checking the percentage of points of the trajectory in walkable area|

### Formula for evaluating the indexes
```
Index         Max Score(100)    Min Score(0)      formula
I_ce       | ce < 1.0   | 30 < ce   | 100 - (100 * (ce - 1.))/29
I_ca       | ca = 0.0   | 10 < ca   | 100 - (10 * ca)
I_eag      | eag < 0.05 | 2.0 < eag | 100 - (100 * (eag - 0.05))/1.95
I_ve       | ve < 0.1   | 2.0 < ve  | 100 - (100 * (ve - 0.1))/1.9
I_obstacle | obs = 1.0  | obs = 0.0 | 100 * obs
```

### about Frequency of the evaluation
Note that frequency of the evaluation depends on the frequency of the ground-truth data.
The frequency of the ground-truth data for xDR Challenge 2023 is about 100Hz.
If the sampling frequency of your estimation is less than 100Hz, your estimation can not be accurately evaluated.
We recommend you to estimate trajectories in 100Hz or to up-sample the trajectories to 100Hz. 

## Requirements (Required python version and packages for running the scripts)
```
python==3.8.5
numpy==1.23.4
pandas==1.5.0
scipy==1.8.1
matplotlib==3.3.2
seaborn==0.10.1
```

## Description of Files

| **Filename** | **Description** |
 ---            |---
| do_evaluation_XC2023.py | Execute evaluation script for index |
| requirements.txt        | File for summarizing the requirements|
<!--
| **evtools**             |--- elemental scripts called by do_evaluation_XC2023.py
| bitmap_tools.py         | Scripts for using bitmap|
| EV_converter.py         | Script of calculating index by using evaluation indicators standardized by PDRBMS|
| evaluate_CA.py          | Evaluation scripts for CA|
| evaluate_CE.py          | Evaluation scripts for CE |
| evaluate_ CE.py         | Evaluation scripts for CE |
| evaluate_OE.py          | Evaluation scripts for OE |
| evaluate_VE.py          | Evaluation scripts for VE |
-->
## Usage
### Step.1  Install
```
git clone --recursive https://github.com/PDR-benchmark-standardization-committee/xDR-Challenge-2023-evaluation
cd xDR-Challenge-2023-evaluation
pip install -r requirements.txt
```

### Step.2 Placing estimation files
Please place (copy) the file of estimated trajectory at [dataset]/[traj]/.
The estimated trajectories with BLE information and without BLE information should be placed in separated folders.
-	with BLE: _est
-	without BLE: _pdr_est
The file structure of the evaluation scripts is shown below.
```
xDR-Challenge-2023-evaluation/
├ dataset/
|   ├ gis/
|   |  ├ beacon_list.csv
|   |  ├ FLD01_0.01_0.01.bmp
|   |  ├ FLU01_0.01_0.01.bmp
|   |  └ FLU02_0.01_0.01.bmp
|   |
|   ├ gt/
|   |  ├ *_*_gt.csv
|   |  └ *_*_gt.csv
|   |
|   └ traj/
|      ├ *_*_est.csv [**estimation with BLE files**]
|      ├ *_*_pdr_est.csv [**estimation files**]_pdr.csv
|
├ evtools/
├ output/
├ do_evaluation_XC2023.py
├ requirements.txt
└ README.md
```

#### Configuration of the contents of the file of estimated trajectory (\*\_\*\_est.csv, \*\_\*\_pdr_est.csv)
The contents of the estimated trajectory file are separated by commas and are as follows.
Note that Headers should not be included in the trajectory file.
| Timestamp (s) | x(m) | y(m) | floor |
| ---      | ---  | ---  | ---   

### Step.3 Running evaluation scripts
You need to select estimation and ground truth folder path for evaluation
```
python do_evaluation_XC2023.py -t [estimation_path]
```
If you want to see the demo estimation score results, you just execute following script
```
python do_evaluation_XC2023.py -t dataset/traj/
```

Results are evaluation indexes and the integrated index. They are evaluated for each trajectory. Average of the indexes of the trajectories in the dataset are used for the competition. The results of indexes are saved in [output] folder.

## Optional Arguments
There are optional arguments in evaluation scripts.

### 1. Outputting the graph and images for CA, EAG, OE
If you add "--draw" option, you can obtain histogram of CA or graph of EAG, Map of obstacle interference for OE. They are saved as folder named the trajectory name in output folder.
```
python do_evaluation_XC2023.py -t [estimation_folder] --draw
```

### 2. Indicating output folder 
If you add "—output_path" option, you can indicate the name of the output folder.
```
python do_evaluation_XC2023.py -t [estimation_folder] --output_path new_output_folder/
```

### 3. Changing the weights for indexes
If you add "--est_weight" option, you can change index weights to calculate competition score in index_weights.ini
Default weight of index is below. These weights are used in xDR Challenge 2023.
```
I_ce = 0.25
I_ca = 0.20
I_eag = 0.25
I_ve = 0.15
I_obstacle = 0.15
```
The weights can be defined by command line arguments.
The indexes are ordered as I_ce, I_ca, I_eag, I_ve, I_obstacle and space separated.
```
python do_evaluation_XC2023.py -t [estimation_folder] --est_weight 0.25 0.2 0.25 0.15 0.15
```
