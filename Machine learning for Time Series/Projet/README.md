<div align="center">
	<h1>
		SoccerCPD
	</h1>
</div>



## Introduction

For the formation clustering (Section 4.3) and role labeling (Section 5.3) steps of the paper, we offer `data/form_periods.pkl`, `data/role_periods.csv`, and `data/role_records.csv` that contains the information of all the detected formations and roles in our dataset.<br>

## Getting Started
**both Python and R need to be installed for executing the code.** The version we have used in this study are as follows:

- Python 3.8
- R 3.6.0

To perform the whole process at once, we utilize the Python package `rpy2` to run the R script with  `gSeg` inside our Python implementation. We found that **the M1 chip by Apple does not support binaries for `rpy2`'s API mode in Python, raising a memory error.** (So please use another processor such as the Intel chip.)

After installing the necessary languages, you need to install the packages listed in `requirements.txt`. Make sure that you are in the correct working directory that contains our `requirements.txt`.
```
pip install -r requirements.txt
```

Subsequently, please download the sample match data (named `17985.ugp`) from the following Google Drive link and move it into the directory `data/ugp`.
- Link for the data: https://bit.ly/3gHhkHG

Finally, you can run the algorithm for the sample match data simply by executing `main.py`.
```
python -m main
```


## Citation
This repository includes the source code for the following paper and tracking data collected from a sample match (`17985.ugp`). 
```
@inproceedings{Kim2022,
  author       = {Kim, Hyunsung and
                  Kim, Bit and
                  Chung, Dongwook and
                  Yoon, Jinsung and
                  Ko, Sang{-}Ki},
  title        = {{SoccerCPD}: Formation and Role Change-Point Detection in Soccer Matches
		  Using Spatiotemporal Tracking Data},
  booktitle    = {The 28th {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  pages	       = {3146--3156},
  publisher    = {{ACM}},
  year         = {2023},
  location     = {Washington, DC, USA}
  isbn         = {979-1-4503-9385-0}
  url          = {https://doi.org/10.1145/3534678.3539150},
  doi          = {10.1145/3534678.3539150},
}
```

## References
- A. Bialkowski, P. Lucey, P. Carr, Y. Yue, S. Sridharan and I. Matthews, Large-Scale Analysis of Soccer Matches Using Spatiotemporal Tracking Data, IEEE International Conference on Data Mining, 2014, DOI: https://doi.org/10.1109/ICDM.2014.133.
- H. Song and H. Chen, Asymptotic distribution-free changepoint detection for data with repeated observations, Biometrika, 2021, asab048, DOI: https://doi.org/10.1093/biomet/asab048.
