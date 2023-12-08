# Transcendent Code

Using conformal evaluation to detect concept drift affecting malware detection.

For more information, you can see the project page: https://s2lab.cs.ucl.ac.uk/projects/transcend/

## What is Transcend and Conformal Evaluation? 

Malware evolves rapidly which makes it hard---if not impossible---to 
generalize learning models to reflect future, previously-unseen behaviors. 
Consequently, most malware classifiers become unsustainable in the long run, 
becoming rapidly antiquated as malware continues to evolve. 

Transcendent is a toolset which, together with a statistical framework called 
conformal evaluation, aims to identify aging classification models in vivo 
during deployment, before the machine learning model's performance starts to 
degrade.

Further details can be found in the paper [*Transcending TRANSCEND: Revisiting 
Malware Classification in the Presence of Concept Drift*](https://arxiv.org/abs/2010.03856). by F. Barbero, F. Pendlebury, F. Pierazzi, and L. Cavallaro (IEEE S&P 2022).

If you end up using Transcendent as part of a project or publication, please include a citation of the S&P paper:

```
@inproceedings{barbero2022transcendent,
author = {Federico Barbero and Feargus Pendlebury and Fabio Pierazzi and Lorenzo Cavallaro},
title = {Transcending Transcend: Revisiting Malware Classification in the Presence of Concept Drift},
booktitle = {{IEEE} Symposium on Security and Privacy},
year = {2022},
}
```

Transcendent is based on Transcend. Further details can be found in the paper [*Transcend: Detecting Concept Drift 
in Malware Classification Models*](https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-jordaney.pdf). by R. Jordaney, K. Sharad, S. K. Dash, Z. Wang, 
D. Papini, I. Nouretdinov, and L. Cavallaro (USENIX Sec 2017). An associated 
presentation can be found at [the Usenix site.](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/jordaney)

If you end up using Transcendent as part of a project or publication, please 
include a citation of the original Transcend Usenix paper as well: 

```
@inproceedings {jordaney2017,
    author = {Roberto Jordaney and Kumar Sharad and Santanu K. Dash and Zhi Wang and Davide Papini and Ilia Nouretdinov and Lorenzo Cavallaro},
    title = {Transcend: Detecting Concept Drift in Malware Classification Models},
    booktitle = {26th {USENIX} Security Symposium ({USENIX} Security 17)},
    year = {2017},
    isbn = {978-1-931971-40-9},
    address = {Vancouver, BC},
    pages = {625--642},
    url = {https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/jordaney},
    publisher = {{USENIX} Association},
}
```

## Getting Started 

### Installation

Transcend requires Python 3 (preferably >= 3.5) as well as the statistical 
learning stack of NumPy, SciPy, and Scikit-learn.

Package dependencies can be installed by using the listing in requirements.txt.

```shell
pip install -r requirements.txt
```

A full installation can be peformed using setup.py:

```shell
pip install -r requirements.txt
python setup.py install 
```

Features to reproduce the Android experiments can be downloaded from [this link](https://www.dropbox.com/sh/8cc6z64rzi1n4br/AAD88BhcF_BjWcT7tO2T53qTa?dl=0)

Features for Marvin and Drebin can be downloaded from [this link](https://www.dropbox.com/s/wj2eoww36ljqpor/transcend-features.tar.gz?dl=0)

### Usage 
    
Conformal evaluation can get a little bit fiddly, so it's advised that you 
become familiar with a typical testing pipeline such as the example given in 
`ce.py` as well as the following functions (which are particularly affected by 
different configuration settings):

* `utils.parse_args()`
* `data.load_features()`
* `thresholding.find_quartile_thresholds()`
* `thresholding.find_random_search_thresholds()`
* `thresholding.sort_by_predicted_label()`
* `thresholding.get_performance_with_rejection()`

### ce.py

An example conformal evaluation pipeline using the Transcend library is given 
in `ce.py`. It can be run with a multitude of command line arguments. 

Comparing quartiles of correct predictions using credibility only: 

```shell
python3 ce.py	                  	    \
    --train drebin              	    \
    --test marvin_full          	    \
    -k 10                       	    \
    -n 10                       	    \
    --pval-consider full-train  	    \
    -t quartiles                	    \
    --q-consider correct                \
    -c cred                     	 
```


Random search for thresholds maximising F1 above threshold and minimising F1 of 
rejected predictions while enforcing thresholds for credibility and confidence: 

```shell
python3 ce.py	                  	    \
    --train drebin              	    \
    --test marvin_full          	    \
    -k 10                       	    \
    -n -2                       	    \
    --pval-consider full-train  	    \
    -t random-search            	    \
    -c cred+conf                  	    \
    --rs-max f1_k           	 	    \
    --rs-min f1_r              		    \
    --rs-limit reject_total_perc:0.25   \
    --rs-samples 500
```

Random search for thresholds maximising F1 above threshold subject to the 
total percentage of rejected elements while enforcing credibility thresholds: 

```shell
python3 ce.py 		                                \
	--train drebin                                  \
	--test marvin_half                              \
	-k 10                                           \
	-n -1                                           \
	--pval-consider full-train                      \
	-t constrained-search                           \
	-c cred                                         \
	--cs-max f1_k:0.95                              \
	--cs-con kept_pos_perc:0.76,kept_neg_perc:0.76  \
	--rs-samples 500
```

## Acknowledgements 

This research has been partially supported by the UK EPSRC grants EP/K033344/1, 
EP/L022710/1, EP/K006266/1, and EP/P009301/1 as well as the NVIDIA Corporation,
NHS England, and Innovate UK. 
