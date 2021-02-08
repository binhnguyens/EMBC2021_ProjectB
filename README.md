# EMBC2021_ProjectB

Project for EMBC 2021 with Michael Nigro and Alice Rueda

### Demo_modified_jan22.m
This file is used and based off of the FSLib which can be found here: https://github.com/forensicanalysis/fslib
<br>
It imports data from the CPSS Statistics Canada, and is then placed into a matrix. Which then it follows by using feature ranking techniques from the FSLib. We found that mRMR is the best resultant.

Followed by the feature ranking, various classification techniques are applied including
- SVM 
- DT
- K-fold classifiers

Within the classification, display ranking is used to display the rank of the features

### Main_fslib
This file loops through all of the feature ranking techniques and tests it with a DT and SVM classifier. It saves the results in a text file.
