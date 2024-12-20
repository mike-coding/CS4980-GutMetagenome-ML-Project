# CS4980-GutMetagenome-ML-Project
Training classification models on human gut metagenomic signatures 

## Testing This Project
ADD: Instructions for building datasets, running experiments, etc

### Dependencies
This project was developed in Python 3.11.0.<br>
From the project root folder, run the following through your command line to install all dependencies:
```
pip install -r requirements.txt
```

### Run Experiments
From project root, you can execute this to start running experiments:
```
Python scripts/ml_interface.py
```


## Project Aims

- Determine if logistic regression or random forest models can be more performant than was reported in [study #2](#2-angelova-iy-kovtun-as-averina-ov-koshenko-ta-danilenko-vn-unveiling-the-connection-between-microbiota-and-depressive-disorder-through-machine-learning-international-journal-of-molecular-sciences-2023-242216459-httpsdoiorg103390ijms242216459)
- Perform some tests to determine if [study #2](#2-angelova-iy-kovtun-as-averina-ov-koshenko-ta-danilenko-vn-unveiling-the-connection-between-microbiota-and-depressive-disorder-through-machine-learning-international-journal-of-molecular-sciences-2023-242216459-httpsdoiorg103390ijms242216459)'s YOLOv8 results are reliable
- Determine if other classifiers (like support vector machines or k-nearest neighbor) can predict competitively on metagenomic signature data
- Compare how these classifiers define their respective decision boundaries


## Study Design
1. **Comparative Analysis: Classical Models**<br>
 **[ <code>Log Reg</code> <code>Random Forest</code> ]**<br>
      * Original 70:30 split with original parameters
      * 10-Fold cross-validation with original parameters
      * 10-Fold CV with grid search-optimized parameters
      * <code>**LogReg Only</code>:** 10-Fold CV + bag grid search-optimized model
2. **Support Vector Machine Exploration**<br>
 **[ <code>SVM(RBF)</code> <code>SVM(polynomial)</code> <code>LinearSVM</code> ]**<br>
   * Original 70:30 split
   * 10-fold cross validation
   * 10-fold CV + grid search-optimized parameters
   * bag models with optimized parameters
   * Examine support vectors
3. **YOLO Comparison: Using Synthetic Data**<br>
 **[ <code>SVM</code> <code>Log Reg</code> <code>Random Forest</code> ]**<br>
   * original dataset split
   * grid search optimal parameters
   * Statistical comparison
4. **Other Experiments**
    * Feature importance comparison<br>
**[ <code>SVM</code> <code>Log Reg</code> <code>Random Forest</code> ]**<br>
    * Comparing with sets that include demographic data
    * Comparing with sets sourced from [study 1](#1-kovtun-as-averina-ov-angelova-iy-et-al-alterations-of-the-composition-and-neurometabolic-profile-of-human-gut-microbiota-in-major-depressive-disorder-biomedicines-2022109-2162-published-2022-sep-2-httpsdoiorg103390biomedicines10092162)


## Foundational Works
These two studies form the foundation for my project: 

#### 1. Kovtun AS, Averina OV, Angelova IY, et al. **Alterations of the Composition and Neurometabolic Profile of Human Gut Microbiota in Major Depressive Disorder.** Biomedicines. 2022;10(9 :2162. Published 2022 Sep 2. https://doi.org/10.3390/biomedicines10092162
- Collected gut microbiome WGS samples from 74 individuals: 36 afflicated with MDD, 38 healthy controls
- Formulated list of key enzymes known to be involved in production/degradation of metabolites associated with depression pathology (eg. Serotonin: Aromatic amino acid decarboxylase)
- For 46 common human gut microbiome genera, WGS samples were probed for orthologous amino acid sequences of key enzymes 
- For each sample, ortholog matches were processed into _metagenomic signatures_: species-enzyme pairs and their corresponding relative abundance


#### 2. Angelova IY, Kovtun AS, Averina OV, Koshenko TA, Danilenko VN. **Unveiling the Connection between Microbiota and Depressive Disorder through Machine Learning.** International Journal of Molecular Sciences. 2023; 24(22):16459. https://doi.org/10.3390/ijms242216459
- Attempted to predict depression using models trained on the _metagenomic signatures_ collected above
- Tested logistic regression, random forest, and YOLOv8 CNN
<br>