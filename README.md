# CS4980-GutMetagenome-ML-Project
Training classification models on human gut metagenomic signatures 

## Testing This Project
ADD: Instructions for building datasets, running experiments, etc

### Dependencies
This project was developed in Python 3.11.0.
To run the scripts in this repository, you'll want to make sure you've installed Python and the modules listed below. This can be accomplished by running the following through your command line:
```
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

## Project Aims

- Determine if logistic regression or random forest models can be more performant than was reported in [study #2](#2-angelova-iy-kovtun-as-averina-ov-koshenko-ta-danilenko-vn-unveiling-the-connection-between-microbiota-and-depressive-disorder-through-machine-learning-international-journal-of-molecular-sciences-2023-242216459-httpsdoiorg103390ijms242216459)
- Perform some tests to determine if [study #2](#2-angelova-iy-kovtun-as-averina-ov-koshenko-ta-danilenko-vn-unveiling-the-connection-between-microbiota-and-depressive-disorder-through-machine-learning-international-journal-of-molecular-sciences-2023-242216459-httpsdoiorg103390ijms242216459)'s YOLOv8 results are reliable
- Determine if other classifiers (like support vector machines or k-nearest neighbor) can predict competitively on metagenomic signature data
- Compare how these classifiers define their respective decision boundaries


## Study Design
1. Comparative Analysis: Classical Models
    a. Logistic Regression
        * Original dataset split, original parameters
        * 10-Fold cross-validation, original parameters
        * 10-Fold cross-validation, grid search for optimal parameters
        * 10-Fold cross-validation, bag best grid search result
    b. Random Forest
        * Original dataset split, original parameters
        * 10-Fold cross-validation, original parameters
        * 10-Fold cross-validation, grid search for optimal parameters
2. Support Vector Machine Exploration
    a. SVC: RBF Kernel
        * original dataset split
        * 10-fold cross validation
        * 10-fold cross validation, grid search
        * bag best result
    b. SVC: Polynomial Kernel
        * original dataset split
        * 10-fold cross validation
        * 10-fold cross validation, grid search
        * bag best result
    c. LinearSVC
        * original dataset split
        * 10-fold cross validation
        * 10-fold cross validation, grid search
        * bag best result

    d. Examine support vectors
3. YOLO Comparison: Using Synthetic Data
    a. Logistic regression with:
        * original dataset split, 
        * grid search optimal parameters
            > reuse part 1 optimal parameters
    b. Random forest with:
        * original dataset split, 
        * grid search optimal parameters
            > reuse part 1 optimal parameters
    c. SVM - best performing
        * original dataset split, 
        * grid search optimal parameters
            > reuse part 2 optimal parameters
    d. Statistical comparison
4. Other Experiments
    a. Feature Importance Comparison
        * SVMs
        * Logistic regression
        * Random forest
    b. Comparing with sets including demographic data
    c. Comparing with sets sourced from study 1


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


