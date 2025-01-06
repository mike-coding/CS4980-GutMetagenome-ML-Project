# Revisiting the Connection Between Gut Microbiota and Depressive Disorder: An Evaluation and Reproduction of Machine Learning Approaches
_Michael Wurth, CS:4980 at the University of Iowa_

## Abstract
Research into the human gut microbiome has increasingly revealed associations with various psychiatric disorders, suggesting a link between microbial composition and mental health outcomes. Angelova et al.² investigated these associations by training machine learning classifiers on metagenomic sequence read data, finding that the YOLOv8 (You Only Look Once) convolutional neural network (CNN) was most effective in classifying depressive states based on gut microbiome composition.
Our study aims to revisit and extend the findings of Angelova et al.² by evaluating the performance of logistic regression, random forest, and support vector machines (SVMs). Additionally, we assess whether these models can achieve performance comparable to YOLOv8 under similar data conditions and parameter optimization.
Using the datasets from the reference study, we re-examined model efficacy before and after parameter tuning and cross-validation. Logistic regression notably outperformed previous reports, achieving a cross-validated accuracy of 0.77 and a ROC AUC of 0.81. On their synthetically expanded dataset, random forest achieved near-perfect classification. SVM models attained perfect accuracy on this dataset, matching YOLOv8's performance as reported by Angelova et al.² These findings suggest that simpler models can achieve competitive results on metagenomic data when conditions are controlled and classifier optimization is appropriately applied.


**Key Words: [Depression, gut microbiome, machine learning, logistic regression, psychiatric disorder]**

## Testing This Project

### Dependencies
This project was developed in Python 3.11.0.<br>
From the project root folder, run the following through your command line to install all dependencies:
```
pip install -r requirements.txt
```

### Run Experiments
From the project root, you can execute this in the terminal to start running experiments:
```
Python scripts/ml_interface.py
```
The above will prompt the user to run [study design experiments 1-4](#study-design) by default.<br> 
To selectively run specific experiments from the study design, you can execute the following in the terminal, assuming the current working directory is the project root:
```
cd scripts
Python

from ml_interface import MLInterface
interface=MLInterface()

interface.perform_experiment_1() # perform functions for part 1 of study design
interface.perform_experiment_2() # perform functions for part 2
interface.perform_experiment_3() # you can probably guess what this one does
interface.perform_experiment_4()

interface.plot_SVM_decision_boundary() # Make SVM decision boundary figures for RBF & poly kernels
```

### Build Your Own Experiment
I have not directly implemented functions to perform the experiments outlined in [section 5: Other Experiments](#study-design).<br>
However, you can still perform these tests (and many others) through MLInterface.<br> 
Here's an example of how to train and test a logistic regression model on a dataset with demographic data (sex, age) included:
```
Python

from ml_interface import MLInterface
interface=MLInterface()

interface.select_model('lg')
interface.load_experiment_set(2, 'demo')
interface.train_model()
results = interface.evaluate_model()

interface.start_new_result_log('My experiment')
interface.write_to_result_log(results, 'Logistic Regression on Demographic Dataset')
interface.dump_result_log()
```

## Project Aims
- Determine if logistic regression or random forest models can be more performant than was reported in [study #2](#2-angelova-iy-kovtun-as-averina-ov-koshenko-ta-danilenko-vn-unveiling-the-connection-between-microbiota-and-depressive-disorder-through-machine-learning-international-journal-of-molecular-sciences-2023-242216459-httpsdoiorg103390ijms242216459)
- Investigate whether [study #2](#2-angelova-iy-kovtun-as-averina-ov-koshenko-ta-danilenko-vn-unveiling-the-connection-between-microbiota-and-depressive-disorder-through-machine-learning-international-journal-of-molecular-sciences-2023-242216459-httpsdoiorg103390ijms242216459)'s YOLOv8 results are reliable
- Find out if other classifiers (like support vector machines) can predict competitively on metagenomic signature data
- Compare how these classifiers define their respective decision boundaries and feature importances


## Study Design
1. **Comparative Analysis: Classical Models**<br>
 **[ <code>Log Reg</code> <code>Random Forest</code> ]**<br>
      * Original 70:30 split with original parameters
      * 10-Fold cross-validation with original parameters
      * 10-Fold CV with grid search-optimized parameters
      * <code>**LogReg Only</code>** 10-Fold CV + bag grid search-optimized model
2. **Support Vector Machine (SVM) Performance on Metagenomic Signature Data**<br>
 **[ <code>SVM(RBF)</code> <code>SVM(polynomial)</code> <code>LinearSVM</code> ]**<br>
   * Original 70:30 split
   * 10-fold cross validation
   * 10-fold CV + grid search-optimized parameters
   * bag models with optimized parameters
3. **YOLO Comparison: Using Synthetic Data**<br>
 **[ <code>SVM</code> <code>Log Reg</code> <code>Random Forest</code> ]**<br>
   * Original dataset split
   * Grid search optimal parameters
   * Statistical comparison
4. **Feature Importance Comparison**<br>
**[ <code>SVM</code> <code>Log Reg</code> <code>Random Forest</code> ]**<br>
	* Compare respective assessments of feature importance between models
5. **Other Experiments**<br>
**[ <code>SVM</code> <code>Log Reg</code> <code>Random Forest</code> ]**<br>
    * Comparing with datasets that include demographic data
    * Comparing with datasets sourced from [study 1](#1-kovtun-as-averina-ov-angelova-iy-et-al-alterations-of-the-composition-and-neurometabolic-profile-of-human-gut-microbiota-in-major-depressive-disorder-biomedicines-2022109-2162-published-2022-sep-2-httpsdoiorg103390biomedicines10092162)


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