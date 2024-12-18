# CS4980-GutMetagenome-ML-Project
Training classification models on human gut metagenomic signatures 

## Testing This Project
Instructions on using the scripts, etc to go here

### Dependencies
This project uses the following Python libraries:
- Pandas
- Requests
- BeautifulSoup4
- Scikit-Learn
- NumPy
- Matplotlib
- Seaborn

To run the scripts included in this project, you'll want to make sure your Python distribution has these modules installed. This can be accomplished through running the following through your command line:
```
pip install pandas
pip install requests
pip install bs4
pip install scikit-learn
pip install numpy
pip install matplotlib
pip install seaborn
```

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
<br>

## Project Goals

- Determine if logistic regression or random forest models can be more performant than was reported in study #2
- Perform some tests to determine if study 2's YOLOv8 results are reliable
- Determine if other classifiers (like support vector machines) can also predict accurately on metagenomic signature data
- Compare how these classifiers define the boundary between a healthy individuals and those with depression
