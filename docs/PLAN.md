# Assignment 4: Generative Models with GAN/cGAN on Adult Dataset

## Project Overview
Implement GAN and Conditional GAN (cGAN) for generating synthetic tabular data from the Adult dataset, then evaluate using detection and efficacy metrics.

---

## Phase 1: Project Setup

### 1.1 Create Virtual Environment
```bash
python -m venv venv
```

### 1.2 Requirements (`requirements.txt`)
```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
jupyter>=1.0.0
```

### 1.3 README Structure
- Project description
- Installation instructions
- Usage guide
- Results summary

---

## Phase 2: Data Loading & Preprocessing

### 2.1 Load ARFF File
Use `scipy.io.arff.loadarff()` to read the Adult dataset:
```python
from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff('adult.arff')
df = pd.DataFrame(data)
```

### 2.2 Dataset Features
**Numeric (6):** age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week

**Categorical (9):** workclass, education, marital-status, occupation, relationship, race, sex, native-country, income (target)

### 2.3 Preprocessing Steps
1. **Handle missing values** (`?` in the data) - **Impute using mode for categorical, median for numeric** (document this choice in the report as a preprocessing decision)
2. **Encode categorical features** - One-hot encoding
3. **Normalize continuous features** - Min-Max scaling to [0, 1] range
4. **Stratified train-test split** - 80/20 maintaining income label ratios
5. **Run 3 experiments** with different random seeds (42, 123, 456)

---

## Phase 3: GAN Architecture

### 3.1 Design Approach (Based on CTGAN principles)
Reference: [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503)

**Generator:**
- Input: Random noise vector (latent_dim=128)
- Architecture: MLP with BatchNorm and LeakyReLU
- Output: Synthetic tabular row (continuous + categorical outputs)
- Use separate output heads:
  - Continuous features: Linear activation
  - Categorical features: Softmax per category group

**Discriminator:**
- Input: Real or synthetic tabular row
- Architecture: MLP with LeakyReLU and Dropout
- Output: Real/Fake probability

### 3.2 Training Configuration
- **Loss function:** Binary Cross-Entropy or Wasserstein loss with gradient penalty
- **Optimizer:** Adam (Generator lr=0.0002, Discriminator lr=0.0002)
- **Batch size:** 64-128
- **Epochs:** 150-300 (monitor for mode collapse)

### 3.3 Optional: Autoencoder Enhancement
Train autoencoder to create embeddings, then:
- Generator produces embeddings
- Decoder reconstructs tabular data
- Discriminator works in embedding space

---

## Phase 4: Conditional GAN (cGAN) Architecture

### 4.1 Modifications from GAN
Reference: [CTAB-GAN+ paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10801038/)

**Generator:**
- Input: Noise vector + One-hot encoded label (income: >50K or <=50K)
- Concatenate condition to noise before feeding to network

**Discriminator:**
- Input: Tabular row + One-hot encoded label
- Receives real samples with their true labels
- Receives fake samples with the requested label

### 4.2 Training Process
- Sample label from training distribution
- Generator creates sample conditioned on that label
- Discriminator evaluates (sample, label) pairs

---

## Phase 5: Evaluation Metrics

### 5.1 Training Monitoring
- Plot Generator and Discriminator losses over epochs
- Monitor for mode collapse (discriminator loss → 0)
- Track gradient norms

### 5.2 Distribution Comparison Visualizations
1. **Histograms** - Compare real vs synthetic for each feature
2. **Correlation matrices** - Compare feature correlations
3. **PCA/t-SNE plots** - Visualize data distributions

### 5.3 Detection Metric (Lower is better)
Goal: Synthetic data should be indistinguishable from real data.

Process:
1. Split original training set into 4 folds
2. Split synthetic data into 4 folds
3. 4-fold cross-validation:
   - Train: 3 folds real + 3 folds synthetic (label: real=0, synthetic=1)
   - Test: 1 fold real + 1 fold synthetic
4. Train Random Forest classifier
5. Report average AUC (target: close to 0.5 = random guessing)

### 5.4 Efficacy Metric (Higher is better, max=1)
Goal: Synthetic data should be useful for training classifiers.

Process:
1. Train RF on **real** training data → evaluate on real test set → AUC_real
2. Train RF on **synthetic** data → evaluate on real test set → AUC_synthetic
3. Efficacy = AUC_synthetic / AUC_real

---

## Phase 6: Implementation Checklist

### Files to Create
```
deep_hw_4/
├── venv/
├── requirements.txt
├── README.md
├── notebook.ipynb          # Single notebook with ALL code (submission)
├── adult.arff              # Dataset
└── report.pdf              # Final report (from .docx)
```

**Implementation: From scratch in PyTorch** (no external GAN libraries)
**Organization: Single Jupyter notebook** containing all code sections:
1. Data loading & preprocessing
2. GAN model definitions (Generator, Discriminator)
3. cGAN model definitions
4. Training loops with logging
5. Evaluation metrics (Detection, Efficacy)
6. Visualizations

### Assignment Substep Mapping

| Assignment Requirement | Implementation |
|------------------------|----------------|
| Load ARFF dataset | `scipy.io.arff.loadarff()` in notebook |
| 80/20 stratified split | `train_test_split(stratify=y)` |
| 3 random seeds | Loop with seeds [42, 123, 456], report averages |
| GAN architecture | Custom PyTorch Generator + Discriminator classes |
| cGAN architecture | Modified Generator/Discriminator with label conditioning |
| Training info (epochs, lr, etc.) | Documented in notebook markdown cells |
| Loss graphs | matplotlib plots of G/D losses per epoch |
| Distribution visualizations | Histograms, correlation matrices (seaborn) |
| Detection metric (4-fold RF) | 4-fold CV with RandomForestClassifier |
| Efficacy metric (RF comparison) | AUC ratio calculation |
| Report (max 6 pages) | PDF export from docx |

---

## Academic Sources

1. **CTGAN Paper**: [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503) - NeurIPS 2019
2. **CTAB-GAN+**: [Enhancing tabular data synthesis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10801038/)
3. **ARFF Loading**: [SciPy loadarff documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.loadarff.html)

---

## Verification Plan

1. **Data loading**: Verify ARFF loads correctly, check shape and dtypes
2. **Preprocessing**: Verify no NaN values, correct encoding dimensions
3. **GAN training**: Monitor loss curves for stability (no mode collapse)
4. **cGAN training**: Verify generated samples have correct label distributions
5. **Synthetic data quality**: Visual inspection of histograms
6. **Detection metric**: Should be close to 0.5 AUC if synthetic data is good
7. **Efficacy metric**: Should be close to 1.0 if synthetic data is useful
