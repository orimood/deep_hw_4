# Assignment 4: Generative Models with GAN/cGAN

Deep Learning course assignment implementing GAN and Conditional GAN for synthetic tabular data generation on the Adult dataset.

## Project Structure

```
deep_hw_4/
├── data/
│   └── adult.arff              # Adult Census Income dataset
├── docs/
│   ├── assignment.docx         # Assignment specification
│   ├── PLAN.md                 # Implementation plan
│   └── experiment_log.md       # Experiment tracking
├── notebooks/
│   ├── main.ipynb              # Main GAN implementation (VGM + WGAN-GP)
│   ├── baseline.ipynb          # Simplified baseline (MinMax scaling)
│   ├── improved.ipynb          # Improved version experiments
│   ├── ctgan.ipynb             # CTGAN-style modifications
│   └── eda.ipynb               # Exploratory Data Analysis
├── outputs/
│   ├── main/                   # Main notebook outputs
│   ├── baseline/               # Baseline outputs
│   ├── improved/               # Improved version outputs
│   ├── ctgan/                  # CTGAN outputs
│   └── comparison_results.png  # Comparison visualization
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebooks:
```bash
jupyter notebook notebooks/main.ipynb
```

## Dataset

The Adult dataset (Census Income) contains demographic information to predict whether income exceeds $50K/year. Features include age, workclass, education, occupation, etc.

## Implementation

- **GAN**: WGAN-GP with MLP-based Generator and Discriminator
- **cGAN**: Conditional GAN with label conditioning for controlled generation
- **Preprocessing**: VGM transformation, zero/peak inflation handling
- **Evaluation**: Detection metric (RF classifier AUC) and Efficacy metric (AUC ratio)

## Notebooks

| Notebook | Description |
|----------|-------------|
| `main.ipynb` | Full implementation with VGM transformation and advanced techniques |
| `baseline.ipynb` | Simplified version with MinMax scaling |
| `improved.ipynb` | Experimental improvements |
| `ctgan.ipynb` | CTGAN-style architecture modifications |
| `eda.ipynb` | Data exploration and visualization |

## References

1. Xu, L. et al. "Modeling Tabular data using Conditional GAN" NeurIPS 2019
2. Zhao, Z. et al. "CTAB-GAN+: Enhancing Tabular Data Synthesis"
3. Gulrajani, I. et al. "Improved Training of Wasserstein GANs" NeurIPS 2017
