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
│   ├── gan_fixed.ipynb         # **RECOMMENDED** Mixture model GAN (fixes detection issue)
│   ├── main.ipynb              # Main GAN implementation (VGM + WGAN-GP)
│   ├── baseline.ipynb          # Simplified baseline (MinMax scaling)
│   ├── improved.ipynb          # Improved version experiments
│   ├── ctgan.ipynb             # CTGAN-style modifications
│   └── eda.ipynb               # Exploratory Data Analysis
├── outputs/
│   ├── gan_fixed/              # Mixture model GAN outputs
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
| `gan_fixed_v2.ipynb` | **LATEST** - V5 with quantile/moment/correlation losses |
| `gan_fixed.ipynb` | V4 - Mixture model GAN with explicit zero/peak handling |
| `main.ipynb` | Full implementation with VGM transformation and advanced techniques |
| `baseline.ipynb` | Simplified version with MinMax scaling |
| `improved.ipynb` | Experimental improvements |
| `ctgan.ipynb` | CTGAN-style architecture modifications |
| `eda.ipynb` | Data exploration and visualization |

## Key Innovation (gan_fixed.ipynb)

The main issue with standard GANs on tabular data is that neural networks cannot model **point masses**:
- `capital-gain`: 91% are exactly 0
- `capital-loss`: 95% are exactly 0
- `hours-per-week`: 47% are exactly 40

The `gan_fixed.ipynb` notebook solves this with **mixture models**:
```
P(x) = p_special * delta(special_value) + (1-p_special) * P_continuous(x)
```

The generator outputs separate indicators for special values and samples from them at generation time.

## References

1. Xu, L. et al. "Modeling Tabular data using Conditional GAN" NeurIPS 2019
2. Zhao, Z. et al. "CTAB-GAN+: Enhancing Tabular Data Synthesis"
3. Gulrajani, I. et al. "Improved Training of Wasserstein GANs" NeurIPS 2017
