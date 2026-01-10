# Assignment 4: Generative Models with GAN/cGAN

Deep Learning course assignment implementing GAN and Conditional GAN for synthetic tabular data generation on the Adult dataset.

## Project Structure

```
deep_hw_4/
├── venv/                   # Virtual environment
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── notebook.ipynb         # Main implementation notebook
├── adult.arff             # Adult dataset
└── report.pdf             # Final report
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook:
```bash
jupyter notebook notebook.ipynb
```

## Dataset

The Adult dataset (Census Income) contains demographic information to predict whether income exceeds $50K/year. Features include age, workclass, education, occupation, etc.

## Implementation

- **GAN**: Standard GAN with MLP-based Generator and Discriminator for tabular data
- **cGAN**: Conditional GAN with label conditioning for controlled generation
- **Evaluation**: Detection metric (RF classifier) and Efficacy metric (AUC ratio)

## References

1. Xu, L. et al. "Modeling Tabular data using Conditional GAN" NeurIPS 2019
2. Zhao, Z. et al. "CTAB-GAN+: Enhancing Tabular Data Synthesis"
