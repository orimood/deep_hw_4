# GAN/cGAN Experiment Log

## Problem Statement
- **Initial Detection AUC**: ~0.998 (goal: close to 0.5)
- **Initial Efficacy**: ~0.99 (good, want to maintain)
- **Issue**: Synthetic data is easily distinguishable from real data despite good efficacy

---

## Root Cause Analysis (From Plots)

### Loss Plot Issues
- Discriminator loss was flat at 0 throughout training
- Generator loss stayed high (~3-5) without proper convergence
- Indicates discriminator winning too easily

### Feature Distribution Issues
| Feature | Problem |
|---------|---------|
| **capital-gain** | Real has ~92% zeros at spike, synthetic spreads values |
| **capital-loss** | Same - real has zero spike, synthetic is broader |
| **education-num** | Real has sharp discrete peaks, synthetic too smooth |
| **hours-per-week** | Real has 40-hour spike, synthetic misses this |

### Conclusion
The GAN cannot capture **zero-inflation** and **discrete-like spikes**. These are trivial fingerprints for RF detection.

---

## Major Architecture Changes (Latest Session)

### 1. Zero-Inflation Handling (CRITICAL FIX)
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **Problem** | capital-gain/capital-loss have ~90% zeros; GAN produced smooth distributions |
| **Solution** |
- Separate binary indicator `is_zero` (0 or 1)
- Only model non-zero values with VGM
- Generator outputs: `(is_zero_prob, normalized_value, mode_one_hot)`
- During sampling: if `is_zero > 0.5`, value is set to 0 |
| **Impact** | Explicitly models the zero spike instead of trying to learn it implicitly |

### 2. Peak-Inflation Handling for hours-per-week (NEW)
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **Problem** | hours-per-week has 46.7% at exactly 40 (huge spike); GAN smoothed it out |
| **Solution** |
- Separate binary indicator `is_peak` (1 if value == 40)
- Only model non-peak values with VGM
- Generator outputs: `(is_peak_prob, normalized_value, mode_one_hot)`
- During sampling: if `is_peak > 0.5`, value is set to exactly 40 |
| **Impact** | Captures the 40-hour workweek spike explicitly |

### 3. education-num as Categorical (NEW)
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **Problem** | education-num is essentially discrete (9=HS-grad 32%, 10=Some-college 22%, 13=Bachelors 16%) |
| **Solution** | Moved from continuous to categorical features |
| **Impact** |
- CONTINUOUS_COLS: 6 → 5 features
- CATEGORICAL_COLS: 8 → 9 features
- education-num now one-hot encoded instead of VGM |

### 4. Special Proportion Loss (Unified)
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **Problem** | Generator wasn't matching proportions of special values |
| **Solution** | Unified loss for both zero and peak inflated features |
| **Implementation** | `g_loss = wgan_loss + λ_corr * corr_loss + λ_special * special_loss` |
| **Parameter** | `LAMBDA_SPECIAL = 0.5` |

### 3. Increased Discriminator Regularization
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **Changes** |
- Dropout increased: 0.4 → **0.5**
- Label smoothing: use 0.9 instead of 1.0 for real samples
- Reduced hidden capacity (fewer parameters) |
| **Impact** | Prevents D from overfitting to subtle artifacts |

### 4. TTUR Learning Rate Adjustment
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **Change** | `LR_G = 0.0002`, `LR_D = 0.0001` (2:1 ratio) |
| **Reason** | Generator needs to learn faster to catch up with discriminator |

### 5. Slower Instance Noise Annealing
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **Change** | Noise anneals over 80% of training instead of 100% |
| **Parameter** | `INSTANCE_NOISE = 0.15` (reduced from 0.2) |

### 6. Plot Saving
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **Change** | All plots saved to `plots/gan/` and `plots/cgan/` folders |
| **Files** | `losses_seed{N}.png`, `features_seed{N}.png`, `correlation_seed{N}.png`, `zero_inflation_seed{N}.png` |

---

## Previously Implemented

### VGM (Variational Gaussian Mixture) Mode-Specific Normalization
| Status | Implemented |
|--------|-------------|
| **Source** | [CTGAN Paper](https://arxiv.org/pdf/1907.00503) |
| **Impact** | Continuous dimension: 6 → 36 (6 features × (1 + 5 modes)) |
| **Note** | Now 38 with zero-inflation (2 extra for is_zero indicators) |

### Rare Category Grouping
| Status | Implemented |
|--------|-------------|
| **Change** | Group categories with <1% frequency into 'Other' |
| **Impact** | native-country: 42 → ~5 categories |

### WGAN-GP (Wasserstein Loss + Gradient Penalty)
| Status | Implemented |
|--------|-------------|
| **Source** | [WGAN-GP Paper](https://arxiv.org/abs/1704.00028) |

### Correlation Loss
| Status | Implemented |
|--------|-------------|
| **Source** | [CTAB-GAN+](https://pmc.ncbi.nlm.nih.gov/articles/PMC10801038/) |
| **Parameter** | `LAMBDA_CORR = 0.1` |

### Gumbel-Softmax with Straight-Through Estimator
| Status | Implemented |
|--------|-------------|
| **Source** | [Gumbel-Softmax Paper](https://arxiv.org/abs/1611.01144) |

### Spectral Normalization
| Status | Implemented |
|--------|-------------|
| **Source** | [Spectral Normalization Paper](https://arxiv.org/abs/1802.05957) |

### Minibatch Discrimination
| Status | Implemented |
|--------|-------------|
| **Source** | [Improved GAN Training](https://arxiv.org/abs/1606.03498) |

---

## Current Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `LATENT_DIM` | 128 | |
| `HIDDEN_DIM` | 128 | Reduced for weaker D |
| `EPOCHS` | 500 | |
| `BATCH_SIZE` | 128 | |
| `LR_G` | 0.0002 | Increased (TTUR) |
| `LR_D` | 0.0001 | Lower than G |
| `N_CRITIC` | 5 | |
| `LAMBDA_GP` | 10 | |
| `LAMBDA_CORR` | 0.1 | |
| `LAMBDA_SPECIAL` | 0.5 | For zero/peak proportion matching |
| `TEMPERATURE` | 0.2 | |
| `DROPOUT` | 0.5 | Increased |
| `INSTANCE_NOISE` | 0.15 | Reduced |
| `N_MODES` | 5 | |
| `RARE_THRESHOLD` | 0.01 | |

---

## Results Log

### Baseline (Before Zero-Inflation Fix)
| Metric | GAN (seed 42) | cGAN (seed 42) |
|--------|---------------|----------------|
| Detection AUC | 0.9984 | 0.9977 |
| Efficacy Ratio | 0.9932 | 0.9874 |

### After Zero-Inflation + All Improvements
| Metric | GAN (seed 42) | cGAN (seed 42) |
|--------|---------------|----------------|
| Detection AUC | *Pending* | *Pending* |
| Efficacy Ratio | *Pending* | *Pending* |

### Final Results (Average over 3 seeds)
| Metric | GAN | cGAN |
|--------|-----|------|
| Detection AUC | *Pending* | *Pending* |
| Efficacy Ratio | *Pending* | *Pending* |

**Target**: Detection AUC < 0.70 while maintaining Efficacy > 0.85

---

## Architecture Summary

### Generator (with Zero/Peak Inflation)
```
Input: latent_dim (128)
  ↓
Linear(128 → 128) + LayerNorm + LeakyReLU
  ↓
Linear(128 → 256) + LayerNorm + LeakyReLU
  ↓
Linear(256 → 256) + LayerNorm + LeakyReLU
  ↓
Linear(256 → output_dim)
  ↓
For zero-inflated features (capital-gain, capital-loss):
  - 1 logit → sigmoid (is_zero probability)
  - 1 value → tanh (normalized value if non-zero)
  - 5 logits → Gumbel-Softmax (mode selection)
For peak-inflated features (hours-per-week):
  - 1 logit → sigmoid (is_peak probability, peak=40)
  - 1 value → tanh (normalized value if not peak)
  - 5 logits → Gumbel-Softmax (mode selection)
For regular continuous features (age, fnlwgt):
  - 1 value → tanh (normalized value)
  - 5 logits → Gumbel-Softmax (mode selection)
For categorical features (including education-num):
  - k logits → Gumbel-Softmax (category selection)
```

### Discriminator (with Increased Regularization)
```
Input: data_dim
  ↓
SpectralNorm(Linear(input → 128)) + LeakyReLU + Dropout(0.5)
  ↓
SpectralNorm(Linear(128 → 128)) + LeakyReLU + Dropout(0.5)
  ↓
MinibatchDiscrimination (32 features)
  ↓
SpectralNorm(Linear(160 → 64)) + LeakyReLU + Dropout(0.5)
  ↓
Linear(64 → 1) (no sigmoid, WGAN)
```

---

## File Structure
```
plots/
├── gan/
│   ├── losses_seed42.png
│   ├── features_seed42.png
│   ├── correlation_seed42.png
│   ├── zero_inflation_seed42.png
│   └── ... (seeds 123, 456)
└── cgan/
    ├── losses_seed42.png
    ├── features_seed42.png
    ├── correlation_seed42.png
    ├── zero_inflation_seed42.png
    └── ... (seeds 123, 456)
```

---

## References

1. **CTGAN**: Xu, L. et al. "Modeling Tabular data using Conditional GAN" NeurIPS 2019
2. **CTAB-GAN+**: Zhao, Z. et al. "CTAB-GAN+: Enhancing Tabular Data Synthesis"
3. **Gumbel-Softmax**: Jang, E. et al. "Categorical Reparameterization with Gumbel-Softmax" ICLR 2017
4. **WGAN-GP**: Gulrajani, I. et al. "Improved Training of Wasserstein GANs" NeurIPS 2017
5. **Spectral Normalization**: Miyato, T. et al. "Spectral Normalization for GANs" ICLR 2018

---

## CTGAN-Style Fixes (V2 Notebook)

**File**: `notebook_ctgan_fixes.ipynb`

Based on research into working CTGAN implementations, the following critical differences were identified:

### Key Findings from CTGAN Research

| Aspect | Original Implementation | CTGAN Default | Issue |
|--------|------------------------|---------------|-------|
| PAC (packing) | None | **10** | Missing mode collapse prevention |
| N_CRITIC | 5 (WGAN-GP) | **1** | Too many D updates |
| Hidden Dim | 128 | **256** | Network too small |
| Batch Size | 128 | **500** | Too small (must be divisible by PAC) |
| LR_D | 0.0001 | **0.0002** | Should match LR_G |
| Architecture | MLP | **Residual blocks** | Missing skip connections |
| Log Transform | None | **Yes** | Long-tail features not compressed |

### V2 Changes Implemented

#### 1. PAC (Packing) - CRITICAL
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **What** | Concatenate 10 samples before feeding to discriminator |
| **Why** | Prevents mode collapse by making D evaluate batches, not individuals |
| **Source** | [CTGAN Paper](https://arxiv.org/abs/1907.00503) |

#### 2. N_CRITIC = 1 (Not 5)
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **What** | Changed from 5 discriminator updates per G update to 1:1 |
| **Why** | CTGAN uses 1:1 ratio, not WGAN-GP's 5:1 |
| **Impact** | Generator gets more training relative to D |

#### 3. Larger Network (256 Hidden)
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **What** | Increased hidden_dim from 128 to 256 |
| **Why** | CTGAN uses (256, 256) for both G and D |

#### 4. Residual Blocks in Generator
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **What** | Added 2 residual blocks in Generator |
| **Why** | Better gradient flow, matches CTGAN architecture |

#### 5. Batch Size = 500
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **What** | Increased from 128 to 500 |
| **Why** | CTGAN default, must be divisible by PAC=10 |

#### 6. Same Learning Rate for G and D
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **What** | LR = 0.0002 for both (was 0.0002/0.0001) |
| **Why** | CTGAN uses same rate for both |

#### 7. Log Transform for Long-Tail Features
| Status | **IMPLEMENTED** |
|--------|-----------------|
| **What** | Apply log(1 + |x|) to capital-gain, capital-loss before VGM |
| **Why** | Compresses extreme values, makes distribution easier to learn |

---

## V2 Hyperparameters (CTGAN-Style)

| Parameter | V1 Value | V2 Value | Notes |
|-----------|----------|----------|-------|
| `LATENT_DIM` | 128 | 128 | |
| `HIDDEN_DIM` | 128 | **256** | CTGAN default |
| `EPOCHS` | 500 | **300** | CTGAN default |
| `BATCH_SIZE` | 128 | **500** | Must be divisible by PAC |
| `LR_G` | 0.0002 | 0.0002 | |
| `LR_D` | 0.0001 | **0.0002** | Same as G (CTGAN) |
| `N_CRITIC` | 5 | **1** | CTGAN uses 1:1 |
| `PAC` | N/A | **10** | NEW - packing |
| `LAMBDA_GP` | 10 | 10 | |
| `LAMBDA_CORR` | 0.1 | 0.1 | |
| `LAMBDA_SPECIAL` | 0.5 | 0.5 | |
| `TEMPERATURE` | 0.2 | 0.2 | |
| `DROPOUT` | 0.5 | 0.5 | |

---

## V2 Architecture Summary

### Generator (CTGAN-Style with Residual Blocks)
```
Input: latent_dim (128)
  ↓
Linear(128 → 256) + BatchNorm + ReLU
  ↓
ResidualBlock(256) [fc→bn→relu→fc→bn + skip]
  ↓
ResidualBlock(256) [fc→bn→relu→fc→bn + skip]
  ↓
Linear(256 → output_dim)
  ↓
Output heads (same as V1):
  - Zero/Peak-inflated: (is_special, value, mode_onehot)
  - Regular: (value, mode_onehot)
  - Categorical: Gumbel-Softmax
```

### Discriminator (CTGAN-Style with PAC)
```
Input: data_dim × PAC (concatenated 10 samples)
  ↓
Linear(input*10 → 256) + LeakyReLU + Dropout(0.5)
  ↓
Linear(256 → 256) + LeakyReLU + Dropout(0.5)
  ↓
Linear(256 → 256) + LeakyReLU + Dropout(0.5)
  ↓
Linear(256 → 1) (WGAN score)
```

---

## V2 File Structure
```
plots_v2/
├── gan/
│   ├── losses_seed42.png
│   ├── features_seed42.png
│   ├── correlation_seed42.png
│   └── ... (seeds 123, 456)
└── cgan/
    ├── losses_seed42.png
    ├── features_seed42.png
    ├── correlation_seed42.png
    └── ... (seeds 123, 456)
```

---

## V2 Results (CTGAN-Style)

### After CTGAN Fixes
| Metric | GAN (seed 42) | cGAN (seed 42) |
|--------|---------------|----------------|
| Detection AUC | *Pending* | *Pending* |
| Efficacy Ratio | *Pending* | *Pending* |

### Final Results (Average over 3 seeds)
| Metric | GAN | cGAN |
|--------|-----|------|
| Detection AUC | *Pending* | *Pending* |
| Efficacy Ratio | *Pending* | *Pending* |

---

## Next Steps

1. ✅ Implement zero-inflation handling (capital-gain, capital-loss)
2. ✅ Implement peak-inflation handling (hours-per-week at 40)
3. ✅ Move education-num to categorical
4. ✅ Add special proportion loss (unified zero/peak)
5. ✅ Increase discriminator dropout to 0.5
6. ✅ Add label smoothing (0.9 for real)
7. ✅ Implement plot saving to folders
8. ✅ Update experiment configuration
9. ✅ Research CTGAN implementation differences
10. ✅ Create V2 notebook with CTGAN fixes (PAC, n_critic=1, larger network)
11. ❌ V2 FAILED: Detection AUC = 1.0, Efficacy = 0.47 (worse than baseline!)
12. ✅ Created V3 Simple Baseline notebook (notebook_simple_baseline.ipynb)
13. ⏳ Run V3 simple baseline and compare results
14. ⏳ If V3 works better, iterate from there
15. ⏳ Run all 3 seeds and report final results

---

## V2 CTGAN-Style Results (FAILED)

| Metric | GAN (seed 42) |
|--------|---------------|
| Detection AUC | **1.0000** (WORSE - was 0.998) |
| Efficacy Ratio | **0.4748** (MUCH WORSE - was 0.99) |

**Conclusion**: The CTGAN-style changes (PAC=10, n_critic=1, residual blocks) caused catastrophic failure. The Generator completely failed to learn useful representations.

---

## V3 Simple Baseline Approach

**File**: `notebook_simple_baseline.ipynb`

**Philosophy**: Strip away complexity, establish working baseline first.

### Key Simplifications from V2:

| Aspect | V2 (CTGAN) | V3 (Simple) | Reason |
|--------|------------|-------------|--------|
| Normalization | VGM (modes) | **MinMax [0,1]** | VGM adds 5x dimensions, may be overkill |
| Architecture | Residual blocks | **Simple MLP** | Residual may cause training instability |
| PAC | 10 | **None** | Complex, may be implemented wrong |
| N_CRITIC | 1 | **5** | Standard WGAN-GP recommendation |
| Zero-inflation | Explicit handling | **None** | Let GAN learn implicitly first |
| Hidden dim | 256 | 256 | Same |
| Batch size | 500 | **256** | Standard |

### V3 Configuration:

| Parameter | Value |
|-----------|-------|
| LATENT_DIM | 128 |
| HIDDEN_DIM | 256 |
| BATCH_SIZE | 256 |
| EPOCHS | 300 |
| LR_G | 0.0002 |
| LR_D | 0.0001 (TTUR) |
| N_CRITIC | 5 |
| LAMBDA_GP | 10 |
| DROPOUT | 0.3 |

### Expected Outcome:
- Detection AUC should be ~0.99 (same as original, not worse)
- Efficacy should be ~0.99 (same as original)
- This establishes a working baseline to iterate from
