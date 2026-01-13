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

---

## V4 Mixture Model GAN (gan_fixed.ipynb)

**File**: `notebooks/gan_fixed.ipynb`

**Philosophy**: Explicitly model point masses that neural networks cannot learn implicitly.

### Core Insight

Standard GANs fail on tabular data because sigmoid/tanh activations produce **smooth continuous distributions**, but real data has **point masses** (delta functions):
- `capital-gain`: 91.7% exactly 0
- `capital-loss`: 95.3% exactly 0
- `hours-per-week`: 46.8% exactly 40

A Random Forest can trivially detect synthetic data with a single split: `if capital_gain > 0.001 → synthetic`

### V4 Architecture: Mixture Model Outputs

```
Generator Output Structure:
├── Zero-inflated (capital-gain, capital-loss):
│   ├── is_zero (sigmoid) → Bernoulli sample at generation → EXACT 0
│   └── value (sigmoid) → only used when is_zero=0
├── Peak-inflated (hours-per-week):
│   ├── is_peak (sigmoid) → Bernoulli sample at generation → EXACT 40
│   └── value (sigmoid) → only used when is_peak=0
├── Continuous (age, fnlwgt):
│   └── value (sigmoid) → [0, 1]
└── Categorical (education-num + 8 others):
    └── Gumbel-softmax → hard one-hot
```

### V4 Key Innovations

| Innovation | Description |
|------------|-------------|
| **Mixture Model** | Separate binary indicator + continuous value for special features |
| **Hard Sampling** | At generation, sample from Bernoulli to produce EXACT special values |
| **Proportion Loss** | Explicit loss to match proportion of zeros/peaks |
| **Categorical Frequency Loss** | Match category distributions |

### V4 Loss Function

```python
g_loss = wgan_loss + λ_prop * proportion_loss + λ_cat * categorical_freq_loss

where:
  - proportion_loss = Σ (actual_prop - target_prop)²  # for zero/peak features
  - categorical_freq_loss = MSE(fake_freq, real_freq)  # per category
```

### V4 Results (Seed 42)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Detection AUC** | 0.9804 | < 0.70 | ❌ Still high |
| **Efficacy** | 0.5395 | > 0.85 | ❌ Dropped significantly |

### V4 What Worked ✅

Special value proportions now match **perfectly**:

| Feature | Real | Synthetic | Target | Match |
|---------|------|-----------|--------|-------|
| capital-gain zeros | 91.7% | 91.7% | 91.7% | ✅ Perfect |
| capital-loss zeros | 95.3% | 95.4% | 95.3% | ✅ Perfect |
| hours-per-week peaks | 46.8% | 46.5% | 46.8% | ✅ Perfect |

### V4 What Still Fails ❌

RF Feature Importance for Detection (what discriminator sees):

| Feature | Importance | Problem |
|---------|------------|---------|
| **hours-per-week_value** | 39% | Non-peak values don't match distribution |
| **fnlwgt** | 14.5% | Distribution mismatch |
| **age** | 10% | Distribution mismatch |
| hours-per-week_is_peak | 6.5% | Minor (indicator is close) |

### V4 Root Cause Analysis

The **proportion** of special values is fixed, but the **distribution of non-special values** is wrong:

1. **hours-per-week non-40 values**: Real has broad distribution, synthetic has narrow spike around 0.5
2. **fnlwgt**: Real is right-skewed, synthetic is shifted
3. **age**: Minor mismatch in tails

The sigmoid activation produces a **unimodal distribution** centered around 0.5, but real data often has:
- Skewed distributions (fnlwgt)
- Multi-modal distributions (hours-per-week: part-time vs overtime)
- Different spreads

### V4 Conclusion

Mixture model successfully handles point masses, but continuous distributions still distinguishable.

---

## V5 Plan: Fix Continuous Distributions

**File**: `notebooks/gan_fixed_v2.ipynb`

**Goal**: Make continuous feature distributions indistinguishable from real data.

### Targeted Fixes (Based on RF Feature Importance)

#### Fix 1: hours-per-week_value (39% importance)

| Problem | Solution |
|---------|----------|
| Sigmoid produces narrow distribution | Use **tanh + scaling** to match real data range |
| Non-peak values have specific distribution | Add **quantile matching loss** |

#### Fix 2: fnlwgt (14.5% importance)

| Problem | Solution |
|---------|----------|
| Right-skewed distribution not captured | Apply **log transform** before scaling |
| Distribution shape mismatch | Add **moment matching loss** (mean, std, skew) |

#### Fix 3: age (10% importance)

| Problem | Solution |
|---------|----------|
| Tail distributions differ | Add **quantile matching loss** |

### New Loss Functions to Add

```python
# 1. Quantile Matching Loss
def quantile_loss(real, fake, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    loss = 0
    for q in quantiles:
        real_q = torch.quantile(real, q, dim=0)
        fake_q = torch.quantile(fake, q, dim=0)
        loss += (real_q - fake_q).pow(2).mean()
    return loss

# 2. Moment Matching Loss
def moment_loss(real, fake):
    # Match mean, std, skewness
    loss = (real.mean(0) - fake.mean(0)).pow(2).mean()
    loss += (real.std(0) - fake.std(0)).pow(2).mean()
    return loss

# 3. Correlation Preservation Loss
def correlation_loss(real, fake):
    real_corr = torch.corrcoef(real.T)
    fake_corr = torch.corrcoef(fake.T)
    return (real_corr - fake_corr).pow(2).mean()
```

### Architecture Changes

1. **Continuous features**: Use `tanh` (range [-1,1]) instead of `sigmoid` (range [0,1]) for better gradient flow
2. **Add skip connections**: Help generator learn identity mapping for well-matched features
3. **Feature-specific output heads**: Separate processing for different feature types

### V5 Expected Improvements

| Metric | V4 | V5 Target |
|--------|-----|-----------|
| Detection AUC | 0.9804 | < 0.80 |
| Efficacy | 0.5395 | > 0.85 |

### V5 Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| LAMBDA_QUANTILE | 1.0 | Match distribution shape |
| LAMBDA_MOMENT | 0.5 | Match mean/std |
| LAMBDA_CORR | 0.5 | Preserve relationships |
| LAMBDA_PROP | 1.0 | Keep proportion matching |
| LOG_TRANSFORM | fnlwgt | Handle right-skew |

---

---

## V5 Results (gan_fixed_v2.ipynb - Initial)

**Implemented all planned V5 fixes:**
- ✅ Quantile matching loss (5 quantiles: 10th, 25th, 50th, 75th, 90th)
- ✅ Moment matching loss (mean, std)
- ✅ Correlation preservation loss
- ✅ Log transform for fnlwgt
- ✅ Skip connections in generator
- ✅ Spectral normalization in discriminator

### V5 Results (Seed 42)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Detection AUC** | 0.9802 | < 0.70 | ❌ Still high |
| **Efficacy** | 0.5692 | > 0.85 | ❌ Slight improvement |

### V5 RF Feature Importance (Detection)

| Feature | Importance | Problem |
|---------|------------|---------|
| **hours-per-week_value** | 48.9% | Distribution shape still wrong |
| **age** | 10.4% | Distribution shifted |
| **fnlwgt** | 8.8% | Still mismatch |
| hours-per-week_is_peak | 6.2% | Minor |

### V5 Analysis

The quantile loss with only 5 points isn't capturing the full distribution shape.
The generator's sigmoid activation produces a narrow, unimodal distribution.

---

## V6 Plan: Histogram Matching (Current)

**File**: `notebooks/gan_fixed_v2.ipynb` (updated)

### V6 Targeted Fixes

| Fix | Description | Target Feature |
|-----|-------------|----------------|
| **Histogram Matching Loss** | Soft KL-divergence over 30 bins | All continuous |
| **20 Quantile Points** | Up from 5, finer shape matching | All continuous |
| **Skewness Matching** | Added to moment loss | age, fnlwgt |
| **Learned Variance** | Per-feature variance in generator | All continuous |
| **Generation Noise** | Add noise during sampling for diversity | hours-per-week |

### V6 New Loss Functions

```python
# Histogram Matching Loss (soft, differentiable)
def histogram_matching_loss(real, fake, n_bins=30):
    # Create soft histogram using Gaussian kernel
    # KL divergence between histograms

# Enhanced Moment Loss with Skewness
def moment_matching_loss(real, fake):
    # Match mean, std, AND skewness
```

### V6 Configuration

| Parameter | V5 Value | V6 Value | Notes |
|-----------|----------|----------|-------|
| LAMBDA_QUANTILE | 1.0 | **2.0** | Increased weight |
| LAMBDA_MOMENT | 0.5 | **1.0** | Increased + skewness |
| LAMBDA_HIST | N/A | **2.0** | NEW histogram loss |
| N_QUANTILES | 5 | **20** | Finer matching |
| Generator | Sigmoid output | **Sigmoid + learned variance** | More flexibility |

### V6 Architecture Changes

```
Generator Changes:
- Output head: sigmoid + learned per-feature scale/bias
- Training: Add noise proportional to learned log_var
- Generation: Add noise for diversity (0.05-0.15 depending on feature)
```

### V6 Results (Seed 42)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Detection AUC** | 0.9732 | < 0.70 | ❌ Still high |
| **Efficacy** | 0.5383 | > 0.85 | ❌ Worse than V5 |

### V6 RF Feature Importance

| Feature | V5 | V6 | Change |
|---------|-----|-----|--------|
| hours-per-week_value | 48.9% | 43.4% | -5.5% |
| age | 10.4% | 11.1% | +0.7% |
| fnlwgt | 8.8% | 8.8% | same |

### V6 Conclusion

Histogram matching provided only marginal improvement (0.7% detection AUC reduction).
The fundamental problem: **generator architecture cannot produce correct distribution shapes**.
Sigmoid/tanh outputs are inherently unimodal and smooth - real data has complex shapes.

---

## V7 Plan: Inverse CDF Transform (Current)

**File**: `notebooks/gan_fixed_v2.ipynb` (updated)

### Core Idea

Instead of training the generator to output the correct distribution (impossible with sigmoid),
use the generator's [0,1] output as a **quantile** and transform via inverse CDF.

```
Generator Output (sigmoid) → [0, 1] = probability/quantile
                ↓
Inverse CDF (from real data) → actual value with correct distribution
```

### Why This Works

| Approach | Problem |
|----------|---------|
| Sigmoid output | Always produces smooth unimodal distribution |
| Loss functions | Can match moments/quantiles but not full shape |
| **Inverse CDF** | **Guarantees exact marginal distribution match** |

### Implementation

```python
# During preprocessing: store empirical quantiles
self.quantile_values[col] = np.percentile(values, np.linspace(0, 100, 1001))

# During generation: transform [0,1] → real value
def inverse_cdf_transform(uniform_val, quantile_values):
    # uniform_val in [0, 1] from sigmoid
    # Returns value from real distribution
    idx = uniform_val * (len(quantile_values) - 1)
    return np.interp(idx, range(len(quantile_values)), quantile_values)
```

### Expected Impact

| Metric | V6 | V7 Target |
|--------|-----|-----------|
| Detection AUC | 0.9732 | < 0.60 |
| Efficacy | 0.5383 | > 0.85 |

The marginal distributions will match **perfectly** by construction.
RF will only be able to detect via feature correlations, not individual distributions.

---

## V7 Results: Inverse CDF Transform

**File**: `notebooks/gan_fixed_v2.ipynb`

### V7 Results (Seed 42)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Detection AUC** | 0.9779 | < 0.70 | ❌ Still high |
| **Efficacy** | 0.5451 | > 0.85 | ❌ Still low |

### V7 RF Feature Importance

| Feature | V6 | V7 | Change |
|---------|-----|-----|--------|
| hours-per-week_value | 43.4% | 24.1% | -19% ✅ |
| fnlwgt | 8.8% | 20.7% | +12% ❌ |
| hours-per-week_is_peak | 7.5% | 14.4% | +7% ❌ |
| age | 11.1% | 14.3% | +3% ❌ |

### V7 Conclusion

Inverse CDF improved individual feature distributions but **broke correlations**.
The RF now detects via correlation patterns instead of individual distributions.

---

## V8: Final Clean Implementation

**File**: `notebooks/final.ipynb`

### Decision: Start Fresh with Simple Approach

After V5-V7 attempts with complex losses (histogram, quantile, inverse CDF),
decided to simplify and focus on what the assignment actually requires:

1. **GAN** - Standard GAN that works
2. **cGAN** - Conditional GAN with label conditioning
3. **Proper evaluation** - 3 seeds, Detection AUC, Efficacy

### V8 Architecture

**Simplified design:**
- WGAN-GP with gradient penalty
- Mixture model for zeros/peaks only
- No complex auxiliary losses
- cGAN adds one-hot label to both G and D inputs

```
GAN:
  Generator: z (128) → MLP → fake_data
  Discriminator: data → MLP → score

cGAN:
  Generator: [z, label_onehot] → MLP → fake_data
  Discriminator: [data, label_onehot] → MLP → score
```

### V8 Key Differences from V5-V7

| Aspect | V5-V7 | V8 |
|--------|-------|-----|
| Losses | WGAN + histogram + quantile + moment + correlation | WGAN + proportion only |
| Complexity | High (many hyperparameters) | Low (simple) |
| cGAN | Not implemented | ✅ Implemented |
| Seeds | 1 seed | 3 seeds (42, 123, 456) |
| Focus | Perfect distributions | Working models |

### V8 Configuration

| Parameter | Value |
|-----------|-------|
| LATENT_DIM | 128 |
| HIDDEN_DIM | 256 |
| BATCH_SIZE | 256 |
| EPOCHS | 300 |
| LR_G | 2e-4 |
| LR_D | 1e-4 |
| N_CRITIC | 5 |
| LAMBDA_GP | 10 |
| LAMBDA_PROP | 1.0 |

---

## Summary of All Versions

| Version | Approach | Detection AUC | Efficacy | Issue |
|---------|----------|---------------|----------|-------|
| Baseline | Simple MinMax | ~0.999 | ~0.99 | RF trivially detects |
| V4 | Mixture model | 0.9804 | 0.5395 | Proportions match, distributions don't |
| V5 | + Quantile/Moment loss | 0.9802 | 0.5692 | Minimal improvement |
| V6 | + Histogram loss | 0.9732 | 0.5383 | Marginal improvement |
| V7 | Inverse CDF | 0.9779 | 0.5451 | Broke correlations |
| V8 | Simple + cGAN | Pending | Pending | Focus on working models |

---

## V8 Initial Run: CRITICAL BUG FOUND

**File**: `notebooks/final.ipynb`

### V8 Initial Results (Seed 42) - BUGGED

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Detection AUC** | **1.0000** | < 0.70 | ❌ CATASTROPHIC |
| **Efficacy** | 0.5843 | > 0.85 | ❌ Low |

### Bug Analysis

The Generator's `_apply_activations` function had incorrect value encoding:

```python
# WRONG (what it was):
if hard:
    is_zero = (torch.rand_like(is_zero) < is_zero).float()
    value = is_zero * 0 + (1 - is_zero) * value  # ← zeros got value=0

# CORRECT (what real data has):
values[~mask] = 0  # Zero-inflated: value=0 when zero
values[~mask] = 0.5  # Peak-inflated: value=0.5 when peak
```

The issue: Real data encodes zeros with `value=0` but peaks with `value=0.5`.
The generator was using `0` for both, then got "fixed" to use `0.5` for both.

RF could trivially detect synthetic data by checking if `value=0` when `is_zero=1`.

---

## V9: Integrate gan_fixed_v2 Elements (Current)

**File**: `notebooks/final.ipynb` (updated)

### Changes Made

After finding the V8 bug, integrated successful elements from `gan_fixed_v2.ipynb`:

#### 1. Generator Architecture Improvements

| Change | Description |
|--------|-------------|
| **Skip connections** | `h3 = h3 + skip_proj(h1)` - helps gradient flow |
| **Learned variance** | Per-feature `log_var` output for diversity |
| **Learnable scale/bias** | `cont_scale`, `cont_bias` parameters per continuous feature |

#### 2. Feature Engineering

| Change | Description |
|--------|-------------|
| **education-num as categorical** | 16 discrete values → one-hot (not continuous) |
| **Log transform for fnlwgt** | `np.log1p(values)` - handles right-skew |
| **Correct zero encoding** | `value=0` when `is_zero=1` (matches real data) |

#### 3. Multiple Auxiliary Losses

| Loss | Weight | Purpose |
|------|--------|---------|
| `proportion_loss` | λ=1.0 | Match zero/peak proportions |
| `quantile_loss` | λ=0.5 | Match 20 quantile points |
| `moment_loss` | λ=0.5 | Match mean, std, skewness |
| `correlation_loss` | λ=1.0 | Preserve feature correlations |
| `categorical_loss` | λ=0.2 | Match category frequencies |

### V9 Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| LATENT_DIM | 128 | |
| HIDDEN_DIM | 256 | |
| BATCH_SIZE | 256 | |
| EPOCHS | 300 | |
| LR_G | 2e-4 | |
| LR_D | 1e-4 | TTUR |
| N_CRITIC | 5 | |
| LAMBDA_GP | 10 | |
| LAMBDA_PROP | 1.0 | Proportion matching |
| LAMBDA_QUANT | 0.5 | Quantile matching |
| LAMBDA_MOMENT | 0.5 | Moment matching |
| LAMBDA_CORR | 1.0 | Correlation preservation |
| LAMBDA_CAT | 0.2 | Categorical matching |

### V9 Data Dimensions

```
Data dimensions: 123
  [  0-  1] capital-gain (zero_inflated)
  [  2-  3] capital-loss (zero_inflated)
  [  4-  5] hours-per-week (peak_inflated)
  [  6-  6] age (continuous)
  [  7-  7] fnlwgt (continuous) [LOG]
  [  8- 23] education-num (categorical) ← Was continuous in V8
  [ 24- 31] workclass (categorical)
  ... (other categoricals)
```

### V9 Generator Architecture

```
Input: z (128) [+ label_onehot for cGAN]
  ↓
Linear(input → 256) + LayerNorm + LeakyReLU
  ↓
Linear(256 → 512) + LayerNorm + LeakyReLU
  ↓
Linear(512 → 512) + LayerNorm + LeakyReLU + Skip Connection
  ↓
Linear(512 → output_dim + n_cont_features)  # Extra for variance
  ↓
Activations:
  - Zero-inflated: sigmoid(is_zero), sigmoid(value) * scale + bias
  - Peak-inflated: sigmoid(is_peak), sigmoid(value) * scale + bias
  - Continuous: sigmoid(value) * scale + bias
  - Categorical: Gumbel-softmax → hard one-hot
```

### V9 Results (Seed 42) ✅ SIGNIFICANT IMPROVEMENT

| Metric | GAN | cGAN | Target |
|--------|-----|------|--------|
| **Detection AUC** | 0.8974 | **0.8694** | < 0.70 |
| **Efficacy** | 0.6089 | **0.9540** | > 0.85 |

### V9 vs Previous Versions

| Version | Detection AUC | Efficacy | Notes |
|---------|---------------|----------|-------|
| V5-V7 | 0.97-0.98 | 0.54-0.57 | Complex losses, no improvement |
| V8 (bugged) | 1.0000 | 0.58 | Wrong encoding |
| **V9 GAN** | **0.8974** | **0.6089** | -10% detection! |
| **V9 cGAN** | **0.8694** | **0.9540** | Best results! |

### What Made V9 Work (Architecture Analysis)

#### 1. Skip Connections in Generator
```python
h3 = h3 + self.skip_proj(h1)  # Residual path from layer 1 to layer 3
```
- Helps gradient flow during training
- Allows generator to learn identity mapping for well-matched features
- Prevents vanishing gradients in deep network

#### 2. Learned Per-Feature Variance
```python
self.cont_scale = nn.Parameter(torch.ones(n_cont))
self.cont_bias = nn.Parameter(torch.zeros(n_cont))
# During forward:
value = value * self.cont_scale[idx] + self.cont_bias[idx]
```
- Each continuous feature gets learnable scale and bias
- Allows different features to have different output ranges
- More flexible than fixed sigmoid [0,1] output

#### 3. Training-Time Noise Injection
```python
if self.training:
    noise = torch.randn_like(base_val) * torch.exp(0.5 * log_var) * 0.1
    value = torch.clamp(base_val + noise, 0, 1)
```
- Adds learned noise during training for regularization
- Wider noise (0.15) for peak-inflated features (hours-per-week)
- Prevents overfitting to specific values

#### 4. Education-num as Categorical
- Changed from continuous (1 dim) to categorical (16 dims one-hot)
- Real data has discrete values (1-16), not continuous
- Gumbel-softmax produces exact discrete outputs

#### 5. Log Transform for fnlwgt
```python
if col in self.log_transform:
    values = np.log1p(values)
```
- fnlwgt is highly right-skewed (range: 12k - 1.5M)
- Log transform compresses range, makes distribution more normal
- Easier for generator to learn

#### 6. Multiple Auxiliary Losses Working Together
```python
g_loss = (g_loss_wgan +
         1.0 * proportion_loss +    # Match zero/peak proportions
         0.5 * quantile_loss +      # Match 20 quantile points
         0.5 * moment_loss +        # Match mean, std, skewness
         1.0 * correlation_loss +   # Preserve feature correlations
         0.2 * categorical_loss)    # Match category frequencies
```
- **Correlation loss (λ=1.0)** is key - preserves relationships between features
- Without it, RF detects via broken correlations (V7 inverse CDF problem)

#### 7. cGAN Label Conditioning (Why cGAN >> GAN)
```python
# Generator: concat label to input
z = torch.cat([z, labels_onehot], dim=1)

# Discriminator: concat label to input
x = torch.cat([x, labels_onehot], dim=1)
```
- cGAN learns **class-conditional distributions**
- Income <=50K and >50K have different feature distributions
- Generator can specialize outputs per class
- Results: Efficacy 0.95 (cGAN) vs 0.61 (GAN)

### Why cGAN Dramatically Outperforms GAN

| Aspect | GAN | cGAN |
|--------|-----|------|
| Label handling | Copies from training set | Generates with correct class patterns |
| Feature distributions | Must average across classes | Learns per-class distributions |
| Efficacy | 0.6089 | **0.9540** |

The Adult dataset has **class-dependent features**:
- High income → higher education, more hours, capital gains
- Low income → different distribution patterns

cGAN learns these conditional patterns; GAN averages them together.

### V9 Results (3-seed average) - Pending

| Metric | GAN (3-seed avg) | cGAN (3-seed avg) |
|--------|------------------|-------------------|
| Detection AUC | *Pending* | *Pending* |
| Efficacy | *Pending* | *Pending* |

---

## Summary of All Versions

| Version | Approach | Detection AUC | Efficacy | Issue |
|---------|----------|---------------|----------|-------|
| Baseline | Simple MinMax | ~0.999 | ~0.99 | RF trivially detects |
| V4 | Mixture model | 0.9804 | 0.5395 | Proportions match, distributions don't |
| V5 | + Quantile/Moment loss | 0.9802 | 0.5692 | Minimal improvement |
| V6 | + Histogram loss | 0.9732 | 0.5383 | Marginal improvement |
| V7 | Inverse CDF | 0.9779 | 0.5451 | Broke correlations |
| V8 | Simple final.ipynb | 1.0000 | 0.5843 | BUG: wrong encoding |
| **V9 GAN** | Skip conn + losses | **0.8974** | **0.6089** | ✅ Major improvement |
| **V9 cGAN** | + Label conditioning | **0.8694** | **0.9540** | ✅ **BEST RESULTS** |

---

## Next Steps

1. ✅ Implement V5 with quantile/moment/correlation losses
2. ✅ V5 tested - still 98% detection AUC
3. ✅ Implement V6 with histogram matching loss
4. ✅ V6 tested - only 0.7% improvement, efficacy dropped
5. ✅ Implement V7 with inverse CDF transform
6. ✅ V7 tested - helped individual features, broke correlations
7. ✅ Created V8 final.ipynb with both GAN and cGAN
8. ❌ V8 had critical bug - Detection AUC = 1.0
9. ✅ Fixed bug and integrated gan_fixed_v2 elements (V9)
10. ✅ V9 seed 42 results: GAN 0.90/0.61, cGAN **0.87/0.95** 🎉
11. ⏳ Run remaining seeds (123, 456) and compute averages
12. ⏳ Write final report with analysis
