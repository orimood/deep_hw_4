# GAN/cGAN Experiment Log

## Problem Statement
- **Initial Detection AUC**: ~0.998 (goal: close to 0.5)
- **Initial Efficacy**: ~0.99 (good, want to maintain)
- **Issue**: Synthetic data is easily distinguishable from real data despite good efficacy

---

## Changes Implemented

### 1. GPU/CUDA Setup
| Status | Worked |
|--------|--------|
| **Change** | Reinstalled PyTorch with CUDA support |
| **Command** | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| **Result** | PyTorch 2.6.0+cu124, CUDA 12.4, GPU: NVIDIA GeForce RTX 2080 |
| **Outcome** | Training now runs on GPU (much faster) |

---

### 2. Instance Noise (Annealed)
| Status | Pending Test |
|--------|--------------|
| **Change** | Added instance noise to real and fake data during training |
| **Source** | [CTGAN Paper](https://arxiv.org/pdf/1907.00503), [CTAB-GAN+](https://pmc.ncbi.nlm.nih.gov/articles/PMC10801038/) |
| **Justification** | Prevents discriminator from becoming too confident by adding noise that decays over training |
| **Implementation** |
```python
# In train_gan() and train_cgan()
current_noise_std = instance_noise * max(0, 1 - epoch / epochs)  # Linear decay
real_data_noisy = real_data + torch.randn_like(real_data) * current_noise_std
fake_data_noisy = fake_data + torch.randn_like(fake_data) * current_noise_std
```
| **Expected Impact** | Medium - should reduce discriminator confidence |

---

### 3. Hyperparameter Adjustments
| Status | Pending Test |
|--------|--------------|
| **Source** | [WGAN-GP Paper](https://arxiv.org/abs/1704.00028), [CTAB-GAN+](https://pmc.ncbi.nlm.nih.gov/articles/PMC10801038/) |

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `EPOCHS` | 300 | 400 | More convergence time for WGAN-GP |
| `LR_D` | 0.00005 | 0.0001 | Faster discriminator adaptation |
| `LAMBDA_GP` | 10 | 5 | Less constraint on discriminator (allow stronger gradients) |
| `TEMPERATURE` | 0.5 | 0.3 | Sharper categorical outputs (closer to one-hot) |
| `INSTANCE_NOISE` | N/A | 0.1 | New parameter for noise injection |

| **Expected Impact** | Medium - combined effect should help detection metric |

---

## Previously Implemented (Before This Session)

### 4. Gumbel-Softmax with Straight-Through Estimator
| Status | Already Implemented |
|--------|---------------------|
| **Source** | [Gumbel-Softmax Paper](https://arxiv.org/abs/1611.01144) (Jang et al., ICLR 2017) |
| **Problem** | Soft categorical probabilities (e.g., [0.3, 0.5, 0.2]) are trivially distinguishable from hard one-hot real data |
| **Solution** | Gumbel-Softmax produces hard one-hot outputs in forward pass while allowing gradients in backward pass |
| **Outcome** | Necessary but not sufficient - still had high detection AUC |

### 5. WGAN-GP (Wasserstein Loss + Gradient Penalty)
| Status | Already Implemented |
|--------|---------------------|
| **Source** | [WGAN-GP Paper](https://arxiv.org/abs/1704.00028) (Gulrajani et al., NeurIPS 2017) |
| **Problem** | BCE loss causes vanishing gradients when discriminator becomes confident |
| **Solution** | Wasserstein loss provides meaningful gradients; gradient penalty enforces Lipschitz constraint |
| **Outcome** | More stable training but didn't solve detection issue alone |

### 6. Log Transform for Skewed Features
| Status | Already Implemented |
|--------|---------------------|
| **Problem** | `capital-gain` and `capital-loss` have ~90% zeros, MinMax scaling compresses poorly |
| **Solution** | `log1p` transform spreads distribution more evenly before scaling |
| **Outcome** | Improved feature distribution matching |

### 7. Zero-Inflated Feature Handling
| Status | Already Implemented |
|--------|---------------------|
| **Problem** | Features with many zeros need special handling |
| **Solution** | Separate "is_zero" prediction head with straight-through estimator |
| **Outcome** | Better handling of capital-gain/capital-loss features |

### 8. Minibatch Discrimination
| Status | Already Implemented |
|--------|---------------------|
| **Source** | [Improved GAN Training](https://arxiv.org/abs/1606.03498) (Salimans et al.) |
| **Problem** | Mode collapse - generator produces limited variety |
| **Solution** | Compute pairwise similarities in batch to encourage diversity |
| **Outcome** | Improved sample diversity |

### 9. Spectral Normalization
| Status | Already Implemented |
|--------|---------------------|
| **Source** | [Spectral Normalization Paper](https://arxiv.org/abs/1802.05957) |
| **Problem** | Discriminator can become unstable |
| **Solution** | Normalize weights by spectral norm to stabilize training |
| **Outcome** | More stable discriminator |

### 10. Classifier-Based Label Assignment (for GAN)
| Status | Already Implemented |
|--------|---------------------|
| **Problem** | Random label assignment destroys feature-label correlations |
| **Solution** | Train RF classifier on real data, use it to predict labels for synthetic samples |
| **Outcome** | Improved efficacy metric |

---

## Not Yet Implemented (Future Options)

### A. Correlation Loss
| Status | Not Implemented |
|--------|-----------------|
| **Source** | [CTAB-GAN+](https://pmc.ncbi.nlm.nih.gov/articles/PMC10801038/) |
| **Idea** | Add auxiliary loss penalizing difference between real and fake correlation matrices |
| **Implementation** |
```python
def correlation_loss(real, fake):
    r_corr = torch.corrcoef(real[:, :6].T)  # Continuous features only
    f_corr = torch.corrcoef(fake[:, :6].T)
    return F.mse_loss(f_corr, r_corr)

# Add to generator loss
g_loss = -d_fake.mean() + 0.1 * correlation_loss(real_data, fake_data)
```
| **Expected Impact** | Medium-High - preserves feature relationships |

### B. Generator Output Noise
| Status | Not Implemented |
|--------|-----------------|
| **Idea** | Add small noise to continuous outputs during training |
| **Implementation** |
```python
# In Generator.forward(), after continuous_out = torch.sigmoid(...)
if self.training:
    continuous_out = continuous_out + torch.randn_like(continuous_out) * 0.02
```
| **Expected Impact** | Low-Medium |

### C. Mode-Specific Normalization (VGM)
| Status | Not Implemented |
|--------|-----------------|
| **Source** | [CTGAN Paper](https://arxiv.org/pdf/1907.00503) |
| **Idea** | Use variational Gaussian mixture to model continuous distributions |
| **Expected Impact** | High - but significant implementation effort |

---

## Results Log

### Baseline (Before Changes)
| Metric | GAN (seed 42) | cGAN (seed 42) |
|--------|---------------|----------------|
| Detection AUC | 0.9985 | 0.9979 |
| Efficacy Ratio | 0.9855 | 0.9777 |

### After Instance Noise + Hyperparameter Changes
| Metric | GAN (seed 42) | cGAN (seed 42) |
|--------|---------------|----------------|
| Detection AUC | *Pending* | *Pending* |
| Efficacy Ratio | *Pending* | *Pending* |

**Target**: Detection AUC < 0.85 while maintaining Efficacy > 0.90

---

## References

1. **CTGAN**: Xu, L. et al. "Modeling Tabular data using Conditional GAN" NeurIPS 2019 - [arXiv](https://arxiv.org/pdf/1907.00503)
2. **CTAB-GAN+**: Zhao, Z. et al. "CTAB-GAN+: Enhancing Tabular Data Synthesis" - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10801038/)
3. **Gumbel-Softmax**: Jang, E. et al. "Categorical Reparameterization with Gumbel-Softmax" ICLR 2017
4. **WGAN-GP**: Gulrajani, I. et al. "Improved Training of Wasserstein GANs" NeurIPS 2017
5. **WCGAN-GP for Tabular**: [CEUR-WS](https://ceur-ws.org/Vol-2771/AICS2020_paper_57.pdf)
6. **Mode Collapse Solutions**: [Spot Intelligence](https://spotintelligence.com/2023/10/11/mode-collapse-in-gans-explained-how-to-detect-it-practical-solutions/)

---

## Next Steps

1. Run notebook with seed 42 for both GAN and cGAN
2. Check if Detection AUC improved (target < 0.85)
3. If still high, implement correlation loss
4. If improved, run remaining seeds (123, 456) for final results
5. Update this log with results
