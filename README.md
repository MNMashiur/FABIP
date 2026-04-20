# FABIP: Fairness-Aware Bias Impact Pruning

**A General Framework for Sustainable and Fair Neural Network Prediction**

> *"Can a model be made fairer and smaller at the same time — without sacrificing accuracy?"*  
> FABIP answers yes, through principled structured pruning guided by a mathematically justified Bias Impact Score.

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Key Contributions](#key-contributions)
4. [Mathematical Framework](#mathematical-framework)
5. [Architecture](#architecture)
6. [Datasets](#datasets)
7. [Baselines](#baselines)
8. [Results](#results)
9. [Visualisations](#visualisations)
10. [Project Structure](#project-structure)
11. [How to Run](#how-to-run)
12. [Ablation Study](#ablation-study)
13. [Sensitivity Analysis](#sensitivity-analysis)
14. [Multi-Dataset Generalisation](#multi-dataset-generalisation)
15. [What "Sustainable" Means](#what-sustainable-means)
16. [Addressing Faculty Feedback](#addressing-faculty-feedback)
17. [Authors](#authors)
18. [References](#references)

---

## Overview

**FABIP** (Fairness-Aware Bias Impact Pruning) is a structured neural network pruning framework that simultaneously reduces demographic bias and computational cost in classification models used for high-stakes decision-making.

Unlike prior fairness-aware pruning methods that zero out individual weights (unstructured), FABIP **physically removes entire neurons** based on a novel per-neuron scoring system — the **Bias Impact Score (BIS)** — resulting in models that are provably smaller, faster, and fairer.

The framework is evaluated on three real-world datasets spanning criminal justice, financial hiring, and credit risk — demonstrating that fairness improvements are not domain-specific but a general property of the approach.

---

## Motivation

Machine learning models are increasingly deployed in critical decision-making contexts: predicting recidivism, assessing loan eligibility, and screening job applicants. These models routinely exhibit demographic bias — producing systematically worse outcomes for minority or historically disadvantaged groups.

Two common responses exist in the literature:

1. **Fairness constraints** during training — but these often hurt accuracy and add complexity.
2. **Post-hoc weight pruning** — but standard pruning methods (e.g., magnitude pruning) ignore bias entirely and sometimes *amplify* it.

**The gap FABIP fills:** There is no unified framework that (a) identifies the specific neurons *causing* bias, (b) removes them structurally rather than symbolically, and (c) does so across general domains rather than only criminal justice or only medical imaging.

---

## Key Contributions

| # | Contribution |
|---|---|
| **C1** | A novel **Bias Impact Score (BIS)** computed per neuron via gradient-based attribution *and* ablation masking — the first dual-verification BIS in the pruning-fairness literature |
| **C2** | A **Fairness Pruning Score (FPS)** that jointly weighs bias contribution and predictive importance, ensuring bias-carrying neurons are removed without destroying accuracy |
| **C3** | True **structured pruning**: entire neurons are removed and the network is physically rebuilt, yielding genuine reductions in FLOPs, parameter count, and inference latency |
| **C4** | A **generalised framework** validated across three domains (criminal justice, financial hiring, credit risk) and three sensitive attributes (race, sex, age) |
| **C5** | An operationally precise definition of **"sustainable"** ML: a model is sustainable when it achieves fairness *without* a computational penalty |

---

## Mathematical Framework

### Step 1 — Gradient-Based BIS

For each hidden neuron $j$, the gradient-based Bias Impact Score measures how differently the neuron responds to the two demographic groups:

$$\text{BIS}_{\text{grad}}(j) = \left| \frac{\partial \mathcal{L}_{\text{priv}}}{\partial h_j} - \frac{\partial \mathcal{L}_{\text{unpriv}}}{\partial h_j} \right|$$

where $\mathcal{L}_{\text{priv}}$ and $\mathcal{L}_{\text{unpriv}}$ are the losses computed separately on the privileged and unprivileged subgroups. A large difference means neuron $j$ encodes group-specific information — it is a **bias carrier**.

### Step 2 — Ablation-Based BIS (Causal Verification)

To causally verify the gradient estimate, each candidate neuron $j$ is ablated (zeroed) and the change in the group accuracy gap is measured:

$$\text{BIS}_{\text{ablate}}(j) = \Delta\text{AccGap}^{\text{baseline}} - \Delta\text{AccGap}^{\text{without } j}$$

If ablating neuron $j$ reduces the gap, then $\text{BIS}_{\text{ablate}}(j) > 0$, confirming it as a bias contributor. This step converts a correlation-based signal into a causal one.

### Step 3 — Ensemble BIS

The two signals are combined with a weighting factor $\alpha$:

$$\text{BIS}(j) = \alpha \cdot \text{BIS}_{\text{grad}}(j) + (1 - \alpha) \cdot \text{BIS}_{\text{ablate}}(j), \quad \alpha = 0.7$$

The gradient component is used as the primary signal (fast, computed over the full dataset); the ablation component serves as targeted causal verification on high-BIS candidates.

### Step 4 — Weight Importance

To prevent accuracy degradation, the predictive importance of each neuron is estimated via the L2 norm of its gradient with respect to the full loss:

$$I(j) = \left\| \frac{\partial \mathcal{L}}{\partial h_j} \right\|_2$$

Neurons with high importance are protected from pruning regardless of their BIS.

### Step 5 — Fairness Pruning Score (FPS)

The FPS combines both signals into a single pruning criterion:

$$\text{FPS}(j) = \lambda_1 \cdot \text{BIS}(j) - \lambda_2 \cdot I(j), \quad \lambda_1 = 0.7,\ \lambda_2 = 0.3$$

Neurons with **high BIS** (strong bias contribution) and **low importance** (safe to remove) receive the highest FPS and are pruned first.

### Step 6 — Structured Pruning

Neurons with FPS above the pruning threshold $\tau$ are removed:

$$\text{Remove neuron } j \iff \text{FPS}(j) > \tau$$

Crucially, this is **structured pruning**: entire rows of the weight matrix are removed and the network is *physically rebuilt* with smaller hidden dimensions. This is why:

$$\text{FLOPs} = 2 \times n_{\text{in}} \times n_{\text{out}} \quad \text{(per layer)}$$

decreases in a way that translates to real hardware speedups — unlike unstructured pruning, which merely zeros weights while keeping the same matrix dimensions.

---

## Architecture

The baseline model is a 3-hidden-layer **MLP with BatchNorm and Dropout**:

```
Input (7 features)
    ↓
Linear(7 → 128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(64 → 32)  + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(32 → 1)   [Binary output via BCEWithLogitsLoss]
```

After FABIP structured pruning at ratio 0.30, the architecture becomes physically smaller (exact dimensions depend on which neurons are identified as bias carriers — typically around `128 → 64 → 32` shrinks to `~90 → ~44 → ~22`).

**Training configuration:**
| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 (baseline), 5e-4 (fine-tune) |
| Weight decay | 1e-4 |
| LR scheduler | ReduceLROnPlateau (patience=4, factor=0.5) |
| Early stopping patience | 10 epochs |
| Batch size | 256 |
| Random seed | 42 |

---

## Datasets

### Primary — COMPAS (Recidivism)

| Property | Detail |
|---|---|
| **Domain** | Criminal Justice |
| **Task** | Binary classification: predict recidivism within 2 years |
| **Sensitive attribute** | Race (African-American = 1, Other = 0) |
| **Features** | Age, prior convictions, juvenile felony/misdemeanour/other counts, charge degree |
| **Target** | `two_year_recid` (0 = did not reoffend, 1 = reoffended) |
| **Source** | ProPublica / [Kaggle — COMPAS Recidivism Racial Bias](https://www.kaggle.com/datasets/danofer/compass) |
| **Total samples** | ~6,172 (after filtering) |
| **Train / Val / Test** | 4,320 / 926 / 926 |
| **Known bias** | Black defendants are assigned higher risk scores than comparable White defendants |

**Dataset filtering applied** (ProPublica methodology):
- `days_b_screening_arrest` between −30 and +30
- `is_recid ≠ −1`
- `c_charge_degree ≠ 'O'`
- `score_text ≠ 'N/A'`

### Secondary — Adult Income (UCI)

| Property | Detail |
|---|---|
| **Domain** | Finance / Hiring |
| **Task** | Predict income > $50K |
| **Sensitive attribute** | Sex (Female = 1, Male = 0) |
| **Source** | UCI ML Repository via `sklearn.datasets.fetch_openml` |
| **Known bias** | Female workers systematically under-predicted for high-income class |

### Secondary — German Credit (UCI)

| Property | Detail |
|---|---|
| **Domain** | Finance / Credit |
| **Task** | Predict credit risk (good = 1, bad = 0) |
| **Sensitive attribute** | Age (Young < 25 = 1, Older = 0) |
| **Source** | UCI ML Repository via `sklearn.datasets.fetch_openml` |
| **Known bias** | Younger applicants assigned worse credit risk regardless of financial history |

---

## Baselines

Three baselines are implemented and compared against FABIP:

### 1. Vanilla (Unpruned)
A standard neural network trained with cross-entropy loss and no fairness intervention. Establishes the accuracy ceiling and the bias floor — the worst-case fairness and best-case efficiency.

### 2. Magnitude Pruning (Standard Unstructured)
The lowest-magnitude weights globally are zeroed using L1-based unstructured pruning (40% of weights). Fine-tuned afterward.

**Key limitation (why it appears in this study):** Zeroed weights are still stored in memory and still participate in matrix multiplication. The network dimensions are unchanged. **FLOPs do not decrease.** This is the most common mistake in the pruning-fairness literature and is explicitly demonstrated in this work.

### 3. FairPrune (Wu et al., 2022)
Neurons whose activations are most correlated with the sensitive attribute $A$ are zeroed out. This is the closest prior work to FABIP, applied originally to dermatological disease diagnosis.

**Key limitation:** Also unstructured — correlation-based zeroing does not change matrix dimensions, so FLOPs remain unchanged. Furthermore, it uses a single signal (correlation) rather than FABIP's dual-verification BIS.

### FABIP (Ours)
Structured pruning guided by the ensemble BIS + FPS. Physically removes neurons and rebuilds the network smaller. The only method in this comparison that achieves genuine reductions in FLOPs, parameters, and inference latency.

---

## Results

All results are on the **COMPAS test set** (926 samples), reported after fine-tuning.

### Main Results Table

| Model | Accuracy | F1 | AUC | DP ↓ | EO ↓ | DI → 1.0 | #Params | FLOPs | Latency (μs) |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla | 0.6771 | 0.6071 | 0.7176 | 0.2718 | 0.4598 | 2.1964 | 11,841 | 22,336 | 23.604 |
| Magnitude Pruning | — | — | — | — | — | — | 11,841 | 22,336 | — |
| FairPrune | — | — | — | — | — | — | 11,841 | 22,336 | — |
| **FABIP (Ours)** | **0.6782** | **0.6140** | **0.7209** | **0.2603** | **0.4340** | **2.0623** | **6,213** | **11,476** | **18.842** |

> Magnitude Pruning and FairPrune are included in the notebook with full metrics. Their FLOPs and #Params columns match Vanilla because both are **unstructured** — this is an intentional design point of the study.

### FABIP vs Vanilla — Summary of Improvements

| Metric | Vanilla | FABIP | Change |
|---|---|---|---|
| Demographic Parity ↓ | 0.2718 | 0.2603 | **−4.2% (fairer)** |
| Equalized Odds ↓ | 0.4598 | 0.4340 | **−5.6% (fairer)** |
| Disparate Impact → 1.0 | 2.1964 | 2.0623 | **Closer to 1.0** |
| Accuracy | 0.6771 | 0.6782 | **+0.1% (unchanged)** |
| F1-Score | 0.6071 | 0.6140 | **+0.7%** |
| AUC | 0.7176 | 0.7209 | **+0.5%** |
| FLOPs | 22,336 | 11,476 | **−48.6%** |
| Parameters | 11,841 | 6,213 | **−47.5%** |
| Inference latency | 23.604 μs | 18.842 μs | **−20.2%** |

The key finding: FABIP improves all three fairness metrics, maintains (and slightly improves) accuracy, and cuts computational cost nearly in half — none of which is achievable with unstructured pruning methods.

### Fairness Metric Definitions

| Metric | Formula | Interpretation |
|---|---|---|
| **Demographic Parity (DP)** | `|P(ŷ=1|A=0) − P(ŷ=1|A=1)|` | Difference in positive prediction rates between groups. Lower = fairer. |
| **Equalized Odds (EO)** | `|TPR_priv − TPR_unpriv| + |FPR_priv − FPR_unpriv|` | Combined difference in error rates between groups. Lower = fairer. |
| **Disparate Impact (DI)** | `P(ŷ=1|A=1) / P(ŷ=1|A=0)` | Ratio of positive prediction rates. Values close to 1.0 = fairer. |

---

## Visualisations

The notebook produces **8 publication-quality figures**, all saved as `.png` files:

| Figure | File | Description |
|---|---|---|
| **Fig 1** | `fig1_main_comparison.png` | 6-panel dashboard: Accuracy, DP, EO, DI, FLOPs, Parameters across all 4 methods |
| **Fig 2** | `fig2_scatter.png` | Fairness–Accuracy scatter plot (bubble size = FLOPs); EO vs F1 scatter |
| **Fig 3** | `fig3_bis_heatmap.png` | Per-neuron BIS heatmap (gradient vs ablation) + FPS bar chart with pruned/kept highlighted |
| **Fig 4** | `fig4_training_curves.png` | Validation loss and accuracy curves: Vanilla training vs FABIP fine-tuning |
| **Fig 5** | `fig5_group_accuracy.png` | Per-group accuracy (privileged vs unprivileged) and accuracy gap across all methods |
| **Fig 6** | `fig6_ablation.png` | Ablation study: effect of removing each FABIP component on accuracy and fairness |
| **Fig 7** | `fig7_sensitivity.png` | Sensitivity analysis: accuracy, DP, and FLOPs as pruning ratio varies from 10% to 50% |
| **Fig 8** | `fig8_multidataset.png` | Multi-dataset generalisation: FABIP vs Vanilla across COMPAS, Adult, German Credit |

---

## Project Structure

```
FABIP_Project.ipynb          ← Main notebook (Google Colab, GPU recommended)
README.md                    ← This file
compas-scores-two-years.csv  ← Primary dataset (upload to /content/ in Colab)
fig1_main_comparison.png     ← Generated after running Section 10
fig2_scatter.png             ← Generated after running Section 10
fig3_bis_heatmap.png         ← Generated after running Section 10
fig4_training_curves.png     ← Generated after running Section 10
fig5_group_accuracy.png      ← Generated after running Section 10
fig6_ablation.png            ← Generated after running Section 11
fig7_sensitivity.png         ← Generated after running Section 12
fig8_multidataset.png        ← Generated after running Section 13
```

---

## 🚀 How to Run

### ▶️ Run Directly in Google Colab (Recommended)

👉 **[Open in Colab](https://colab.research.google.com/github/MNMashiur/FABIP-Fairness-Aware-Bias-Impact-Pruning-for-Sustainable-and-Fair-Recidivism-Prediction/blob/main/FABIP.ipynb)**


> Replace `YOUR_USERNAME/YOUR_REPO` with your actual GitHub repository link.

---

### Steps

1. Click the **Open in Colab** link above  
2. *(Optional)* Enable GPU for faster execution:  
   **Runtime → Change runtime type → T4 GPU**  
3. Click:  
   **Runtime → Run all**

---

### ✅ Ready to Go

- No dataset upload required  
- No setup required  
- Everything is pre-configured  

---

### ⏱ Runtime

- Approximately **20–40 minutes** on GPU  
- Includes:
  - Baseline training  
  - FABIP pruning  
  - Fairness evaluation  
  - Ablation study  
  - Sensitivity analysis  

---

### 🧠 Notes

- Outputs (metrics, plots, comparisons) are **already visible in GitHub preview**
- You can **rerun and modify everything directly in Colab**
- Designed for **one-click reproducibility**

---

## 💻 Optional: Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

python -m venv fabip_env
source fabip_env/bin/activate       # Linux / macOS
fabip_env\Scripts\activate          # Windows

pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn fairlearn fvcore

jupyter notebook FABIP_Project.ipynb

> ⚠️ A CUDA-capable GPU is strongly recommended. CPU-only inference is supported but the BIS ablation loop over neuron candidates is significantly slower.

### Switching Datasets

In **Section 2**, change one line:

```python
ACTIVE_DATASET = 'compas'   # Options: 'compas' | 'adult' | 'german'
```

Then re-run all cells from Section 2 onwards. Adult Income and German Credit are downloaded automatically via `sklearn.datasets.fetch_openml` — no manual file upload required.

### Dependencies

| Package | Version tested | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | Model training, pruning, gradient computation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `pandas` | ≥ 2.0 | Data loading and manipulation |
| `scikit-learn` | ≥ 1.3 | Metrics, preprocessing, dataset loading |
| `matplotlib` | ≥ 3.7 | All visualisations |
| `seaborn` | ≥ 0.12 | Heatmap styling |
| `fairlearn` | ≥ 0.9 | Fairness metric reference |
| `fvcore` | ≥ 0.1 | FLOPs counting utility |

---

## Ablation Study

To rigorously justify each component of FABIP, three degraded variants are evaluated:

| Variant | BIS Used | Importance Used | Pruning Type | Purpose |
|---|---|---|---|---|
| No-BIS | ✗ | ✓ | Structured | Proves BIS is necessary — removing unimportant neurons alone does not improve fairness |
| No-Importance | ✓ | ✗ | Structured | Proves importance weighting is necessary — BIS-only pruning can hurt accuracy |
| No-Ablation | Gradient only | ✓ | Structured | Proves ablation verification adds value over gradient BIS alone |
| **FABIP (Full)** | Gradient + Ablation | ✓ | Structured | Best fairness with minimal accuracy cost |

All four variants use the same pruning ratio (0.30) for a fair comparison. Results confirm that every component contributes — removing any single element degrades either fairness or accuracy.

---

## Sensitivity Analysis

FABIP is evaluated across five pruning ratios to characterise the fairness–efficiency–accuracy trade-off:

| Pruning Ratio | Behaviour |
|---|---|
| 0.10 | Minimal fairness improvement; FLOPs barely change |
| 0.20 | Moderate fairness gains; small FLOPs reduction |
| **0.30** | **Best balance — chosen as the default** |
| 0.40 | Further FLOPs reduction; minor additional accuracy cost |
| 0.50 | Aggressive pruning; accuracy begins to degrade noticeably |

The chosen default of **30% pruning** achieves the best balance between fairness improvement, accuracy preservation, and computational reduction across all three datasets.

---

## Multi-Dataset Generalisation

A full pipeline (vanilla training → FABIP) is run on all three datasets independently. This confirms that FABIP is not overfit to the COMPAS dataset or recidivism prediction specifically, but is a general fairness-aware pruning framework.

**Summary of FABIP improvements over Vanilla across datasets:**

| Dataset | Sensitive Attr. | DP Improvement | EO Improvement | FLOPs Reduction |
|---|---|---|---|---|
| COMPAS | Race | −4.2% | −5.6% | ~48% |
| Adult Income | Sex | Consistent reduction | Consistent reduction | ~45–50% |
| German Credit | Age | Consistent reduction | Consistent reduction | ~45–50% |

Exact numbers for Adult and German Credit are printed in **Section 13** of the notebook.

---

## What "Sustainable" Means

The term **"sustainable"** in the project title is defined operationally and precisely throughout the code and this README. It is not used loosely.

A model is sustainable if it satisfies all three of the following:

1. **Fair** — demonstrates reduced demographic disparity (lower DP and EO) compared to an unpruned baseline
2. **Accurate** — maintains predictive performance within an acceptable threshold (< 2% accuracy loss in our experiments)
3. **Computationally efficient** — achieves genuine reductions in FLOPs, parameter count, and inference latency through structured (not unstructured) pruning

Standard unstructured pruning methods (including FairPrune) fail condition 3 because zeroing weights does not change the physical dimensions of weight matrices. Only FABIP's structured approach satisfies all three conditions simultaneously.

---

## Addressing Faculty Feedback

This section explicitly maps the project revisions to the four pieces of feedback received on the original proposal (score: 5.6/10).

### Feedback 1: "BIS formula never explained — how is accuracy attributed to a single weight?"

**Resolution:** The BIS is now computed at the neuron level using two complementary and independently justifiable methods:

- **Gradient attribution** (`compute_bis_gradient`): Computes the absolute difference in per-group gradients at each hidden neuron. This directly measures how differently the neuron responds to the two demographic groups during loss minimisation — a principled information-theoretic interpretation of bias contribution.
- **Ablation masking** (`compute_bis_ablation`): Physically zeros each candidate neuron and measures the change in the group accuracy gap. This is a causal intervention, not a correlation, and directly measures whether the neuron is causally responsible for the observed bias.

Both methods are implemented with full code, mathematical definitions, and explanatory comments in Sections 8 and 9 of the notebook.

### Feedback 2: "FLOPs do not decrease from zeroed weights alone. Only structured pruning achieves this."

**Resolution:** FABIP uses exclusively **structured pruning** (`structured_prune` function in Section 9). The implementation physically removes neuron rows from weight matrices and calls `FABIPNet.__init__` with updated `hidden_dims` to rebuild a genuinely smaller network. The FLOPs formula `2 × n_in × n_out` (per layer) is explicitly computed and reported before and after pruning. The notebook demonstrates that Magnitude Pruning and FairPrune — both unstructured — show *identical FLOPs to the Vanilla model*, while FABIP shows a ~48.6% FLOPs reduction.

### Feedback 3: "No dedicated experimental plan section. No baselines defined."

**Resolution:** Three baselines are fully implemented and compared: Vanilla (no intervention), Magnitude Pruning (standard unstructured), and FairPrune (Wu et al., 2022). Results are presented in a structured table across 10 metrics. An ablation study (Section 11) justifies each component of FABIP. A sensitivity analysis (Section 12) characterises behaviour across pruning ratios. A multi-dataset generalisation (Section 13) validates the approach beyond a single domain.

### Feedback 4: "Too limited in scope — make it general. What do you mean by 'sustainable'?"

**Resolution:** The framework is generalised to three datasets across three different domains and three different sensitive attributes (race, sex, age). "Sustainable" is defined precisely and operationally in the notebook title cell, the final summary cell, and throughout this README: a model is sustainable when it achieves fairness without a computational penalty, demonstrated through genuine FLOPs, parameter, and latency reductions.

---

## Authors

| Name | Student ID | Contribution |
|---|---|---|
| Mashiur Rahaman Mollah Niran | 2022-3-60-183 | 40% |
| Nipu Basher | 2022-3-60-265 | 30% |
| Tasmia Rahman | 2022-3-60-164 | 30% |

**Submitted To:** Ahmed Abdal Shafi Rasel, Senior Lecturer, Department of CSE, East West University

---

## References

1. Kasyap, H., et al. (2024). Mitigating Bias: Model Pruning for Enhanced Model Fairness and Efficiency. *ECAI 2024*, pp. 995–1002.
2. Wu, Y., et al. (2022). FairPrune: Achieving Fairness through Pruning for Dermatological Disease Diagnosis. *MICCAI 2022*, pp. 743–753.
3. Zayed, A., et al. (2024). Fairness-aware Structured Pruning in Transformers. *AAAI 2024*, Vol. 38, No. 20, pp. 22484–22492.
4. Zhang, L., et al. (2023). Towards Fairness-Aware Adversarial Network Pruning. *ICCV 2023*, pp. 5168–5177.
5. Dai, Y., et al. (2023). Integrating Fairness and Model Pruning Through Bi-level Optimization. *arXiv:2312.10181*.
6. Iofinova, E., et al. (2023). Bias in Pruned Vision Models: In-depth Analysis and Countermeasures. *CVPR 2023*, pp. 24364–24373.
7. Qin, Q., & Merlo, E. (2026). Prune Bias from the Root. *Information and Software Technology*, 189, 107906.
8. Kondapalli, P., et al. (2025). A Literature Review: Bias Detection and Mitigation in Criminal Justice. *Engineering Proceedings*, 107(1), 72.
9. Pagano, T. P., et al. (2023). Bias and Unfairness in Machine Learning Models. *BDCC*, 7(1), 15.
10. Chen, Z., et al. (2023). A Comprehensive Empirical Study of Bias Mitigation Methods. *ACM TOSEM*, 32(4), 1–30.
11. Xue, Y., et al. (2024). BMFT: Achieving Fairness via Bias-based Weight Masking Fine-tuning. *MICCAI Workshop on Fairness of AI in Medical Imaging*, pp. 98–108.
12. Jain, B., et al. (2019). Singular Race Models: Addressing Bias and Accuracy in Predicting Prisoner Recidivism. *PETRA 2019*, pp. 599–607.
13. Shah, M., & Sureja, N. (2025). A Comprehensive Review of Bias in Deep Learning Models. *Archives of Computational Methods in Engineering*, 32(1), 255–267.

---

*Department of Computer Science and Engineering, East West University — 2026*
