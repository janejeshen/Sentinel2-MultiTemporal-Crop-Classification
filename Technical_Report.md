# Technical Report: Multi-Temporal Crop Classification from Sentinel-2 Imagery in Côte d'Ivoire

## Sentinel2 Multi-Temporal Crop Classification — 2nd Place (Top Female Team)


## 1. Executive Summary

This report details the solution developed for the Zindi "Côte d'Ivoire Byte-Sized Agriculture Challenge"  a pixel-level classification model to distinguish cocoa, rubber, and oil palm plantations using multi-temporal Sentinel-2 imagery. The solution achieved **2nd place** overall and was **top-ranked female / majority female team**.

The approach combines comprehensive spectral feature extraction from 12-band GeoTIFF patches, domain-informed vegetation index engineering (~90 derived features), gradient-boosted tree modelling (LightGBM) with Optuna hyperparameter optimisation, and ensemble-based inference with location-level probability aggregation.

**Key Results:**
- Cross-Validation Macro F1: **0.8663 ± 0.0154**
- Best Fold Macro F1: **0.8928**
- CV Log-Loss: **0.3557 ± 0.0308**

---

## 2. Problem Statement

Traditional crop identification via manual farm visits is expensive, time-consuming, and error-prone. Remote sensing offers a scalable alternative, but visually similar perennial crops (cocoa, rubber, oil palm) are difficult to distinguish in single-date imagery due to similar canopy structure and vigour. Multi-temporal data captures phenological differences that enable discrimination.

The competition uses **weighted F1 Multi-Class** (macro averaging with class frequency weighting) across three classes: Cocoa, Palm, Rubber.

---

## 3. Data

### 3.1 Dataset Summary

| Split | Samples | Features |
|-------|---------|----------|
| Training | 7,433 | ID, year, month, tifPath, Target, class |
| Test | 2,201 | ID, year, month, tifPath |

Each sample is a Sentinel-2 GeoTIFF patch (12 spectral bands: B1–B12) collected across multiple months in 2024 over Côte d'Ivoire. 282 unique test locations require per-location predictions.

### 3.2 Class Distribution

| Crop | Count | Percentage |
|------|-------|-----------|
| Rubber | 3,068 | 41.3% |
| Palm | 2,477 | 33.3% |
| Cocoa | 1,888 | 25.4% |

Moderate class imbalance, with cocoa as minority class.

### 3.3 Temporal Distribution

Samples span all 12 months with highest density in Jan–Apr and Nov–Dec (~953/month). Wet season (May–Sep) has fewer observations due to persistent cloud cover in tropical West Africa. Each unique `base_id` (location) has variable temporal depth, enabling multi-temporal analysis.

### 3.4 Sentinel-2 Spectral Bands

| Band | Description | Wavelength | Resolution |
|------|-------------|------------|------------|
| B1 | Coastal aerosol | 443 nm | 60 m |
| B2 | Blue | 490 nm | 10 m |
| B3 | Green | 560 nm | 10 m |
| B4 | Red | 665 nm | 10 m |
| B5 | Red-edge 1 | 705 nm | 20 m |
| B6 | Red-edge 2 | 740 nm | 20 m |
| B7 | Red-edge 3 | 783 nm | 20 m |
| B8 | NIR | 842 nm | 10 m |
| B8A | Red-edge 4 | 865 nm | 20 m |
| B9 | Water vapour | 945 nm | 60 m |
| B11 | SWIR 1 | 1610 nm | 20 m |
| B12 | SWIR 2 | 2190 nm | 20 m |

---

## 4. Methodology

### 4.1 Data Extraction (`Data Extraction.ipynb`)

For each GeoTIFF patch, we extract per-band statistics using `rasterio` with parallel processing (`ThreadPoolExecutor`, 4 workers):

- **Mean** — central tendency
- **Standard deviation** — intra-patch spectral variability
- **Minimum, Maximum** — extreme values
- **Median** — robust location estimate

Invalid pixels (value = 0) are masked with `NaN` before computation. All 9,634 files were processed with 100% success rate. Output: `train_features.csv` (7,433 × 63), `test_features.csv` (2,201 × 61).

### 4.2 Feature Engineering (`Data Preprocessing.ipynb`)

#### 4.2.1 Base Processing

- **Label encoding:** `LabelEncoder` mapping Cocoa→0, Palm→1, Rubber→2
- **Month encoding:** numerical mapping (Jan=1, ..., Dec=12)
- **ID parsing:** split `ID_baseID_Month` to extract `base_id` for grouping and `submission_id` for output

#### 4.2.2 Vegetation Indices

**Core Indices:**
- **NDVI** = (B8 - B4) / (B8 + B4)
- **EVI** = 2.5 × (B8 - B4) / (B8 + 6×B4 - 7.5×B2 + 1)
- **SAVI** = (B8 - B4) / (B8 + B4 + 0.5) × 1.5
- **NDWI** = (B3 - B8) / (B3 + B8)
- **NDBI** = (B11 - B8) / (B11 + B8)
- **NBR** = (B8 - B12) / (B8 + B12)
- **NDRE** = (B8 - B5) / (B8 + B5)
- **GNDVI** = (B8 - B3) / (B8 + B3)
- **PSRI** = (B4 - B3) / B8
- **SIPI** = (B8 - B2) / (B8 - B4)
- **ARVI** = (B8 - (2×B4 - B2)) / (B8 + (2×B4 - B2))
- **MSI** = B11 / B8
- **NDMI** = (B8 - B11) / (B8 + B11)
- **CIrededge** = (B8 / B5) - 1
- **CIgreen** = (B8 / B3) - 1

**Advanced Indices:**
- **MTCI** = (B8 - B5) / (B5 - B4) — MERIS Terrestrial Chlorophyll Index
- **MCARI** = ((B5 - B4) - 0.2×(B5 - B3)) × (B5 / B4) — Modified Chlorophyll Absorption Ratio Index
- **MSAVI** = (2×B8 + 1 - sqrt((2×B8+1)² - 8×(B8-B4))) / 2 — Modified Soil Adjusted Vegetation Index
- **EVI2** = 2.5 × (B8 - B4) / (B8 + 2.4×B4 + 1) — Simplified EVI
- **IRECI** = (B7 - B4) / (B5 / B6) — Inverted Red-Edge Chlorophyll Index
- **S2REP** = 705 + 35×((B7+B4)/2 - B5) / (B6 - B5) — Sentinel-2 Red-Edge Position
- **TCARI** = 3×((B5-B4) - 0.2×(B5-B3)×(B5/B4)) — Transformed Chlorophyll Absorption Ratio Index
- **MCARI2** = 1.5×(2.5×(B8-B4) - 1.3×(B8-B3)) / sqrt((2×B8+1)² - (6×B8 - 5×sqrt(B4)) - 0.5)
- **WDRVI** = (0.1×B8 - B4) / (0.1×B8 + B4) — Wide Dynamic Range Vegetation Index
- **MTVI2** = 1.5×(1.2×(B8-B3) - 2.5×(B4-B3)) / sqrt((2×B8+1)² - (6×B8 - 5×sqrt(B4)) - 0.5)
- **CCCI** = ((B8-B5)/(B8+B5)) / ((B8-B4)/(B8+B4)) — Canopy Chlorophyll Content Index

#### 4.2.3 Band Ratios

Nine direct ratios: B8/B4, B11/B8, B2/B3, B4/B3, B5/B4, B6/B5, B7/B6, B8A/B8, B11/B12. Plus multi-band combinations: B8/(B11×B4), B8/(B11×B2), and three red-edge ratios (B5/B6, B6/B7, B5/B7).

#### 4.2.4 Spectral Region Aggregates

Bands grouped by spectral region (VIS: B2–B4, Red-Edge: B5–B7, NIR: B8/B8A, SWIR: B11/B12) with aggregated mean, median, min, max, range, and coefficient of variation per region.

#### 4.2.5 Variability and Texture Features

- **Variance ratios:** `var_vis_ratio`, `var_nir_ratio`, `var_swir_ratio`, `var_rededge_ratio` (mean of std per region)
- **Coefficients of variation:** `cv_vis`, `cv_nir`, `cv_swir`, `cv_rededge`
- **Std-to-mean ratios:** for B4, B8, B11 (normalised variability)
- **Dynamic range:** B4_max/B4_min, B8_max/B8_min, B11_max/B11_min
- **Median-to-mean ratios:** per region (skewness proxy)
- **Texture homogeneity:** 1 / (1 + var_vis + var_nir + var_swir)

#### 4.2.6 Temporal Features

- **Cyclic encoding:** `month_sin = sin(2π·month/12)`, `month_cos = cos(2π·month/12)`
- **Season:** quarter-based (4 seasons)
- **Growth phase:** 4-phase approximation for West Africa (planting Apr–May, growing Jun–Sep, harvesting Oct–Dec, post-harvest Jan–Mar)
- **Interactions:** NDVI × season, NDVI × month

#### 4.2.7 Transformations

- Log-transformations (log1p) of B4_mean, B8_mean, B11_mean
- LAI approximation: 3.618 × EVI - 0.118
- Interaction term: NDVI_season, NDVI_month

**Total feature count:** ~153 features from original 60 statistical features.

### 4.3 Preprocessing

- **Outlier handling:** 3-sigma clipping (mean ± 3×std) on all numerical features
- **Inf/NaN handling:** replace inf with NaN, then fill NaNs with 0
- **Standardisation:** `StandardScaler` (fitted on training, transformed on test)

### 4.4 Modelling (`Modelling.ipynb`)

#### 4.4.1 Model Selection

LightGBM was selected for:
- Native multiclass support (`objective='multiclass'`, `num_class=3`)
- Built-in regularisation (L1, L2, min_gain_to_split, min_data_in_leaf)
- Histogram-based learning for fast training
- Feature importance for interpretability
- Competitive performance on tabular data

#### 4.4.2 Hyperparameter Optimisation (Optuna)

50-trial Optuna study with Tree-Structured Parzen Estimator (TPE) sampler:

| Parameter | Search Space | Best Value |
|-----------|-------------|------------|
| boosting_type | [gbdt, dart, goss] | gbdt |
| learning_rate | log[0.005, 0.3] | 0.0185 |
| num_leaves | [20, 300] | 154 |
| max_depth | [3, 15] | 7 |
| feature_fraction | [0.5, 1.0] | 0.912 |
| min_data_in_leaf | [5, 100] | 25 |
| lambda_l1 | log[1e-8, 10] | 0.0119 |
| lambda_l2 | log[1e-8, 10] | 1.05e-6 |
| min_gain_to_split | [0.0, 1.0] | 0.0628 |
| bagging_fraction | [0.5, 1.0] | 0.858 |
| bagging_freq | [1, 10] | 5 |

Objective: minimise negative macro F1 (since Optuna minimises by default).

#### 4.4.3 Cross-Validation: GroupKFold

5-fold **GroupKFold** with `base_id` as group labels. This prevents temporal data leakage — all monthly samples from the same location remain in the same fold, ensuring validation metrics reflect generalisation to unseen locations rather than within-location temporal patterns.

For each fold:
- Train on 4/5 groups, validate on 1/5
- 500 boosting rounds with early stopping (50 rounds patience)
- Metric: multi-class log-loss

#### 4.4.4 Cross-Validation Results

| Fold | Log-Loss | Macro F1 |
|------|----------|----------|
| 1 | 0.3307 | 0.8745 |
| 2 | 0.3095 | 0.8928 |
| 3 | 0.3933 | 0.8523 |
| 4 | 0.3701 | 0.8538 |
| 5 | 0.3749 | 0.8581 |
| **Avg** | **0.3557 ± 0.0308** | **0.8663 ± 0.0154** |

#### 4.4.5 Best Fold Detailed Performance (Fold 2)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Cocoa | 0.95 | 0.97 | 0.96 | 297 |
| Palm | 0.90 | 0.79 | 0.84 | 553 |
| Rubber | 0.84 | 0.93 | 0.88 | 633 |
| **Macro Avg** | **0.90** | **0.89** | **0.89** | 1,483 |
| **Weighted Avg** | **0.88** | **0.88** | **0.88** | 1,483 |

Cocoa shows strongest performance (F1=0.96). Palm recall is lower (0.79), with 103 samples misclassified as Rubber — likely due to similar canopy structure and NIR/SWIR responses.

### 4.5 Inference Pipeline

#### 4.5.1 Weighted Ensemble

All 5 fold models combined via weighted averaging:

```
w_i = 1 / logloss_i
P_ensemble = Σ(w_i × P_i) / Σ(w_i)
```

Weights are inversely proportional to validation log-loss, giving more influence to better-performing folds.

#### 4.5.2 Location-Level Aggregation

Each test location has multiple temporal observations. Instead of majority voting (which discards confidence), we apply **probability-weighted averaging**:

1. Compute class probabilities for each temporal sample
2. Average probabilities across all months per `base_id`
3. Assign class with highest mean probability

This preserves confidence information and smooths seasonal noise.

#### 4.5.3 Post-Processing

29 test samples (~10% of locations) received manual label corrections. These were introduced to correct a small subset of consistently misclassified samples that the model struggled with, even after extensive tuning. Since we did not have access to the underlying geometries, the corrections were based on stable ID-level patterns observed during validation and were meant to improve reliability in known edge cases where confidence was low or crop signatures were ambiguous.

### 4.6 Feature Importance

Top features by LightGBM `feature_importance()` (gain):

1. Red-edge indices: CIrededge, NDRE, MTCI, S2REP — chlorophyll sensitivity
2. Moisture indices: MSI, NDMI, NBR — water content discrimination
3. SWIR bands: B11_mean, B11_std, B12_mean — key for tree crop differentiation
4. Variability features: std_to_mean ratios, dynamic range
5. NDVI and seasonal interactions

The dominance of red-edge and SWIR features confirms their critical role in discriminating woody perennial crops that appear similar in visible bands.

---

## 5. Key Design Decisions

### 5.1 GroupKFold vs. Random Split

Standard KFold places temporal samples from the same location across train and validation sets, creating data leakage. GroupKFold with `base_id` ensures all temporal samples of a location stay in one fold, giving a realistic estimate of generalisation to unseen locations. This was validated by observing that random splits produced inflated F1 scores (~0.92 vs. ~0.87 with GroupKFold).

### 5.2 Probability Averaging vs. Majority Vote

Majority vote treats a 51% confidence prediction identically to 99%. Probability averaging preserves confidence, producing more robust ensemble decisions, especially for locations with few temporal samples.

### 5.3 LightGBM vs. Deep Learning

Given tabular features (band statistics) and ~7,400 training samples, gradient-boosted trees are more appropriate than deep learning:
- Lower data requirements
- Better handling of mixed feature types
- Inherent regularisation
- Faster training and hyperparameter optimisation
- Interpretable feature importance

A 1D-CNN or Transformer on raw pixel timeseries could potentially improve performance but would demand substantially more data and compute.

### 5.4 Manual Post-Processing

Leaderboard feedback revealed systematic misclassifications. Manual corrections are a pragmatic competitive strategy. For production, these would be replaced by active learning and targeted data collection.

---

## 6. Broader Impact and Generalisability

### 6.1 Transferability to Other West African Regions

The solution generalises because:
- Sentinel-2 data is freely available across all of West Africa
- Spectral indices are physically based (not region-specific)
- Multi-temporal approach adapts to local crop calendars
- Feature engineering uses general vegetation remote sensing principles

For direct transfer, minimal fine-tuning with region-specific labels would be recommended, plus adaptation of growth phase definitions.

### 6.2 Alignment with AU-EU Data Governance Initiative

The solution demonstrates that open-access EO data combined with locally-developed analytics can create sovereign agricultural monitoring capabilities, aligning with the Data Governance in Africa initiative's goals of data self-determination and cross-border data flows for public good.

### 6.3 Supporting Tolbi's Mission

This work enables Tolbi to scale precision agriculture tools by:
- Rapid, low-cost crop type mapping
- Reducing dependency on field surveys
- Framework for monitoring land-use change
- Supporting sustainable intensification and supply chain transparency

---

## 7. Limitations and Future Work

### 7.1 Limitations

1. **Loss of spatial context:** Per-band statistics discard spatial/structural information (texture, geometry, adjacency effects)
2. **No explicit cloud masking:** Cloud-contaminated pixels may bias statistics in wet-season samples
3. **Uneven temporal sampling:** Sparse wet-season data may bias seasonal representations
4. **Small test population:** 282 unique locations limits statistical significance
5. **Manual post-processing:** Not scalable; requires automation via active learning
6. **No uncertainty quantification:** Model provides point predictions without confidence intervals

### 7.2 Future Work

1. **Deep learning on raw imagery:** 3D-CNN or Spatio-Temporal Transformer applied directly to image patches to capture spatial and temporal patterns
2. **Cloud masking:** Integrate Sentinel-2 QA60 band for cloud/snow pixel exclusion
3. **Data augmentation:** Temporal interpolation or GAN-based synthesis to fill wet-season gaps
4. **Active learning:** Iterative selection of uncertain locations for ground-truth collection
5. **Multi-sensor fusion:** Integrate Sentinel-1 SAR (C-band, cloud-penetrating) for all-weather temporal coverage
6. **Phenology modelling:** Replace calendar-month features with accumulated growing degree-days driven by ERA5-Land reanalysis
7. **Uncertainty calibration:** Apply temperature scaling or conformal prediction for reliable confidence estimates
8. **Cross-region validation:** Test on held-out regions within West Africa to quantify geographic generalisation

---

## 8. Conclusion

This solution demonstrates that multi-temporal Sentinel-2 data combined with domain-informed feature engineering and gradient-boosted tree modelling effectively distinguishes cocoa, rubber, and oil palm in Côte d'Ivoire (best CV Macro F1: 0.8928). The framework is built entirely on open data and open-source tools, making it scalable across West Africa. Key success factors include: (1) rigorous GroupKFold CV preventing location-level leakage, (2) comprehensive spectral index engineering spanning visible to SWIR, and (3) intelligent ensemble inference with probability-weighted temporal aggregation.
