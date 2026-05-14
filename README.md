# Sentinel2 Multi-Temporal Crop Classification

**Zindi Competition - Côte d'Ivoire Byte-Sized Agriculture Challenge**

**Award:** 2nd Place - Top Female

A pixel-level classification model that distinguishes between cocoa, rubber, and oil palm plantations using multi-temporal Sentinel-2 satellite imagery over Côte d'Ivoire.

## Approach

### Pipeline

1. **Data Extraction** — Read 12-band Sentinel-2 GeoTIFFs and extract per-band statistics (mean, std, min, max, median) using parallel processing with `ThreadPoolExecutor`.

2. **Feature Engineering** — Computed a rich set of spectral indices and derived features:

   - **Vegetation Indices:** NDVI, EVI, SAVI, NDWI, NDBI, NBR, NDRE, GNDVI, ARVI, MSI, NDMI, MTCI, MCARI, PSRI, SIPI, WDRVI, MTVI2, CCCI, TCARI
   - **Band Ratios:** B8/B4, B11/B8, B2/B3, B4/B3, B5/B4, B6/B5, B7/B6, B8A/B8, B11/B12, plus multi-band combinations
   - **Texture/Variance:** Coefficients of variation per spectral region, dynamic range, median-to-mean ratios
   - **Seasonal Encoding:** Sin/cos month encoding, growth phase approximation for West African crop calendars
   - **Spatial Statistics:** Aggregated statistics by spectral region (VIS, NIR, SWIR, Red-Edge)

3. **Modelling** — LightGBM with Optuna hyperparameter optimisation:

   - 5-fold GroupKFold cross-validation (grouped by base location ID to prevent leakage)
   - Early stopping with 50 rounds patience
   - Ensemble of all 5 fold models weighted by inverse log-loss
   - Probability-weighted averaging across temporal samples per location

4. **Post-Processing** — Manual adjustments informed by leaderboard feedback for edge cases.

### Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Cocoa | 0.95 | 0.97 | 0.96 |
| Palm | 0.90 | 0.79 | 0.84 |
| Rubber | 0.84 | 0.93 | 0.88 |

- **Cross-Validation Log-Loss:** 0.3557 ± 0.0308
- **Cross-Validation Macro F1:** 0.8663 ± 0.0154
- **Best Fold F1 (macro):** 0.8928


## Data

The dataset contains multi-temporal Sentinel-2 imagery (12 spectral bands) collected across multiple months in 2024 for three crop types: cocoa, oil palm, and rubber. Each sample is a GeoTIFF patch with corresponding metadata (year, month, location ID).

## Dependencies

Python 3.11, pandas, numpy, rasterio, lightgbm, scikit-learn, optuna, matplotlib, seaborn.
