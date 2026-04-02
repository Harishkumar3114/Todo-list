

## Phase 1: 

After manual verification of samples from all 8 languages and the problems figured out, I took 8 metrics to evaluate our data and these metrics capture the majority of bad samples from the data-

* **SNR (Signal-to-Noise Ratio)**: Measures constant background hums (like AC or traffic) to prevent the model from hallucinating words from noise.
* **VAD Ratio (Voice Activity)**: Calculates the percentage of actual human speech, ensuring the model aligns text to voice rather than dead air.
* **ELR (Early-to-Late Ratio)**: Detects room echo and reverberation, preventing the model from confusing smeared word boundaries.
* **Clipping Rate**: Identifies digital distortion from audio being recorded too loud, catching "blown-out" files where speech data is permanently destroyed.
* **Waveform Kurtosis**: Detects sudden, extreme audio spikes, preventing the model from mistaking mic pops or table bumps for hard consonants.
* **Spectral Flatness**: Differentiates structured human speech from pure white noise, filtering out files that are mostly drone sounds or static.
* **ZCR (Zero-Crossing Rate)**: Measures sound wave vibration frequency to catch excessive hissing or high-pitched electronic whines from faulty equipment.
* **Silence Ratio**: Calculates the percentage of absolute dead air to flag recordings with excessively long pauses.

This [threshold_decision.py](./threshold_decision.py) takes the samples of 4000 audio (~10% of total audio samples from 8 languages) from each language and resamples them to 16KHz for mathematical comparison.  It chops the audio into 30ms frame and calculates 8 distinct acoustic 
metrics.  Use Parallel processing to handle thousands of files rapidly.

It uses 6 Analytical plots - Boxplots, Histogram, Retention Curves, Heatmaps, IQR spread and Radar chart are applied on each of the 8 metrics and for each language are plotted and plots are saved.  It are made to help me decide between Global vs Per-language quality thresholds.


### Key Inferences from Acoustic Visualizations

#### A. Global vs. Per-Language Behavior
* **Visual Evidence:** [Boxplots per Language](./plots/01_boxplots_per_language.png) & [Histograms](./plots/02_histograms_all_languages.png)
* **Insight:** Metrics like `Waveform Kurtosis`, `Spectral Flatness`, and `ELR` are structurally stable across all languages (CV < 0.25). However, **VAD Ratio** (Voice Activity) and **SNR** (Signal-to-Noise) vary wildly. For instance, applying a global VAD threshold of >0.6 would preserve almost all Tamil data but delete over 40% of Hindi data.
* **Decision:** We must use **Per-Language** thresholds and normalization for VAD Ratio and SNR, while relying on **Global** thresholds for the remaining metrics.

#### B. Redundant Metrics & Hard Failures
* **Visual Evidence:** [Metric Correlation Heatmap](./plots/04_correlation_heatmap.png)
* **Insight:** `VAD Ratio` and `Silence Ratio` are almost perfectly inversely correlated ($r = -0.91). Using both over-penalizes the exact same audio flaw. Additionally, `Clipping Rate` has roughly zero correlation with acoustic metrics because it is a hardware-level digital error.
* **Decision:** Completely drop `Silence Ratio` from the evaluation score. Treat `Clipping Rate` as a binary "Hard Reject" rather than a weighted metric.

#### C. Language-Specific Quirks
* **Visual Evidence:** [Language Quality Fingerprints (Radar)](./plots/06_radar_language_profiles.png) & [Retention Curves](./plots/03_retention_curves.png)
* **Insight:** Bengali exhibits a uniquely high Zero-Crossing Rate (ZCR) compared to the median dataset shape. 
* **Decision:** ZCR must be moved to the **Per-Language** normalization group so Bengali audio is only evaluated against its own linguistic baseline.

---


### The Final Curation Pipeline (Soft Score)

Based on the inferences above, we process and filter the audio in three distinct steps:

#### Step 1: The Hard Filters (Immediate Rejection)
Before calculating any complex scores, we drop audio:
**Drop if** `duration > 0.3`
* **Drop if** `clipping_rate > 0.05` (Heavy digital distortion / blown-out audio).
* **Drop if** `vad_ratio < lang_median_vad × 0.4` (Essentially dead air; negligible human speech).

#### Step 2: Hybrid Normalization (0.0 to 1.0)
To combine different units (dB, percentages, raw floats) into a single score, we normalize all metrics to a `0.0` (Worst) to `1.0` (Best) scale. 
* **Per-Language Normalization:** `vad_ratio`, `snr_db`, and `zcr`. Required because these metrics have high natural variance across languages (CV > 0.25). A global scale would unfairly penalize and disproportionately delete data from outlier languages like Hindi and Bengali.

* **Global Normalization:** `c50_db` (ELR), `spectral_flatness`, and `kurtosis`.  These metrics showed structural stability across all languages (CV < 0.25), meaning a reverberant recording sounds equally degraded regardless of the language being spoken. Normalizing these globally is both correct and efficient.

#### Step 3: The Weighted Soft Score
We calculate a final Audio Quality Soft Score using the following weights, prioritized by their impact on Speech Model training:

| Metric | Weight | Importance | Normalization Type |
| :--- | :--- | :--- | :--- |
| **SNR (WADA)** | **40%** | Background noise is the #1 cause of model hallucinations. | **Per-Language** |
| **VAD Ratio** | **30%** | Ensures the model trains on actual phonemes, not silence. | **Per-Language** |
| **ELR (c50_db)**| **10%** | Rejects heavily reverberant (echoey) room recordings. | Global |
| **Spectral Flatness** | **10%** | Identifies pure white-noise files. | Global |
| **ZCR** | **5%** | Minor penalty for excessive hissing/static. | **Per-Language** |
| **Kurtosis** | **5%** | Minor penalty for sudden mic pops/clicks. | Global |

**Execution:** After computing the soft scores, we apply a specific, data-driven quality threshold independently for each language. Because highly variable metrics (VAD, SNR, and ZCR) are normalized per language, the soft score accurately reflects a file's quality relative to its own linguistic baseline. To determine the exact cutoff for each language, we analyze its retention curve to identify the natural inflection point—the precise score where a genuine drop in acoustic quality occurs. Any file scoring below this threshold is discarded.
