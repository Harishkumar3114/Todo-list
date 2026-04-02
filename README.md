
After manual verification of samples from all 8 languages and the problems figured out, I took 7 metrics to evaluate our data in phase 1-

SNR (Signal-to-Noise Ratio): Measures constant background hums (like AC or traffic) to prevent the AI from hallucinating words from noise.

VAD Ratio (Voice Activity): Calculates the percentage of actual human speech, ensuring the model aligns text to voice rather than dead air.

ELR (Early-to-Late Ratio): Detects room echo and reverberation, preventing the model from confusing smeared word boundaries.

Clipping Rate: Identifies digital distortion from audio being recorded too loud, catching "blown-out" files where speech data is permanently destroyed.

Waveform Kurtosis: Detects sudden, extreme audio spikes, preventing the model from mistaking mic pops or table bumps for hard consonants (like 'P' or 'T').

Spectral Flatness: Differentiates structured human speech from pure white noise, filtering out files that are mostly drone sounds or static.

ZCR (Zero-Crossing Rate): Measures sound wave vibration frequency to catch excessive hissing or high-pitched electronic whines from faulty equipment.

Silence Ratio: Calculates the percentage of absolute dead air to flag recordings with excessively long pauses (though it's usually redundant if you use VAD).

To ensure high-quality Automatic Speech Recognition (ASR) training for the IndicVoices dataset, we ran a comprehensive acoustic profiling suite across 8 languages. Because recording environments and phonetic structures vary drastically across languages, we analyzed 6 key visualizations to design a **Hybrid Data Curation Pipeline**. This approach removes noisy or broken audio without accidentally introducing language bias (e.g., unfairly deleting all data from a specific language simply because it was recorded in the field).

Here are the core inferences and our final curation strategy.

### 🔍 1. Key Inferences from Acoustic Visualizations

#### A. Global vs. Per-Language Behavior
* **Visual Evidence:** [Boxplots per Language](./plots/01_boxplots_per_language.png) & [Histograms](./plots/02_histograms_all_languages.png)
* **Mathematical Proof:** [Cross-Language IQR Spread](./plots/05_iqr_spread_per_metric.png)
* **Insight:** Metrics like `ZCR`, `Waveform Kurtosis`, `Spectral Flatness`, and `ELR` are structurally stable across all languages (CV < 0.25). However, **VAD Ratio** (Voice Activity) and **SNR** (Signal-to-Noise) vary wildly. For instance, applying a global VAD threshold of >0.6 would preserve almost all Tamil data but delete over 40% of Hindi data.
* **Decision:** We must use **Per-Language** thresholds and normalization for VAD Ratio and SNR, while relying on **Global** thresholds for the remaining metrics.

#### B. Redundant Metrics & Hard Failures
* **Visual Evidence:** [Metric Correlation Heatmap](./plots/04_correlation_heatmap.png)
* **Insight:** `VAD Ratio` and `Silence Ratio` are almost perfectly inversely correlated ($r = -0.91$). Using both over-penalizes the exact same audio flaw. Additionally, `Clipping Rate` has roughly zero correlation with acoustic metrics because it is a hardware-level digital error.
* **Decision:** Completely drop `Silence Ratio` from the evaluation score. Treat `Clipping Rate` as a binary "Hard Reject" rather than a weighted metric.

#### C. Language-Specific Quirks
* **Visual Evidence:** [Language Quality Fingerprints (Radar)](./plots/06_radar_language_profiles.png) & [Retention Curves](./plots/03_retention_curves.png)
* **Insight:** Bengali exhibits a uniquely high Zero-Crossing Rate (ZCR) compared to the median dataset shape. 
* **Decision:** When scoring ZCR globally, the penalty weight must be gentle enough not to disproportionately decimate the Bengali subset.

---

### 🛠️ 2. The Final Curation Pipeline (The "Soft Score")

Based on the inferences above, we process and filter the audio in three distinct steps:

#### Step 1: The Hard Filters (Immediate Rejection)
Before calculating any complex scores, we drop audio that is structurally unusable for ASR:
* **Drop if** `clipping_rate > 0.05` (Heavy digital distortion / blown-out audio).
* **Drop if** `vad_ratio < 0.10` (Essentially dead air; negligible human speech).

#### Step 2: Hybrid Normalization (0.0 to 1.0)
To combine different units (dB, percentages, raw floats) into a single score, we normalize all metrics to a `0.0` (Worst) to `1.0` (Best) scale. 
* **Per-Language Normalization:** `vad_ratio` and `snr_db`. (e.g., A Hindi file is only compared against the Min/Max of other Hindi files).
* **Global Normalization:** `c50_db` (ELR), `spectral_flatness`, `zcr`, and `kurtosis`. (Compared against the Min/Max of the entire dataset).

#### Step 3: The Weighted Soft Score
We calculate a final Audio Quality Soft Score using the following weights, prioritized by their impact on ASR training:

| Metric | Weight | Importance for ASR | Normalization Type |
| :--- | :--- | :--- | :--- |
| **SNR (WADA)** | **40%** | Background noise is the #1 cause of ASR hallucinations. | Per-Language |
| **VAD Ratio** | **30%** | Ensures the model trains on actual phonemes, not silence. | Per-Language |
| **ELR (c50_db)**| **10%** | Rejects heavily reverberant (echoey) room recordings. | Global |
| **Spectral Flatness** | **10%** | Identifies pure white-noise files. | Global |
| **ZCR** | **5%** | Minor penalty for excessive hissing/static. | Global |
| **Kurtosis** | **5%** | Minor penalty for sudden mic pops/clicks. | Global |

**Execution:** We sort the resulting dataset by this combined Soft Score and prune the bottom 20% globally. Because VAD and SNR were normalized per-language, this safely removes the lowest-quality audio *without* erasing inherently quieter or different-paced languages.
