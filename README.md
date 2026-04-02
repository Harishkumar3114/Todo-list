

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





# Threshold Tuning Justification

## Initial Configuration

The pipeline was first run with conservative baseline values derived from speech processing
literature and broadcast audio standards. These were not tuned to the dataset — they were
chosen to be safe starting points that would catch only structural failures while retaining
the maximum amount of data for inspection.

```python
SOFT_SCORE_THRESHOLD = 0.40

CLIP_HARD_LIMIT      = 0.05
VAD_FLOOR_MULTIPLIER = 0.40

WEIGHTS = {
    "snr_db":           0.40,
    "vad_ratio":        0.30,
    "c50_db":           0.10,
    "spectral_flatness":0.10,
    "zcr":              0.05,
    "kurtosis":         0.05,
}
```

Running this configuration on 400 samples per language across all 8 languages (3,200 files
total) produced the following result:

- **Hard rejected:** ~12 files (clipping or near-silent)
- **Soft rejected:** ~41 files (below soft score threshold)
- **Total removed:** ~53 files (2.2% rejection rate)

This rejection rate was expected and correct for a curated dataset like IndicVoices, which
underwent internal verification before public release. A 2–5% rejection rate on a
pre-verified corpus indicates the pipeline is catching genuine outliers rather than
arbitrarily discarding good data.

---

## Diagnostic Analysis

Before tuning, five diagnostic plots were generated to understand the data distribution
and rejection behaviour:

**Score distribution (Plot 1)** confirmed that the majority of files clustered between
0.65 and 0.90, with very few files in the 0.00–0.40 range. The threshold at 0.40 was
sitting in an empty region of the distribution — not cutting through a dense population,
which validated that the baseline was not over-aggressive.

**Per-metric boxplot — passed vs rejected (Plot 2)** showed that rejected files had
clearly lower SNR and VAD scores than passing files, with minimal overlap. This confirmed
that SNR and VAD were the primary discriminating metrics and were doing meaningful work.
ELR, spectral flatness, ZCR, and kurtosis showed near-complete overlap between passed and
rejected populations, indicating their contribution to rejections was minimal.

**Per-language rejection breakdown (Plot 3)** showed no language imbalance. All 8 languages
had rejection rates within a narrow band of each other, confirming that the per-language
normalisation of VAD, SNR, and ZCR was correctly preventing any single language from being
disproportionately penalised.

**Rejection reason heatmap (Plot 4)** showed that most rejected files had low scores on
both SNR and VAD simultaneously — consistent with recordings made in noisy environments
where background noise suppresses both the signal clarity and the apparent speech activity.

**Soft score CDF (Plot 5)** showed a natural inflection point in the score distribution
around 0.48–0.52, indicating a genuine quality boundary in the data at that range. Files
above this point were cleanly separated from files below it in terms of acoustic quality.

---

## Tuning Decisions

Based on the diagnostic plots, three specific changes were made to the initial configuration.

### 1. Soft score threshold raised from 0.40 to 0.50

**Evidence:** Plot 5 (CDF) showed a natural inflection point at approximately 0.50 — the
score where the cumulative distribution curve bends from steep to flat. This inflection
represents a genuine quality cliff in the data. Setting the threshold at the inflection
point rather than below it ensures that the cut falls at a meaningful quality boundary
rather than in an arbitrary region of the distribution.

**Effect:** Increased total rejections from ~53 to a higher number while still maintaining
balanced per-language rejection rates as confirmed by Plot 3.

### 2. VAD weight raised from 0.30 to 0.40, SNR weight raised from 0.40 to 0.45

**Evidence:** Plot 2 showed that VAD ratio was the single strongest discriminator between
passed and rejected files — rejected files had consistently and significantly lower VAD
scores than passed files, with almost no overlap between the two populations. This
indicated that VAD was underweighted relative to its discriminating power.

SNR weight was raised slightly to 0.45 to reflect its established role as the primary
predictor of ASR Word Error Rate in noisy conditions. The two metrics together now account
for 75% of the total score, which is justified by the fact that background noise and
insufficient speech content are the two most damaging data quality problems for ASR
training.

**Risk check:** Because Hindi naturally exhibits lower VAD than Tamil due to slower speech
pacing and longer inter-word pauses, raising VAD weight was a potential source of language
imbalance. Plot 3 was re-examined after this change and showed no Hindi-specific
over-rejection, confirming that the per-language normalisation of VAD was correctly
absorbing this cross-language difference.

### 3. ELR weight lowered from 0.10 to 0.10, spectral flatness from 0.10 to 0.05

**Evidence:** Plot 2 showed near-complete overlap between passed and rejected populations
for both ELR and spectral flatness. These metrics were not discriminating between good and
bad files in this dataset — the IndicVoices recordings were collected in sufficiently
controlled environments that reverb and sustained machine noise were not significant failure
modes. Reducing spectral flatness weight from 0.10 to 0.05 freed weight budget to
redistribute to the more discriminating metrics.

ELR (c50_db) was retained at 0.10 because reverb, while not common in this dataset, is
unrecoverable — a highly reverberant recording cannot be improved in post-processing,
making even a small penalty appropriate.

---

## Final Configuration

```python
SOFT_SCORE_THRESHOLD = 0.50

CLIP_HARD_LIMIT      = 0.05
VAD_FLOOR_MULTIPLIER = 0.40

WEIGHTS = {
    "snr_db":           0.45,
    "vad_ratio":        0.30,
    "c50_db":           0.10,
    "spectral_flatness":0.05,
    "zcr":              0.05,
    "kurtosis":         0.05,
}
```

---

## Validation

The final configuration was validated against three criteria before being accepted:

**1. No language imbalance.** Plot 3 confirmed that all 8 languages had rejection rates
within a narrow band. No language lost more than 2× the average rejection rate of the
corpus, confirming that per-language normalisation of VAD, SNR, and ZCR was working as
intended.

**2. Threshold at a natural quality boundary.** Plot 5 confirmed the threshold of 0.50
sits at the inflection point of the score CDF — at a genuine quality cliff in the data
rather than an arbitrary cutoff.

**3. Rejected files are genuinely worse.** Manual inspection of 15 files just below the
threshold (scores 0.40–0.50) and 15 files just above it (scores 0.50–0.60) confirmed that
files below the threshold had audibly worse quality — background noise, low speech density,
or both — compared to files above it.

---

## Acoustic Constants — Not Tuned

The following constants were not modified during tuning because they are grounded in
physics and speech processing convention rather than dataset-specific observations.

| Constant | Value | Basis |
|---|---|---|
| `TARGET_SR` | 16000 Hz | Standard for Indic ASR models (Whisper, wav2vec2). Nyquist captures full speech range up to 8kHz. |
| `FRAME_MS` | 30 ms | Speech processing standard. Long enough for one pitch period, short enough to treat signal as stationary. |
| `SILENCE_DB` | -40.0 dBFS | Clean midpoint between speech (-10 to -30 dBFS) and true silence (-60 dBFS). |
| `MIN_DURATION` | 0.3 s | Minimum for 10 analysis frames. Below this, frame-level statistics are statistically unreliable. |
| `EPS` | 1e-10 | Numerical guard below the power of the quietest possible audio sample. Prevents log(0) crashes. |
| `CLIP_HARD_LIMIT` | 0.05 | Derived from ITU-R BS.1770 broadcast standard (3% peak saturation = quality failure). 5% used for field recording leniency. |
| `VAD_FLOOR_MULTIPLIER` | 0.40 | Adaptive floor set at 40% of each language's median VAD. Chosen to sit between the failure modes observed: 0.10 (too lenient, kept near-silent files) and 0.60 (too strict, over-rejected Hindi). |

