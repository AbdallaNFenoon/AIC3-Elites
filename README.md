
## Elites
# MTC-AIC3 BCI Competition Solution (Score: 0.71)

This repository contains the complete code, trained models, and documentation for our winning solution to the **MTC-AIC3: Egypt National Artificial Intelligence Competition**. Our final approach achieved a leaderboard score of **0.71**.

## 1. Executive Summary

Our investigation led to a critical discovery: the competition was not a traditional Brain-Computer Interface (BCI) problem, but rather a **motion artifact classification task**. The primary predictive signal was found not in the noisy EEG channels, but in the **6 motion sensor channels** (`Accelerometer` and `Gyroscope`).

The final, high-scoring model was built by engineering a comprehensive set of time-series features from the motion data using the `tsfresh` library and training specialized `LightGBM` classifiers for the MI and SSVEP tasks. This repository documents the methodology, code, and key findings from our investigation.

---

## 2. Repository Structure

This repository contains all the necessary assets to reproduce our results, from raw data processing to final model inference.


```plaintext
├── AIC3-Notebook.ipynb # The primary Jupyter Notebook for exploration and development.
├── Configs.txt # A text file version of the configuration script.
├── Kaggle_Score_Screenshot.png # Screenshot of our final leaderboard score (0.71).
├── Model-pipeline.py # The main, consolidated script to run the full training pipeline.
├── predict.py # A lightweight script to load saved models and predict on new data.
├── Note_on_Submission(please read).txt # Important notes regarding the different submission files.
├── requirements.txt # A list of all Python dependencies required to run the code.
├── System Description Paper.pdf # The detailed technical paper summarizing our methodology.
│
├── model_assets/ # Directory containing all saved model components.
│ ├── MI_model.pkl # The trained LightGBM model for the MI task.
│ ├── MI_scaler.pkl # The StandardScaler object for MI features.
│ ├── MI_selector.pkl # The SelectKBest object for MI feature selection.
│ ├── MI_label_encoder.pkl # The LabelEncoder for MI classes.
│ ├── MI_features.json # A list of the final selected feature names for the MI model.
│ │
│ ├── SSVEP_model.pkl # The trained LightGBM model for the SSVEP task.
│ ├── SSVEP_scaler.pkl # The StandardScaler object for SSVEP features.
│ ├── SSVEP_selector.pkl # The SelectKBest object for SSVEP feature selection.
│ ├── SSVEP_label_encoder.pkl # The LabelEncoder for SSVEP classes.
│ └── SSVEP_features.json # A list of the final selected feature names for the SSVEP model.
│
├── submission_outputs/
│ ├── Submission.csv # The final submission file that scored 0.71.
│ └── predictions_with_confidence.csv # A detailed output with prediction confidence scores.
```
---

## 3. How to Use

There are two primary ways to use this repository: training the model from scratch or generating predictions from the pre-trained models.

### A. Training the Full Pipeline from Scratch

This will re-run the entire data processing, feature extraction, and model training pipeline. **Note:** This is a very computationally expensive process and can take several hours.

1.  **Setup:** Place the original competition data in a folder structure as described in the `System Description Paper.pdf`. Update the `base_path` in `Model-pipeline.py` accordingly.
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Pipeline:**
    ```bash
    python Model-pipeline.py
    ```
4.  **Output:** This will generate a new submission file and save all the trained model assets in the `model_assets` directory.

### B. Generating Predictions from Pre-Trained Models (Fast)

This script loads the already-trained models and assets to quickly generate a submission file for new test data. This is the recommended approach for inference.

1.  **Setup:** Ensure the `model_assets` directory is present and contains all the `.pkl` and `.json` files. Place the new test data in the appropriate directory structure.
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Prediction Script:**
    ```bash
    python predict.py
    ```
4.  **Output:** This will generate a `submission_from_saved_model.csv` file in seconds.

---

## 4. Key Discoveries & Methodology

Our final solution was the result of a long, systematic investigation. The key findings were:

*   **EEG Data is a Red Herring:** The EEG channels contain no discernible neural signals and act as noise, harming model performance if not handled correctly.
*   **Motion Sensors are the Signal:** The gyroscope and accelerometer data contain strong, clean, and highly predictive patterns related to the task labels.
*   **`tsfresh` is the Key:** The breakthrough in performance came from using `tsfresh` with `ComprehensiveFCParameters` to extract thousands of "signal shape" features from the motion data. Simple statistics (`mean`, `std`) were not sufficient.
*   **The Power of FFT:** The feature importance analysis revealed that Fast Fourier Transform (FFT) coefficients, which describe the frequency components of the *movements*, were among the most powerful predictors for both tasks.
*   **A Universal Model is Best:** All attempts at personalization (subject-specific proxies) or advanced stacking architectures failed. The data noise profile favors a single, robust model trained on all available data.

This project underscores the importance of deep exploratory data analysis and demonstrates that the most powerful predictive signals are not always found where they are expected.
