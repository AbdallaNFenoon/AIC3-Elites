# --- INFERENCE SCRIPT: LOAD & PREDICT (V60) ---

import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import os
import pickle
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
print("✅ Libraries imported. This is the INFERENCE script.")

# --- Configuration ---
# You can adjust these paths to where your data and saved models are.
BASE_PATH = 'D:/Documents/AIC/mtcaic3/'
SUBMISSION_PATH = 'D:/Documents/AIC/Submission/'
ASSETS_PATH = os.path.join(SUBMISSION_PATH, 'model_assets')

# --- Helper Functions (Copied from winning script) ---
def load_trial_data(row, dataset_type='test'):
    """Load a single trial's data"""
    eeg_path = f"{BASE_PATH}/{row['task']}/{dataset_type}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
    eeg_data = pd.read_csv(eeg_path)
    trial_num = int(row['trial'])
    samples_per_trial = 2250 if row['task'] == 'MI' else 1750
    start_idx = (trial_num - 1) * samples_per_trial
    end_idx = start_idx + samples_per_trial
    return eeg_data.iloc[start_idx:end_idx].reset_index(drop=True)

def prepare_tsfresh_data(trial_data, trial_id):
    """Prepare data for tsfresh in long format"""
    motion_channels = ['AccX', 'AccY', 'AccZ', 'Gyro1', 'Gyro2', 'Gyro3']
    dfs = []
    for ch in motion_channels:
        if ch in trial_data.columns:
            data = trial_data[ch].values
            data = (data - np.mean(data)) / (np.std(data) + 1e-6) # Z-score normalization
            df = pd.DataFrame({'id': trial_id, 'time': range(len(data)), 'value': data, 'channel': ch})
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# --- Main Inference Function ---
def predict_from_saved_models():
    print("="*80)
    print("GENERATING SUBMISSION FROM SAVED MODELS")
    print("="*80)

    test_df = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
    all_predictions = []

    for task in ['MI', 'SSVEP']:
        print(f"\n{'='*60}")
        print(f"Processing {task}...")
        print('='*60)
        
        # --- Step 1: Load all saved assets for the task ---
        print("Loading saved model and assets...")
        with open(os.path.join(ASSETS_PATH, f'{task}_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(ASSETS_PATH, f'{task}_label_encoder.pkl'), 'rb') as f:
            le = pickle.load(f)
        with open(os.path.join(ASSETS_PATH, f'{task}_features.json'), 'r') as f:
            selected_feature_names = json.load(f)
        with open(os.path.join(ASSETS_PATH, f'{task}_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(ASSETS_PATH, f'{task}_selector.pkl'), 'rb') as f:
            selector = pickle.load(f)
        print("✅ All assets loaded successfully.")

        # --- Step 2: Process Test Data and Extract Features ---
        print(f"\nPredicting test samples for {task}...")
        task_test = test_df[test_df['task'] == task]
        
        test_data_list = []
        for _, row in tqdm(task_test.iterrows(), total=len(task_test)):
            trial_data = load_trial_data(row, 'test')
            tsfresh_data = prepare_tsfresh_data(trial_data, row['id'])
            test_data_list.append(tsfresh_data)
        
        print("Extracting test features (this will take some time)...")
        test_full_data = pd.concat(test_data_list, ignore_index=True)
        test_features = extract_features(
            test_full_data, column_id='id', column_sort='time', column_value='value',
            column_kind='channel', default_fc_parameters=ComprehensiveFCParameters(),
            n_jobs=4, disable_progressbar=False
        )
        test_features = impute(test_features)
        
        # --- Step 3: Apply Saved Transformations and Predict ---
        print("Applying transformations and making predictions...")
        # CRITICAL: Ensure the test set has the exact same columns as the training set before selection
        # We reindex based on the original feature columns before selection was applied
        # Assuming the selector was fit on a DataFrame with all tsfresh features
        all_training_cols = selector.feature_names_in_
        test_features_reindexed = test_features.reindex(columns=all_training_cols, fill_value=0)
        
        # Now, transform using the loaded selector and scaler
        test_features_selected = selector.transform(test_features_reindexed)
        test_features_scaled = scaler.transform(test_features_selected)
        
        # Predict
        y_pred = model.predict(test_features_scaled)
        
        # Convert predictions to original labels
        final_labels = le.inverse_transform(y_pred)
        
        for i, test_id in enumerate(test_features.index):
            all_predictions.append({'id': test_id, 'label': final_labels[i]})

    # --- Step 4: Create Final Submission File ---
    submission_df = pd.DataFrame(all_predictions).sort_values('id')[['id', 'label']]
    submission_path = os.path.join(SUBMISSION_PATH, 'submission_from_saved_model.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Submission saved to: {submission_path}")
    print(f"{'='*60}")
    print("\nPrediction distribution:")
    print(submission_df['label'].value_counts())

if __name__ == "__main__":
    predict_from_saved_models()