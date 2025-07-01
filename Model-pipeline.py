import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
import os
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# change the base path and submission path as per your directory structure
class FinalApproachClassifier:
    def __init__(self, base_path='D:/Documents/AIC/mtcaic3/', submission_path='D:/Documents/AIC/Submission/'):
        self.base_path = base_path
        self.submission_path = submission_path
        self.models = {}
        self.label_encoders = {}
        self.feature_columns = {}
        self.scalers = {}
        self.selectors = {}
        
    def save_model_assets(self, task):
        """Save all model assets for a specific task"""
        assets_dir = os.path.join(self.submission_path, 'model_assets')
        os.makedirs(assets_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(assets_dir, f'{task}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.models[task], f)
        print(f"Model saved to: {model_path}")
        
        # Save label encoder
        le_path = os.path.join(assets_dir, f'{task}_label_encoder.pkl')
        with open(le_path, 'wb') as f:
            pickle.dump(self.label_encoders[task], f)
        
        # Save feature columns
        features_path = os.path.join(assets_dir, f'{task}_features.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns[task], f)
        print(f"Feature names saved to: {features_path}")
        
        # Save scaler and selector if they exist
        if task in self.scalers:
            scaler_path = os.path.join(assets_dir, f'{task}_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[task], f)
                
        if task in self.selectors:
            selector_path = os.path.join(assets_dir, f'{task}_selector.pkl')
            with open(selector_path, 'wb') as f:
                pickle.dump(self.selectors[task], f)
    
    def load_trial_data(self, row, dataset_type='train'):
        """Load a single trial's data"""
        eeg_path = f"{self.base_path}/{row['task']}/{dataset_type}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
        eeg_data = pd.read_csv(eeg_path)
        
        trial_num = int(row['trial'])
        if row['task'] == 'MI':
            samples_per_trial = 2250
        else:
            samples_per_trial = 1750
        
        start_idx = (trial_num - 1) * samples_per_trial
        end_idx = start_idx + samples_per_trial
        
        return eeg_data.iloc[start_idx:end_idx].reset_index(drop=True)
    
    def check_impossible_combinations(self):
        """Check if certain label combinations don't exist"""
        print("Checking label combinations across tasks...")
        
        train_df = pd.read_csv(os.path.join(self.base_path, 'train.csv'))
        
        # Group by subject and check their labels in both tasks
        subject_patterns = []
        
        for subject in train_df['subject_id'].unique():
            subject_data = train_df[train_df['subject_id'] == subject]
            
            mi_data = subject_data[subject_data['task'] == 'MI']
            ssvep_data = subject_data[subject_data['task'] == 'SSVEP']
            
            if len(mi_data) > 0 and len(ssvep_data) > 0:
                # Get dominant labels for this subject
                mi_dominant = mi_data['label'].value_counts().index[0]
                ssvep_dominant = ssvep_data['label'].value_counts().index[0]
                
                subject_patterns.append({
                    'subject': subject,
                    'mi_dominant': mi_dominant,
                    'ssvep_dominant': ssvep_dominant
                })
        
        pattern_df = pd.DataFrame(subject_patterns)
        
        print("\nCross-task label patterns:")
        cross_patterns = pattern_df.groupby(['mi_dominant', 'ssvep_dominant']).size().unstack(fill_value=0)
        print(cross_patterns)
        
        # Check if test subjects follow patterns
        test_df = pd.read_csv(os.path.join(self.base_path, 'test.csv'))
        print("\nTest subjects:")
        print(test_df['subject_id'].unique())
    
    def prepare_tsfresh_data(self, trial_data, trial_id):
        """Prepare data for tsfresh in long format"""
        motion_channels = ['AccX', 'AccY', 'AccZ', 'Gyro1', 'Gyro2', 'Gyro3']
        
        dfs = []
        for ch in motion_channels:
            if ch in trial_data.columns:
                # Apply session normalization
                data = trial_data[ch].values
                # Z-score normalization
                data = (data - np.mean(data)) / (np.std(data) + 1e-6)
                
                df = pd.DataFrame({
                    'id': trial_id,
                    'time': range(len(data)),
                    'value': data,
                    'channel': ch
                })
                dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
    
    def extract_optimized_tsfresh_features(self, task='MI'):
        """Extract tsfresh features with optimized parameters"""
        print(f"\nExtracting optimized tsfresh features for {task}...")
        
        train_df = pd.read_csv(os.path.join(self.base_path, 'train.csv'))
        val_df = pd.read_csv(os.path.join(self.base_path, 'validation.csv'))
        
        # Combine train and validation
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        task_data = combined_df[combined_df['task'] == task]
        
        # Prepare data for tsfresh
        all_data = []
        labels = {}
        
        print("Preparing data for tsfresh...")
        for _, row in tqdm(task_data.iterrows(), total=len(task_data)):
            try:
                trial_data = self.load_trial_data(
                    row, 
                    'train' if row['id'] <= 4800 else 'validation'
                )
                tsfresh_data = self.prepare_tsfresh_data(trial_data, row['id'])
                all_data.append(tsfresh_data)
                labels[row['id']] = row['label']
            except Exception as e:
                print(f"Error processing {row['id']}: {e}")
                continue
        
        # Combine all data
        full_data = pd.concat(all_data, ignore_index=True)
        
        # Try different parameter sets
        print("\nTrying different tsfresh parameter sets...")
        
        # 1. Comprehensive parameters (might find the missing patterns)
        print("Extracting with ComprehensiveFCParameters...")
        features_comprehensive = extract_features(
            full_data,
            column_id='id',
            column_sort='time',
            column_value='value',
            column_kind='channel',
            default_fc_parameters=ComprehensiveFCParameters(),
            n_jobs=4,
            disable_progressbar=False
        )
        
        # Clean features
        features_comprehensive = impute(features_comprehensive)
        
        # Create labels series
        y = pd.Series(labels)
        y = y[features_comprehensive.index]
        
        return features_comprehensive, y
    
    def train_final_model(self):
        """Train the final optimized model"""
        print("="*80)
        print("TRAINING FINAL OPTIMIZED MODEL")
        print("="*80)
        
        test_df = pd.read_csv(os.path.join(self.base_path, 'test.csv'))
        all_predictions = []
        
        for task in ['MI', 'SSVEP']:
            print(f"\n{'='*60}")
            print(f"Processing {task}...")
            print('='*60)
            
            # Extract features
            X, y = self.extract_optimized_tsfresh_features(task)
            
            print(f"\nFeature shape: {X.shape}")
            print(f"Number of samples: {len(y)}")
            
            # Feature selection - keep top features
            print("\nSelecting top features...")
            
            # Encode labels for feature selection
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.label_encoders[task] = le
            
            # Select top K features
            k_features = min(1000, X.shape[1])  # Use top 1000 features
            selector = SelectKBest(f_classif, k=k_features)
            X_selected = selector.fit_transform(X, y_encoded)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_columns[task] = selected_features
            self.selectors[task] = selector
            
            print(f"Selected {len(selected_features)} features")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
            self.scalers[task] = scaler
            
            # Train model with optimized parameters
            model = lgb.LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.01,
                max_depth=10,
                num_leaves=100,
                min_child_samples=20,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring='accuracy')
            print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Train final model
            print("Training final model...")
            model.fit(X_scaled, y_encoded)
            self.models[task] = model
            
            # Save model assets
            self.save_model_assets(task)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            print("\nTop 20 features:")
            print(feature_importance)
            
            # Predict test data
            print(f"\nPredicting test samples for {task}...")
            task_test = test_df[test_df['task'] == task]
            
            # Prepare test data for tsfresh
            test_data_list = []
            test_ids = []
            
            for _, row in tqdm(task_test.iterrows(), total=len(task_test)):
                try:
                    trial_data = self.load_trial_data(row, 'test')
                    tsfresh_data = self.prepare_tsfresh_data(trial_data, row['id'])
                    test_data_list.append(tsfresh_data)
                    test_ids.append(row['id'])
                except Exception as e:
                    print(f"Error processing test {row['id']}: {e}")
                    # Add default prediction
                    all_predictions.append({
                        'id': row['id'],
                        'label': 'Left' if task == 'MI' else 'Forward'
                    })
            
            if test_data_list:
                # Extract features for test data
                print("Extracting test features...")
                test_full_data = pd.concat(test_data_list, ignore_index=True)
                
                test_features = extract_features(
                    test_full_data,
                    column_id='id',
                    column_sort='time',
                    column_value='value',
                    column_kind='channel',
                    default_fc_parameters=ComprehensiveFCParameters(),
                    n_jobs=4,
                    disable_progressbar=False
                )
                
                test_features = impute(test_features)
                
                # CRITICAL: Ensure test features match training features exactly
                # First, get all training feature columns (before selection)
                all_feature_columns = X.columns.tolist()
                
                # Reindex test features to match training features
                test_features = test_features.reindex(columns=all_feature_columns, fill_value=0)
                
                # Now apply the same feature selection
                test_features_selected = selector.transform(test_features)
                
                # Scale using the same scaler
                test_features_scaled = scaler.transform(test_features_selected)
                
                # Predict
                y_pred = model.predict(test_features_scaled)
                y_proba = model.predict_proba(test_features_scaled)
                
                # Convert predictions to labels
                for idx, test_id in enumerate(test_features.index):
                    label = le.inverse_transform([y_pred[idx]])[0]
                    confidence = np.max(y_proba[idx])
                    
                    all_predictions.append({
                        'id': test_id,
                        'label': label,
                        'confidence': confidence
                    })
        
        # Create submission
        submission_df = pd.DataFrame(all_predictions).sort_values('id')[['id', 'label']]
        submission_path = os.path.join(self.submission_path, 'submission_tsfresh_optimized.csv')
        submission_df.to_csv(submission_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"Submission saved to: {submission_path}")
        print(f"{'='*60}")
        
        print("\nPrediction distribution:")
        print(submission_df['label'].value_counts())
        
        # Calculate confidence statistics
        conf_df = pd.DataFrame(all_predictions)
        if 'confidence' in conf_df.columns:
            print(f"\nAverage prediction confidence: {conf_df['confidence'].mean():.4f}")
        
        # Save predictions with confidence for analysis
        conf_path = os.path.join(self.submission_path, 'predictions_with_confidence.csv')
        conf_df.to_csv(conf_path, index=False)
        print(f"Predictions with confidence saved to: {conf_path}")
        
        return submission_df

class SimpleBestApproach:
    """Recreate the 0.64 score approach with tsfresh EfficientFCParameters"""
    
    def __init__(self, base_path='D:/Documents/AIC/mtcaic3/', submission_path='D:/Documents/AIC/Submission/'):
        self.base_path = base_path
        self.submission_path = submission_path
        self.models = {}
        self.label_encoders = {}
        self.feature_columns = {}
        
    def save_model_assets(self, task):
        """Save model assets"""
        assets_dir = os.path.join(self.submission_path, 'model_assets_efficient')
        os.makedirs(assets_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(assets_dir, f'{task}_model_efficient.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.models[task], f)
        print(f"Model saved to: {model_path}")
        
        # Save label encoder
        le_path = os.path.join(assets_dir, f'{task}_label_encoder_efficient.pkl')
        with open(le_path, 'wb') as f:
            pickle.dump(self.label_encoders[task], f)
        
        # Save feature columns
        features_path = os.path.join(assets_dir, f'{task}_features_efficient.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns[task], f)
        print(f"Feature names saved to: {features_path}")
    
    def load_trial_data(self, row, dataset_type='train'):
        """Load a single trial's data"""
        eeg_path = f"{self.base_path}/{row['task']}/{dataset_type}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
        eeg_data = pd.read_csv(eeg_path)
        
        trial_num = int(row['trial'])
        if row['task'] == 'MI':
            samples_per_trial = 2250
        else:
            samples_per_trial = 1750
        
        start_idx = (trial_num - 1) * samples_per_trial
        end_idx = start_idx + samples_per_trial
        
        return eeg_data.iloc[start_idx:end_idx].reset_index(drop=True)
    
    def train_and_predict(self):
        """Train using EfficientFCParameters like the 0.64 approach"""
        print("="*80)
        print("RECREATING 0.64 SCORE APPROACH")
        print("="*80)
        
        train_df = pd.read_csv(os.path.join(self.base_path, 'train.csv'))
        val_df = pd.read_csv(os.path.join(self.base_path, 'validation.csv'))
        test_df = pd.read_csv(os.path.join(self.base_path, 'test.csv'))
        
        # Combine train and validation
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        
        all_predictions = []
        
        for task in ['MI', 'SSVEP']:
            print(f"\nProcessing {task}...")
            
            task_data = combined_df[combined_df['task'] == task]
            task_test = test_df[test_df['task'] == task]
            
            # Prepare data for tsfresh
            all_data = []
            labels = {}
            
            print("Preparing training data...")
            for _, row in tqdm(task_data.iterrows(), total=len(task_data)):
                try:
                    trial_data = self.load_trial_data(
                        row, 
                        'train' if row['id'] <= 4800 else 'validation'
                    )
                    
                    # Prepare in tsfresh format
                    motion_channels = ['AccX', 'AccY', 'AccZ', 'Gyro1', 'Gyro2', 'Gyro3']
                    dfs = []
                    
                    for ch in motion_channels:
                        if ch in trial_data.columns:
                            df = pd.DataFrame({
                                'id': row['id'],
                                'time': range(len(trial_data)),
                                'value': trial_data[ch].values,
                                'channel': ch
                            })
                            dfs.append(df)
                    
                    all_data.append(pd.concat(dfs, ignore_index=True))
                    labels[row['id']] = row['label']
                    
                except Exception as e:
                    continue
            
            # Combine all data
            full_data = pd.concat(all_data, ignore_index=True)
            
            # Extract features with EfficientFCParameters
            print("Extracting features with EfficientFCParameters...")
            features = extract_features(
                full_data,
                column_id='id',
                column_sort='time',
                column_value='value',
                column_kind='channel',
                default_fc_parameters=EfficientFCParameters(),
                n_jobs=4,
                disable_progressbar=False
            )
            
            # Clean features
            features = impute(features)
            
            # Create labels series
            y = pd.Series(labels)
            y = y[features.index]
            
            print(f"Extracted {features.shape[1]} features for {len(y)} samples")
            
            # Save feature columns
            self.feature_columns[task] = features.columns.tolist()
            
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.label_encoders[task] = le
            
            # Train LightGBM model
            model = lgb.LGBMClassifier(
                n_estimators=1500,
                learning_rate=0.01,
                max_depth=10,
                num_leaves=80,
                min_child_samples=20,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            print("Training model...")
            model.fit(features, y_encoded)
            self.models[task] = model
            
            # Save model assets
            self.save_model_assets(task)
            
            # Feature importance
            importance_df = pd.DataFrame({
                'feature': features.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            print("\nTop 20 features:")
            print(importance_df)
            
            # Save feature importance
            importance_path = os.path.join(self.submission_path, f'feature_importance_{task}_efficient.csv')
            importance_df.to_csv(importance_path, index=False)
            
            # Predict test data
            print(f"\nPredicting test samples for {task}...")
            
            # Prepare test data
            test_data_list = []
            test_ids = []
            
            for _, row in tqdm(task_test.iterrows(), total=len(task_test)):
                try:
                    trial_data = self.load_trial_data(row, 'test')
                    
                    # Prepare in tsfresh format
                    dfs = []
                    for ch in ['AccX', 'AccY', 'AccZ', 'Gyro1', 'Gyro2', 'Gyro3']:
                        if ch in trial_data.columns:
                            df = pd.DataFrame({
                                'id': row['id'],
                                'time': range(len(trial_data)),
                                'value': trial_data[ch].values,
                                'channel': ch
                            })
                            dfs.append(df)
                    
                    test_data_list.append(pd.concat(dfs, ignore_index=True))
                    test_ids.append(row['id'])
                    
                except Exception as e:
                    print(f"Error processing test {row['id']}: {e}")
                    all_predictions.append({
                        'id': row['id'],
                        'label': 'Left' if task == 'MI' else 'Forward'
                    })
            
            if test_data_list:
                # Extract features for test data
                print("Extracting test features...")
                test_full_data = pd.concat(test_data_list, ignore_index=True)
                
                test_features = extract_features(
                    test_full_data,
                    column_id='id',
                    column_sort='time',
                    column_value='value',
                    column_kind='channel',
                    default_fc_parameters=EfficientFCParameters(),
                    n_jobs=4,
                    disable_progressbar=False
                )
                
                test_features = impute(test_features)
                
                # Ensure same features as training
                test_features = test_features.reindex(columns=features.columns, fill_value=0)
                
                # Predict
                y_pred = model.predict(test_features)
                y_proba = model.predict_proba(test_features)
                
                # Convert predictions to labels
                for idx, test_id in enumerate(test_features.index):
                    label = le.inverse_transform([y_pred[idx]])[0]
                    confidence = np.max(y_proba[idx])
                    
                    all_predictions.append({
                        'id': test_id,
                        'label': label,
                        'confidence': confidence
                    })
        
        # Create submission
        submission_df = pd.DataFrame(all_predictions).sort_values('id')[['id', 'label']]
        submission_path = os.path.join(self.submission_path, 'submission_tsfresh_efficient.csv')
        submission_df.to_csv(submission_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"Submission saved to: {submission_path}")
        print(f"{'='*60}")
        
        print("\nPrediction distribution:")
        print(submission_df['label'].value_counts())
        
        # Save predictions with confidence
        conf_df = pd.DataFrame(all_predictions)
        conf_path = os.path.join(self.submission_path, 'predictions_efficient_with_confidence.csv')
        conf_df.to_csv(conf_path, index=False)
        print(f"Predictions with confidence saved to: {conf_path}")
        
        return submission_df

def save_investigation_summary():
    """Save a summary of our investigation"""
    summary = {
        "investigation_summary": {
            "best_score": 0.64,
            "approach": "LightGBM + tsfresh (EfficientFCParameters)",
            "key_findings": [
                "Motion sensors contain the signal, not EEG",
                "Gyroscope channels (especially Gyro2) are most informative",
                "Final angles and cumulative positions are important features",
                "Cross-channel correlations (off-diagonal energy) matter",
                "No hidden patterns in metadata (Battery, Validation, Counter)"
            ],
            "critical_time_points": {
                "MI": {
                    "Gyro1": [32, 36, 55, 486, 489],
                    "Gyro2": [21, 28, 185, 757, 762],
                    "Gyro3": [260, 450, 455, 459, 461]
                },
                "SSVEP": {
                    "Gyro1": [120, 125, 163, 165, 169],
                    "Gyro2": [107, 109, 122, 131, 136],
                    "Gyro3": [120, 124, 131, 133, 135]
                }
            },
            "failed_approaches": [
                "Pure EEG analysis (CSP, EEGNet)",
                "Subject-specific models",
                "Battery/Validation column patterns",
                "Simple threshold rules"
            ],
            "recommendations_for_0.8": [
                "Try ComprehensiveFCParameters instead of EfficientFCParameters",
                "Apply subject/session normalization before feature extraction",
                "Use different algorithms (XGBoost, CatBoost)",
                "Ensemble multiple approaches",
                "Add derivative features before tsfresh",
                "Focus on phase relationships between gyroscopes"
            ]
        }
    }
    
    summary_path = 'D:/Documents/AIC/Submission/investigation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\nInvestigation summary saved to: {summary_path}")

def final_investigation():
    """Final investigation and model training"""
    
    print("="*80)
    print("FINAL INVESTIGATION FOR 0.8 SCORE")
    print("="*80)
    
    # First check patterns
    classifier = FinalApproachClassifier()
    classifier.check_impossible_combinations()
    
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    # Try the optimized approach
    print("\n1. Trying optimized tsfresh with ComprehensiveFCParameters...")
    print("This will take 2-3 hours with your hardware...")
    print("Press Ctrl+C to skip to the faster approach")
    
    try:
        optimized_submission = classifier.train_final_model()
    except KeyboardInterrupt:
        print("\nSkipping ComprehensiveFCParameters approach...")
        optimized_submission = None
    
    # Also try the simpler approach that got 0.64
    print("\n2. Trying EfficientFCParameters approach (0.64 baseline)...")
    print("This will take ~40-60 minutes...")
    simple_classifier = SimpleBestApproach()
    simple_submission = simple_classifier.train_and_predict()
    
    # Save investigation summary
    save_investigation_summary()
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS TO REACH 0.8:")
    print("="*80)
    
    print("\n1. FEATURE ENGINEERING:")
    print("   - The ComprehensiveFCParameters might find patterns EfficientFCParameters missed")
    print("   - Pay attention to FFT coefficients and autocorrelation features")
    print("   - Consider adding derivative features before tsfresh")
    
    print("\n2. MODEL OPTIMIZATION:")
    print("   - Try XGBoost or CatBoost instead of LightGBM")
    print("   - Use Optuna for hyperparameter tuning")
    print("   - Consider stacking multiple models")
    
    print("\n3. PREPROCESSING:")
    print("   - Try different normalization strategies (per-subject, per-session)")
    print("   - Apply bandpass filtering to motion sensors")
    print("   - Remove outliers before feature extraction")
    
    print("\n4. ENSEMBLE APPROACH:")
    print("   - Combine predictions from multiple parameter sets")
    print("   - Weight models based on cross-validation performance")
    print("   - Use different random seeds and average predictions")
    
    print("\n5. ANALYSIS:")
    print("   - Check the saved model assets and feature importance files")
    print("   - Look for patterns in the confidence scores")
    print("   - Compare feature importance between approaches")
    
    print("\n" + "="*80)
    print("ALL ASSETS SAVED TO:")
    print("="*80)
    print(f"- Submissions: D:/Documents/AIC/Submission/")
    print(f"- Model assets: D:/Documents/AIC/Submission/model_assets/")
    print(f"- Feature importance: D:/Documents/AIC/Submission/feature_importance_*.csv")
    print(f"- Investigation summary: D:/Documents/AIC/Submission/investigation_summary.json")

if __name__ == "__main__":
    # Run the final investigation
    final_investigation()           