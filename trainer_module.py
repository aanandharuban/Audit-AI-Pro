import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def run_autonomous_training(new_audit_csv="audit_report.csv", master_data="master_audit_dataset.csv"):
    if not os.path.exists(new_audit_csv):
        return
    
    try:
        base_df = pd.read_csv(master_data)
        new_df = pd.read_csv(new_audit_csv)
        
        # Map app columns to master columns
        new_df['Is_Fraud'] = new_df['Status'].apply(lambda x: 1 if "RISK" in str(x) else 0)
        if 'Std_Total' in new_df.columns:
            new_df = new_df.rename(columns={'Std_Total': 'Total', 'Std_Narr': 'Narration'})
            
        combined_df = pd.concat([base_df, new_df[['Total', 'Narration', 'Is_Fraud']]], ignore_index=True)
        combined_df = combined_df.reset_index(drop=True)
        
        # ALIGNED FEATURE ENGINEERING
        combined_df['is_round'] = (combined_df['Total'] % 1000 == 0).astype(int)
        combined_df['is_q_end'] = 0 
        combined_df['v_count'] = 1 
        kw_list = ['cash', 'gift', 'personal', 'split', 'adjustment', 'urgent', 'repair', 'misc']
        combined_df['kw_flag'] = combined_df['Narration'].str.lower().apply(lambda x: 1 if any(w in str(x) for w in kw_list) else 0)
        
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        X_text = tfidf.fit_transform(combined_df['Narration'].astype(str)).toarray()
        X_num = combined_df[['Total', 'is_round', 'is_q_end', 'v_count', 'kw_flag']].values
        X_final = np.hstack((X_num, X_text))
        
        model = XGBClassifier(n_estimators=50).fit(X_final, combined_df['Is_Fraud'])
        joblib.dump(model, 'audit_brain_xgb.pkl')
        joblib.dump(tfidf, 'text_processor.pkl')
        combined_df.to_csv(master_data, index=False)
        
    except Exception as e:
        print(f"Error: {e}")