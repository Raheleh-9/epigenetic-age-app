import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Configuration
st.set_page_config(page_title="Epigenetic Age Estimator", page_icon="🧬")

# Scientific Header
st.title("DNA Methylation Age Predictor")
st.markdown("""
This application estimates biological age based on **DNA methylation (DNAm) profiles**. 
The model is trained on the **GSE40279** dataset using an **ElasticNetCV** regressor 
optimized across 500 high-variance CpG sites.
""")

# 1. Load Model, Scaler, and Features
@st.cache_resource
def load_artifacts():
    with open("trained_model.pkl", "rb") as f:
        # Loading the model, scaler, and the 500 specific feature names
        model, scaler, feature_names = pickle.load(f)
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
    st.sidebar.success("Model Artifacts Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# 2. File Upload
st.subheader("Upload Genomic Data")
uploaded_file = st.file_uploader("Choose a CSV or TXT file (Beta Values)", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        # Load user data
        user_data = pd.read_csv(uploaded_file, index_col=0)
        
        with st.spinner("Analyzing Epigenetic Markers..."):
            # 3. Feature Matching Logic
            # Identifying which of our 500 trained features are present in the upload
            available_features = [f for f in feature_names if f in user_data.columns]
            
            # Reindexing ensures the columns are in the EXACT order the model expects
            # Missing features are filled with the median/zero to prevent crashes
            input_df = user_data.reindex(columns=feature_names, fill_value=0)

            # 4. Prediction Pipeline
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)

            # 5. Professional Results Display
            st.divider()
            col1, col2 = st.columns(2)
            
            col1.metric("Predicted Biological Age", f"{prediction[0]:.2f} Years")
            col2.metric("Marker Coverage", f"{(len(available_features)/len(feature_names))*100:.1f}%")

            # Warning if coverage is too low
            if len(available_features) < len(feature_names) * 0.7:
                st.warning("Low marker coverage detected. Accuracy may be significantly impacted.")

            st.info(f"Analysis completed using {len(available_features)} recognized CpG sites.")
            
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

# Footer
st.divider()
st.caption("Developed for Longevity Research | Data Source: NCBI GEO GSE40279")
