
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
XtX_inv = joblib.load("XtX_inv.pkl")
h_star = joblib.load("h_star.pkl")

def smiles_to_ecfp(smiles, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    arr = np.zeros((nBits,))
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def leverage(x):
    return x @ XtX_inv @ x.T

def predict_smiles(smiles):
    fp = smiles_to_ecfp(smiles)
    if fp is None:
        return None

    X = fp.reshape(1, -1)

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    lev = leverage(X_pca[0])
    in_domain = lev < h_star

    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)

    return {
        "SMILES": smiles,
        "In_domain": in_domain,
        "Leverage": lev,
        "Probability_active": proba,
        "Predicted_class": pred
    }

st.title("GSK-3β ML Predictor")

smiles_input = st.text_input("Enter SMILES")

if st.button("Predict"):
    result = predict_smiles(smiles_input)
    if result:
        st.json(result)
    else:
        st.error("Invalid SMILES")

uploaded_file = st.file_uploader("Upload CSV with SMILES")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    results = []

    for sm in df["SMILES"]:
        res = predict_smiles(sm)
        if res:
            results.append(res)

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    st.download_button("Download Results", results_df.to_csv(index=False), "results.csv")
