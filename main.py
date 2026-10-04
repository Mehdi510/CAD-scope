import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.decomposition import PCA

st.set_page_config(page_title="CAD Transcriptomics", page_icon="🫀", layout="wide")

CAD_GENES = [
    "IL6","IL1B","TNF","CXCL8","CCL2","ICAM1","VCAM1","SELE","NFKB1",
    "NFKBIA","STAT3","JUN","FOS","EGR1","EGR3","CCL5","CCR7","CD3D",
    "CD3E","CD4","CD8A","FOXP3","GZMB","MMP9","MMP2","TLR4","SOD2",
    "HMOX1","NOS2","ABCA1","LDLR","APOE","LPL","PPARG","TGFB1"
]

PATHWAYS = {
    "Inflammatory signaling": ["IL6","IL1B","TNF","NFKB1","NFKBIA","STAT3","JUN","FOS"],
    "Chemokine / immune recruitment": ["CXCL8","CCL2","CCL5","CCR7","ICAM1","VCAM1"],
    "T-cell program": ["CD3D","CD3E","CD4","CD8A","FOXP3","GZMB"],
    "Endothelial activation": ["ICAM1","VCAM1","SELE","NOS2"],
    "Oxidative stress": ["SOD2","HMOX1","NOS2"],
    "Lipid handling": ["ABCA1","LDLR","APOE","LPL","PPARG"],
    "Matrix remodeling": ["MMP2","MMP9","TGFB1"],
}

def demo_data(n=120, seed=7):
    rng = np.random.default_rng(seed)
    genes = sorted(set(CAD_GENES + [
        f"GENE_{i:04d}" for i in range(1, 301)
    ]))
    y = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, len(genes)))
    for g, effect in [("IL6",0.9),("CXCL8",0.7),("ICAM1",0.6),("NFKB1",0.5),
                      ("CD3D",-0.6),("FOXP3",-0.7),("SOD2",0.35)]:
        X[:, genes.index(g)] += y * effect
    return pd.DataFrame(X, columns=genes), pd.Series(y, name="CAD")

def read_table(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_csv(uploaded, sep="\t")

def orient_matrix(df):
    # Accept either genes x samples or samples x genes.
    gene_col_candidates = [c for c in df.columns if str(c).lower() in
                           {"gene","gene_id","geneid","symbol","gene_symbol","genes"}]
    if gene_col_candidates:
        gc = gene_col_candidates[0]
        out = df.set_index(gc)
        out = out.apply(pd.to_numeric, errors="coerce")
        out = out.dropna(axis=1, how="all").dropna(axis=0, how="all")
        return out.T, "genes-as-rows"
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] >= max(5, numeric.shape[0] // 2):
        return numeric, "samples-as-rows"
    # If first column looks like gene IDs, use it.
    first = df.iloc[:,0].astype(str)
    if first.nunique() == len(first):
        out = df.iloc[:,1:].copy()
        out.index = first
        out = out.apply(pd.to_numeric, errors="coerce")
        return out.T, "genes-as-rows"
    raise ValueError("Could not identify a usable numeric expression matrix.")

def align_labels(labels_df, sample_index):
    if labels_df is None:
        return None
    cols = {str(c).lower(): c for c in labels_df.columns}
    sample_col = next((cols[k] for k in ["sample","sample_id","sampleid","id"] if k in cols), None)
    label_col = next((cols[k] for k in ["cad","label","status","group","phenotype","class"] if k in cols), None)
    if label_col is None:
        raise ValueError("Metadata needs a CAD/label/status/group column.")
    if sample_col is not None:
        s = labels_df.set_index(sample_col)[label_col].reindex(sample_index)
    else:
        s = labels_df[label_col].copy()
        if len(s) != len(sample_index):
            raise ValueError("Metadata row count does not match expression samples.")
        s.index = sample_index
    vals = s.astype(str).str.lower().str.strip()
    mapping = {
        "cad":1, "case":1, "disease":1, "1":1, "yes":1, "true":1,
        "control":0, "ctrl":0, "normal":0, "healthy":0, "0":0, "no":0, "false":0
    }
    y = vals.map(mapping)
    if y.isna().any():
        # Numeric 0/1 may already be present
        try:
            y = pd.to_numeric(s)
        except Exception:
            pass
    if not set(pd.Series(y).dropna().unique()).issubset({0,1}):
        raise ValueError("CAD labels must resolve to 0/1 or recognizable CAD/control text.")
    return pd.Series(y, index=sample_index, name="CAD").astype(int)

def signature_score(X):
    common = [g for g in CAD_GENES if g in X.columns]
    if not common:
        return None, []
    Z = (X[common] - X[common].mean()) / X[common].std(ddof=0).replace(0, 1)
    weights = pd.Series(1.0, index=common)
    for g in ["CD3D","CD3E","CD4","FOXP3"]:
        if g in weights.index:
            weights[g] = -1.0
    score = Z.mul(weights, axis=1).mean(axis=1)
    # Convert to 0-100 display score using logistic transform.
    pct = 100/(1+np.exp(-score))
    return pct, common

st.title("🫀 CAD Transcriptomics Prototype")
st.caption("Research prototype — transcriptomic similarity, not a clinical diagnosis or medical recommendation.")

with st.sidebar:
    st.header("Data")
    mode = st.radio("Input mode", ["Demo data", "Upload expression matrix"])
    uploaded = None
    meta = None
    if mode == "Upload expression matrix":
        uploaded = st.file_uploader("Expression CSV/TSV", type=["csv","tsv","txt"])
        meta = st.file_uploader("Optional metadata CSV/TSV", type=["csv","tsv","txt"])
    st.divider()
    st.markdown("**Recommended GEO cohorts**")
    st.markdown("- GSE180081 — discovery")
    st.markdown("- GSE180082 — validation")
    st.markdown("- GSE221911 — additional whole-blood RNA-seq")
    st.divider()
    st.info("For real clinical use, the model requires independent clinical validation, calibration, regulatory review, and clinician oversight.")

try:
    if mode == "Demo data":
        X, y = demo_data()
        source = "Synthetic demo data"
    else:
        if uploaded is None:
            st.info("Upload an expression matrix to begin. A demo mode is available in the sidebar.")
            st.stop()
        raw = read_table(uploaded)
        X, orientation = orient_matrix(raw)
        labels_df = read_table(meta) if meta else None
        y = align_labels(labels_df, X.index) if labels_df is not None else None
        source = f"Uploaded data ({orientation})"

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().sum() > 0]
    X = X.loc[:, X.nunique(dropna=True) > 1]

    st.success(f"Loaded {source}: **{X.shape[0]} samples × {X.shape[1]} genes/features**")

    score, sig_genes = signature_score(X)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "CAD model", "Molecular biology", "Validation"])

    with tab1:
        c1,c2,c3 = st.columns(3)
        c1.metric("Samples", X.shape[0])
        c2.metric("Features", X.shape[1])
        c3.metric("Signature genes present", len(sig_genes))
        if score is not None:
            st.subheader("CAD molecular similarity score")
            display_score = float(score.mean())
            st.metric("Cohort mean score", f"{display_score:.1f}/100")
            st.caption("This score is a prototype molecular similarity index derived from the included signature genes; it is not a clinical probability of CAD.")
        pca = PCA(n_components=2)
        Z = StandardScaler().fit_transform(X.fillna(X.median()))
        pcs = pca.fit_transform(Z)
        pcdf = pd.DataFrame({"PC1":pcs[:,0],"PC2":pcs[:,1]}, index=X.index)
        if y is not None:
            pcdf["CAD"] = y.map({0:"Control",1:"CAD"}).values
            fig = px.scatter(pcdf, x="PC1", y="PC2", color="CAD", title="PCA of expression profiles")
        else:
            fig = px.scatter(pcdf, x="PC1", y="PC2", title="PCA of expression profiles")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Interpretable CAD classifier")
        if y is None:
            st.warning("Upload metadata containing CAD/control labels to train and evaluate the classifier.")
        elif y.nunique() < 2:
            st.warning("At least two classes are required.")
        else:
            common = [g for g in CAD_GENES if g in X.columns]
            if len(common) < 5:
                st.warning("Too few CAD signature genes are present; use a gene-symbol expression matrix.")
            else:
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
                ])
                cv = StratifiedKFold(n_splits=min(5, int(y.value_counts().min())), shuffle=True, random_state=42)
                aucs = cross_val_score(model, X[common], y, cv=cv, scoring="roc_auc")
                model.fit(X[common], y)
                pred = model.predict_proba(X[common])[:,1]
                st.metric("Cross-validated AUROC", f"{aucs.mean():.3f} ± {aucs.std():.3f}")
                coef = pd.Series(model.named_steps["clf"].coef_[0], index=common).sort_values()
                coef_df = pd.DataFrame({"Gene":coef.index, "Coefficient":coef.values})
                fig = px.bar(coef_df.tail(15), x="Coefficient", y="Gene", orientation="h",
                             title="Features increasing model score")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("The classifier is trained only on the uploaded labeled matrix. Feature selection here is restricted to the predefined prototype signature to keep the demo transparent.")

    with tab3:
        st.subheader("Pathway-level molecular view")
        rows = []
        for pathway, genes in PATHWAYS.items():
            present = [g for g in genes if g in X.columns]
            if present:
                z = (X[present] - X[present].mean()) / X[present].std(ddof=0).replace(0,1)
                val = float(z.mean().mean())
                rows.append((pathway, val, len(present)))
        if rows:
            pdf = pd.DataFrame(rows, columns=["Pathway","Activity","Genes represented"]).sort_values("Activity")
            fig = px.bar(pdf, x="Activity", y="Pathway", orientation="h", title="Relative pathway activity")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pdf, use_container_width=True, hide_index=True)
        else:
            st.info("No predefined pathway genes were found in the uploaded matrix.")
        if sig_genes:
            st.subheader("Prototype CAD signature genes detected")
            st.write(", ".join(sig_genes))

    with tab4:
        st.subheader("Validation checklist")
        st.markdown("""
        **For a real study, do not report this demo score as clinical risk.**

        1. Train on a discovery cohort such as GSE180081.
        2. Freeze the signature and model.
        3. Validate on GSE180082 and GSE221911 without refitting preprocessing on the test set.
        4. Evaluate AUROC, AUPRC, sensitivity, specificity and calibration.
        5. Investigate batch effects and population differences.
        6. Separately test any stenosis/severity association.
        """)
        st.markdown("**Research-only disclaimer:** This prototype does not diagnose CAD, estimate future cardiovascular events, or recommend treatment.")

except Exception as e:
    st.error(f"Could not process the input: {e}")
    st.exception(e)
