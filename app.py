import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Customer Segmentation", layout="wide", page_icon="🧑‍🤝‍🧑")

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #6C5CE7 0%, #00B894 50%, #FD79A8 100%);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
}
.main-header h1 {
    color: white !important;
    margin: 0;
    font-size: 34px;
}
.main-header p {
    color: #f1f1f1;
    margin: 6px 0 0 0;
    font-size: 15px;
}
.section-card {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    margin-bottom: 20px;
    border-left: 6px solid #6C5CE7;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #6C5CE7 0%, #00B894 100%);
    border-radius: 12px;
    padding: 12px;
}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] div {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧑‍🤝‍🧑 Customer Segmentation Studio</h1>
    <p>Upload data → Unsupervised model chuno (KMeans / Hierarchical / DBSCAN / GMM) → Poora dashboard + segment insights paao</p>
</div>
""", unsafe_allow_html=True)

# Colorful palette used across all charts
PALETTE = ["#6C5CE7", "#00B894", "#FD79A8", "#0984E3", "#FDCB6E", "#E17055", "#00CEC9", "#D63031"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALETTE)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "#FAFAFA"
plt.rcParams["axes.edgecolor"] = "#DDDDDD"

SEGMENT_COLORS = {"Low": "#0984E3", "Medium": "#FDCB6E", "High": "#E17055", "Noise / Outlier": "#B2BEC3"}

def color_for(seg, idx):
    return SEGMENT_COLORS.get(seg, PALETTE[idx % len(PALETTE)])

# ---------------- 1. FILE UPLOAD ----------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📂 Apni data file upload karo (CSV ya Excel)", type=["csv", "xlsx", "xls"])
st.markdown('</div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    from io import BytesIO
    if file_name.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))
    else:
        return pd.read_excel(BytesIO(file_bytes))

if uploaded_file is not None:
    df = load_data(uploaded_file.getvalue(), uploaded_file.name)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📄 Data Preview (head)")
    st.dataframe(df.head())
    st.caption(f"Total rows: **{len(df)}** &nbsp;|&nbsp; Total columns: **{df.shape[1]}**")
    st.markdown('</div>', unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # ---- Auto-detect & exclude ID-like columns (e.g. customer_id) ----
    def is_id_like(col):
        name_flag = any(kw in col.lower() for kw in ["_id", "id_", "customerid", "index", "unnamed"]) or col.lower() == "id"
        is_integer_col = pd.api.types.is_integer_dtype(df[col])
        uniqueness_flag = is_integer_col and (df[col].nunique() / len(df) > 0.95)
        return name_flag or uniqueness_flag

    id_like_cols = [c for c in numeric_cols if is_id_like(c)]
    feature_cols_all = [c for c in numeric_cols if c not in id_like_cols]

    if len(feature_cols_all) < 2:
        feature_cols_all = numeric_cols

    if len(feature_cols_all) < 2:
        st.error("Clustering ke liye kam se kam 2 numeric (non-ID) columns chahiye.")
    else:
        # ============================================================
        # FEATURE SELECTION (works with as many columns as the file has —
        # zyada data / zyada columns aane par bhi dashboard sahi se scale hota hai)
        # ============================================================
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Feature & Model Setup")

        if id_like_cols:
            st.caption(f"ℹ️ ID-jaise columns auto-skip kiye gaye hain (feature nahi banaye): {', '.join(id_like_cols)}")

        selected_features = st.multiselect(
            "🧮 Clustering ke liye features chuno (jitna zyada relevant data, utna behtar segmentation)",
            feature_cols_all,
            default=feature_cols_all,
        )
        if len(selected_features) < 2:
            st.warning("Kam se kam 2 features select karo.")
            st.stop()

        st.markdown("**🤖 Unsupervised model chuno:**")
        model_choice = st.radio(
            "Model Selection",
            ["KMeans (Clustering)", "Hierarchical (Agglomerative)", "DBSCAN (Density-based)", "Gaussian Mixture (GMM)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        X = df[selected_features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_samples = X_scaled.shape[0]
        max_k = min(10, n_samples - 1) if n_samples > 2 else 2
        max_k = max(max_k, 2)

        # Visualization ke liye PCA use nahi kar rahe — seedha pehle 2 selected
        # features (scaled) ka scatter dikhate hain. Asli clustering hamesha
        # saare selected features par hoti hai, ye sirf 2D plot ke liye hai.
        use_pca_view = False
        X_view = X_scaled[:, :2]
        view_x_label = selected_features[0]
        view_y_label = selected_features[1]
        if len(selected_features) > 2:
            st.caption(f"ℹ️ Scatter plot sirf **{selected_features[0]}** vs **{selected_features[1]}** dikha raha hai "
                       f"(2D view ke liye) — clustering baaki sab {len(selected_features)} features par bhi hui hai.")

        # ============================================================
        # NEW: PREVIEW SCATTER — cluster form hone se PEHLE ka data
        # ============================================================
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🔎 Data Before Clustering (Raw Preview)")
        st.caption("Ye scatter clustering model chalne se **pehle** ka hai — sab points abhi ek hi color "
                   "mein hain, koi segment assign nahi hua hai.")
        fig0, ax0 = plt.subplots(figsize=(6, 4))
        ax0.scatter(X_view[:, 0], X_view[:, 1], s=90, color=PALETTE[0],
                    edgecolor="white", linewidth=1.0, alpha=0.85)
        ax0.set_xlabel(view_x_label)
        ax0.set_ylabel(view_y_label)
        ax0.set_title("Raw Data (Before Clustering)", fontweight="bold")
        fig_col0, _ = st.columns([2, 1])
        with fig_col0:
            st.pyplot(fig0, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        SIL_SAMPLE_SIZE = 2000  # cap silhouette cost on big datasets (O(n^2) otherwise)

        def safe_silhouette(labels):
            valid = labels != -1
            unique_labels = set(labels[valid])
            if len(unique_labels) < 2 or valid.sum() < 2:
                return None
            try:
                sample = SIL_SAMPLE_SIZE if valid.sum() > SIL_SAMPLE_SIZE else None
                return silhouette_score(X_scaled[valid], labels[valid],
                                         sample_size=sample, random_state=42)
            except Exception:
                return None

        @st.cache_data(show_spinner=False)
        def compute_kmeans_elbow(X_scaled, k_range_tuple):
            wcss_ = []
            for i in k_range_tuple:
                km_ = KMeans(n_clusters=i, random_state=42, n_init=10)
                km_.fit(X_scaled)
                wcss_.append(km_.inertia_)
            return wcss_

        @st.cache_data(show_spinner=False)
        def compute_hierarchical_silhouette(X_scaled, k_range_tuple, linkage_):
            scores = []
            sample = SIL_SAMPLE_SIZE if X_scaled.shape[0] > SIL_SAMPLE_SIZE else None
            for i in k_range_tuple:
                model_i = AgglomerativeClustering(n_clusters=i, linkage=linkage_)
                labels_i = model_i.fit_predict(X_scaled)
                try:
                    scores.append(silhouette_score(X_scaled, labels_i, sample_size=sample, random_state=42))
                except Exception:
                    scores.append(-1)
            return scores

        @st.cache_data(show_spinner=False)
        def compute_gmm_bic(X_scaled, k_range_tuple):
            scores = []
            for i in k_range_tuple:
                gm_i = GaussianMixture(n_components=i, random_state=42)
                gm_i.fit(X_scaled)
                scores.append(gm_i.bic(X_scaled))
            return scores

        def rank_labels_to_segments(cluster_ids, raw_labels):
            """Map arbitrary cluster ids -> Low/Medium/High (…or Level N) based on
            the overall (scaled) feature intensity of each cluster, so the labels
            stay meaningful regardless of how many features/clusters are used."""
            score_by_cluster = {}
            for cid in cluster_ids:
                mask = raw_labels == cid
                score_by_cluster[cid] = X_scaled[mask].mean()
            order = sorted([c for c in cluster_ids if c != -1], key=score_by_cluster.get)
            n = len(order)
            if n <= 2:
                pool = ["Low", "High"]
            elif n == 3:
                pool = ["Low", "Medium", "High"]
            else:
                pool = [f"Level {i+1} (Low→High)" for i in range(n)]
            lmap = {cid: pool[r] for r, cid in enumerate(order)}
            if -1 in cluster_ids:
                lmap[-1] = "Noise / Outlier"
            return lmap

        # ============================================================
        # MODEL 1: KMEANS
        # ============================================================
        if model_choice == "KMeans (Clustering)":
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📉 Elbow Method Graph")
            cluster_mode = st.radio(
                "Cluster count kaise decide karein?",
                ["Manual (khud select karo)", "Auto (Elbow method se best value)"]
            )

            k_range = range(1, max_k + 1)
            wcss = compute_kmeans_elbow(X_scaled, tuple(k_range))

            fig1, ax1 = plt.subplots(figsize=(6, 3.3))
            ax1.plot(list(k_range), wcss, marker="o", linewidth=2, color=PALETTE[0])
            ax1.set_xlabel("No. of Clusters")
            ax1.set_ylabel("WCSS")
            ax1.set_title("Elbow Method", fontweight="bold")
            fig_col, _ = st.columns([2, 1])
            with fig_col:
                st.pyplot(fig1, use_container_width=True)

            if cluster_mode == "Manual (khud select karo)":
                n_clusters = st.slider("Number of clusters chuno", min_value=2, max_value=max_k, value=min(3, max_k))
            else:
                diffs = [wcss[i - 1] - wcss[i] for i in range(1, len(wcss))]
                n_clusters = diffs.index(max(diffs)) + 2 if diffs else 2
                n_clusters = max(2, min(n_clusters, max_k))
                st.info(f"Auto-selected clusters: **{n_clusters}**")
            st.markdown('</div>', unsafe_allow_html=True)

            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            raw_labels = model.fit_predict(X_scaled)
            algo_summary = f"KMeans with k = {n_clusters}"

        # ============================================================
        # MODEL 2: HIERARCHICAL (AGGLOMERATIVE)
        # ============================================================
        elif model_choice == "Hierarchical (Agglomerative)":
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📉 Silhouette Score by Cluster Count")
            linkage = st.selectbox("Linkage method", ["ward", "average", "complete", "single"], index=0)
            cluster_mode = st.radio(
                "Cluster count kaise decide karein?",
                ["Manual (khud select karo)", "Auto (best silhouette score se)"]
            )

            k_range = range(2, max_k + 1)
            sil_scores = compute_hierarchical_silhouette(X_scaled, tuple(k_range), linkage)

            fig1, ax1 = plt.subplots(figsize=(6, 3.3))
            ax1.plot(list(k_range), sil_scores, marker="o", linewidth=2, color=PALETTE[1])
            ax1.set_xlabel("No. of Clusters")
            ax1.set_ylabel("Silhouette Score")
            ax1.set_title("Hierarchical Clustering — Score by k", fontweight="bold")
            fig_col, _ = st.columns([2, 1])
            with fig_col:
                st.pyplot(fig1, use_container_width=True)

            if cluster_mode == "Manual (khud select karo)":
                n_clusters = st.slider("Number of clusters chuno", min_value=2, max_value=max_k, value=min(3, max_k))
            else:
                best_idx = int(np.argmax(sil_scores))
                n_clusters = list(k_range)[best_idx]
                st.info(f"Auto-selected clusters: **{n_clusters}** (best silhouette score)")
            st.markdown('</div>', unsafe_allow_html=True)

            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            raw_labels = model.fit_predict(X_scaled)
            algo_summary = f"Agglomerative ({linkage} linkage) with k = {n_clusters}"

        # ============================================================
        # MODEL 3: DBSCAN
        # ============================================================
        elif model_choice == "DBSCAN (Density-based)":
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("⚙️ DBSCAN Parameters")
            st.caption("DBSCAN khud decide karta hai kitne clusters banane hain — density ke aadhar par. "
                       "Bahar ke points ko 'Noise / Outlier' maana jaata hai.")
            col_a, col_b = st.columns(2)
            with col_a:
                eps = st.slider("eps (neighbourhood radius)", min_value=0.1, max_value=3.0, value=0.5, step=0.05)
            with col_b:
                min_samples = st.slider("min_samples", min_value=2, max_value=max(2, min(20, n_samples - 1)), value=5)
            st.markdown('</div>', unsafe_allow_html=True)

            model = DBSCAN(eps=eps, min_samples=min_samples)
            raw_labels = model.fit_predict(X_scaled)
            n_found = len(set(raw_labels) - {-1})
            n_noise = int((raw_labels == -1).sum())
            st.info(f"DBSCAN ne **{n_found} clusters** dhoonde, aur **{n_noise} points** ko noise/outlier maana.")
            algo_summary = f"DBSCAN (eps={eps}, min_samples={min_samples}) → {n_found} clusters found"

        # ============================================================
        # MODEL 4: GAUSSIAN MIXTURE MODEL (GMM)
        # ============================================================
        else:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📉 BIC Score by Component Count")
            cluster_mode = st.radio(
                "Component (segment) count kaise decide karein?",
                ["Manual (khud select karo)", "Auto (sabse kam BIC se)"]
            )

            k_range = range(1, max_k + 1)
            bic_scores = compute_gmm_bic(X_scaled, tuple(k_range))

            fig1, ax1 = plt.subplots(figsize=(6, 3.3))
            ax1.plot(list(k_range), bic_scores, marker="o", linewidth=2, color=PALETTE[2])
            ax1.set_xlabel("No. of Components")
            ax1.set_ylabel("BIC (lower is better)")
            ax1.set_title("Gaussian Mixture — BIC Curve", fontweight="bold")
            fig_col, _ = st.columns([2, 1])
            with fig_col:
                st.pyplot(fig1, use_container_width=True)

            if cluster_mode == "Manual (khud select karo)":
                n_clusters = st.slider("Number of segments chuno", min_value=2, max_value=max_k, value=min(3, max_k))
            else:
                n_clusters = list(k_range)[int(np.argmin(bic_scores))]
                n_clusters = max(2, n_clusters)
                st.info(f"Auto-selected segments: **{n_clusters}** (lowest BIC)")
            st.markdown('</div>', unsafe_allow_html=True)

            model = GaussianMixture(n_components=n_clusters, random_state=42)
            raw_labels = model.fit_predict(X_scaled)
            algo_summary = f"Gaussian Mixture with {n_clusters} components"

        # ============================================================
        # SHARED DASHBOARD — scales automatically with however much data
        # was uploaded (more rows/columns → richer profile table & charts)
        # ============================================================
        cluster_ids = sorted(set(raw_labels))
        label_map = rank_labels_to_segments(cluster_ids, raw_labels)
        segment_series = pd.Series(raw_labels, index=X.index).map(label_map)

        result_df = df.loc[X.index].copy()
        result_df["Cluster"] = raw_labels
        result_df["Segment"] = segment_series.values

        sil = safe_silhouette(np.array(raw_labels))
        n_segments_found = len([c for c in cluster_ids if c != -1])

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📊 Segmentation Dashboard")
        st.caption(f"Model used: **{algo_summary}**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Customers Segmented", f"{len(result_df)}")
        m2.metric("Segments Found", n_segments_found)
        m3.metric("Silhouette Score", f"{sil:.3f}" if sil is not None else "N/A")
        m4.metric("Features Used", len(selected_features))

        # ---- Main scatter (raw axes if 2 features, PCA projection otherwise) ----
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for i, seg in enumerate(pd.unique(segment_series)):
            mask = (segment_series == seg).values
            ax2.scatter(X_view[mask, 0], X_view[mask, 1], s=110, label=seg,
                        color=color_for(seg, i), edgecolor="white", linewidth=1.1, alpha=0.9)
        ax2.set_xlabel(view_x_label)
        ax2.set_ylabel(view_y_label)
        ax2.set_title("Customer Segments (After Clustering)", fontweight="bold")
        ax2.legend(title="Segment")
        fig_col2, fig_col3 = st.columns(2)
        with fig_col2:
            st.pyplot(fig2, use_container_width=True)

        # ---- Segment size distribution ----
        counts = segment_series.value_counts()
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        colors_pie = [color_for(seg, i) for i, seg in enumerate(counts.index)]
        ax3.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors_pie,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5}, textprops={"fontsize": 9})
        ax3.set_title("Segment Distribution", fontweight="bold")
        with fig_col3:
            st.pyplot(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Cluster profile table: mean of every selected feature per segment ----
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🧬 Segment Profile (feature-wise averages)")
        profile = result_df.groupby("Segment")[selected_features].mean().round(2)
        profile["Customer Count"] = result_df["Segment"].value_counts()
        profile["% of Total"] = (profile["Customer Count"] / len(result_df) * 100).round(1)
        st.dataframe(profile.style.background_gradient(cmap="PuBuGn", subset=selected_features), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Grouped bar chart comparing feature averages across segments ----
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📈 Feature Comparison Across Segments")
        segs_order = [s for s in profile.index]
        n_feat = len(selected_features)
        x_pos = np.arange(n_feat)
        width = min(0.8 / max(len(segs_order), 1), 0.25)
        fig4, ax4 = plt.subplots(figsize=(max(6, n_feat * 1.2), 4))
        for i, seg in enumerate(segs_order):
            vals = profile.loc[seg, selected_features].values
            ax4.bar(x_pos + i * width, vals, width=width, label=seg, color=color_for(seg, i), edgecolor="white")
        ax4.set_xticks(x_pos + width * (len(segs_order) - 1) / 2)
        ax4.set_xticklabels(selected_features, rotation=25, ha="right")
        ax4.set_ylabel("Average value")
        ax4.set_title("Segment-wise Feature Averages", fontweight="bold")
        ax4.legend(title="Segment")
        st.pyplot(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Final data + download ----
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📋 Final Data with Segments")
        st.dataframe(result_df, use_container_width=True)
        csv_download = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Result CSV", csv_download, "segmented_customers.csv", "text/csv")
        st.markdown('</div>', unsafe_allow_html=True)