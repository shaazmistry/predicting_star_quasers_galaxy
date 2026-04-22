import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AstroClassifier",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

/* ── BACKGROUND: deep purple nebula with blue patches ── */
.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 15% 25%, rgba(59,7,120,0.75) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 70%, rgba(29,50,180,0.6) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 55% 10%, rgba(100,20,180,0.45) 0%, transparent 50%),
        radial-gradient(ellipse 70% 55% at 30% 80%, rgba(20,60,200,0.4) 0%, transparent 55%),
        radial-gradient(ellipse 40% 35% at 90% 20%, rgba(80,0,160,0.5) 0%, transparent 50%),
        linear-gradient(135deg, #0d0020 0%, #0a001a 40%, #050015 100%);
    background-attachment: fixed;
}

/* Starfield overlay */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(1px 1px at 8%  18%, rgba(255,255,255,0.9) 0%, transparent 100%),
        radial-gradient(1px 1px at 22% 55%, rgba(200,180,255,0.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 45% 12%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 67% 78%, rgba(180,200,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 83% 35%, rgba(255,255,255,0.9) 0%, transparent 100%),
        radial-gradient(1px 1px at 91% 62%, rgba(220,200,255,0.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 13% 88%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 37% 43%, rgba(200,210,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 72% 22%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 55% 91%, rgba(190,170,255,0.5) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 30% 5%,  rgba(255,255,255,0.95) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 78% 95%, rgba(210,200,255,0.8) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* Global font */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    color: #e2d4f8;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.08em;
    color: #e2d4f8;
}

/* Main title */
.main-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #c084fc 0%, #a78bfa 35%, #818cf8 65%, #e0d7ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    letter-spacing: 0.12em;
    margin-bottom: 0.2em;
}

.subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    color: #a78bfa;
    text-align: center;
    letter-spacing: 0.2em;
    margin-bottom: 2rem;
    text-transform: uppercase;
}

/* ── TRANSPARENT GLASS PANELS ── */
/* Main content area blocks */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] {
    background: rgba(30, 5, 60, 0.35) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 14px;
    border: 1px solid rgba(167,139,250,0.2);
}

/* Tab content panels */
[data-testid="stTabsContent"] {
    background: rgba(20, 4, 50, 0.4) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 0 12px 12px 12px;
    border: 1px solid rgba(167,139,250,0.18);
    padding: 1rem;
}

/* Metric / custom cards */
.metric-card {
    background: rgba(50, 10, 100, 0.35);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 24px rgba(120,50,220,0.12), inset 0 1px 0 rgba(255,255,255,0.06);
}

/* Prediction result box */
.prediction-box {
    background: rgba(60, 10, 120, 0.45);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 2px solid rgba(192,132,252,0.5);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(167,139,250,0.2), 0 0 80px rgba(120,50,220,0.1);
    margin: 1rem 0;
}

.pred-class {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: 0.15em;
}

.pred-class.STAR   { color: #fde68a; text-shadow: 0 0 20px rgba(253,230,138,0.7); }
.pred-class.GALAXY { color: #c084fc; text-shadow: 0 0 20px rgba(192,132,252,0.7); }
.pred-class.QSO    { color: #67e8f9; text-shadow: 0 0 20px rgba(103,232,249,0.7); }

/* Sidebar — frosted purple glass */
[data-testid="stSidebar"] {
    background: rgba(15, 3, 40, 0.7) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid rgba(167,139,250,0.2) !important;
}
[data-testid="stSidebar"] * { color: #d4c0f8 !important; }

/* Tabs bar */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(30, 5, 65, 0.55);
    backdrop-filter: blur(12px);
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
    border: 1px solid rgba(167,139,250,0.15);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.07em;
    color: #9d77e0;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(192,132,252,0.3), rgba(129,140,248,0.2)) !important;
    color: #c084fc !important;
    border: 1px solid rgba(192,132,252,0.35) !important;
}

/* Input labels */
.stSlider label, .stNumberInput label, .stSelectbox label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9rem;
    color: #b39ddb;
    letter-spacing: 0.05em;
}

/* Input fields — glassy */
input[type="number"], .stSelectbox > div > div {
    background: rgba(40, 8, 80, 0.5) !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 8px !important;
    color: #e2d4f8 !important;
    backdrop-filter: blur(8px);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(109,40,217,0.75), rgba(79,20,180,0.8));
    backdrop-filter: blur(8px);
    color: #f3e8ff;
    border: 1px solid rgba(192,132,252,0.5);
    border-radius: 8px;
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    padding: 0.6rem 2rem;
    transition: all 0.25s;
    box-shadow: 0 4px 18px rgba(109,40,217,0.4);
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(139,60,247,0.85), rgba(109,40,217,0.9));
    box-shadow: 0 4px 28px rgba(192,132,252,0.45);
    border-color: rgba(216,180,254,0.7);
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: rgba(25, 5, 55, 0.45) !important;
    backdrop-filter: blur(10px);
    border-radius: 10px;
    border: 1px solid rgba(167,139,250,0.2);
}

/* Divider */
hr { border-color: rgba(167,139,250,0.2) !important; }

/* Section labels */
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    color: #c084fc;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    opacity: 0.9;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(10,0,25,0.8); }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.35); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(192,132,252,0.55); }
</style>
""", unsafe_allow_html=True)


# ─── DATA + MODEL LOADING ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    path = kagglehub.dataset_download("pratikjadhav31/skyservercsv")
    df = pd.read_csv(os.path.join(path, "skyserver.csv"))
    df.drop(['objid', 'specobjid'], axis=1, inplace=True)
    return df

@st.cache_resource(show_spinner=False)
def train_models(df):
    le = LabelEncoder()
    df = df.copy()
    df['class'] = le.fit_transform(df['class'])
    class_names = list(le.classes_)

    x = df.drop('class', axis=1)
    y = df['class']

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.3, random_state=120)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_leaf_nodes=15, max_depth=3),
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(n_neighbors=3),
    }

    results = {}
    for name, m in models.items():
        m.fit(x_train, y_train)
        preds = m.predict(x_test)
        results[name] = {
            "model": m,
            "accuracy": round(accuracy_score(y_test, preds) * 100, 2),
            "preds": preds,
        }

    return results, scaler, class_names, x_test, y_test, df

# ─── LOAD ─────────────────────────────────────────────────────────────────────
with st.spinner("🔭 Syncing with the cosmos..."):
    df_raw = load_data()
    model_results, scaler, class_names, x_test, y_test, df_enc = train_models(df_raw)

features = [c for c in df_raw.columns if c != 'class']

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">✦ ASTROCLASSIFIER ✦</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Stars · Galaxies · Quasars — Sloan Digital Sky Survey</div>', unsafe_allow_html=True)
st.markdown("---")

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">⚙ Configuration</div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox("Active Model", list(model_results.keys()), index=2)
    st.markdown("---")

    st.markdown('<div class="section-label">📡 Dataset Stats</div>', unsafe_allow_html=True)
    total = len(df_raw)
    class_counts = df_raw['class'].value_counts()
    for cls in class_counts.index:
        pct = class_counts[cls] / total * 100
        icon = "⭐" if cls == "STAR" else ("🌌" if cls == "GALAXY" else "💫")
        st.markdown(f"**{icon} {cls}** — {class_counts[cls]:,} ({pct:.1f}%)")

    st.markdown("---")
    st.markdown('<div class="section-label">🏆 Model Scores</div>', unsafe_allow_html=True)
    for name, res in model_results.items():
        bar = "█" * int(res['accuracy'] / 5)
        st.markdown(f"`{name[:3]}` **{res['accuracy']}%**")

    st.markdown("---")
    st.markdown('<div class="section-label">ℹ About</div>', unsafe_allow_html=True)
    st.caption("Classifies celestial objects using photometric survey data (u,g,r,i,z bands + redshift) from SDSS SkyServer.")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔭  PREDICT",
    "📊  CLASS DIST",
    "🔥  CORRELATION",
    "🏆  MODEL ACCURACY",
    "🌌  3D EXPLORER",
    "🎨  OBJECT VISUALIZER",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Photometric Parameters")
    st.caption("Adjust the sliders or type values for each SDSS band feature + redshift.")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        feat_cols = st.columns(2)
        user_inputs = {}
        for i, feat in enumerate(features):
            col = feat_cols[i % 2]
            min_val = float(df_raw[feat].min())
            max_val = float(df_raw[feat].max())
            mean_val = float(df_raw[feat].mean())
            user_inputs[feat] = col.number_input(
                label=feat.upper(),
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                format="%.4f",
                step=(max_val - min_val) / 1000,
            )

        predict_btn = st.button("⟡  CLASSIFY OBJECT", use_container_width=True)

    with col_right:
        if predict_btn:
            model = model_results[selected_model_name]["model"]
            input_arr = np.array([[user_inputs[f] for f in features]])
            input_scaled = scaler.transform(input_arr)
            pred_idx = model.predict(input_scaled)[0]
            pred_label = class_names[pred_idx]

            # Confidence via predict_proba if available
            conf_str = ""
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_scaled)[0]
                conf_str = f"{proba[pred_idx]*100:.1f}% confidence"

            icons = {"STAR": "⭐", "GALAXY": "🌌", "QSO": "💫"}
            descriptions = {
                "STAR": "A self-luminous ball of plasma held together by gravity. Identified by sharp spectral lines and near-zero redshift.",
                "GALAXY": "A massive gravitationally bound system of stars, gas, dust, and dark matter containing billions of stellar objects.",
                "QSO": "A quasi-stellar object — an extremely luminous active galactic nucleus powered by a supermassive black hole."
            }

            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size:3.5rem; margin-bottom:0.5rem;">{icons.get(pred_label,'🔭')}</div>
                <div class="pred-class {pred_label}">{pred_label}</div>
                <div style="font-size:1rem; color:#5ba3c9; margin:0.6rem 0; font-family:Rajdhani; letter-spacing:0.1em;">
                    {conf_str}
                </div>
                <hr style="border-color:rgba(79,195,247,0.15); margin:1rem 0;">
                <div style="font-size:0.95rem; color:#8fb8d0; line-height:1.6; font-family:Rajdhani;">
                    {descriptions[pred_label]}
                </div>
                <div style="margin-top:1rem; font-size:0.8rem; color:#3a6a88; font-family:Orbitron; letter-spacing:0.1em;">
                    MODEL: {selected_model_name.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show all model predictions side by side
            st.markdown("##### All Models")
            mcols = st.columns(3)
            for idx, (mname, mres) in enumerate(model_results.items()):
                m = mres["model"]
                p = class_names[m.predict(input_scaled)[0]]
                mcols[idx].markdown(f"""
                <div class="metric-card" style="text-align:center;">
                    <div style="font-size:0.65rem;font-family:Orbitron;color:#4fc3f7;letter-spacing:0.1em;">{mname}</div>
                    <div style="font-size:1.4rem;margin-top:0.3rem;">{icons.get(p,'🔭')}</div>
                    <div style="font-family:Orbitron;font-size:0.9rem;color:#c8d8f0;">{p}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem 1rem; color:#2a5a7a;">
                <div style="font-size:4rem; margin-bottom:1rem; opacity:0.5;">🔭</div>
                <div style="font-family:Orbitron; font-size:0.85rem; letter-spacing:0.15em; color:#2a5a7a;">
                    AWAITING INPUT<br>
                    <span style="font-size:0.7rem; opacity:0.6;">Adjust parameters and click CLASSIFY</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — CLASS DISTRIBUTION
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Class Distribution")
    c1, c2 = st.columns(2)

    with c1:
        counts = df_raw['class'].value_counts().reset_index()
        counts.columns = ['Class', 'Count']
        colors = ['#fff176', '#ce93d8', '#80cbc4']
        fig_bar = px.bar(
            counts, x='Class', y='Count', color='Class',
            color_discrete_sequence=colors,
            template='plotly_dark',
            title='Object Count by Class'
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(5,15,40,0.8)',
            paper_bgcolor='rgba(5,15,40,0)',
            font=dict(family='Rajdhani', color='#c8d8f0'),
            title_font=dict(family='Orbitron', size=14),
            showlegend=False,
        )
        fig_bar.update_traces(marker_line_color='rgba(79,195,247,0.3)', marker_line_width=1)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        fig_pie = px.pie(
            counts, names='Class', values='Count',
            color_discrete_sequence=colors,
            template='plotly_dark',
            title='Class Proportions',
            hole=0.45,
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(5,15,40,0)',
            paper_bgcolor='rgba(5,15,40,0)',
            font=dict(family='Rajdhani', color='#c8d8f0'),
            title_font=dict(family='Orbitron', size=14),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Per-class feature distributions
    st.markdown("### Feature Distribution by Class")
    feat_choice = st.selectbox("Select feature", features, key="dist_feat")
    fig_hist = px.histogram(
        df_raw, x=feat_choice, color='class',
        color_discrete_map={"STAR": "#fff176", "GALAXY": "#ce93d8", "QSO": "#80cbc4"},
        template='plotly_dark', barmode='overlay', nbins=80,
        title=f'{feat_choice.upper()} Distribution per Class',
        opacity=0.75,
    )
    fig_hist.update_layout(
        plot_bgcolor='rgba(5,15,40,0.8)',
        paper_bgcolor='rgba(5,15,40,0)',
        font=dict(family='Rajdhani', color='#c8d8f0'),
        title_font=dict(family='Orbitron', size=14),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 3 — CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Feature Correlation Heatmap")

    numeric_df = df_enc[features + ['class']]
    corr = numeric_df.corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[
            [0.0, '#0a1628'],
            [0.3, '#1565c0'],
            [0.5, '#01579b'],
            [0.7, '#4fc3f7'],
            [1.0, '#e1f5fe'],
        ],
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 11, "family": "Rajdhani"},
        hoverongaps=False,
        colorbar=dict(
            tickfont=dict(family='Rajdhani', color='#c8d8f0'),
            title=dict(text="r", font=dict(family='Orbitron', color='#4fc3f7'))
        )
    ))
    fig_corr.update_layout(
        plot_bgcolor='rgba(5,15,40,0.8)',
        paper_bgcolor='rgba(5,15,40,0)',
        font=dict(family='Rajdhani', color='#c8d8f0'),
        title=dict(text='Pearson Correlation Matrix', font=dict(family='Orbitron', size=14, color='#4fc3f7')),
        height=500,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Top Correlated Feature Pairs")
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            pairs.append({
                "Feature A": corr.columns[i],
                "Feature B": corr.columns[j],
                "Correlation": round(corr.iloc[i, j], 4)
            })
    pairs_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False).head(10)
    st.dataframe(
        pairs_df.style.background_gradient(subset=["Correlation"], cmap="Blues"),
        use_container_width=True,
        hide_index=True
    )

# ════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL ACCURACY
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Model Accuracy Comparison")

    model_names = list(model_results.keys())
    accuracies = [model_results[m]["accuracy"] for m in model_names]

    fig_acc = go.Figure()
    bar_colors = ['#4fc3f7', '#81d4fa', '#b3e5fc']
    for i, (name, acc) in enumerate(zip(model_names, accuracies)):
        fig_acc.add_trace(go.Bar(
            x=[name], y=[acc],
            marker_color=bar_colors[i],
            marker_line_color='rgba(79,195,247,0.4)',
            marker_line_width=1,
            text=[f"{acc}%"],
            textposition='outside',
            textfont=dict(family='Orbitron', size=13, color='#e1f5fe'),
            name=name,
        ))
    fig_acc.add_hline(y=90, line_dash="dot", line_color="rgba(255,241,118,0.4)",
                      annotation_text="90% threshold", annotation_font_color="#fff176")
    fig_acc.update_layout(
        plot_bgcolor='rgba(5,15,40,0.8)',
        paper_bgcolor='rgba(5,15,40,0)',
        font=dict(family='Rajdhani', color='#c8d8f0'),
        title=dict(text='Test Set Accuracy (%)', font=dict(family='Orbitron', size=14, color='#4fc3f7')),
        yaxis=dict(range=[0, 105], gridcolor='rgba(79,195,247,0.1)', ticksuffix="%"),
        xaxis=dict(gridcolor='rgba(79,195,247,0.05)'),
        showlegend=False,
        height=400,
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # Per-class F1 scores
    st.markdown("### Per-Class F1 Score Breakdown")
    sel_model = st.selectbox("Model", model_names, key="f1_model")
    preds = model_results[sel_model]["preds"]
    report = classification_report(y_test, preds, target_names=class_names, output_dict=True)

    f1_data = {cls: report[cls]['f1-score'] for cls in class_names}
    f1_fig = go.Figure(go.Bar(
        x=list(f1_data.keys()),
        y=[v*100 for v in f1_data.values()],
        marker_color=['#fff176', '#ce93d8', '#80cbc4'],
        marker_line_color='rgba(79,195,247,0.3)',
        marker_line_width=1,
        text=[f"{v*100:.1f}%" for v in f1_data.values()],
        textposition='outside',
        textfont=dict(family='Orbitron', size=12),
    ))
    f1_fig.update_layout(
        plot_bgcolor='rgba(5,15,40,0.8)',
        paper_bgcolor='rgba(5,15,40,0)',
        font=dict(family='Rajdhani', color='#c8d8f0'),
        title=dict(text=f'F1 Score by Class — {sel_model}', font=dict(family='Orbitron', size=13, color='#4fc3f7')),
        yaxis=dict(range=[0, 110], gridcolor='rgba(79,195,247,0.1)', ticksuffix="%"),
        height=360,
    )
    st.plotly_chart(f1_fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 5 — 3D EXPLORER
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 3D Feature Space Explorer")
    st.caption("Visualize objects in 3D photometric space. Drag to rotate, scroll to zoom.")

    feat_options = features
    c1, c2, c3 = st.columns(3)
    x_axis = c1.selectbox("X Axis", feat_options, index=feat_options.index('g') if 'g' in feat_options else 0)
    y_axis = c2.selectbox("Y Axis", feat_options, index=feat_options.index('r') if 'r' in feat_options else 1)
    z_axis = c3.selectbox("Z Axis", feat_options, index=feat_options.index('redshift') if 'redshift' in feat_options else 2)

    sample_n = st.slider("Sample size (larger = slower)", 500, min(5000, len(df_raw)), 2000, step=500)
    df_sample = df_raw.sample(n=sample_n, random_state=42)

    color_map = {"STAR": "#fff176", "GALAXY": "#ce93d8", "QSO": "#80cbc4"}
    fig_3d = px.scatter_3d(
        df_sample,
        x=x_axis, y=y_axis, z=z_axis,
        color='class',
        color_discrete_map=color_map,
        opacity=0.65,
        size_max=5,
        template='plotly_dark',
        title=f'3D: {x_axis.upper()} × {y_axis.upper()} × {z_axis.upper()}',
        labels={x_axis: x_axis.upper(), y_axis: y_axis.upper(), z_axis: z_axis.upper()}
    )
    fig_3d.update_traces(marker=dict(size=2.5))
    fig_3d.update_layout(
        paper_bgcolor='rgba(5,15,40,0)',
        font=dict(family='Rajdhani', color='#c8d8f0'),
        title_font=dict(family='Orbitron', size=14, color='#4fc3f7'),
        scene=dict(
            bgcolor='rgba(5,15,40,0.85)',
            xaxis=dict(gridcolor='rgba(79,195,247,0.15)', color='#5ba3c9'),
            yaxis=dict(gridcolor='rgba(79,195,247,0.15)', color='#5ba3c9'),
            zaxis=dict(gridcolor='rgba(79,195,247,0.15)', color='#5ba3c9'),
        ),
        legend=dict(font=dict(family='Orbitron', size=11)),
        height=580,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("### 2D Projection")
    c1, c2 = st.columns(2)
    px_feat = c1.selectbox("X", feat_options, index=0, key="px")
    py_feat = c2.selectbox("Y", feat_options, index=2, key="py")
    fig_2d = px.scatter(
        df_sample, x=px_feat, y=py_feat, color='class',
        color_discrete_map=color_map,
        opacity=0.6,
        template='plotly_dark',
        title=f'{px_feat.upper()} vs {py_feat.upper()}'
    )
    fig_2d.update_traces(marker=dict(size=3))
    fig_2d.update_layout(
        plot_bgcolor='rgba(5,15,40,0.8)',
        paper_bgcolor='rgba(5,15,40,0)',
        font=dict(family='Rajdhani', color='#c8d8f0'),
        title_font=dict(family='Orbitron', size=13, color='#4fc3f7'),
    )
    st.plotly_chart(fig_2d, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 6 — OBJECT VISUALIZER
# ════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 🎨 Astronomical Object Visualizer")
    st.caption("Uses photometric band ratios & redshift to procedurally render how the object would appear. Not a real image — a data-driven artistic reconstruction.")

    col_v1, col_v2 = st.columns([1, 2])

    with col_v1:
        st.markdown("##### Input Parameters")
        vis_inputs = {}
        for feat in features:
            mn = float(df_raw[feat].min())
            mx = float(df_raw[feat].max())
            mv = float(df_raw[feat].mean())
            vis_inputs[feat] = st.number_input(
                feat.upper(), min_value=mn, max_value=mx,
                value=mv, format="%.4f", key=f"vis_{feat}"
            )

        vis_btn = st.button("✦ RENDER OBJECT", use_container_width=True, key="vis_btn")

    with col_v2:
        if vis_btn:
            # Predict class from inputs
            model = model_results[selected_model_name]["model"]
            inp = np.array([[vis_inputs[f] for f in features]])
            inp_scaled = scaler.transform(inp)
            pred_idx = model.predict(inp_scaled)[0]
            pred_label = class_names[pred_idx]

            # Derive color from band values (u=ultraviolet, g=green, r=red, i=infrared, z=near-IR)
            u = vis_inputs.get('u', 20)
            g = vis_inputs.get('g', 20)
            r = vis_inputs.get('r', 20)
            i_band = vis_inputs.get('i', 20)
            z_band = vis_inputs.get('z', 20)
            redshift = vis_inputs.get('redshift', 0.0)

            # Normalize bands to 0-1 for color mapping
            def norm_band(val, lo=13, hi=25):
                return max(0.0, min(1.0, (hi - val) / (hi - lo)))

            rn = norm_band(r)
            gn = norm_band(g)
            bn = norm_band(u)

            # Redshift nudges color warmer
            rn = min(1.0, rn + redshift * 0.3)
            bn = max(0.0, bn - redshift * 0.2)

            r_hex = int(rn * 255)
            g_hex = int(gn * 200)
            b_hex = int(bn * 255)

            brightness = norm_band(r, 10, 26)
            size = 80 + brightness * 120

            color_main = f"rgb({r_hex},{g_hex},{b_hex})"
            color_glow = f"rgba({r_hex},{g_hex},{b_hex},0.25)"
            color_core = f"rgba({min(255,r_hex+80)},{min(255,g_hex+80)},{min(255,b_hex+80)},0.9)"

            # Build description
            color_temp = "cool red giant" if r_hex > g_hex and r_hex > b_hex else \
                         "hot blue-white" if b_hex > r_hex else "yellow main-sequence"

            descriptions = {
                "STAR": f"A {color_temp} star. Spectral bands suggest {'high' if brightness > 0.6 else 'moderate'} luminosity with redshift z={redshift:.4f}.",
                "GALAXY": f"{'Elliptical' if g_hex > r_hex else 'Spiral'} galaxy at z={redshift:.4f}. Band ratios suggest {'older stellar population' if r_hex > b_hex else 'active star formation'}.",
                "QSO": f"Quasi-stellar object at z={redshift:.4f}. Extreme luminosity with {'broad emission lines implied by blue excess' if b_hex > r_hex else 'redshifted emission spectrum'}.",
            }

            icons = {"STAR": "⭐", "GALAXY": "🌌", "QSO": "💫"}

            # Build Three.js 3D interactive visualizer
            # TYPE is ALWAYS from the ML prediction, not from band colors
            js_type    = pred_label          # "STAR" | "GALAXY" | "QSO"
            is_spiral  = g_hex > r_hex       # only relevant when GALAXY
            arm_count  = 4 if b_hex > r_hex else 2

            canvas_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ background:#0a0015; font-family:'Courier New',monospace; width:100%; height:100%; overflow:hidden; }}

  #wrap {{ position:relative; width:100%; height:100%; }}

  /* canvas always fills whatever height the iframe has */
  #c {{ display:block; width:100%; height:calc(100% - 0px); cursor:grab; }}
  #c:active {{ cursor:grabbing; }}

  /* when fullscreen class active, hide databar space and give canvas full height */
  body.normal #c   {{ height:calc(100% - 90px); }}
  body.normal #databar {{ display:flex; }}
  body.full   #c   {{ height:100%; }}
  body.full   #databar {{ display:none; }}

  #controls {{
    position:absolute; top:10px; left:12px; z-index:20;
  }}
  #fsBtn {{
    background:rgba(80,20,140,0.7); border:1px solid rgba(167,139,250,0.45);
    color:#c084fc; border-radius:6px; padding:5px 13px;
    font-size:0.62rem; letter-spacing:0.1em; cursor:pointer;
    font-family:'Courier New',monospace; transition:background 0.2s;
  }}
  #fsBtn:hover {{ background:rgba(130,50,220,0.85); color:#f3e8ff; }}

  #hint {{ position:absolute; top:10px; right:12px; font-size:0.58rem;
           color:rgba(167,139,250,0.4); letter-spacing:0.07em;
           pointer-events:none; z-index:10; }}

  #overlay {{
    position:absolute; left:0; right:0;
    background:linear-gradient(transparent, rgba(8,0,22,0.94));
    padding:14px 18px 10px; pointer-events:none; z-index:5;
    bottom:90px;
  }}
  body.full #overlay {{ bottom:0; }}

  #label {{
    font-size:1.1rem; font-weight:bold; letter-spacing:0.18em;
    color:{color_main}; text-shadow:0 0 14px {color_main};
  }}
  #desc {{ font-size:0.66rem; color:#b39ddb; margin-top:3px; line-height:1.5; }}

  #databar {{
    display:none; flex-direction:column;
    background:rgba(8,0,22,0.97);
    border-top:1px solid rgba(167,139,250,0.2);
    padding:8px 18px 10px; gap:5px;
    position:absolute; bottom:0; left:0; right:0; height:90px; z-index:10;
  }}
  .dtitle {{ font-size:0.55rem; color:#6b4fa0; letter-spacing:0.18em; text-transform:uppercase; }}
  .drow   {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:2px; }}
  .band   {{
    font-size:0.6rem; padding:3px 10px; border-radius:5px;
    border:1px solid rgba(167,139,250,0.22); background:rgba(30,5,60,0.6);
  }}
</style>
</head>
<body class="normal">
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="controls">
    <button id="fsBtn" onclick="toggleFS()">⛶ FULLSCREEN</button>
  </div>
  <div id="hint">🖱 DRAG · SCROLL · RIGHT-DRAG PAN</div>
  <div id="overlay">
    <div id="label">{icons.get(pred_label,'')} {pred_label}</div>
    <div id="desc">{descriptions[pred_label]}</div>
  </div>
  <div id="databar">
    <div class="dtitle">📡 Photometric Data</div>
    <div class="drow">
      <span class="band" style="color:#c084fc">u = {u:.4f}</span>
      <span class="band" style="color:#86efac">g = {g:.4f}</span>
      <span class="band" style="color:#fca5a5">r = {r:.4f}</span>
      <span class="band" style="color:#fdba74">i = {i_band:.4f}</span>
      <span class="band" style="color:#e2e8f0">z = {z_band:.4f}</span>
      <span class="band" style="color:#67e8f9">redshift = {redshift:.4f}</span>
      <span class="band" style="color:#a5f3fc">brightness = {brightness:.3f}</span>
      <span class="band" style="color:#fde68a">class = {pred_label}</span>
      <span class="band" style="color:#d8b4fe">model = {selected_model_name}</span>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ── FULLSCREEN via iframe resize ──────────────────────────
// Streamlit iframes can't use position:fixed to escape, but we CAN
// resize the iframe element from inside using window.frameElement
const NORMAL_H = 710;   // matches st.components height
const FULL_H   = window.screen.height;

let isFull = false;
function toggleFS() {{
  isFull = !isFull;
  const btn  = document.getElementById('fsBtn');
  const body = document.body;

  if (isFull) {{
    body.className = 'full';
    btn.textContent = '✕ EXIT';
    // Resize the iframe itself so it takes over the page
    try {{
      if (window.frameElement) {{
        window.frameElement.style.height = FULL_H + 'px';
        window.frameElement.style.position = 'fixed';
        window.frameElement.style.top = '0';
        window.frameElement.style.left = '0';
        window.frameElement.style.width = '100vw';
        window.frameElement.style.zIndex = '99999';
      }}
    }} catch(e) {{}}
  }} else {{
    body.className = 'normal';
    btn.textContent = '⛶ FULLSCREEN';
    try {{
      if (window.frameElement) {{
        window.frameElement.style.height = NORMAL_H + 'px';
        window.frameElement.style.position = '';
        window.frameElement.style.top = '';
        window.frameElement.style.left = '';
        window.frameElement.style.width = '';
        window.frameElement.style.zIndex = '';
      }}
    }} catch(e) {{}}
  }}
  setTimeout(onResize, 50);
}}

// ── SETUP ──────────────────────────────────────────────────
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{ canvas, antialias:true, alpha:true }});
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.setClearColor(0x000000, 0);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 2000);
camera.position.set(0, 30, 90);

// ── DATA-DRIVEN PARAMS ─────────────────────────────────────
// TYPE comes from the ML model prediction — NEVER from band colors
const TYPE       = "{js_type}";
const CR         = {r_hex}/255;
const CG         = {g_hex}/255;
const CB         = {b_hex}/255;
const BRIGHTNESS = {brightness:.3f};
const REDSHIFT   = {redshift:.4f};
const IS_SPIRAL  = {str(is_spiral).lower()};
const ARMS       = {arm_count};
const mainColor  = new THREE.Color(CR, CG, CB);
const coreColor  = new THREE.Color(
  Math.min(1,CR+0.35), Math.min(1,CG+0.25), Math.min(1,CB+0.35));

// ── ORBIT CONTROLS (manual) ────────────────────────────────
let isDragging=false, isRightDrag=false;
let prevMouse={{x:0,y:0}};
let spherical={{theta:0.3, phi:1.1, r:90}};
let panOffset={{x:0,y:0}};

canvas.addEventListener('mousedown', e=>{{
  isDragging=true;
  isRightDrag=(e.button===2);
  prevMouse={{x:e.clientX,y:e.clientY}};
}});
canvas.addEventListener('contextmenu', e=>e.preventDefault());
window.addEventListener('mouseup', ()=>isDragging=false);
window.addEventListener('mousemove', e=>{{
  if(!isDragging) return;
  const dx=(e.clientX-prevMouse.x)*0.005;
  const dy=(e.clientY-prevMouse.y)*0.005;
  if(isRightDrag){{ panOffset.x-=dx*30; panOffset.y+=dy*30; }}
  else {{ spherical.theta-=dx; spherical.phi=Math.max(0.1,Math.min(Math.PI-0.1,spherical.phi+dy)); }}
  prevMouse={{x:e.clientX,y:e.clientY}};
}});
canvas.addEventListener('wheel', e=>{{
  spherical.r=Math.max(10,Math.min(400,spherical.r+e.deltaY*0.08));
  e.preventDefault();
}}, {{passive:false}});

let lastTouch=null;
canvas.addEventListener('touchstart', e=>{{ lastTouch={{x:e.touches[0].clientX,y:e.touches[0].clientY}}; }});
canvas.addEventListener('touchmove', e=>{{
  if(!lastTouch) return;
  const dx=(e.touches[0].clientX-lastTouch.x)*0.007;
  const dy=(e.touches[0].clientY-lastTouch.y)*0.007;
  spherical.theta-=dx; spherical.phi=Math.max(0.1,Math.min(Math.PI-0.1,spherical.phi+dy));
  lastTouch={{x:e.touches[0].clientX,y:e.touches[0].clientY}};
  e.preventDefault();
}},{{passive:false}});

function updateCamera(){{
  const x=spherical.r*Math.sin(spherical.phi)*Math.sin(spherical.theta);
  const y=spherical.r*Math.cos(spherical.phi);
  const z=spherical.r*Math.sin(spherical.phi)*Math.cos(spherical.theta);
  camera.position.set(x+panOffset.x, y+panOffset.y, z);
  camera.lookAt(panOffset.x, panOffset.y, 0);
}}

// ── BACKGROUND STARS ───────────────────────────────────────
(function(){{
  const geo = new THREE.BufferGeometry();
  const pos=[], col=[];
  for(let i=0;i<4000;i++){{
    const r=300+Math.random()*700;
    const th=Math.random()*Math.PI*2;
    const ph=Math.acos(2*Math.random()-1);
    pos.push(r*Math.sin(ph)*Math.cos(th), r*Math.sin(ph)*Math.sin(th), r*Math.cos(ph));
    const t=Math.random();
    col.push(0.6+t*0.4, 0.6+t*0.35, 0.7+t*0.3);
  }}
  geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geo.setAttribute('color',  new THREE.Float32BufferAttribute(col,3));
  const mat=new THREE.PointsMaterial({{size:0.6,vertexColors:true,transparent:true,opacity:0.85}});
  scene.add(new THREE.Points(geo,mat));
}})();

// ── SPRITE HELPER ──────────────────────────────────────────
function makeGlow(size, color, opacity){{
  const c2=document.createElement('canvas'); c2.width=c2.height=64;
  const ctx=c2.getContext('2d');
  const g=ctx.createRadialGradient(32,32,0,32,32,32);
  const hex='#'+color.getHexString();
  g.addColorStop(0,'rgba(255,255,255,'+opacity+')');
  g.addColorStop(0.2,hex.replace('#','rgba(').replace(/(..)(..)(..)$/,(m,r,g,b)=>
    `${{parseInt(r,16)}},${{parseInt(g,16)}},${{parseInt(b,16)}},${{opacity*0.8}})`));
  g.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=g; ctx.fillRect(0,0,64,64);
  const tex=new THREE.CanvasTexture(c2);
  const mat=new THREE.SpriteMaterial({{map:tex,transparent:true,depthWrite:false,blending:THREE.AdditiveBlending}});
  const sprite=new THREE.Sprite(mat);
  sprite.scale.set(size,size,1);
  return sprite;
}}

// ── PARTICLE FIELD HELPER ──────────────────────────────────
function makeParticles(count, posFunc, colorFunc, size){{
  const geo=new THREE.BufferGeometry();
  const pos=[], col=[];
  for(let i=0;i<count;i++){{
    const p=posFunc(i,count); pos.push(...p);
    const c=colorFunc(i,count); col.push(...c);
  }}
  geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geo.setAttribute('color',  new THREE.Float32BufferAttribute(col,3));
  const mat=new THREE.PointsMaterial({{
    size, vertexColors:true, transparent:true, opacity:0.9,
    blending:THREE.AdditiveBlending, depthWrite:false
  }});
  return new THREE.Points(geo,mat);
}}

// ══════════════════════════════════════════════════════════
// STAR
// ══════════════════════════════════════════════════════════
const objects=[];
if(TYPE==='STAR'){{
  // Core sphere
  const coreGeo=new THREE.SphereGeometry(3+BRIGHTNESS*3,32,32);
  const coreMat=new THREE.MeshBasicMaterial({{color:coreColor}});
  const coreMesh=new THREE.Mesh(coreGeo,coreMat);
  scene.add(coreMesh);

  // Glow layers
  [40,28,18,10].forEach((s,i)=>{{
    const g=makeGlow(s,mainColor,0.18-i*0.03);
    scene.add(g);
  }});
  makeGlow(8,new THREE.Color(1,1,1),0.9); // white core sprite
  const wg=makeGlow(8,new THREE.Color(1,1,1),0.9); scene.add(wg);

  // Corona particles
  const corona=makeParticles(600,
    (i)=>{{
      const a=Math.random()*Math.PI*2, b=Math.acos(2*Math.random()-1);
      const r=5+Math.random()*12+BRIGHTNESS*6;
      return [r*Math.sin(b)*Math.cos(a), r*Math.sin(b)*Math.sin(a), r*Math.cos(b)];
    }},
    ()=>[Math.min(1,CR+0.4), Math.min(1,CG+0.3), Math.min(1,CB+0.4)],
    0.4
  );
  scene.add(corona);

  // Diffraction spikes (line segments)
  const spikeLen=28+BRIGHTNESS*18;
  [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0.7,0.7,0],[-0.7,-0.7,0],[0.7,-0.7,0],[-0.7,0.7,0]]
  .forEach(dir=>{{
    const pts=[new THREE.Vector3(0,0,0),
               new THREE.Vector3(dir[0]*spikeLen,dir[1]*spikeLen,dir[2]*spikeLen)];
    const lg=new THREE.BufferGeometry().setFromPoints(pts);
    const lm=new THREE.LineBasicMaterial({{
      color:mainColor,transparent:true,opacity:0.5,blending:THREE.AdditiveBlending
    }});
    const line=new THREE.Line(lg,lm);
    scene.add(line);
    objects.push({{type:'spike',line,dir,baseLen:spikeLen}});
  }});

  // Surface detail dots (sunspots region)
  const spots=makeParticles(80,
    ()=>{{
      const a=Math.random()*Math.PI*2, b=(Math.random()-0.5)*0.8;
      const r=3.1+BRIGHTNESS*3;
      return [r*Math.cos(b)*Math.cos(a), r*Math.sin(b), r*Math.cos(b)*Math.sin(a)];
    }},
    ()=>[0.05,0.02,0.05], 1.2
  );
  scene.add(spots);
  objects.push({{type:'star', corona, coreMesh}});
}}

// ══════════════════════════════════════════════════════════
// GALAXY
// ══════════════════════════════════════════════════════════
else if(TYPE==='GALAXY'){{
  const galaxyGroup=new THREE.Group();
  scene.add(galaxyGroup);
  objects.push({{type:'galaxyGroup',group:galaxyGroup}});

  if(IS_SPIRAL){{
    // Spiral arm particles
    const armPts=[], armCols=[];
    for(let arm=0;arm<ARMS;arm++){{
      const baseAngle=(arm/ARMS)*Math.PI*2;
      for(let p=0;p<2500;p++){{
        const pct=p/2500;
        const angle=baseAngle+pct*Math.PI*4;
        const dist=2+pct*38;
        const spread=(1-pct)*1.8+0.3;
        const x=(dist+((Math.random()-0.5)*spread))*Math.cos(angle);
        const z=(dist+((Math.random()-0.5)*spread))*Math.sin(angle);
        const y=(Math.random()-0.5)*spread*0.35;
        armPts.push(x,y,z);
        const fade=1-pct*0.7;
        const twinkle=0.8+Math.random()*0.2;
        armCols.push(CR*fade*twinkle+0.05, CG*fade*twinkle+0.03, CB*fade*twinkle+0.08);
      }}
    }}
    // Disk haze
    for(let i=0;i<3000;i++){{
      const r=Math.random()*42, a=Math.random()*Math.PI*2;
      const x=r*Math.cos(a), z=r*Math.sin(a), y=(Math.random()-0.5)*1.2;
      armPts.push(x,y,z);
      const fade=1-r/42;
      armCols.push(CR*fade*0.4, CG*fade*0.35, CB*fade*0.5);
    }}
    const armGeo=new THREE.BufferGeometry();
    armGeo.setAttribute('position',new THREE.Float32BufferAttribute(armPts,3));
    armGeo.setAttribute('color',   new THREE.Float32BufferAttribute(armCols,3));
    const armMat=new THREE.PointsMaterial({{
      size:0.45, vertexColors:true, transparent:true, opacity:0.85,
      blending:THREE.AdditiveBlending, depthWrite:false
    }});
    galaxyGroup.add(new THREE.Points(armGeo,armMat));

    // Dust lane ring
    const dustCurve=new THREE.EllipseCurve(0,0,20,5,0,Math.PI*2,false,0);
    const dustPts=dustCurve.getPoints(120);
    const dustGeo=new THREE.BufferGeometry().setFromPoints(
      dustPts.map(p=>new THREE.Vector3(p.x,0,p.y)));
    const dustMat=new THREE.LineBasicMaterial({{
      color:0x000000,transparent:true,opacity:0.45,blending:THREE.NormalBlending
    }});
    galaxyGroup.add(new THREE.Line(dustGeo,dustMat));

  }} else {{
    // Elliptical: concentric shells of stars
    const ePts=[], eCols=[];
    for(let i=0;i<12000;i++){{
      const r=Math.pow(Math.random(),0.5)*35;
      const a=Math.random()*Math.PI*2, b=Math.acos(2*Math.random()-1);
      const x=r*Math.sin(b)*Math.cos(a)*1.4;
      const y=r*Math.sin(b)*Math.sin(a)*0.65;
      const z=r*Math.cos(b);
      ePts.push(x,y,z);
      const fade=1-r/35;
      eCols.push(Math.min(1,CR*fade+0.12), Math.min(1,CG*fade*0.8+0.08), Math.min(1,CB*fade*0.5+0.04));
    }}
    const eGeo=new THREE.BufferGeometry();
    eGeo.setAttribute('position',new THREE.Float32BufferAttribute(ePts,3));
    eGeo.setAttribute('color',   new THREE.Float32BufferAttribute(eCols,3));
    const eMat=new THREE.PointsMaterial({{
      size:0.38, vertexColors:true, transparent:true, opacity:0.9,
      blending:THREE.AdditiveBlending, depthWrite:false
    }});
    galaxyGroup.add(new THREE.Points(eGeo,eMat));
  }}

  // Central bulge glow
  [16,10,6,3].forEach((s,i)=>{{
    const g=makeGlow(s,coreColor,0.55-i*0.1);
    galaxyGroup.add(g);
  }});
}}

// ══════════════════════════════════════════════════════════
// QSO / QUASAR
// ══════════════════════════════════════════════════════════
else {{
  const qGroup=new THREE.Group();
  scene.add(qGroup);
  objects.push({{type:'qGroup',group:qGroup}});

  // Accretion disk — flat torus of particles
  const diskPts=[], diskCols=[];
  for(let i=0;i<8000;i++){{
    const r=3+Math.pow(Math.random(),0.6)*28;
    const a=Math.random()*Math.PI*2;
    const h=(Math.random()-0.5)*Math.max(0.3,(1-r/30)*2.5);
    diskPts.push(r*Math.cos(a), h, r*Math.sin(a));
    const heat=1-r/31;
    diskCols.push(Math.min(1,CR+heat*0.5), Math.min(1,CG+heat*0.2), Math.min(1,CB*0.5+heat*0.1));
  }}
  const diskGeo=new THREE.BufferGeometry();
  diskGeo.setAttribute('position',new THREE.Float32BufferAttribute(diskPts,3));
  diskGeo.setAttribute('color',   new THREE.Float32BufferAttribute(diskCols,3));
  const diskMat=new THREE.PointsMaterial({{
    size:0.4, vertexColors:true, transparent:true, opacity:0.9,
    blending:THREE.AdditiveBlending, depthWrite:false
  }});
  qGroup.add(new THREE.Points(diskGeo,diskMat));

  // Relativistic jets — particle streams along Y axis
  ['up','down'].forEach(dir=>{{
    const jPts=[], jCols=[];
    const sign=dir==='up'?1:-1;
    for(let i=0;i<3000;i++){{
      const t=Math.pow(Math.random(),0.7);
      const h=sign*(3+t*70);
      const spread=(1-t)*1.8;
      const a=Math.random()*Math.PI*2;
      jPts.push(spread*Math.cos(a), h, spread*Math.sin(a));
      const fade=1-t*0.8;
      jCols.push(Math.min(1,CR*fade+0.2), Math.min(1,CG*fade+0.1), Math.min(1,CB*fade+0.3));
    }}
    const jGeo=new THREE.BufferGeometry();
    jGeo.setAttribute('position',new THREE.Float32BufferAttribute(jPts,3));
    jGeo.setAttribute('color',   new THREE.Float32BufferAttribute(jCols,3));
    const jMat=new THREE.PointsMaterial({{
      size:0.35, vertexColors:true, transparent:true, opacity:0.75,
      blending:THREE.AdditiveBlending, depthWrite:false
    }});
    qGroup.add(new THREE.Points(jGeo,jMat));
  }});

  // Core glows
  [20,12,7,3].forEach((s,i)=>{{
    const g=makeGlow(s,mainColor,0.6-i*0.1); qGroup.add(g);
  }});
  const wg=makeGlow(5,new THREE.Color(1,1,1),0.95); qGroup.add(wg);
}}

// ── AMBIENT NEBULA HAZE ───────────────────────────────────
const nebPts=[], nebCols=[];
for(let i=0;i<500;i++){{
  const r=60+Math.random()*80, a=Math.random()*Math.PI*2, b=Math.acos(2*Math.random()-1);
  nebPts.push(r*Math.sin(b)*Math.cos(a), r*Math.sin(b)*Math.sin(a), r*Math.cos(b));
  nebCols.push(CR*0.15+0.02, CG*0.1+0.01, CB*0.2+0.05);
}}
const nGeo=new THREE.BufferGeometry();
nGeo.setAttribute('position',new THREE.Float32BufferAttribute(nebPts,3));
nGeo.setAttribute('color',   new THREE.Float32BufferAttribute(nebCols,3));
scene.add(new THREE.Points(nGeo,new THREE.PointsMaterial({{
  size:2.5,vertexColors:true,transparent:true,opacity:0.3,
  blending:THREE.AdditiveBlending,depthWrite:false
}})));

// ── ANIMATE ───────────────────────────────────────────────
let t=0;
function animate(){{
  requestAnimationFrame(animate);
  t+=0.008;

  objects.forEach(o=>{{
    if(o.type==='galaxyGroup'||o.type==='qGroup'){{
      o.group.rotation.y=t*0.12;
    }}
    if(o.type==='star'){{
      o.corona.rotation.y=t*0.05;
      o.corona.rotation.x=t*0.03;
    }}
    if(o.type==='spike'){{
      const pulse=1+0.12*Math.sin(t*2.5);
      const [dx,dy,dz]=o.dir;
      const pts=[new THREE.Vector3(0,0,0),
                 new THREE.Vector3(dx*o.baseLen*pulse,dy*o.baseLen*pulse,dz*o.baseLen*pulse)];
      o.line.geometry.setFromPoints(pts);
      o.line.material.opacity=0.35+0.2*Math.sin(t*2);
    }}
  }});

  if(!isDragging){{ spherical.theta+=0.003; }}
  updateCamera();
  renderer.render(scene,camera);
}}
animate();

function onResize() {{
  const nW = canvas.parentElement.clientWidth || 700;
  const nH = canvas.clientHeight || 560;
  renderer.setSize(nW, nH);
  camera.aspect = nW / nH;
  camera.updateProjectionMatrix();
}}
window.addEventListener('resize', onResize);
onResize();
</script>
</body>
</html>"""
            st.components.v1.html(canvas_html, height=710, scrolling=False)

        else:
            st.markdown("""
            <div style="text-align:center; padding:5rem 1rem; color:#1e3a5a;">
                <div style="font-size:5rem; margin-bottom:1rem; opacity:0.4;">🌌</div>
                <div style="font-family:Orbitron; font-size:0.8rem; letter-spacing:0.2em; color:#1e4a6a;">
                    SET PARAMETERS<br>
                    <span style="font-size:0.65rem; opacity:0.5; font-family:Rajdhani;">
                    Adjust values on the left and click RENDER OBJECT
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🔬 **How it works:** The u/g/r/i/z band magnitudes map to RGB color channels (brighter band = more of that color). Redshift warms the palette. The shape (spiral/elliptical/jets) is driven by band ratios. This is a creative data-driven render, not a telescope image.")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:Orbitron; font-size:0.65rem; color:#1e3a5a; letter-spacing:0.2em; padding: 1rem 0;">
    ASTROCLASSIFIER · SDSS SKYSERVER · ML PIPELINE: DT · LR · KNN
</div>
""", unsafe_allow_html=True)