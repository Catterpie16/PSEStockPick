import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="PSEi Machine Learning Project", page_icon="🎓", layout="wide"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .badge-buy { color: #2ecc71; font-weight: bold; }
    .badge-watch { color: #f1c40f; font-weight: bold; }
    .badge-sell { color: #e74c3c; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎓 PSEi 30 Machine Learning Predictor")
st.caption("An Academic Study on Predicting Short-Term Stock Momentum using Tree-Based Ensembles")
st.divider()

# Create Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Live Recommendations", 
    "🧠 Methodology & Pipeline", 
    "📈 Model Performance"
])

# ==========================================
# TAB 1: LIVE RECOMMENDATIONS
# ==========================================
with tab1:
    @st.cache_data(ttl=3600)
    def load_data():
        # Looks for the file in the same GitHub folder
        return pd.read_json("latest_recommendations.json")

    try:
        df = load_data()
        
        col1, col2, col3 = st.columns(3)
        strong_buys = len(df[df["Signal"] == "STRONG BUY"])
        
        col1.metric("Tracked Blue-Chip Stocks", len(df))
        col2.metric("Active 'Strong Buy' Signals", strong_buys)
        col3.metric("Latest Data Date", df['Data_Date'].iloc[0])
        
        st.subheader("Today's Ranked Watchlist")
        
        # Format Dataframe
        display_df = df[['Ticker', 'Close_Price', 'Target_Price_5pct', 'Probability_Pct', 'Signal']].copy()
        display_df.columns = ["Ticker", "Current Price (₱)", "5% Target Price (₱)", "Buy Probability (%)", "Model Signal"]
        
        def style_signal(val):
            if val == 'STRONG BUY': return 'color: #2ecc71; font-weight: bold;'
            elif val == 'NEUTRAL / WATCH': return 'color: #f1c40f; font-weight: bold;'
            else: return 'color: #e74c3c; font-weight: bold;'

        st.dataframe(
            display_df.style.map(style_signal, subset=["Model Signal"]),
            use_container_width=True,
            hide_index=True
        )
    except FileNotFoundError:
        st.warning("⚠️ Prediction data not found. Please ensure 'latest_recommendations.json' is uploaded to GitHub.")

# ==========================================
# TAB 2: METHODOLOGY & PIPELINE
# ==========================================
with tab2:
    st.header("How the Model Works")
    
    st.subheader("1. The Objective (Target Variable)")
    st.write(
        "The model is framed as a **Binary Classification** problem. Rather than predicting exact future prices, "
        "the model answers a strict quantitative question:"
    )
    st.info("**Will this stock gain at least 5% within the next 10 trading days?** (1 = Yes, Buy | 0 = No, Avoid)")

    st.subheader("2. Feature Engineering")
    st.write(
        "Raw price data is highly noisy. We engineered relative momentum and institutional flow metrics to give the model a predictive edge. "
        "Key features include:"
    )
    
    st.markdown(
        """
        * **Distance from Moving Averages:** Converting raw SMAs into relative percentages to measure trend extension.
        * **Proximity to 52-Week High:** A core momentum indicator measuring overhead supply. 
        * **Foreign Flow Intensity:** Institutional buying relative to daily volume.
        * **Market Breadth (Regime Filter):** The rolling 5-day average return of the entire PSEi 30 to prevent the model from buying during macro market crashes.
        """
    )
    
    st.latex(r"\text{Dist\_SMA50} = \frac{\text{Close} - \text{Price\_SMA50}}{\text{Price\_SMA50}}")

    st.subheader("3. Handling Class Imbalance")
    st.write(
        "A 5% gain in 10 days is a rare event for mega-cap stocks. To prevent the model from always safely predicting '0', "
        "we calculated the exact class imbalance ratio and applied mathematical penalties (`scale_pos_weight` and `class_weight='balanced'`) "
        "to force the algorithms to hunt for minority-class buy signals."
    )

# ==========================================
# TAB 3: MODEL PERFORMANCE
# ==========================================
with tab3:
    st.header("Evaluation Results")
    st.write("The dataset was split chronologically (80% Training / 20% Testing) to prevent look-ahead bias. "
             "Models were tuned using RandomizedSearchCV with 5-fold Time Series Cross-Validation.")
    
    st.subheader("Algorithm Comparison")
    
    # Results from your training run
    results_data = {
        "Model": ["Random Forest (Balanced)", "XGBoost (Balanced)"],
        "Train Accuracy": ["66.18%", "66.17%"],
        "Test Accuracy": ["59.18%", "59.66%"],
        "ROC-AUC Score": ["0.5913", "0.5897"]
    }
    st.table(pd.DataFrame(results_data))
    
    st.write(
        "While XGBoost had a slightly higher raw accuracy, **Random Forest** achieved the superior ROC-AUC score (0.5913), "
        "making it the stronger model for separating winning setups from losing setups."
    )
    
    st.divider()
    
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Random Forest Classification Report")
        st.text("""
              Precision    Recall  F1-Score   Support

Class 0 (Sell)     0.87      0.61      0.72     19,413
Class 1 (Buy)      0.19      0.51      0.27      3,458

Accuracy: 59.18%
        """)
        
    with colB:
        st.subheader("Interpreting the Results")
        st.write(
            "The model achieves an **87% precision** on Sell/Avoid signals, making it an excellent risk-management tool. "
            "While the Class 1 (Buy) precision is lower due to market volatility, the model successfully captures over half (51% recall) "
            "of all true 5% breakouts in the test set."
        )
