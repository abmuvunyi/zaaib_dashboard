import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

# --- Page Config & Styling ---
st.set_page_config(
    page_title="Zamara: Operational Dashboard",
    page_icon="zamara_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to mimic Power BI/Modern Dashboard look
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 5px solid #2ECC71; /* Zamara Green accent */
    }
    .metric-title {
        color: #7f8c8d;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #2c3e50;
        font-size: 28px;
        font-weight: 700;
    }
    .metric-delta {
        font-size: 14px;
        font-weight: 500;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1.5rem;
    }
    div[data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    file_path = "Data/New Business report/New business report till July 2021.xlsx"
    demo_path = "demo_data.csv"
    
    # Try loading Real Data (Local)
    try:
        # List of sheets that appear to be Agent/Transaction logs
        target_sheets = [
            'Patrick', 'Sheila', 'Sandra', 'Expery', 'Agnes', 
            'Pascal', 'Direct Business', 'Ornella', 'Olivier', 
            'Jacqueline', 'Viateur'
        ]
        
        all_data = []
        
        # Check if local Excel exists
        if os.path.exists(file_path):
             xl = pd.read_excel(file_path, sheet_name=target_sheets, header=0)
             for sheet, df in xl.items():
                df.columns = [str(c).strip() for c in df.columns]
                needed_cols = ['Date', 'Month', 'Client Name', 'Insurer Name', 'Policy Type', 'Premium', 'Commission', 'Policy No']
                available_cols = [c for c in needed_cols if c in df.columns]
                
                if len(available_cols) > 3: 
                    subset = df[available_cols].copy()
                    subset['SourceSheet'] = sheet 
                    all_data.append(subset)
             
             if all_data:
                 full_df = pd.concat(all_data, ignore_index=True)
             else:
                 full_df = pd.DataFrame()
                 
        else:
            # Fallback to Demo Data (Cloud)
            if os.path.exists(demo_path):
                st.toast("Using Anonymized Demo Data", icon="ℹ️")
                full_df = pd.read_csv(demo_path)
            else:
                st.error("No data found. Please upload data or add demo_data.csv")
                return pd.DataFrame()

        # Cleaning Pipeline (Shared)
        if 'Date' in full_df.columns:
            full_df['Date'] = pd.to_datetime(full_df['Date'], errors='coerce', dayfirst=True)
            
        t_cols = ['Premium', 'Commission']
        for c in t_cols:
            if c in full_df.columns:
                full_df[c] = pd.to_numeric(full_df[c], errors='coerce').fillna(0)
                
        full_df['YearMonth'] = full_df['Date'].dt.to_period('M')
        return full_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


df = load_data()

# --- Sidebar Filters ---
st.sidebar.image("zamara_logo.png", width=150)
st.sidebar.title("📊 Filters")

if not df.empty:
    # Date Range
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    if pd.notnull(min_date) and pd.notnull(max_date):
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        date_range = (None, None)

    # Insurer Filter
    all_insurers = sorted(df['Insurer Name'].dropna().unique().tolist())
    selected_insurers = st.sidebar.multiselect("Select Insurers", all_insurers, default=all_insurers[:5] if all_insurers else [])

    # Product Filter
    all_products = sorted(df['Policy Type'].dropna().unique().tolist())
    selected_products = st.sidebar.multiselect("Select Products", all_products, default=all_products[:5] if all_products else [])
    
    # Filter Logic
    mask = pd.Series(True, index=df.index)
    if date_range[0] and date_range[1]:
        mask &= (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))
    if selected_insurers:
        mask &= df['Insurer Name'].isin(selected_insurers)
    if selected_products:
        mask &= df['Policy Type'].isin(selected_products)
        
    filtered_df = df[mask]
else:
    st.warning("No data loaded.")
    filtered_df = df

# --- Main Dashboard ---

st.title("🛡️ Zamara Actuaries, Administrators & Insurance Brokers: Executive Dashboard")
st.markdown("Operational Overview: Portfolio Performance, Client Segments, and Revenue Projections.")

if filtered_df.empty:
    st.info("No data available for the selected filters.")
else:
    # --- Row 1: KPI Cards ---
    col1, col2, col3, col4 = st.columns(4)
    
    current_revenue = filtered_df['Premium'].sum()
    total_clients = filtered_df['Client Name'].nunique()
    total_policies = len(filtered_df)
    total_commission = filtered_df['Commission'].sum() if 'Commission' in filtered_df.columns else 0
    
    def metric_card(title, value, prefix="", suffix="", col=None):
        formatted_value = f"{prefix}{value:,.0f}{suffix}"
        html = f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{formatted_value}</div>
        </div>
        """
        if col:
            col.markdown(html, unsafe_allow_html=True)
            
    metric_card("Total Premium", current_revenue, "RWF ", "", col1)
    metric_card("Total Commission", total_commission, "RWF ", "", col2)
    metric_card("Active Clients", total_clients, "", "", col3)
    metric_card("Policies Sold", total_policies, "", "", col4)
    
    st.markdown("---")

    # --- Row 2: Revenue Trend & Forecasting (ML) ---
    st.subheader("📈 Revenue Trend & Forecast")
    
    # Prep Daily/Monthly Data
    revenue_trend = filtered_df.groupby('YearMonth')['Premium'].sum().reset_index()
    revenue_trend['YearMonth'] = revenue_trend['YearMonth'].astype(str)
    
    # Simple ML Forecast
    if len(revenue_trend) > 2:
        revenue_trend['PeriodIndex'] = range(len(revenue_trend))
        X = revenue_trend[['PeriodIndex']]
        y = revenue_trend['Premium']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Project 3 Months ahead
        future_periods = 3
        last_idx = revenue_trend['PeriodIndex'].max()
        future_indices = np.array(range(last_idx + 1, last_idx + 1 + future_periods)).reshape(-1, 1)
        future_preds = model.predict(future_indices)
        
        # Construct Future DataFrame
        # Get last date string
        last_date_str = revenue_trend['YearMonth'].iloc[-1]
        try:
            last_date = datetime.datetime.strptime(last_date_str, "%Y-%m")
            future_dates = []
            for i in range(1, 4):
                next_month = last_date.month + i
                next_year = last_date.year + (next_month - 1) // 12
                next_month = (next_month - 1) % 12 + 1
                future_dates.append(f"{next_year}-{next_month:02d}")
        except:
             future_dates = [f"Future {i}" for i in range(1,4)]
             
        future_df = pd.DataFrame({
            'YearMonth': future_dates,
            'Premium': future_preds,
            'Type': 'Forecast'
        })
        
        revenue_trend['Type'] = 'Actual'
        combined_trend = pd.concat([revenue_trend, future_df], ignore_index=True)
    else:
        combined_trend = revenue_trend
        combined_trend['Type'] = 'Actual'

    fig_trend = px.line(combined_trend, x='YearMonth', y='Premium', color='Type', 
                        markers=True, line_shape='spline',
                        color_discrete_map={'Actual': '#2E86C1', 'Forecast': '#E74C3C'})
    fig_trend.update_layout(plot_bgcolor="white", height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- Row 3: Product Mix & Insurer Share ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("☂️ Product Portfolio")
        prod_mix = filtered_df.groupby('Policy Type')['Premium'].sum().reset_index()
        fig_pie = px.pie(prod_mix, values='Premium', names='Policy Type', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.subheader("🏢 Insurer Market Share")
        ins_mix = filtered_df.groupby('Insurer Name')['Premium'].sum().reset_index().sort_values('Premium', ascending=True)
        fig_bar = px.bar(ins_mix, x='Premium', y='Insurer Name', orientation='h',
                         text_auto='.2s', color='Premium', color_continuous_scale='Greens')
        fig_bar.update_layout(plot_bgcolor="white", height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- Row 4: Client Segmentation (ML - Clustering) ---
    st.markdown("---")
    st.subheader("💎 Client Value Segmentation")
    st.markdown("""
    *Strategic segmentation of our client base to identify High-Net-Worth partners and opportunities for dedicated account management.*
    """)
    
    # 1. Feature Engineering per Client
    client_features = filtered_df.groupby('Client Name').agg({
        'Premium': 'sum',
        'Policy No': 'count'  # Proxy for frequency
    }).rename(columns={'Premium': 'TotalValue', 'Policy No': 'Frequency'}).reset_index()
    
    if len(client_features) > 3:
        # Scale Data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(client_features[['TotalValue', 'Frequency']])
        
        # K-Means (k=3 for Low, Med, High value)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        client_features['Cluster'] = kmeans.fit_predict(scaled_features)
        
        # Map Clusters to Human Names based on Mean Value
        cluster_means = client_features.groupby('Cluster')['TotalValue'].mean().sort_values()
        cluster_map = {
            cluster_means.index[0]: 'Standard',
            cluster_means.index[1]: 'Premium',
            cluster_means.index[2]: 'VIP'
        }
        client_features['Segment'] = client_features['Cluster'].map(cluster_map)
        
        # Scatter Plot
        fig_cluster = px.scatter(
            client_features, 
            x='Frequency', 
            y='TotalValue', 
            color='Segment',
            hover_data=['Client Name'],
            log_y=True,  # Log scale often better for financial data
            title="Client Value Matrix (Log Scale)",
            color_discrete_map={'Standard': '#95A5A6', 'Premium': '#F1C40F', 'VIP': '#27AE60'}
        )
        fig_cluster.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        fig_cluster.update_layout(plot_bgcolor="white", height=500)
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Show Top VIPs
        with st.expander("🏆 View Top VIP Clients"):
            vips = client_features[client_features['Segment'] == 'VIP'].sort_values('TotalValue', ascending=False)
            st.dataframe(vips[['Client Name', 'TotalValue', 'Frequency']].head(10))
    else:
        st.warning("Not enough data points for clustering.")

