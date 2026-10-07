import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Extracted Logic Functions ---

def clean_and_combine_data(file_path: str, demo_path: str) -> tuple[pd.DataFrame, str]:
    target_sheets = [
        'Patrick', 'Sheila', 'Sandra', 'Expery', 'Agnes',
        'Pascal', 'Direct Business', 'Ornella', 'Olivier',
        'Jacqueline', 'Viateur'
    ]
    needed_cols = ['Date', 'Month', 'Client Name', 'Insurer Name', 'Policy Type', 'Premium', 'Commission', 'Policy No']
    
    source_type = "none"
    all_data = []

    try:
        if os.path.exists(file_path):
            xl = pd.read_excel(file_path, sheet_name=target_sheets, header=0, engine='openpyxl')
            for sheet, df in xl.items():
                df.columns = [str(c).strip() for c in df.columns]
                available_cols = [c for c in needed_cols if c in df.columns]
                
                if len(available_cols) > 3: 
                    subset = df[available_cols].copy()
                    subset['SourceSheet'] = sheet 
                    all_data.append(subset)

            if all_data:
                full_df = pd.concat(all_data, ignore_index=True)
                source_type = "real"
            else:
                full_df = pd.DataFrame()
        else:
            if os.path.exists(demo_path):
                # Pre-filter columns on read
                full_df = pd.read_csv(demo_path)
                avail_cols = [c for c in needed_cols if c in full_df.columns]
                if avail_cols:
                    full_df = full_df[avail_cols]
                source_type = "demo"
            else:
                return pd.DataFrame(), "missing"

        if 'Date' in full_df.columns:
            full_df['Date'] = pd.to_datetime(full_df['Date'], errors='coerce', dayfirst=True)
            
        for c in ['Premium', 'Commission']:
            if c in full_df.columns:
                full_df[c] = pd.to_numeric(full_df[c], errors='coerce').fillna(0)
                
        if 'Date' in full_df.columns:
            full_df['YearMonth'] = full_df['Date'].dt.to_period('M')
        return full_df, source_type

    except Exception as e:
        return pd.DataFrame(), "error"

def filter_dataframe(df: pd.DataFrame, date_range: tuple, selected_insurers: list, selected_products: list) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1] and 'Date' in df.columns:
        mask &= (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))
    if selected_insurers and 'Insurer Name' in df.columns:
        mask &= df['Insurer Name'].isin(selected_insurers)
    if selected_products and 'Policy Type' in df.columns:
        mask &= df['Policy Type'].isin(selected_products)
    return df[mask]

def forecast_revenue_trend(filtered_df: pd.DataFrame, future_periods: int = 3) -> pd.DataFrame:
    if 'YearMonth' not in filtered_df.columns or 'Premium' not in filtered_df.columns:
        return pd.DataFrame()

    revenue_trend = filtered_df.groupby('YearMonth')['Premium'].sum().reset_index()
    if len(revenue_trend) <= 2:
        revenue_trend['Type'] = 'Actual'
        revenue_trend['YearMonth'] = revenue_trend['YearMonth'].astype(str)
        return revenue_trend

    revenue_trend['PeriodIndex'] = range(len(revenue_trend))
    X = revenue_trend[['PeriodIndex']]
    y = revenue_trend['Premium']

    model = LinearRegression()
    model.fit(X, y)

    last_idx = revenue_trend['PeriodIndex'].max()
    future_indices = np.array(range(last_idx + 1, last_idx + 1 + future_periods)).reshape(-1, 1)
    future_preds = model.predict(future_indices)

    last_date = revenue_trend['YearMonth'].iloc[-1]

    future_dates = [(last_date + i) for i in range(1, future_periods + 1)]

    future_df = pd.DataFrame({
        'YearMonth': future_dates,
        'Premium': future_preds,
        'Type': 'Forecast'
    })

    revenue_trend['Type'] = 'Actual'
    combined_trend = pd.concat([revenue_trend.drop(columns=['PeriodIndex']), future_df], ignore_index=True)
    combined_trend['YearMonth'] = combined_trend['YearMonth'].astype(str)

    return combined_trend

def cluster_clients(filtered_df: pd.DataFrame) -> pd.DataFrame:
    if 'Client Name' not in filtered_df.columns or 'Premium' not in filtered_df.columns or 'Policy No' not in filtered_df.columns:
        return pd.DataFrame()

    client_features = filtered_df.groupby('Client Name').agg({
        'Premium': 'sum',
        'Policy No': 'count'
    }).rename(columns={'Premium': 'TotalValue', 'Policy No': 'Frequency'}).reset_index()

    if len(client_features) <= 3:
        return client_features

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(client_features[['TotalValue', 'Frequency']])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    client_features['Cluster'] = kmeans.fit_predict(scaled_features)

    cluster_means = client_features.groupby('Cluster')['TotalValue'].mean().sort_values()
    cluster_map = {
        cluster_means.index[0]: 'Standard',
        cluster_means.index[1]: 'Premium',
        cluster_means.index[2]: 'VIP'
    }
    client_features['Segment'] = client_features['Cluster'].map(cluster_map)
    return client_features

# --- Main App ---

st.set_page_config(
    page_title="Zamara: Operational Dashboard",
    page_icon="zamara_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    file_path = "Data/New Business report/New business report till July 2021.xlsx"
    demo_path = "demo_data.csv"
    return clean_and_combine_data(file_path, demo_path)

df, data_source = load_data()

if data_source == "demo":
    st.toast("Using Anonymized Demo Data", icon="ℹ️")
elif data_source == "missing":
    st.error("No data found. Please upload data or add demo_data.csv")
elif data_source == "error":
    st.error("Error loading data.")

# --- Sidebar Filters ---
st.sidebar.image("zamara_logo.png", width=150)
st.sidebar.title("📊 Filters")

if not df.empty:
    min_date = df['Date'].min() if 'Date' in df.columns else pd.NaT
    max_date = df['Date'].max() if 'Date' in df.columns else pd.NaT
    
    if pd.notnull(min_date) and pd.notnull(max_date):
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        date_range = (None, None)

    all_insurers = sorted(df['Insurer Name'].dropna().unique().tolist()) if 'Insurer Name' in df.columns else []
    selected_insurers = st.sidebar.multiselect("Select Insurers", all_insurers, default=all_insurers[:5] if all_insurers else [])

    all_products = sorted(df['Policy Type'].dropna().unique().tolist()) if 'Policy Type' in df.columns else []
    selected_products = st.sidebar.multiselect("Select Products", all_products, default=all_products[:5] if all_products else [])
    
    filtered_df = filter_dataframe(df, date_range, selected_insurers, selected_products)
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
    
    current_revenue = filtered_df['Premium'].sum() if 'Premium' in filtered_df.columns else 0
    total_clients = filtered_df['Client Name'].nunique() if 'Client Name' in filtered_df.columns else 0
    total_policies = len(filtered_df)
    total_commission = filtered_df['Commission'].sum() if 'Commission' in filtered_df.columns else 0
    
    with col1:
        st.metric("Total Premium", f"RWF {current_revenue:,.0f}")
    with col2:
        st.metric("Total Commission", f"RWF {total_commission:,.0f}")
    with col3:
        st.metric("Active Clients", f"{total_clients:,}")
    with col4:
        st.metric("Policies Sold", f"{total_policies:,}")
    
    st.markdown("---")

    # --- Row 2: Revenue Trend & Forecasting (ML) ---
    st.subheader("📈 Revenue Trend & Forecast")
    
    combined_trend = forecast_revenue_trend(filtered_df)
    if not combined_trend.empty:
        fig_trend = px.line(combined_trend, x='YearMonth', y='Premium', color='Type',
                            markers=True, line_shape='spline',
                            color_discrete_map={'Actual': '#2E86C1', 'Forecast': '#E74C3C'})
        fig_trend.update_layout(plot_bgcolor="white", height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Not enough data to forecast revenue.")

    # --- Row 3: Product Mix & Insurer Share ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("☂️ Product Portfolio")
        if 'Policy Type' in filtered_df.columns and 'Premium' in filtered_df.columns:
            prod_mix = filtered_df.groupby('Policy Type')['Premium'].sum().reset_index()
            fig_pie = px.pie(prod_mix, values='Premium', names='Policy Type', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.subheader("🏢 Insurer Market Share")
        if 'Insurer Name' in filtered_df.columns and 'Premium' in filtered_df.columns:
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
    
    client_features = cluster_clients(filtered_df)
    
    if len(client_features) > 3 and 'Segment' in client_features.columns:
        fig_cluster = px.scatter(
            client_features, 
            x='Frequency', 
            y='TotalValue', 
            color='Segment',
            hover_data=['Client Name'],
            log_y=True,
            title="Client Value Matrix (Log Scale)",
            color_discrete_map={'Standard': '#95A5A6', 'Premium': '#F1C40F', 'VIP': '#27AE60'}
        )
        fig_cluster.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        fig_cluster.update_layout(plot_bgcolor="white", height=500)
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        with st.expander("🏆 View Top VIP Clients"):
            vips = client_features[client_features['Segment'] == 'VIP'].sort_values('TotalValue', ascending=False)
            st.dataframe(vips[['Client Name', 'TotalValue', 'Frequency']].head(10))
    else:
        st.warning("Not enough data points for clustering.")