# One Acre Fund: Advanced Analytics & ML Dashboard Implementation Plan

## Goal
Create a "Power BI-style" interactive dashboard using Python (Streamlit) that demonstrates Data Science and Machine Learning capabilities for the Rwanda Agricultural Research Data Scientist role.

## Technical Stack
- **Framework**: Streamlit (Python)
- **Visualization**: Plotly Interactive Charts (closest to Power BI interactivity)
- **Data Manipulation**: Pandas
- **Machine Learning**: Scikit-learn
    - *Clustering*: K-Means for Customer Segmentation (High Value vs Standard)
    - *Forecasting*: Linear Regression / Time-series decomposition for Revenue Forecasting

## Proposed Components

### 1. Data Pipeline
- Reuse the extraction logic from `process_sales_data.py`.
- Strict type cleaning and handling of missing dates.

### 2. Dashboard Layout (Power BI Style)
- **Sidebar**: Global Filters (Date Range, Department, Product Type).
- **Top Row**: KPI Cards (Total Revenue, Active Clients, MoM Growth) with Delta indicators.
- **Middle Row**:
    - *Col 1*: Interactive Line Chart (Revenue Trend + Forecast).
    - *Col 2*: Pie/Donut Chart (Product Mix).
- **Bottom Row**:
    - *Col 1*: Scatter Plot (Customer Segmentation - ML Feature).
    - *Col 2*: Data Grid/Table (Recent Transactions).

### 3. Machine Learning Implementations
- **Customer Segmentation (Clustering)**:
    - Feature Engineering: Calculate `TotalPremium` and `TransactionCount` per client.
    - Model: K-Means Clustering (`k=3`).
    - Viz: Scatter plot coloring clients by their cluster (e.g., "VIP", "Mid-Tier", "Standard").
- **Revenue Forecasting**:
    - Resample data to Monthly/Weekly.
    - Fit a Trend Line.
    - Project next 3 months of revenue.

## Verification Plan
- Run `streamlit run app.py` locally to verify UI and Interactivity.
- Verify ML model outputs (sanity check clusters and forecast values).
