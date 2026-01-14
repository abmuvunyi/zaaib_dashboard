# Zamara Insurance Brokers - Executive Operational Dashboard

![Zamara Dashboard](zamara_logo.png)

## Overview

This repository contains the **Strategic Analytics Dashboard** developed for **Zamara Insurance Brokers**. This application transforms raw operational data (premiums, commissions, policies) into actionable business intelligence.

It is designed to give the Senior Leadership Team a real-time pulse on the portfolio's health, forecast revenue trends, and identify high-value client segments for targeted retention strategies.

## Key Features

*   **KPI Scorecard**: Instant view of Total Premium, Commissions, Active Clients, and Policies Sold.
*   **Revenue Forecasting**: Linear Regression models (Scikit-Learn) project cash flow for the upcoming quarter based on historical trends.
*   **Client Segmentation**: Unsupervised Machine Learning (K-Means Clustering) automatically categorizes clients into `VIP`, `Premium`, and `Standard` tiers to optimize account management.
*   **Interactive Analytics**: Drill-down capabilities by Date, Insurer, and Product Line.

## Technology Stack

*   **Core**: Python 3.9+
*   **Frontend**: Streamlit
*   **Visualization**: Plotly Express
*   **Machine Learning**: Scikit-Learn
*   **Data Processing**: Pandas / NumPy

## How to Run Locally

1.  Clone the repository:
    ```bash
    git clone https://github.com/abmuvunyi/zaaib_dashboard.git
    cd zaaib_dashboard
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the application:
    ```bash
    streamlit run app.py
    ```

## Data Privacy

*   **Confidentiality**: Real client data (Excel files) is **excluded** from this repository via `.gitignore`.
*   **Demo Mode**: The application includes an anonymized `demo_data.csv` dataset that automatically loads if the secure Excel source is not detected, allowing for safe public demonstration of the dashboard's capabilities.

---
*Confidential - Internal Use Only*
