import pytest
import pandas as pd
import numpy as np
from app import clean_and_combine_data, filter_dataframe, forecast_revenue_trend, cluster_clients

@pytest.fixture
def sample_df():
    data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']),
        'Client Name': ['Client A', 'Client B', 'Client A', 'Client C'],
        'Insurer Name': ['Insurer X', 'Insurer Y', 'Insurer X', 'Insurer Z'],
        'Policy Type': ['Type 1', 'Type 2', 'Type 1', 'Type 3'],
        'Premium': [1000, 2000, 1500, 3000],
        'Commission': [100, 200, 150, 300],
        'Policy No': ['P1', 'P2', 'P3', 'P4']
    }
    df = pd.DataFrame(data)
    df['YearMonth'] = df['Date'].dt.to_period('M')
    return df

def test_clean_and_combine_data_missing_files():
    df, source_type = clean_and_combine_data("nonexistent.xlsx", "nonexistent.csv")
    assert df.empty
    assert source_type in ["missing", "error"]

def test_filter_dataframe(sample_df):
    date_range = (pd.to_datetime('2023-01-01'), pd.to_datetime('2023-01-31'))
    filtered_df = filter_dataframe(sample_df, date_range, selected_insurers=[], selected_products=[])
    assert len(filtered_df) == 2
    assert all(filtered_df['Date'].dt.month == 1)

    filtered_df = filter_dataframe(sample_df, date_range=None, selected_insurers=['Insurer X'], selected_products=[])
    assert len(filtered_df) == 2
    assert all(filtered_df['Insurer Name'] == 'Insurer X')

def test_forecast_revenue_trend(sample_df):
    # Need enough points for linear regression, mock more data
    data = {
        'YearMonth': pd.period_range(start='2023-01', periods=5, freq='M'),
        'Premium': [1000, 1200, 1100, 1500, 1600]
    }
    df = pd.DataFrame(data)

    result = forecast_revenue_trend(df, future_periods=2)
    assert not result.empty
    assert 'Type' in result.columns
    assert len(result[result['Type'] == 'Forecast']) == 2
    assert len(result[result['Type'] == 'Actual']) == 5

def test_cluster_clients(sample_df):
    # Create enough clients to test clustering (requires >3)
    data = {
        'Client Name': ['A', 'B', 'C', 'D', 'E'],
        'Premium': [100, 1000, 10000, 50000, 100000],
        'Policy No': ['P1', 'P2', 'P3', 'P4', 'P5']
    }
    df = pd.DataFrame(data)
    result = cluster_clients(df)
    assert not result.empty
    assert 'Segment' in result.columns
    assert 'Cluster' in result.columns
    # Ensure all segments are mapped
    assert set(result['Segment'].unique()).issubset({'Standard', 'Premium', 'VIP'})
