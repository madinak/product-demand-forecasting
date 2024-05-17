import pandas as pd
import numpy as np
import os
import zipfile
import requests
from scipy import stats

def get_data():
    # URL of the Online Retail Dataset zip file
    url = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'
    zip_file = 'Online Retail.zip'
    extract_dir = 'Online_Retail'

    # Download the zip file
    response = requests.get(url)
    with open(zip_file, 'wb') as file:
        file.write(response.content)

    # Extract the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Get the path to the extracted Excel file
    file_path = os.path.join(extract_dir, 'Online Retail.xlsx')

    # 4. Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    return df


def clean_data(df):
    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Identify returns and their corresponding sold items
    def find_matching_returns(df):
        grouped = df.groupby(['StockCode', 'Description', 'UnitPrice', 'Country'])
        matching_returns = []
        for _, group in grouped:
            if len(group) > 1 and group['Quantity'].sum() == 0:
                if (group['Quantity'] < 0).any() and (group['Quantity'] > 0).any():
                    matching_returns.extend(group.index.tolist())
        return matching_returns

    # Find and remove matching returns
    matching_returns_indices = find_matching_returns(df)
    df.drop(index=matching_returns_indices, inplace=True)

    # Remove rows where InvoiceNo contains non-numeric characters (letters)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[df['InvoiceNo'].str.isnumeric()]

    # Remove rows with negative Quantity values and UnitPrice greater than 0.1
    df = df[(df['Quantity'] >= 0) & (df['UnitPrice'] > 0.1)]

    # Remove outliers
    def remove_outliers(df, columns, z_score_threshold=1):
        for column in columns:
            z_scores = stats.zscore(df[column])  # Calculate z-scores for the column
            outliers_indices = (z_scores > z_score_threshold)  # Find outlier indices
            df = df[~outliers_indices]  # Remove outliers from the DataFrame

        return df

    # Remove outliers from numeric columns
    df = remove_outliers(df, ['Quantity', 'UnitPrice'])


    # Extract only date (we don't need time)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date

    # Drop NA values
    df.dropna(subset=['Description'], inplace=True)

    # Drop categorical columns we won't be using in the analysis
    df.drop(columns=['InvoiceNo', 'StockCode', 'CustomerID', 'Country'], inplace=True)

    # Reset df indices
    df.reset_index(drop=True, inplace=True)

    return df


def test_function():
    return "It works!"