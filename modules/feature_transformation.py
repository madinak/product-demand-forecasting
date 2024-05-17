import pandas as pd
import numpy as np

class FeatureTransformer:
    def __init__(self, encode_cols=None, transform_cols=None):
        self.encode = encode_cols
        self.transform = transform_cols
        self.transformed_columns = []

    def encode_categorical(self, df):
        if self.encode is None:
            self.encode = ['Category']
        elif 'Category' not in self.encode:
            self.encode.append('Category')
        
        # Check if the column exists in the DataFrame
        if not set(self.encode).issubset(df.columns):
            raise ValueError("One or more columns specified for encoding are not present in the DataFrame.")
        
        df_encoded = pd.get_dummies(df, columns=self.encode, drop_first=True, dtype=float)
        return df_encoded

    def transform_columns(self, df, columns):
        for column in columns:
            transformed_column = column + '_log'
            df[transformed_column] = np.log1p(df[column])  # Apply log transformation
            self.transformed_columns.append(transformed_column)
        return df, self.transformed_columns





def encode_categorical(df, encode_cols=None):
    if encode_cols is None:
        encode_cols = []
    if 'Category' not in encode_cols:
        encode_cols.append('Category')
    
    df_encoded = pd.get_dummies(df, columns=encode_cols, drop_first=True, dtype=float)
    return df_encoded