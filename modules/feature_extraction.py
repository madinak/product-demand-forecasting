import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday

class EuropeanHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1, observance=nearest_workday),
        Holiday('Labour Day', month=5, day=1, observance=nearest_workday),
        Holiday('Europe Day', month=5, day=9, observance=nearest_workday),
        Holiday('Assumption of Mary', month=8, day=15, observance=nearest_workday),
        Holiday('All Saints Day', month=11, day=1, observance=nearest_workday),
        Holiday('Armistice Day', month=11, day=11, observance=nearest_workday),
        Holiday('Christmas Day', month=12, day=25, observance=nearest_workday),
        Holiday('St. Stephen\'s Day', month=12, day=26, observance=nearest_workday),
    ]


class FeatureExtractor:
    def __init__(self):
        self.holidays = EuropeanHolidays()


    def derive_features(self, df):
        # Derive TotalPrice from Quantity and UnitPrice
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

        # Aggregate by day and category
        df_agg = df.groupby([df['InvoiceDate'], 'Category']).agg({
            'Quantity': 'sum',          # Sum of quantities
            'TotalPrice': 'sum'         # Sum of total prices
            }).reset_index()

        df_agg['InvoiceDate'] = pd.to_datetime(df_agg['InvoiceDate'], format='%Y-%m-%d')
        df_agg['Year'] = pd.to_datetime(df_agg['InvoiceDate'], format='%Y-%m-%d').dt.year
        df_agg['Month'] = df_agg['InvoiceDate'].dt.month
        df_agg['Day'] = df_agg['InvoiceDate'].dt.day

        # Determine if the day is a weekend
        df_agg['Weekend'] = df_agg['InvoiceDate'].dt.dayofweek.isin([5, 6]).astype(int)

        # Add days of the week (0=Monday, 1=Tuesday, ..., 6=Sunday)
        df_agg['DayOfWeek'] = df_agg['InvoiceDate'].dt.dayofweek

        # Initialize Holiday column with zeros
        df_agg['Holiday'] = 0

        # Iterate over holiday dates and mark holidays in the Holiday column
        for holiday_date in self.holidays.holidays().date:
            df_agg.loc[df_agg['InvoiceDate'].dt.date == holiday_date, 'Holiday'] = 1

        # Add previous sales to the next row for each category
        df_agg['PrevQuantity'] = df_agg.groupby('Category')['Quantity'].shift(1)

        # Drop the null values (first row for each category) and calculate the difference
        df_agg = df_agg.dropna()
        df_agg['Diff'] = df_agg['Quantity'] - df_agg['PrevQuantity']

        # Create dataframe for transformation from time series to supervised
        df_agg = df_agg.drop(['PrevQuantity'], axis=1)

        # Adding lag features for each day for each category
        lag_columns = []
        for lag in range(1, 8):  # Lag for 7 days
            field_name = 'lag' + str(lag)
            df_agg[field_name] = df_agg.groupby('Category')['Diff'].shift(lag)
            lag_columns.append(field_name)

        # Adding lag features for each week for each category
        for lag in range(7, 15, 7):  # Lag for 7 and 14 days
            field_name = 'lag' + str(lag)
            df_agg[field_name] = df_agg.groupby('Category')['Diff'].shift(lag)
            lag_columns.append(field_name)

        # Drop null values
        df_agg = df_agg.dropna().reset_index(drop=True)

        # Drop InvoiceDate
        df_agg.drop(columns=['InvoiceDate'], inplace=True)

        return df_agg



