def create_features(df):
    import holidays
    sri_lanka_holidays = holidays.CountryHoliday('LK')
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >=5 else 0)
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in sri_lanka_holidays else 0)
    df.set_index('date', inplace=True)
    df['rolling_1h'] = df['Appliances'].rolling(window=6).mean()
    df['rolling_3h'] = df['Appliances'].rolling(window=18).mean()
    df['rolling_1h_std'] = df['Appliances'].rolling(window=6).std()
    df['rolling_3h_std'] = df['Appliances'].rolling(window=18).std()
    df['lag_10m'] = df['Appliances'].shift(1)
    df['lag_30m'] = df['Appliances'].shift(3)
    df['lag_1h'] = df['Appliances'].shift(6)
    df['T2_RH2_interaction'] = df['T2'] * df['RH_2']
    df['hour_weekend_interaction'] = df['hour'] * df['is_weekend']
    df.dropna(inplace=True)
    return df