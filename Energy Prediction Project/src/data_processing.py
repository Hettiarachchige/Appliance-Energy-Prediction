def load_and_clean_data(path):
    import pandas as pd
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.interpolate(method='linear')
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df

def cap_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    for col in df.columns:
        lower = Q1[col] - 1.5 * IQR[col]
        upper = Q3[col] + 1.5 * IQR[col]
        df[col] = df[col].clip(lower, upper)
    return df