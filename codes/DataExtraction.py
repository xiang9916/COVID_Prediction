import pandas as pd

countries_without_full_data = [
    "Democratic People's Republic of Korea",
    'Micronesia (Federated States of)',
    'Nauru',
    'Pitcairn Islands',
    'Saint Helena',
    'Tokelau',
    'Turkmenistan',
    'Tuvalu',
    'Other'
]

def table():
    df = pd.read_csv('./data/raw/GlobalData.csv')
    new_df = pd.DataFrame()

    current_country = '00'
    for i in df.index:
        if df.loc[i, 'Country'] in countries_without_full_data:
            continue
        if current_country != df.loc[i, 'Country']:
            new_df.insert(new_df.shape[1], df.loc[i, 'Country'], -1)
            current_country = df.loc[i, 'Country']

    return new_df


def cumulative():
    df = pd.read_csv('./data/raw/GlobalData.csv')
    s = []
    first_time = True
    for i in df.index:
        if first_time:
            first_time = False
        elif df.iloc[i, 0] == df.iloc[0, 0]:
            break
        s.append(df.iloc[i, 0])

    df_cum = table()
    for i in df.index:
        if df.loc[i, 'Country'] in countries_without_full_data:
            continue
        if int(df.loc[i, 'Cumulative_cases']) > 0:
            df_cum.loc[df.loc[i, 'Date_reported'], df.loc[i, 'Country']] = int(df.loc[i, 'Cumulative_cases'])
        else:
            df_cum.loc[df.loc[i, 'Date_reported'], df.loc[i, 'Country']] = 0
    df_cum.to_csv('./data/data/Cumulative_cases.csv', index = False)
    df_cum = (df_cum - df_cum.mean()) / df_cum.std()
    df_cum.to_csv('./data/data/Cumulative_cases_normalized.csv', index = False)
    return

def new():
    df = pd.read_csv('./data/raw/GlobalData.csv')
    s = []
    first_time = True
    for i in df.index:
        if first_time:
            first_time = False
        elif df.iloc[i, 0] == df.iloc[0, 0]:
            break
        s.append(df.iloc[i, 0])

    df_new = table()
    for i in df.index:
        if df.loc[i, 'Country'] in countries_without_full_data:
            continue
        if int(df.loc[i, 'New_cases']) > 0:
            df_new.loc[df.loc[i, 'Date_reported'], df.loc[i, 'Country']] = int(df.loc[i, 'New_cases'])
        else:
            df_new.loc[df.loc[i, 'Date_reported'], df.loc[i, 'Country']] = 0
    df_new.to_csv('./data/data/New_cases.csv', index = False)
    df_new = (df_new - df_new.mean()) / df_new.std()
    df_new.to_csv('./data/data/New_cases_normalized.csv', index = False)
    return


if __name__ == '__main__':
    pass
