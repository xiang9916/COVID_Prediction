import pandas as pd
from datetime import datetime
import pickle

countries_without_full_data = [
    'American Samoa',
    'Cook Islands',
    "Democratic People's Republic of Korea",
    'Kiribati',
    'Micronesia (Federated States of)',
    'Nauru',
    'Niue',
    'Pitcairn Islands',
    'Saint Helena',
    'Tokelau',
    'Tonga',
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
        if datetime.strptime(df.loc[i, 'Date_reported'], '%Y-%m-%d') > datetime.strptime('2021-08-31', '%Y-%m-%d'): continue
        if first_time:
            first_time = False
        elif df.iloc[i, 0] == df.iloc[0, 0]:
            break
        s.append(df.iloc[i, 0])

    df_cum = table()
    for i in df.index:
        if datetime.strptime(df.loc[i, 'Date_reported'], '%Y-%m-%d') > datetime.strptime('2021-08-31', '%Y-%m-%d'): continue
        if df.loc[i, 'Country'] in countries_without_full_data:
            continue
        if int(df.loc[i, 'Cumulative_cases']) > 0:
            df_cum.loc[df.loc[i, 'Date_reported'], df.loc[i, 'Country']] = int(df.loc[i, 'Cumulative_cases'])
        else:
            df_cum.loc[df.loc[i, 'Date_reported'], df.loc[i, 'Country']] = 0
    df_cum.to_csv('./data/data/Cumulative_cases.csv', index = False)

    df_cum.mean().to_pickle('./data/data/df_cum.mean().pkl')
    df_cum.std().to_pickle('./data/data/df_cum.std().pkl')
    df_cum = (df_cum - df_cum.mean()) / df_cum.std()
    df_cum.to_csv('./data/data/Cumulative_cases_normalized.csv', index = False)
    return

def new():
    df = pd.read_csv('./data/raw/GlobalData.csv')
    s = []
    first_time = True
    for i in df.index:
        if datetime.strptime(df.loc[i, 'Date_reported'], '%Y-%m-%d') > datetime.strptime('2021-08-31', '%Y-%m-%d'): continue
        if first_time:
            first_time = False
        elif df.iloc[i, 0] == df.iloc[0, 0]:
            break
        s.append(df.iloc[i, 0])

    df_new = table()
    for i in df.index:
        if datetime.strptime(df.loc[i, 'Date_reported'], '%Y-%m-%d') > datetime.strptime('2021-08-31', '%Y-%m-%d'): continue
        if df.loc[i, 'Country'] in countries_without_full_data:
            continue
        if int(df.loc[i, 'New_cases']) > 0:
            df_new.loc[df.loc[i, 'Date_reported'], df.loc[i, 'Country']] = int(df.loc[i, 'New_cases'])
        else:
            df_new.loc[df.loc[i, 'Date_reported'], df.loc[i, 'Country']] = 0
    df_new.to_csv('./data/data/New_cases.csv', index = False)

    df_new.mean().to_pickle('./data/data/df_new.mean().pkl')
    df_new.std().to_pickle('./data/data/df_new.std().pkl')
    df_new = (df_new - df_new.mean()) / df_new.std()
    df_new.to_csv('./data/data/New_cases_normalized.csv', index = False)
    return


if __name__ == '__main__':
    pass
