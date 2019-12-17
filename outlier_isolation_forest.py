import pandas as pd
import collections
from numpy import mean
from numpy import std
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from functools import partial
from scipy.stats import zscore

def local(filename):
    return pd.read_csv(filename)
load_data_methods = dict(local=partial(local), remote=partial(local), database=partial(local))

def z_score_transformation(df):
    df['Date'] = df['Date'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y').date())
    # plt.plot(df['Date'],df['Price'])
    total_seconds_per_day = (df['Date'].values[1] - df['Date'].values[0]).total_seconds()
    first_date = df['Date'].values[0]
    df['Date'] = df['Date'].apply(lambda x: (x - first_date).total_seconds() / total_seconds_per_day)
    for col in df.columns:
        df['z_' + col] = zscore(df[col])
    return df


def detect_outliers_isolation_forest(df):
    X_train = df.values
    # fit the model
    rng = np.random.RandomState(42)
    clf = IsolationForest(max_samples=len(df), random_state=rng, contamination=0.05)
    clf.fit(X_train)
    outliers = clf.predict(X_train)
    outliers = pd.Series(outliers)
    outliers = outliers.apply(lambda x: True if x == -1 else False)

    return outliers, df


def visualize(outliers, df):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(df['Date'], df['Price'], c='blue')
    bad_prices = df[outliers]['Price']
    bad_days = df[outliers]['Date']
    ax.scatter(bad_days, bad_prices, c='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.show()


def remove_bad_ids(outliers, df):
    df.drop(df[outliers].index, axis=0, inplace=True)
    df.reset_index(inplace=True)
    return df


def remove_outliers(filename):
    df = load_data_methods['local'](filename=filename)
    df = z_score_transformation(df)
    outliers, df = detect_outliers_isolation_forest(df)
    visualize(outliers, df)
    df_clean = remove_outliers(outliers, df)
    df_clean.to_csv(df_clean, index=False)


if __name__ == '__main__':
    remove_outliers(filename='Outliers.csv')

