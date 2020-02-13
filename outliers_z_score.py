import pandas as pd
import collections
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from functools import partial


def detect_outliers(df, lookback_window_size=18, partially_bad_threshold=20.0, fully_bad_threshold=30.0):
    lookback_prices = collections.deque(maxlen=lookback_window_size)
    bad_ids = []
    fully_bad_ids = []
    for ids, row in df.iterrows():
        current_price = row['Price']
        if len(lookback_prices) < 2:
            lookback_prices.append(current_price)
        else:
            check_partially_bad = abs(current_price - mean(lookback_prices)) > partially_bad_threshold * std(
                lookback_prices)
            check_fully_bad = abs(current_price - mean(lookback_prices)) > fully_bad_threshold * std(lookback_prices)
            if check_partially_bad:
                bad_ids.append(ids)
            elif check_fully_bad:
                fully_bad_ids.append(ids)
            if not check_fully_bad and current_price != lookback_prices[-1]:
                lookback_prices.append(current_price)
    outlier_ids = bad_ids + fully_bad_ids
    return outlier_ids, df


def visualize(bad_ids, df):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(df['Date'], df['Price'], c='blue')
    ax.scatter(df.loc[bad_ids, 'Date'], df.loc[bad_ids, 'Price'], c='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.show()


def remove_bad_ids(bad_ids, df):
    df.drop(bad_ids, axis=0, inplace=True)
    df.reset_index(inplace=True)
    return df


algos = {
    'detect_outliers': partial(detect_outliers, lookback_window_size=18, partially_bad_threshold=20.0,
                               fully_bad_threshold=30.0)
}


def load_data(filename, method='local'):
    df = pd.DataFrame()
    if method == 'local':
        df = pd.read_csv(filename)
    elif method == 'remote':
        # add method to load from remote
        df = pd.DataFrame()
    elif method == 'database':
        # add method to load from a database
        df = pd.DataFrame()

    return df


def remove_outliers(filename='Outliers.csv'):
    df = load_data(filename)
    bad_ids, df = algos['detect_outliers'](df)
    visualize(bad_ids, df)
    print('We have found {} outliers:\n'.format(len(bad_ids)))
    print(df.loc[bad_ids])
    print('Removing these prices')
    df = remove_bad_ids(bad_ids, df)
    clean_file = 'cleaned_' + filename
    print('Saving to the file {}'.format(clean_file))
    df.to_csv(clean_file)


if __name__ == '__main__':
    remove_outliers()
