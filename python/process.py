import numpy as np
import pandas as pd
import datetime

def run(mode):
    # Step 0: Parameters
    window = 7

    # Step 1: Initialization
    scores = []

    # Step 2: Countries of interest
    df = pd.read_csv('../tmp/csv/future{}.csv'.format(mode))
    cur_date = max(df['date'].drop_duplicates().values)
    date = datetime.datetime.strftime(datetime.datetime.strptime(cur_date, '%Y-%m-%d') - datetime.timedelta(days=window-1), '%Y-%m-%d')
    print('Averaging from date {} to date {}'.format(date, cur_date))
    df = df[df['date'] >= date]
    countries = df['country'].drop_duplicates().values
    for country in countries:
        df_country = df[df['country'] == country]
        scores.append((country, np.mean(df_country['ypred'])))

    # Step 3: Spain
    df_spain = pd.read_csv('../tmp/csv/future_spain.csv')
    df_spain = df[df['date'] >= date]
    scores.append(('ESP', np.mean(df_spain[df_spain['country'] == 'ESP']['ypred'])))

    # Step 4: Sort scores
    scores = sorted(scores, key=lambda s: -s[1])

    # Step 5: Write results to file
    f = open('../tmp/csv/ranking{}.csv'.format(mode), 'w')
    f.write('country,score\n')
    for i, (country, score) in enumerate(scores):
        f.write('{},{}\n'.format(country, score))
    f.close()

if __name__ == '__main__':
    run('')
    run('_allcountriestest')
