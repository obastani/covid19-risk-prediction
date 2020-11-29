import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
sns.set()

def load_data(mode):
    # Step 1: Read OWID data
    df_owid = pd.read_csv('../data/owid.csv')
    df_owid = df_owid[df_owid['date'] >= '2020-07-01']

    # Step 2: Read results data
    df_res = pd.read_csv('../tmp/csv/futurealt{}.csv'.format(mode))
    df_res = df_res[df_res['date'] >= '2020-07-01']

    # Step 3: Merge
    df = pd.merge(df_owid, df_res, how='inner', left_on=['iso_code', 'date'], right_on=['country', 'date'])

    # Step 4: Compute score
    countries = df['country'].drop_duplicates().values
    window = 7
    df['score'] = 0.0
    for country in countries:
        df_country = df[df['country'] == country]
        for i in range(window, len(df_country)):
            score = np.mean(df_country.iloc[i-window:i]['ypred'])
            df.loc[(df['country'] == country) & (df['date'] == df_country.iloc[i]['date']), 'score'] = score

    # Step 5: Compute smoothed cases
    df['cases_smooth'] = 0.0
    for country in countries:
        df_country = df[df['country'] == country]
        for i in range(window, len(df_country)):
            cases = np.mean(df_country.iloc[i-window:i]['new_cases_per_million'])
            df.loc[(df['country'] == country) & (df['date'] == df_country.iloc[i]['date']), 'cases_smooth'] = cases

    return df

def plot(df, country, save_name):
    colors = sns.color_palette('husl', 3)
    with sns.axes_style('darkgrid'):
        # Step 1: Preprocess
        df_country = df[df['country'] == country]
        xs = df_country['date'].values

        # Step 2: Plot cases
        ys_cases = df_country['new_cases_per_million'].values
        #ys_cases = df_country['cases_smooth'].values
        plt.plot(xs, ys_cases, color=colors[0])
        plt.ylabel('# Cases / Million', color=colors[0])

        # Step 3: Label for x-axis
        plt.xlabel('Date')

        # Step 4: Include second y-axis
        plt.twinx()

        # Step 5: Plot true data
        ys_pred = df_country['score'].values
        plt.plot(xs, ys_pred, color=colors[1], linestyle='dashed', label='Predicted Score')

        # Step 6: Plot true data
        ys_true = df_country['ytrue'].values
        plt.plot(xs, ys_true, color=colors[1], label='Testing % Positive')

        # Step 7: Label for y-axis
        plt.ylabel('Testing % Positive', color=colors[1])

        # Step 6: Cleanup
        plt.tight_layout()

        # Step 8: Legend
        plt.legend()

    # Step 9: x-axis labels
    dates = list(sorted(df['date'].drop_duplicates().values))
    dates = [date[5:] for date in dates]
    dates = [(datetime.datetime.strptime(date, '%m-%d') + datetime.timedelta(days=10)).strftime("%m-%d") for date in dates]
    date_skip = 14
    plt.xticks(list(range(0, len(dates), date_skip)), dates[::date_skip])

    # Step 4: Save plot
    if save_name is None:
        plt.show()
    else:
        plt.savefig('../tmp/plots_alt/{}.png'.format(save_name))
        plt.close()

def plot_all():
    df = load_data('')
    countries = df['country'].drop_duplicates().values
    #countries = ['MLT', 'ROU', 'JPN', 'CHE', 'CZE', 'BGR', 'NLD', 'BEL', 'SWE']
    for country in countries:
        plot(df, country, country)

    df = load_data('_spain')
    plot(df, 'ESP', 'ESP')

if __name__ == '__main__':
    plot_all()
