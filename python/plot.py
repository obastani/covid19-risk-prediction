import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
sns.set()

def plot(countries, mode_name, save_name):
    # Step 1: Load data
    df_all = pd.read_csv('../tmp/csv/all{}.csv'.format(mode_name))
    df_future = pd.read_csv('../tmp/csv/future{}.csv'.format(mode_name))

    # Step 2: Plotting
    colors = sns.color_palette('husl', len(countries))
    with sns.axes_style('darkgrid'):
        for i, country in enumerate(countries):
            # Step 2a: Plot predicted data
            df_future_country = df_future[df_future['country'] == country]
            xs_pred = df_future_country['date'].values
            ys_pred = df_future_country['ypred'].values
            plt.plot(xs_pred, ys_pred, color=colors[i], linestyle='dashed')

            # Step 2b: Plot true data
            df_all_country = df_all[df_all['country'] == country]
            xs_true = df_all_country['date'].values
            ys_true = df_all_country['ytrue'].values
            plt.plot(xs_true, ys_true, color=colors[i], label=country)

            # Step 2c: Plot predicted errors
            ys_err = df_future_country['yerr'].values
            ys_err_min = np.where(ys_pred - ys_err < 0.0, 0.0, ys_pred - ys_err)
            ys_err_max = ys_pred + ys_err
            plt.fill_between(xs_pred, ys_err_min, ys_err_max, alpha=0.3, facecolor=colors[i])

    # Step 3: Plotting details

    # Step 3a: Legend
    plt.legend()

    # Step 3b: x-axis labels
    dates = list(sorted(df_future['date'].drop_duplicates().values))
    dates = [date[5:] for date in dates]
    dates = [(datetime.datetime.strptime(date, '%m-%d') + datetime.timedelta(days=10)).strftime("%m-%d") for date in dates]
    date_skip = 14
    plt.xticks(list(range(0, len(dates), date_skip)), dates[::date_skip])

    # Step 3c: Axis titles
    plt.xlabel('Date')
    plt.ylabel('Predicted # Deaths / Million People')

    # Step 4: Save plot
    if save_name is None:
        plt.show()
    else:
        plt.savefig('../tmp/plots/{}{}.png'.format(save_name, mode_name))
        plt.close()

def plot_all(mode_name):
    df_future = pd.read_csv('../tmp/csv/future{}.csv'.format(mode_name))
    countries = df_future['country'].drop_duplicates().values
    for country in countries:
        plot([country], mode_name, country)

if __name__ == '__main__':
    #mode_names = ['', '_allcountriestest', '_allcountries', '_spain', '_cases']
    mode_names = ['']
    for mode_name in mode_names:
        plot_all(mode_name)

    #countries = ['GBR', 'DEU', 'FRA', 'ISR', 'USA']
    #plot(countries, '_allcountries', 'plot')

    #countries = ['GRC', 'CYP']
    #plot(countries, '', 'plot_cg')

    plot(['ESP'], '_spain', 'ESP')
