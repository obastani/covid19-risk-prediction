import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Mode guide:
#
#   'spain': Build a predictor that omits new deaths as a feature. Include Spain in future data (but not training data).
#   'allcountries': Include all countries in the dataset.
#   'allcountriestest': Include all countries in the dataset, but only for testing.
#   'cases': Predict new cases instead of new deaths.
#   'localdata': Load the OWID data from the local copy instead of the web.
#

def check_mode(mode):
    if 'allcountries' in mode and 'allcountriestest' in mode:
        raise Exception()
    if 'allcountries' in mode and 'morecountriestest' in mode:
        raise Exception()
    if 'allcountriestest' in mode and 'morecountriestest' in mode:
        raise Exception()

def _get_mode_name(mode, s):
    s = mode.intersection(s)
    if len(s) == 0:
        return ''
    else:
        return '_' + '_'.join(s)

def get_model_name(mode):
    s = ['spain', 'allcountries', 'cases']
    mode_name = _get_mode_name(mode, s)
    return '../models/model{}.h5'.format(mode_name)

def get_results_name(mode, fname):
    s = ['spain', 'allcountries', 'allcountriestest', 'morecountriestest', 'cases']
    mode_name = _get_mode_name(mode, s)
    return '../tmp/csv/{}{}.csv'.format(fname, _get_mode_name(mode, s))

def get_all_features():
    features = ['new_cases_per_million', # 0
                'new_deaths_per_million', # 1
                'new_tests', # 2
                'total_tests', # 3
                'total_tests_per_thousand', # 4
                'new_tests_per_thousand', # 5
                'new_tests_smoothed', # 6
                'new_tests_smoothed_per_thousand', # 7
                #'tests_units', # This one causes a crash
                'stringency_index', # 8
                'population', # 9
                'population_density', # 10
                'median_age', # 11
                'aged_65_older', # 12
                'aged_70_older', # 13
                'gdp_per_capita', # 14
                'extreme_poverty', # 15
                'cardiovasc_death_rate', # 16
                'diabetes_prevalence', # 17
                'female_smokers', # 18
                'male_smokers', # 19
                'handwashing_facilities', # 20
                'hospital_beds_per_thousand', # 21
                'life_expectancy', # 22
                'numTests', # 23
                'eb_prev', # 24
                ]
    return features

def get_features(mode):
    features = get_all_features()
    return [features[8]] + [features[10]] + features[14:16]

def get_features_log(mode):
    features = get_all_features()
    if 'spain' in mode:
        return [features[0]] + features[6:8]
    else:
        return features[:2] + features[6:8]

def get_feature_pred(mode):
    features = get_all_features()
    if 'cases' in mode:
        return features[0]
    else:
        return features[1]

def get_feature_pred_alt(mode):
    features = get_all_features()
    return features[24]

def preprocess_dataset(df_owid, df_interest, mode):
    # Step 1: Get features
    features = list(set([get_feature_pred(mode)] + [get_feature_pred_alt(mode)] + get_features_log(mode) + get_features(mode)))

    # Step 2: Preprocess OWID data
    df_owid['date'] = pd.to_datetime(df_owid['date'], format='%Y-%m-%d')
    df_owid = df_owid.loc[df_owid['date'] >= pd.to_datetime('04/01/2020')]
    df_owid.loc[df_owid['new_cases_per_million'] < 0.0, 'new_cases_per_million'] = 0.0
    df_owid.loc[df_owid['new_deaths_per_million'] < 0.0, 'new_deaths_per_million'] = 0.0

    # Step 3: Preprocess data for countries of interest
    df_interest = df_interest.drop_duplicates()

    # Step 4: Preprocess PLF data
    df_plf = pd.read_csv('../../plf/aggPLF.csv')
    df_iso = pd.read_csv('../../plf/country-codes.csv')
    df_plf = pd.merge(df_plf, df_iso, how='inner', left_on='country', right_on=['ISO3166-1-Alpha-2'])
    df_plf['date'] = pd.to_datetime(df_plf['date'], format='%Y-%m-%d')
    #df_plf['eb_prev'] = 1.0 + 200.0 * df_plf['eb_prev']

    # Step 4: Merge data
    df = pd.merge(df_interest, df_owid, how='inner', left_on='ISO', right_on='iso_code')
    df = pd.merge(df, df_plf, how='left', left_on=['ISO', 'date'], right_on=['ISO3166-1-Alpha-3', 'date'])
    df = df[['ISO',  'date'] + features]
    df.fillna(-1.0, inplace=True)
    df = df.sort_values(by=['ISO','date'])

    # Step 5: Handle Spain
    if not 'spain' in mode:
        df = df[df['ISO'] != 'ESP']

    return df

def get_countries_interest(flag):
    if flag == 'all':
        df_interest = pd.read_csv('../data/interest_all.csv')
    elif flag == 'more':
        df_interest = pd.read_csv('../data/interest_more.csv')
    elif flag == '':
        df_interest = pd.read_csv('../data/interest.csv')
    else:
        raise Exception()
    return df_interest

def get_countries_interest_set(flag):
    df_interest = get_countries_interest(flag)
    return set(df_interest['ISO'].values)

def get_countries_interest_flag(mode, is_train):
    if is_train:
        if 'allcountries' in mode:
            return 'all'
        else:
            return ''
    else:
        if 'allcountries' in mode or 'allcountriestest' in mode:
            return 'all'
        elif 'morecountriestest' in mode:
            return 'more'
        else:
            return ''

def get_dataset(mode):
    # Step 1: Get OWID data
    if 'localdata' in mode:
        df_owid = pd.read_csv('../data/owid.csv')
    else:
        url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
        df_owid = pd.read_csv(url)
        df_owid.to_csv('../data/owid.csv', index=False)

    # Step 2: Get countries of interest
    df_interest = get_countries_interest('all')

    return preprocess_dataset(df_owid, df_interest, mode)

# Build the prediction dataset (x_1, ..., x_T) -> (x_{T+N}, ..., x_{T+N+M}), where
#
#   history_size = T - 1
#   future_window_start = N
#   future_window_end = N+M+1
#
# start index: int (earliest point in zip(df_x, df_y) to consider)
# end_index: int | None (last point in zip(df_x, df_y) to consider, consider all points if None)
# history_size: int (the length of the history to use)
def build_dataset_country(df_x, df_z, df_y, start_ind, end_ind, history_len, future_offset, future_len):
    # Step 1: Initialization
    x = []
    z = []
    y = []
    d = []

    # Step 3: Handle unspecified history
    if start_ind is None:
        start_ind = 0
    if end_ind is None:
        end_ind = len(df_x)

    # Step 2: Only consider points where we have the whole history
    start_ind = max(start_ind, start_ind + history_len)
    end_ind = min(end_ind, len(df_x) - future_offset - future_len)

    # Step 4: Construct data points
    for i in range(start_ind, end_ind):
        # Step 4a: Inputs are indices i-T, ..., i
        x.append(df_x[i-history_len:i+1])

        # Step 4b: Alternate output is index i
        z.append(df_z[i][0])

        # Step 4c: Outputs are indices i+N, ..., i+N+M
        y.append(np.sum(df_y[i+future_offset:i+future_offset+future_len]))

        # Step 4c: Append data
        d.append(i)

    return x, z, y, d

def build_dataset(df, mode):
    # Step 1: Setup
    countries = df['ISO'].drop_duplicates().values
    countries_interest_train = get_countries_interest_set(get_countries_interest_flag(mode, True))
    countries_interest = get_countries_interest_set(get_countries_interest_flag(mode, False))
    n_train = max([len(df.loc[df['ISO'] == country]) for country in countries]) - 30
    feature_pred = get_feature_pred(mode)
    feature_pred_alt = get_feature_pred_alt(mode)
    features = get_features(mode)
    features_log = get_features_log(mode)

    # Step 2: Initialization
    xs_train = []
    zs_train = []
    ys_train = []
    ds_train = []
    xs_test = []
    zs_test = []
    ys_test = []
    ds_test = []
    xs_future = []
    ds_future = []
    zs_future = []
    xs_all = []
    ys_all = []
    zs_all = []
    ds_all = []

    # Step 3: Construct data for each country
    for country in countries:
        # Step 3a: Ignore country if not of interest
        if not country in countries_interest:
            continue

        # Step 3b: Select data for the current country
        df_country = df.loc[df['ISO'] == country]
        dates = [pd.to_datetime(date) for date in df_country['date'].values]

        # Step 3c: Build features for which we want to use their logarithm
        df_x_log = df_country[features_log].values
        df_x_log = np.where(df_x_log >= 0.0, np.log(df_x_log + 1.0), df_x_log)

        # Step 3d: Build features
        df_x = np.concatenate([df_x_log, df_country[features].values], axis=1)

        # Step 3e: Build targets
        df_y = df_country[[feature_pred]].values
        df_z = df_country[[feature_pred_alt]].values

        # Step 3f: Build future data
        x_future, z_future, _, d_future = build_dataset_country(df_x, df_z, df_y, None, None, 10, 0, 0)
        xs_future += x_future
        zs_future += z_future
        ds_future += [(country, dates[d]) for d in d_future]

        # Step 3g: Ignore Spain except for future data
        if country == 'ESP':
            continue

        # Step 3h: Build validation data
        x_test, z_test, y_test, d_test = build_dataset_country(df_x, df_z, df_y, n_train, None, 10, 5, 10)
        xs_test += x_test
        zs_test += z_test
        ys_test += y_test
        ds_test += [(country, dates[d]) for d in d_test]

        # Step 3i: Build all data
        x_all, z_all, y_all, d_all = build_dataset_country(df_x, df_z, df_y, 0, None, 10, 5, 10)
        xs_all += x_all
        zs_all += z_all
        ys_all += y_all
        ds_all += [(country, dates[d]) for d in d_all]

        # Step 3j: Ignore country if not of interest for training
        if not country in countries_interest_train:
            continue

        # Step 3k: Build training data
        x_train, z_train, y_train, d_train = build_dataset_country(df_x, df_z, df_y, None, n_train, 10, 5, 10)
        xs_train += x_train
        zs_train += z_train
        ys_train += y_train
        ds_train += [(country, dates[d]) for d in d_train]

    # Step 4: Postprocessing
    xs_train = np.array(xs_train)
    zs_train = np.array(zs_train)
    ys_train = np.array(ys_train)
    xs_test = np.array(xs_test)
    zs_test = np.array(zs_test)
    ys_test = np.array(ys_test)
    xs_future = np.array(xs_future)
    zs_future = np.array(zs_future)
    xs_all = np.array(xs_all)
    zs_all = np.array(zs_all)
    ys_all = np.array(ys_all)

    print('Training x dimension: {}'.format(xs_train.shape))
    print('Training z dimension: {}'.format(zs_train.shape))
    print('Training y dimension: {}'.format(ys_train.shape))
    print('# Training countries: {}'.format(len(set([country for country, date in ds_train]))))
    print('# Training dates: {}'.format(len(set([date for country, date in ds_train]))))
    print('Test x dimension: {}'.format(xs_test.shape))
    print('Test z dimension: {}'.format(zs_test.shape))
    print('Test y dimension: {}'.format(ys_test.shape))
    print('# Test countries: {}'.format(len(set([country for country, date in ds_test]))))
    print('# Test dates: {}'.format(len(set([date for country, date in ds_test]))))
    print('All x dimension: {}'.format(xs_all.shape))
    print('All z dimension: {}'.format(zs_all.shape))
    print('All y dimension: {}'.format(ys_all.shape))
    print('# All countries: {}'.format(len(set([country for country, date in ds_all]))))
    print('# All dates: {}'.format(len(set([date for country, date in ds_all]))))
    print('Future x dimension: {}'.format(xs_future.shape))
    print('Future z dimension: {}'.format(zs_future.shape))
    print('# Future countries: {}'.format(len(set([country for country, date in ds_future]))))
    print('# Future dates: {}'.format(len(set([date for country, date in ds_future]))))

    return countries, xs_train, zs_train, ys_train, ds_train, xs_test, zs_test, ys_test, ds_test, xs_future, zs_future, ds_future, xs_all, zs_all, ys_all, ds_all

# Helper class for the model
class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(ScaleLayer, self).__init__(name=name)
        self.scale = tf.Variable(1.0, trainable=True)

    def call(self, inputs):
        return tf.stack([inputs[:,0], inputs[:,1] * self.scale], axis=1)

# Helper function for the model
def tf_gaussian(t):
    return tfp.distributions.Normal(loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))

def build_model(shape):
    inputs = tf.keras.Input(shape=shape)
    vals = (tf.keras.layers.BatchNormalization())(inputs)
    vals = (tf.keras.layers.LSTM(20, return_sequences=True))(vals)
    vals = (tf.keras.layers.LSTM(20, activation='relu'))(vals)
    outputs_mean = (tf.keras.layers.Dense(1))(vals)
    vals = (tf.keras.layers.Dense(1, name='std_dense'))(vals)
    vals = tf.keras.layers.concatenate([outputs_mean, vals])
    vals = (ScaleLayer(name='std_scale'))(vals)
    outputs = (tfp.layers.DistributionLambda(tf_gaussian))(vals)
    return inputs, outputs_mean, outputs

def train_model(xs_train, ys_train, xs_test, ys_test, mode):
    # Step 0: Parameters
    BATCH_SIZE = 1000
    BUFFER_SIZE = 1000
    EPOCHS_MEAN_MIN = 5
    EPOCHS_MEAN = 20
    EPOCHS_STD = 20
    EPOCHS_CALIBRATE = 20
    EVALUATION_INTERVAL = 200

    # Step 1: Calibration training and validation sets
    np.random.seed(13)
    xs_train, xs_val, ys_train, ys_val = train_test_split(xs_train, ys_train, test_size=0.1)

    # Step 2: Get best hyperparameters of model for predicting mean
    best = None
    for seed in range(10, 15):
        # Step 2a: Set seed
        tf.random.set_seed(seed)

        # Step 2b: Training, validation, calibration, and test data
        train_data = tf.data.Dataset.from_tensor_slices((xs_train, ys_train))
        train_data = train_data.cache().batch(BATCH_SIZE).repeat()
        val_data = tf.data.Dataset.from_tensor_slices((xs_val, ys_val))
        val_data = val_data.batch(BATCH_SIZE).repeat()

        # Step 3c: Model architecture
        inputs, outputs_mean, outputs = build_model(xs_train.shape[1:])

        # Step 3d: Train model for predicting the mean
        model = tf.keras.Model(inputs=inputs, outputs=outputs_mean)
        model.compile(optimizer='adam', loss='MeanSquaredError')
        history_mean = model.fit(train_data,
                                 epochs=EPOCHS_MEAN,
                                 steps_per_epoch=EVALUATION_INTERVAL,
                                 validation_data=val_data,
                                 validation_steps=50)

        # Step 3e: Save best
        best_cur_arg = EPOCHS_MEAN_MIN + np.argmin(history_mean.history['val_loss'][EPOCHS_MEAN_MIN:])
        best_cur = np.min(history_mean.history['val_loss'][EPOCHS_MEAN_MIN:])
        if best is None or best_cur < best[2]:
            best = (seed, best_cur_arg, best_cur)
        print(best_cur_arg, best_cur)
        print(best)

    # Step 4: Set seed
    tf.random.set_seed(best[0])

    # Step 5: Training and validation data
    train_data = tf.data.Dataset.from_tensor_slices((xs_train, ys_train))
    train_data = train_data.cache().batch(BATCH_SIZE).repeat()
    val_data = tf.data.Dataset.from_tensor_slices((xs_val, ys_val))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    # Step 6: Train model for predicting mean
    inputs, outputs_mean, outputs = build_model(xs_train.shape[1:])
    model = tf.keras.Model(inputs=inputs, outputs=outputs_mean)
    model.compile(optimizer='adam', loss='MeanSquaredError')
    history_mean = model.fit(train_data,
                             epochs=best[1]+1,
                             steps_per_epoch=EVALUATION_INTERVAL,
                             validation_data=val_data,
                             validation_steps=50)

    # Step 7: Calibration and test data
    xs_train_cal, xs_val_cal, ys_train_cal, ys_val_cal = train_test_split(xs_val, ys_val, test_size=0.5)
    cal_data = tf.data.Dataset.from_tensor_slices((xs_train_cal, ys_train_cal))
    cal_data = cal_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    test_data = tf.data.Dataset.from_tensor_slices((xs_val_cal, ys_val_cal))
    test_data = test_data.batch(BATCH_SIZE).repeat()

    # Step 8: Train model for predicting std (i.e., fix layers for mean)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    for layer in model.layers:
        layer.trainable = False
    model.get_layer('std_dense').trainable = True
    negloglik = lambda y, p_y: -p_y.log_prob(y)
    model.compile(optimizer='adam', loss=negloglik)
    history_std = model.fit(train_data,
                            epochs=EPOCHS_STD,
                            steps_per_epoch=EVALUATION_INTERVAL,
                            validation_data=val_data,
                            validation_steps=50)

    # Step 9: Calibrate model
    model.get_layer('std_dense').trainable = False
    model.get_layer('std_scale').trainable = True
    model.compile(optimizer='adam', loss=negloglik)
    history_cal = model.fit(cal_data,
                            epochs=EPOCHS_CALIBRATE,
                            steps_per_epoch=EVALUATION_INTERVAL,
                            validation_data=test_data,
                            validation_steps=50)

    # Step 10: Save model
    path = get_model_name(mode)
    print('Saving model to path: {}'.format(path))
    model.save_weights(path)

    # Step 11: Concatenate history
    history = pd.concat([pd.DataFrame(history_mean.history), pd.DataFrame(history_std.history), pd.DataFrame(history_cal.history)])

    # Step 12: Test model
    test_model(model, xs_train, ys_train, xs_test, ys_test)

    return model, history

def load_model(xs_train, ys_train, xs_test, ys_test, mode):
    # Step 1: Calibration training and validation sets
    np.random.seed(13) # np seed should stay constant so the validation set does not change
    xs_train, xs_val, ys_train, ys_val = train_test_split(xs_train, ys_train, test_size=0.1)

    # Step 2: Load model
    inputs, _, outputs = build_model(xs_train.shape[1:])
    path = get_model_name(mode)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print('Loading model from path: {}'.format(path))
    model.load_weights(path)

    # Step 3: Test model
    test_model(model, xs_train, ys_train, xs_test, ys_test)

    return model

def test_model(model, xs_train, ys_train, xs_test, ys_test):
    def loss(ys_pred):
        return np.mean(np.square(ys_pred - ys_test))

    # Neural network
    ys_test_pred = model(xs_test).mean().numpy()[:,0]
    print('NN MSE: {}'.format(loss(ys_test_pred)))

    # Constant predictor (for comparison)
    ys_const = np.mean(ys_train)
    print('Constant Prediction: {}'.format(ys_const))
    print('Constant Prediction MSE: {}'.format(loss(ys_const)))

    # Sum predictor (for comparison)
    ys_sum_mult = np.median(ys_train / (np.sum(xs_train[:, :, 1], axis=1) + 1e-5))
    ys_sum = np.sum(xs_test[:, :, 1], axis=1) * ys_sum_mult
    print('Sum Prediction MSE: {}'.format(loss(ys_sum)))

def run_model(model, xs, fix_neg, mode):
    means = model(xs).mean().numpy()
    if fix_neg:
        means = np.where(means < 0.0, 0.0, means)
    stddevs = model(xs).stddev().numpy()
    result = np.concatenate([means, stddevs], axis=1)
    return result

def run_results_alt(model, xs_future, zs_future, ds_future, mode):
    # Step 1: Get countries of interest
    countries = sorted(get_countries_interest_set(get_countries_interest_flag(mode, False)))

    # Step 2: Get predicted outcomes
    ys_pred_future = run_model(model, xs_future, True, mode)

    # Step 3: Compute the average normalization constant
    inds = zs_future >= 0.0
    score_norm_default = np.sum(zs_future[inds]) / (1e-5 + np.sum(ys_pred_future[inds,0]))

    # Step 4: Construct normalization constants by country
    score_norm = {}
    for country in countries:
        inds = np.array([z >= 0.0 and d[0] == country for z, d in zip(zs_future, ds_future)])
        if np.sum(inds) == 0:
            score_norm[country] = score_norm_default
        else:
            score_norm[country] = np.sum(zs_future[inds]) / (1e-5 + np.sum(ys_pred_future[inds,0]))

    # Step 5: Get scores
    ss_future = np.array([score_norm[d[0]] for d in ds_future])

    # Step 6: Get predicted value
    zs_pred_future = np.array([s * y for s, y in zip(ss_future, ys_pred_future)])

    # Step 7: Compute confidence set
    c = learn_conf_alt(model, 0.05, 0.05, xs_future, zs_future, ss_future)

    # Step 8: Compute results
    result = [(country, str(date.date()), mean, c * stddev, max(z, 0.0)) for (country, date), (mean, stddev), z in zip(ds_future, zs_pred_future, zs_future)]
    path = get_results_name(mode, 'futurealt')
    print('Writing results to: {}'.format(path))
    f = open(path, 'w')
    f.write('country,date,ypred,yerr,ytrue\n')
    for tuple in result:
        f.write(','.join([str(t) for t in tuple]) + '\n')
    f.close()

def learn_conf_alt(model, e, d, xs, zs, ss):
    # Step 1: Check inputs
    if len(xs) != len(zs) or len(xs) != len(ss):
        raise Exception()

    # Step 2: Compute k
    k = compute_k(len(zs), e, d)
    if k is None:
        return np.inf

    # Step 3: Compute scores
    scores = []
    for i, (x, z, s) in enumerate(zip(xs, zs, ss)):
        if i%1000 == 0:
            print(i, len(xs))
        if z < 0.0:
            continue
        scores.append(float(model(np.array([x])).log_prob(z / s)[0,0]))

    # Step 4: Compute threshold
    scores = sorted(scores)
    c = np.sqrt(-2.0 * scores[k] - np.log(2.0 * np.pi))

    return c

def write_results(model, c, fname, xs, ys, ds, mode):
    if ys is None:
        ys = ['NA' for d in ds]
    result = run_model(model, xs, True, mode)
    result = [(country, str(date.date()), mean, c * stddev, y) for (country, date), (mean, stddev), y in zip(ds, result, ys)]
    path = get_results_name(mode, fname)
    print('Writing results to: {}'.format(path))
    f = open(path, 'w')
    f.write('country,date,ypred,yerr,ytrue\n')
    for tuple in result:
        f.write(','.join([str(t) for t in tuple]) + '\n')
    f.close()

def run_results(model, xs_test, ys_test, ds_test, xs_all, ys_all, ds_all, xs_future, ds_future, mode):
    # Step 1: Compute confidence set
    c = learn_conf(model, 0.05, 0.05, xs_test, ys_test)

    # Step 2: Write results
    write_results(model, c, 'all', xs_all, ys_all, ds_all, mode)
    write_results(model, c, 'future', xs_future, None, ds_future, mode)

# Learn the confidence set threshold.
#
# model: [X] -> [Y]
# e: float
# d: float
# xs: [X]
# ys: [Y]
# X = np.array([input_dim]) (input)
# Y = np.array([ground_truth_input_dim]) (ground truth input)
# return: float (multiply the stddev by this value to get the confidence set)
def learn_conf(model, e, d, xs, ys):
    # Step 1: Check inputs
    if len(xs) != len(ys):
        raise Exception()

    # Step 2: Compute k
    k = compute_k(len(xs), e, d)
    if k is None:
        return np.inf

    # Step 3: Compute scores
    zs = score(model, xs, ys)

    # Step 4: Compute threshold
    zs = sorted(zs)
    c = np.sqrt(-2.0 * zs[k] - np.log(2.0 * np.pi))

    return c

# Compute the log-likelihood of the true labels
# predicted by the model.
#
# model: X -> Y
# xs: [X]
# ys: [Y]
def score(model, xs, ys):
    return [float(model(np.array([x])).log_prob(y)[0,0]) for x, y in zip(xs, ys)]

# Compute k using the recursive equations
#
# r_0 = n * log(1 - e)
# r_h = (log(n) + ... + log(n - h + 1)) - (log(h) + ... + log(1)) + h * log(e) + (n - h) * log(1 - e)
#     = r_{h-1} + log(n - h + 1) - log(h) + log(e) - log(1 - e)
# s_{-1} = 0
# s_h    = exp(r_0) + ... + exp(r_h)
#        = s_{h-1} + exp(r_h)
def compute_k(n, e, d):
    r = 0.0
    s = 0.0
    for h in range(n + 1):
        if h == 0:
            r = n * np.log(1.0 - e)
        else:
            r += np.log(n - h + 1) - np.log(h) + np.log(e) - np.log(1.0 - e)
        s += np.exp(r)
        if s > d:
            if h == 0:
                return None
            else:
                return h - 1
    return n
