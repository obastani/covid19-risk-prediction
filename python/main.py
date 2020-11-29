from train import *

def run(is_train, mode):
    # Step 1: Check mode
    check_mode(mode)

    # Step 2: Get dataset
    df = get_dataset(mode)

    # Step 3: Build dataset
    countries, xs_train, zs_train, ys_train, ds_train, xs_test, zs_test, ys_test, ds_test, xs_future, zs_future, ds_future, xs_all, zs_all, ys_all, ds_all = build_dataset(df, mode)

    # Step 4: Train model
    if is_train:
        model, history = train_model(xs_train, ys_train, xs_test, ys_test, mode)
    else:
        model = load_model(xs_train, ys_train, xs_test, ys_test, mode)

    # Step 5: Run results
    run_results(model, xs_test, ys_test, ds_test, xs_all, ys_all, ds_all, xs_future, ds_future, mode)

    # Step 6: Get normalization constants
    #run_results_alt(model, xs_future, zs_future, ds_future, mode)

if __name__ == '__main__':
    #mode = set()
    #mode = set(['cases'])
    #mode = set(['allcountries'])
    #mode = set(['allcountriestest'])
    #mode = set(['spain'])

    run(True, set())
    run(False, set(['morecountriestest']))
    run(False, set(['allcountriestest']))
    run(True, set(['spain']))
