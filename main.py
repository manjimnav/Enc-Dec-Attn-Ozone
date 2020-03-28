import argparse
import os
import time

import pandas as pd

from utils.datautils import load_data, split_data_generator
from utils.metricsutils import calculate_metrics
from utils.modelutils import build_model, initialize_callbacks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Path to the data", required=True)
    parser.add_argument("-c", "--checkpoint", help="Checkpoint path", required=True)
    parser.add_argument("-w", "--window", help="Window size", default=24)
    parser.add_argument("-ho", "--horizon", help="Horizon size", default=24)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=100)
    parser.add_argument("-b", "--batch", help="Batch size", default=32)
    parser.add_argument("-y", "--year", help="Initial year", default=2006)

    args = parser.parse_args()

    df = load_data(args.data)

    window = args.window
    horizon = args.horizon
    batch_size = args.batch
    checkpoint_path = args.checkpoint
    epochs = args.epochs

    metrics_file = os.path.splitext(os.path.basename(args.data))[0] + '_metrics.csv'
    if os.path.isfile(metrics_file):
        total_metrics = pd.read_csv(metrics_file)
    else:
        total_metrics = None

    for training_generator, test_generator, validation_generator, scaler, scaler_o3, year in split_data_generator(df,
                                                                                                                  horizon,
                                                                                                                  window,
                                                                                                                  batch_size,
                                                                                                                  args.year):
        print("Training for year: " + str(year))
        start_time = time.time()

        model = build_model(window, horizon)
        reduce_lr, es, checkpoint = initialize_callbacks(checkpoint_path, year)

        model.fit(training_generator,
                  validation_data=validation_generator,
                  use_multiprocessing=True,
                  callbacks=[reduce_lr, es, checkpoint],
                  epochs=epochs)

        metrics = calculate_metrics(model, test_generator, year, scaler_o3)

        if total_metrics is None:
            total_metrics = metrics
        else:
            total_metrics = pd.concat([total_metrics, metrics])

        total_metrics.to_csv(metrics_file)
        print("--- Execution time %s minutes ---" % ((time.time() - start_time) / 60))
