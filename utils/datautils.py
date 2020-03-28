import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .datagenerator import DataGenerator


def load_data(path):
    df = pd.read_csv(path)

    df['FECHA_HORA'] = pd.to_datetime(df['FECHA_HORA'])
    df = df.set_index('FECHA_HORA')

    df = pd.concat([df.filter(regex=r) for r in ['O3', 'PM10', 'TMP']], axis=1)

    return df


def split_data_generator(df, horizont, window, batch_size):
    for i in range(10):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler_o3 = MinMaxScaler(feature_range=(0, 1))
        year = 2006 + i

        train = df[df.index.year != year]
        test = df[df.index.year == year]

        data_train = scaler.fit_transform(train.values)
        data_test = scaler.transform(test.values)

        scaler_o3.fit(train.iloc[:, 0].values.reshape(-1, 1))

        df_train = pd.DataFrame(data=data_train,
                                columns=train.columns)
        df_test = pd.DataFrame(data=data_test,
                               columns=test.columns)

        validation_row = int(len(df_train) * 0.9)

        train_generator = DataGenerator(df_train.iloc[:validation_row, :].values,
                                        df_train.iloc[:validation_row, 0].values, length=window,
                                        batch_size=batch_size, n_outputs=horizont)

        validation_generator = DataGenerator(df_train.iloc[validation_row:, :].values,
                                             df_train.iloc[validation_row:, 0].values, length=window,
                                             batch_size=batch_size, n_outputs=horizont)

        test_generator = DataGenerator(df_test.values, df_test.iloc[:, 0].values, length=window,
                                       batch_size=batch_size, n_outputs=horizont)

        yield train_generator, test_generator, validation_generator, scaler, scaler_o3, year
