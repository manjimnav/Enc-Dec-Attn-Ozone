import tensorflow as tf


def build_model(window, horizont):
    inp = tf.keras.layers.Input(shape=(window, 3))
    out = tf.keras.layers.LSTM(256, activation='relu', return_sequences=True, return_state=True)(inp)
    out = tf.keras.layers.Attention()(out)
    out = tf.keras.layers.LSTM(256, activation='relu')(out)
    out = tf.keras.layers.Dense(horizont)(out)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


def initialize_callbacks(path, year):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                     patience=5, min_lr=0.0002)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto',
                                                      restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(path + '/cp-' + str(year) + '-{epoch:04d}.ckpt', monitor='val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=False, mode='auto', period=1)

    return reduce_lr, early_stopping, checkpoint
