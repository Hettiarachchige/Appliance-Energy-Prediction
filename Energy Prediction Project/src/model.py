def build_lstm_model(hp, input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.regularizers import l2
    import tensorflow as tf
    model = Sequential()
    model.add(Input(shape=input_shape))
    for i in range(hp.Int("num_layers", 1, 2)):
        units = hp.Int(f"units_{i}", 32, 128, step=32)
        return_seq = i < hp.Int("num_layers", 1, 2) - 1
        reg = l2(hp.Float("l2", 0.0, 0.01, step=0.001))
        model.add(LSTM(units, return_sequences=return_seq, activation='tanh', kernel_regularizer=reg))
        model.add(Dropout(hp.Float(f"dropout_{i}", 0.1, 0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
                  loss="mse", metrics=["mae"])
    return model