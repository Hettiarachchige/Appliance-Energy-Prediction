def run_tuning(X_train, y_train, build_model_fn, tuner_path):
    import keras_tuner as kt
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    import os
    if os.path.exists(tuner_path):
        tf.io.gfile.rmtree(tuner_path)
    tuner = kt.BayesianOptimization(build_model_fn, objective='val_loss', max_trials=10, directory=tuner_path, project_name='energy_prediction')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tuner.search(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32, callbacks=[early_stop])
    return tuner.get_best_models(1)[0], tuner.get_best_hyperparameters(1)[0]
