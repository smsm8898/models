import tarfile
import yaml
from model.DeepFM import DeepFM


def get_data():
    pass


def get_loader():
    pass


def get_loss(loss_type):
    if loss_type == "BCE":
        return tf.keras.losses.BinaryCrossentropy()
    elif loss_type == "BFE":
        return tf.keras.losses.BinaryFocalCrossentropy()
    else:
        raise KeyError(f"{loss_type} is invalid")
    

def get_model(config):
    return DeepFM(config)


def get_metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.AUC(name="prc", curve="PR"),
    ]


def get_callbacks(config):
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", factor=config.decay_rate, patience=3, min_lr=0.0001),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        ...
    ]
    return callbacks


def get_config():
    with open(...) as f:
        default = yaml.load(f, Loader=yaml.FullLoader)
    return config


def train(args):
    # config
    config = get_config(args)
    
    # data
    data = get_data()
    ds = get_loader()
    
    # loss & optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    loss = get_loss(config.loss_type)

    # model
    model = get_model(config)
    
    metrics = get_metrics()

    # compile
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )

    # callbacks
    callbacks = get_callbacks(config)

    # train
    history = model.fit(
        ds,
        epochs=config.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    # save model as models
    tf.keras.models.save_model(model, config.model_base_path, save_format="tf")

    return model



if __name__ == "__main__":
    set_seed(42)
    args = get_args()

    train(args)
