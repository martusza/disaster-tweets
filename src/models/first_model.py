import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def model_1(num_words, embedding):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(num_words, embedding),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=3,
                               activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=5,
                               activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.summary()
    return model


def prepare_text_datasets(train_dataset,
                          tokenizer=None,
                          pad=None
                          ):

    tokenizer = Tokenizer(
        **tokenizer
    )

    tokenizer.fit_on_texts(train_dataset)
    train_sequences = tokenizer.texts_to_sequences(train_dataset)
    train_padded = pad_sequences(
        sequences=train_sequences,
        **pad
    )

    return train_padded
