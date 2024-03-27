import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text as text
# from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt
from transformers import TFBertModel
from keras import utils


MODEL_CHECKPOINT = "bert-base-uncased"
# bert = TFBertModel.from_pretrained(model_checkpoint)
# bert.trainable = True

def getData():
    tf.get_logger().setLevel('ERROR')

    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                      untar=True, cache_dir='.',
                                      cache_subdir='')

    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

    train_dir = os.path.join(dataset_dir, 'train')

    # remove unused folders to make it easier to load the data
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

def train_val_test_split():
    AUTOTUNE = tf.data.AUTOTUNE
    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test',
        batch_size=batch_size)

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names

def choose_model():
    bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'

    map_name_to_handle = {
        'bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
        'bert_en_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
        'bert_multi_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
        'small_bert/bert_en_uncased_L-2_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-2_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-2_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-2_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-4_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-4_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-4_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-4_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-6_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-6_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-6_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-6_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-8_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-8_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-8_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-8_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-10_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-10_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-10_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-10_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-12_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-12_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-12_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
        'albert_en_base':
            'https://tfhub.dev/tensorflow/albert_en_base/2',
        'electra_small':
            'https://tfhub.dev/google/electra_small/2',
        'electra_base':
            'https://tfhub.dev/google/electra_base/2',
        'experts_pubmed':
            'https://tfhub.dev/google/experts/bert/pubmed/2',
        'experts_wiki_books':
            'https://tfhub.dev/google/experts/bert/wiki_books/2',
        'talking-heads_base':
            'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
    }

    map_model_to_preprocess = {
        'bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'bert_en_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
        'small_bert/bert_en_uncased_L-2_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-2_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-2_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-2_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-4_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-4_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-4_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-4_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-6_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-6_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-6_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-6_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-8_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-8_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-8_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-8_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-10_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-10_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-10_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-10_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-12_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-12_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-12_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'small_bert/bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'bert_multi_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
        'albert_en_base':
            'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
        'electra_small':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'electra_base':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'experts_pubmed':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'experts_wiki_books':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'talking-heads_base':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    }

    tfhub_handle_encoder = map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

    print(f'BERT model selected           : {tfhub_handle_encoder}')
    print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
    return tfhub_handle_encoder,tfhub_handle_preprocess

def prepare_label(label):
    # convert integers to dummy variables (i.e. one hot encoded)
    encoded_y = utils.to_categorical(label)
    return encoded_y

def prepare_dataset(dataset):
    signal = np.array([])
    label = np.array([])
    for text_batch, label_batch in dataset:
        signal = np.append(signal, text_batch.numpy())
        label = np.append(label, label_batch.numpy())
    return signal, label

def classifier_model(num_classes):
    inputs = tf.keras.Input(shape=(None,768))
    # lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))
    # lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
    # dropout = tf.keras.layers.Dropout(0.2)(inputs)
    net1 = tf.keras.layers.Dense(768, activation='relu', name='classifier1')(inputs)
    # dropout2 = tf.keras.layers.Dropout(rate=0.2)(net1)
    net2 = tf.keras.layers.Dense(512, activation='relu', name='classifier2')(net1)
    # dropout3 = tf.keras.layers.Dropout(rate=0.2)(net2)
    net3 = tf.keras.layers.Dense(256, name='classifier3')(net2)
    # dropout4 = tf.keras.layers.Dropout(rate=0.2)(net3)
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(net3)
    model_head = tf.keras.Model(inputs, output)
    return model_head

def finetuned_bert_and_classifier(num_classes):
    bert = TFBertModel.from_pretrained(MODEL_CHECKPOINT)
    bert.trainable = True

    input_ids = tf.keras.layers.Input(
        shape=(128,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(128,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(128,), dtype=tf.int32, name="token_type_ids"
    )
    bert_output = bert(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids,
    )
    embeddings = bert_output['pooler_output']
    model_embed = tf.keras.Model(inputs=[input_ids, attention_masks, token_type_ids], outputs= embeddings)

    # lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(embeddings)
    # lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(lstm1)
    dropout = tf.keras.layers.Dropout(0.1)(embeddings)
    net1 = tf.keras.layers.Dense(768, name='classifier1')(dropout)
    # dropout2 = tf.keras.layers.Dropout(rate=0.2)(net1)
    # net2 = tf.keras.layers.Dense(512, name='classifier2')(dropout2)
    # dropout3 = tf.keras.layers.Dropout(rate=0.2)(net2)
    # net3 = tf.keras.layers.Dense(256, name='classifier3')(dropout3)
    # dropout4 = tf.keras.layers.Dropout(rate=0.2)(net3)
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(net1)
    model_head = tf.keras.Model(dropout,output)

    whole_model = tf.keras.Model(inputs=[input_ids, attention_masks, token_type_ids], outputs=output)
    return whole_model, model_embed, model_head

def pretrained_bert_model():
    bert = TFBertModel.from_pretrained(MODEL_CHECKPOINT)
    bert.trainable = False
    input_ids = tf.keras.layers.Input(
        shape=(128,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(128,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(128,), dtype=tf.int32, name="token_type_ids"
    )
    bert_output = bert(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids,
    )
    embeddings = bert_output['pooler_output']
    model_embed = tf.keras.Model(inputs=[input_ids, attention_masks, token_type_ids], outputs=embeddings)
    return model_embed

def main():
    train, val, test, classes = train_val_test_split()

    model_embed, model_head, whole_model = bert_model()

    x_val, y_val = prepare_dataset(val)
    #Get the embedded data
    n_samples = 20
    sample_ids = np.random.choice(len(x_val), n_samples)

    # Encode the digits with the first two layers
    embedded_words = model_embed.predict(x_val, batch_size=64)

    print(embedded_words[0].shape)

    X = np.array([ embedded_words[i] for i in sample_ids ])
    print(X.shape)
    Y = np.array([ y_val[i] for i in sample_ids ])
    print(Y.shape)

    from deepview import DeepView
    # create a wrapper function for deepview
    # Here, because random forest can handle numpy lists and doesn't
    # need further preprocessing or conversion into a tensor datatype
    pred_wrapper = DeepView.create_simple_wrapper(model_head)
    #
    # this is the alternative way of defining the prediction wrapper
    # for deep learning frameworks. In this case it's PyTorch.
    # def torch_wrapper(x):
    #     with torch.no_grad():
    #         x = np.array(x, dtype=np.float32)
    #         tensor = torch.from_numpy(x).to(device)
    #         pred = model(tensor).cpu().numpy()
    #     return pred

    #--- Deep View Parameters ----
    classes = np.arange(2)
    batch_size = 64
    max_samples = 500
    data_shape = (512,)
    resolution = 100
    N = 10
    lam = 0.6
    cmap = 'tab10'
    # to make sure deepview.show is blocking,
    # disable interactive mode
    interactive = False
    title = 'movie-reviews BERT'

    deepview = DeepView(pred_wrapper, classes, max_samples, batch_size, data_shape,
        N, lam, resolution, cmap, interactive, title)


    deepview.add_samples(X,Y)


    deepview.show()



