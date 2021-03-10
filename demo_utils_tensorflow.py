# --------- Tensorflow ---------------------
import tensorflow as tf



def create_tf_model_intermediate():
    dense1 = tf.keras.layers.Dense(64, activation='elu')
    dense2 = tf.keras.layers.Dense(64, activation='elu')
    dense3 = tf.keras.layers.Dense(10, activation='softmax')
    # in order to access the intermediate embedding, split the model into two models
    # model_embd will embed samples
    model_embd = tf.keras.Sequential([dense1])
    # model_head can make predictions based on the embedding from model_embd
    model_head = tf.keras.Sequential([dense2, dense3])
    whole_model = tf.keras.Sequential([model_embd, model_head])
    return model_embd, model_head, whole_model
