import argparse
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing

FLAGS = None

def get_input_data(_):
    attrition = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

    # Undersample Attrition == No
    attrition_yes = attrition[attrition['Attrition'] == 'Yes']
    attrition_no = attrition[attrition['Attrition'] == 'No'].iloc[:550]
    attrition = pd.concat([attrition_yes, attrition_no])

    # Change categorical data to numerical values
    numerical = attrition._get_numeric_data()
    categorical = attrition.select_dtypes(include=['object'])
    categorical_converted = pd.get_dummies(categorical)
    input_data = pd.concat([numerical, categorical], axis=1)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    np_scaled = min_max_scaler.fit_transform(input_data)
    input_data_normalized = pd.DataFrame(np_scaled, columns = input_data.columns.values)

    assert input_data_normalized.isnull().any().any() == False

    return input_data_normalized.drop(['Attrition_Yes', 'Attrition_No'], axis=1), input_data_normalized[['Attrition_No', 'Attrition_Yes']]


def feed_forward(x, hidden_layer_units):
    w_1 = tf.Variable(tf.truncated_normal([55, hidden_layer_units], mean=0.5, stddev=0.3))
    b_1 = tf.Variable(tf.zeros[hidden_layer_units])

    z_1 = tf.matmul(x, w_1) + b_1
    a_1 = tf.sigmoid(z_1)

    w_L = tf.Variable(tf.truncated_normal([hidden_layer_units, 2], mean=0.5, stddev=0.3))
    b_L = tf.Variable(tf.zeros[2])

    z_L = tf.matmul(a_1, w_L) + b_L
    return z_L


def main(argv=None):
    input_data, input_labels = get_input_data()
    training_step = argv['training-step']

    x = tf.placeholder(tf.float32, [None, 55])
    y = tf.placeholder(tf.float32, [None, 2])

    z_L = feed_forward(x)
    a_L = tf.nn.softmax(z_L)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=z_L))

    train_step = tf.train.GradientDescentOptimizer(training_step).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(a_L, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-step', type=float32, default=0.3)
    parser.add_argument('--hidden-layer-units', type=int, default=25)
    args = vars(parser.parse_args())
    tf.app.run(main=main, argv=[sys.argv[0]] + args)
