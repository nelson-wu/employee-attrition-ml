import argparse
import sys
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.utils import shuffle

FLAGS = None


def get_input_data():
    attrition = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    attrition = shuffle(attrition)

    # Undersample Attrition == No
    attrition_yes = attrition[attrition['Attrition'] == 'Yes']
    attrition_no = attrition[attrition['Attrition'] == 'No'].iloc[:600]
    attrition = pd.concat([attrition_yes, attrition_no])

    # Change categorical data to numerical values
    numerical = attrition._get_numeric_data()
    categorical = attrition.select_dtypes(include=['object'])
    categorical_converted = pd.get_dummies(categorical)
    input_data = pd.concat([numerical, categorical_converted], axis=1)
    input_data = shuffle(input_data)

    # Normalize
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    np_scaled = min_max_scaler.fit_transform(input_data)
    input_data_normalized = pd.DataFrame(np_scaled, columns=input_data.columns.values)

    assert input_data_normalized.isnull().any().any() == False

    input_data_nums = input_data_normalized.drop(['Attrition_Yes', 'Attrition_No'], axis=1)
    input_data_labels = input_data_normalized[['Attrition_No', 'Attrition_Yes']]

    training_data = input_data_nums[0:FLAGS.training_size]
    training_labels = input_data_labels[0:FLAGS.training_size]

    test_data = input_data_nums[FLAGS.training_size:]
    test_labels = input_data_labels[FLAGS.training_size:]

    return training_data.values, training_labels.values, test_data.values, test_labels.values


def feed_forward(x, hidden_layer_units):
    w_1 = tf.Variable(tf.truncated_normal([55, hidden_layer_units], mean=0.0, stddev=0.3))
    b_1 = tf.Variable(tf.zeros([hidden_layer_units]))
    #x = tf.Print(x, ["x", x, tf.shape(x)])

    z_1 = tf.matmul(x, w_1) + b_1
    a_1 = tf.sigmoid(z_1)
    #a_1 = tf.Print(a_1, ["a_1", a_1, tf.shape(a_1)])

    w_L = tf.Variable(tf.truncated_normal([hidden_layer_units, 2], mean=0.5, stddev=0.3))
    b_L = tf.Variable(tf.zeros([2]))

    z_L = tf.matmul(a_1, w_L) + b_L
    #z_L = tf.Print(z_L, ["z_L", z_L, tf.shape(z_L)])
    return z_L


def main(argv=None):
    training_step = FLAGS.training_step
    hidden_layer_units = FLAGS.hidden_layer_units
    batch_num = FLAGS.batch_num

    training_data, training_labels, test_data, test_labels = get_input_data()
    training_data_batches = np.array_split(training_data, batch_num)
    training_labels_batches = np.array_split(training_labels, batch_num)

    x = tf.placeholder(tf.float32, [None, 55])
    y = tf.placeholder(tf.float32, [None, 2])

    z_L = feed_forward(x, hidden_layer_units)
    a_L = tf.nn.softmax(z_L)
    #a_L = tf.Print(a_L, [a_L, tf.shape(a_L)])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=z_L))
    cross_entropy = tf.Print(cross_entropy, ["cross_entropy", str(cross_entropy)])

    train_step = tf.train.AdamOptimizer(training_step).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(a_L, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_accuracy = accuracy.eval(feed_dict={x: test_data, y: test_labels})
        print("Starting accuracy: %g"%(test_accuracy))

        # 2000 epochs
        for i in range(2000):
            for (data_batch, label_batch) in zip(training_data_batches, training_labels_batches):
                train_step.run(feed_dict={x: data_batch, y: label_batch})

            test_accuracy = accuracy.eval(feed_dict={x: test_data, y: test_labels})
            print("Epoch %d: %g accuracy"%(i, test_accuracy))
            #print(test_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--training_step', type=float, default=0.003)
    parser.add_argument('--hidden_layer_units', type=int, default=25)
    parser.add_argument('--batch_num', type=int, default=100)
    parser.add_argument('--training_size', type=int, default=400)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
