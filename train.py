import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from datetime import datetime
import networks
import augmentation


# HYPERPARAMETERS
MODEL_NAME = "_".join(str(datetime.now()).split(" "))
DATASET_SIZE = 60000
TEST_SIZE = 10000
BATCH_SIZE_TRAINING = 50
BATCH_SIZE_VALIDATION = 100
EPOCHS = 5
LR = 0.01
VALIDATION_STEP = 50

HORIZONTAL_FLIP = False
MIX_UP = True
BETA_MIX_UP = 0.2
RANDOM_TRANS = False
TRANS_W = 2


def train():

    # Load Data
    datasets = tfds.load("fashion_mnist", as_supervised=True)
    train_dataset = datasets["train"].batch(BATCH_SIZE_TRAINING)
    x_train, y_train = train_dataset.repeat().shuffle(buffer_size=100).make_one_shot_iterator().get_next()

    test_dataset = datasets["test"].batch(BATCH_SIZE_VALIDATION)
    x_test, y_test = test_dataset.repeat().make_one_shot_iterator().get_next()

    # Preprocess and augmentation
    x_train = tf.divide(tf.cast(x_train, dtype=tf.float32), 255.0)
    y_train = tf.one_hot(y_train, 10)
    x_test = tf.divide(tf.cast(x_test, dtype=tf.float32), 255.0)
    y_test = tf.one_hot(y_test, 10)

    if HORIZONTAL_FLIP:
        x_train = augmentation.horizontal_flip(x_train)
    if MIX_UP:
        x_train, y_train = augmentation.mix_up(x_train, y_train, BETA_MIX_UP)
    if RANDOM_TRANS:
        x_train = augmentation.random_translation(x_train, TRANS_W)

    # Build model
    with tf.variable_scope("prediction"):
        logits = networks.wide_resnet(x_train, is_training=True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

    with tf.variable_scope("prediction", reuse=tf.AUTO_REUSE):
        logits_test = networks.wide_resnet(x_test, is_training=True)

    # Configure saver
    os.mkdir(os.path.join("./models", MODEL_NAME))
    saver = tf.train.Saver()
    train_logger = open('./models/{}/train.txt'.format(MODEL_NAME), 'w')
    validation_logger = open('./models/{}/validation.txt'.format(MODEL_NAME), 'w')
    accuracy_logger = open('./models/{}/accuracy.txt'.format(MODEL_NAME), 'w')

    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)
    exit()
    """


    with tf.Session() as session:

        # Train model
        session.run(tf.global_variables_initializer())
        step = 0
        while step * BATCH_SIZE_TRAINING / DATASET_SIZE < EPOCHS:

            _, loss_np = session.run([train_op, loss])

            if step%10 == 0:
                print("Training step: {}.\t Loss: {}".format(step, loss_np))
                train_logger.write(str(step) + '\t' + str(loss_np) + '\n')
            step += 1

            if step % VALIDATION_STEP == 0:

                logits_test_np, y_test_np = session.run([logits_test, y_test])

                predictions = np.argmax(logits_test_np, axis=1)
                labels = np.argmax(y_test_np, axis=1)

                batch_accuracy = np.mean(np.equal(predictions, labels))
                print("Validation step: {}.\t Accuracy: {}".format(step, batch_accuracy))
                validation_logger.write(str(step) + '\t' + str(batch_accuracy) + '\n')

        saver.save(session, 'models/{}/model'.format(MODEL_NAME))

        # Compute final accuracy over the whole dataset
        acc = []
        for _ in range(int(TEST_SIZE/BATCH_SIZE_VALIDATION)):

            logits_test_np, y_test_np = session.run([logits_test, y_test])

            predictions = np.argmax(logits_test_np, axis=1)
            labels = np.argmax(y_test_np, axis=1)
            batch_accuracy = np.mean(np.equal(predictions, labels))
            acc.append(batch_accuracy)

        accuracy_logger.write(str(np.mean(acc)))


if __name__ == "__main__":

    train()