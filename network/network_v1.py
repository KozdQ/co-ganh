import numpy as np
import pandas as pd
import tensorflow as tf


class Network:
    DATA_FILE = "Data/self_play.csv"
    MODEL_DIR = "Model/"
    EXPORT_DIR = "Export/"

    RESIDUAL_BLOCKS = 2

    def __init__(self):
        self.__nn = tf.compat.v1.estimator.Estimator(model_dir=self.MODEL_DIR, model_fn=self.model_fn, params={})

    def train(self, epochs=None, shuffle=True, steps=1):
        self.__nn.train(
            input_fn=self.input_fn(
                data_file=self.DATA_FILE,
                num_epochs=epochs,
                shuffle=shuffle
            ),
            steps=steps
        )

    def export(self):
        serving_input_receiver_fn = tf.compat.v1.estimator.export.build_raw_serving_input_receiver_fn(
            features={'x': tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[1, 5 * 5 * 3])}
        )
        self.__nn.export_savedmodel(self.EXPORT_DIR, serving_input_receiver_fn)

    def model_fn(self, features, labels, mode, params):
        mode = tf.compat.v1.estimator.ModeKeys.PREDICT
        # if mode != tf.compat.v1.estimator.ModeKeys.PREDICT:
        policy_labels, value_labels = tf.compat.v1.split(labels, [4444, 1], axis=1)
        value_labels = tf.compat.v1.cast(value_labels, tf.compat.v1.float32)
        policy_labels = tf.compat.v1.cast(policy_labels, tf.compat.v1.float32)

        # Input layer comes from features, which come from input_fn
        input_layer = tf.compat.v1.cast(features["x"], tf.compat.v1.float32)
        board_image = tf.compat.v1.reshape(input_layer, [-1, 5, 5, 3])

        pre_activation = self.custom_conv(board_image, 3, 3, 3, 100, name="conv1")

        norm1 = self.custom_batch_norm(pre_activation, name="norm1")

        relu1 = self.custom_relu(norm1, name="relu1")

        curr_input = relu1
        for i in range(self.RESIDUAL_BLOCKS):
            id_str = str(2 + i)
            conv_layer = self.custom_conv(curr_input, 3, 3, 100, 100, name="conv" + id_str)
            norm_layer = self.custom_batch_norm(conv_layer, name="norm" + id_str)
            relu_layer = self.custom_relu(norm_layer, name="relu" + id_str)
            conv_layer_2 = self.custom_conv(relu_layer, 3, 3, 100, 100, name="2conv" + id_str)
            norm_layer_2 = self.custom_batch_norm(conv_layer_2, name="2norm" + id_str)
            residual_layer = tf.compat.v1.add(curr_input, norm_layer_2)
            relu_layer_2 = self.custom_relu(residual_layer, name="2relu" + id_str)
            curr_input = relu_layer_2

        residual_tower_out = curr_input

        # policy_conv = self.custom_conv(residual_tower_out, 1, 1, 100, 2, name="policy_conv")
        # policy_norm = self.custom_batch_norm(policy_conv, name="policy_norm")
        # policy_relu = self.custom_relu(policy_norm, name="policy_relu")
        # # policy_relu should have shape [batch, 5, 5, 2] so I want [batch, 50]
        # policy_relu = tf.compat.v1.reshape(policy_relu, [-1, 50])
        # policy_output_layer = tf.compat.v1.layers.(inputs=policy_relu, units=4444, activation=tf.compat.v1.nn.sigmoid)

        # value_conv = self.custom_conv(residual_tower_out, 1, 1, 100, 1, name="value_conv")
        # value_norm = self.custom_batch_norm(value_conv, name="value_norm")
        # value_relu = self.custom_relu(value_norm, name="value_relu")
        # # value_relu should have shape [batch, 5, 5, 1] so I want [batch, 25]
        # value_relu = tf.compat.v1.reshape(value_relu, [-1, 25])
        # value_hidden = tf.compat.v1.keras.layers.Dense(inputs=value_relu, units=100, activation=tf.compat.v1.nn.relu)
        # value_output_layer = tf.compat.v1.keras.layers.Dense(inputs=value_hidden, units=1, activation=tf.compat.v1.nn.tanh)

        predictions = tf.compat.v1.concat([policy_labels, value_labels], axis=1)
        if mode == tf.compat.v1.estimator.ModeKeys.PREDICT:
            return tf.compat.v1.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={"policy": tf.compat.v1.estimator.export.PredictOutput({"policy": policy_labels}),
                                tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.compat.v1.estimator.export.PredictOutput(
                                    {"policy": policy_labels,
                                     "value": value_labels})}
            )

        loss = tf.compat.v1.reduce_mean(
            tf.compat.v1.square(tf.compat.v1.add(
                tf.compat.v1.square(tf.compat.v1.subtract(tf.compat.v1.cast(value_labels, tf.compat.v1.float32), value_labels)),
                tf.compat.v1.reshape(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=policy_labels,
                                                                                labels=policy_labels),
                           [-1, 1]))
            ))

        eval_metric_ops = {}

        global_step_v1 = tf.compat.v1.Variable(0, trainable=False)
        learning_rate = tf.compat.v1.train.exponential_decay(0.01, global_step_v1,
                                                             100000, 0.96, staircase=True)

        optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9)

        train_op = optimizer.minimize(
            loss=loss, global_step=global_step_v1)

        return tf.compat.v1.estimator.EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops,
                                          export_outputs={"policy": tf.compat.v1.estimator.export.PredictOutput(
                                              {"policy": policy_labels}),
                                              tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.compat.v1.estimator.export.PredictOutput(
                                                  {"value": value_labels})})

    def custom_conv(self, input_layer, f_height, f_width, in_channels, out_channels, stride=None, name="conv"):
        if stride is None:
            stride = [1, 1, 1, 1]
        with tf.compat.v1.variable_scope(name) as scope:
            # need a filter/kernel with size 3x3, 256 output channels
            kernel = tf.compat.v1.get_variable(
                name + "_filter",
                [f_height, f_width, in_channels, out_channels],
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=5e-2, dtype=tf.compat.v1.float32)
            )

            # convolution!
            conv1 = tf.compat.v1.nn.conv2d(input_layer, kernel, stride, padding='SAME', name=scope.name)
            biases1 = tf.compat.v1.get_variable(name + "_biases", [out_channels],
                                                initializer=tf.compat.v1.constant_initializer(0.0))
            pre_activation = tf.compat.v1.nn.bias_add(conv1, biases1)

            return pre_activation

    def custom_batch_norm(self, inputs, training=True, name="norm"):
        with tf.compat.v1.variable_scope(name) as scope:
            norm1 = tf.compat.v1.layers.batch_normalization(inputs=inputs, training=training, name=scope.name)
        return norm1

    def custom_relu(self, inputs, name="relu"):
        with tf.compat.v1.variable_scope(name) as scope:
            relu = tf.compat.v1.nn.relu(inputs, name=scope.name)
        return relu

    def input_fn(self, data_file, num_epochs, batch_size=32, shuffle=False, num_threads=4):
        feature_cols = self.feature_col_names()
        target_cols = self.target_col_names()
        dataset = pd.read_csv(
            data_file,
            header=0,
            usecols=feature_cols + target_cols,
            skipinitialspace=True,
            engine="python")

        dataset.dropna(how="any")

        policy_labels = dataset.probs.apply(lambda x: self.create_policy(self.decode(x)))
        policy_labels = policy_labels.apply(pd.Series)

        value_labels = dataset.value

        dataset.pop('probs')
        dataset.pop('value')

        return tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={"x": np.array(dataset)},
            y=np.array(pd.concat([policy_labels, value_labels], axis=1)),
            batch_size=batch_size,
            num_epochs=num_epochs,
            shuffle=shuffle,
            num_threads=num_threads)

    # [0_0, 0_1, ..., 1_24, turn_0, turn_1, ..., turn_24]
    def feature_col_names(self):
        feature_columns = []
        for i in range(2):
            for x in range(25):
                feature_columns.append(str(i) + "_" + str(x))
        for x in range(25):
            feature_columns.append('turn_' + str(x))
        return feature_columns

    def target_col_names(self):
        return ['probs', 'value']

    def create_policy(self, labels):
        policy = [float(0) for x in range(4444)]
        for move, prob in labels:
            policy[move] = float(prob)
        return np.array(policy)

    # Move probabilities are encoded into a string format as follows:
    # "(from_square!to_square:probability)#(...)"
    def decode(self, labels):
        new_labels = []
        labels = labels.split('#')
        for label in labels:
            label = label.strip('(').strip(')').split(':')
            label = (int(label[0]), float(label[1]))
            new_labels.append(label)
        return new_labels
