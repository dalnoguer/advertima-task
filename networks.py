#!/usr/bin/python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Specifications of the neural networks used."""

from __future__ import division
import numpy as np
import tensorflow as tf


def wide_resnet(inputs, is_training, name=None, update_batch_stats=False):
    """A wide resnet model.

    Based on the implementation at
    https://github.com/tensorflow/models/tree/master/research/resnet
    """

    # hparams
    lrelu_leakiness = 0.1
    width = 2
    num_residual_units = 2
    num_classes = 10

    # If `update_batch_stats` is false, have batch norm update a dummy
    # collection whose ops are never run.
    batch_norm_updates_collections = (
        tf.GraphKeys.UPDATE_OPS if update_batch_stats else "_unused"
    )

    # Helper functions
    def _conv(name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                "DW",
                [filter_size, filter_size, in_filters, out_filters],
                tf.float32,
                initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)
                ),
            )
            return tf.nn.conv2d(x, kernel, strides, padding="SAME")

    def _relu(x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name="leaky_relu")

    def _residual(
        x, in_filter, out_filter, stride, activate_before_residual=False
    ):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope("shared_activation"):
                x = tf.contrib.layers.batch_norm(
                    x,
                    scale=True,
                    updates_collections=batch_norm_updates_collections,
                    is_training=is_training,
                )
                x = _relu(x, lrelu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope("residual_only_activation"):
                orig_x = x
                x = tf.contrib.layers.batch_norm(
                    x,
                    scale=True,
                    updates_collections=batch_norm_updates_collections,
                    is_training=is_training,
                )
                x = _relu(x, lrelu_leakiness)

        with tf.variable_scope("sub1"):
            x = _conv("conv1", x, 3, in_filter, out_filter, stride)

        with tf.variable_scope("sub2"):
            x = tf.contrib.layers.batch_norm(
                x,
                scale=True,
                updates_collections=batch_norm_updates_collections,
                is_training=is_training,
            )
            x = _relu(x, lrelu_leakiness)
            x = _conv("conv2", x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope("sub_add"):
            if in_filter != out_filter:
                orig_x = _conv(
                    "conv1x1", orig_x, 1, in_filter, out_filter, stride
                )
            x += orig_x

        tf.logging.debug("image after unit %s", x.get_shape())
        return x

    with tf.name_scope(name, "net"):
        x = inputs
        x = _conv("init_conv", x, 3, 1, 16, [1, 1, 1, 1])

        activate_before_residual = [True, False, False]
        res_func = _residual
        filters = [16, 16 * width, 32 * width, 64 * width]

        with tf.variable_scope("unit_1_0"):
            x = res_func(
                x,
                filters[0],
                filters[1],
                [1, 1, 1, 1],
                activate_before_residual[0],
            )
        for i in range(1, num_residual_units):
            with tf.variable_scope("unit_1_%d" % i):
                x = res_func(x, filters[1], filters[1], [1, 1, 1, 1], False)

        with tf.variable_scope("unit_2_0"):
            x = res_func(
                x,
                filters[1],
                filters[2],
                [1, 2, 2, 1],
                activate_before_residual[1],
            )
        for i in range(1, num_residual_units):
            with tf.variable_scope("unit_2_%d" % i):
                x = res_func(x, filters[2], filters[2], [1, 1, 1, 1], False)

        with tf.variable_scope("unit_3_0"):
            x = res_func(
                x,
                filters[2],
                filters[3],
                [1, 2, 2, 1],
                activate_before_residual[2],
            )
        for i in range(1, num_residual_units):
            with tf.variable_scope("unit_3_%d" % i):
                x = res_func(x, filters[3], filters[3], [1, 1, 1, 1], False)

        with tf.variable_scope("unit_last"):
            x = tf.contrib.layers.batch_norm(
                x,
                scale=True,
                updates_collections=batch_norm_updates_collections,
                is_training=is_training,
            )
            x = _relu(x, lrelu_leakiness)
            # Global average pooling
            x = tf.reduce_mean(x, [1, 2])

        with tf.variable_scope("logit"):
            w_init = tf.glorot_normal_initializer()
            logits = tf.layers.dense(
                x, num_classes, kernel_initializer=w_init
            )

    return logits


def conv_net(inputs, is_training, name=None, update_batch_stats=False):

    def _conv(name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                "DW",
                [filter_size, filter_size, in_filters, out_filters],
                tf.float32,
                initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)
                ),
            )
            return tf.nn.conv2d(x, kernel, strides, padding="SAME")

    def _relu(x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name="leaky_relu")

    # hparams
    lrelu_leakiness = 0.1
    num_residual_units = 2
    num_classes = 10

    batch_norm_updates_collections = (
        tf.GraphKeys.UPDATE_OPS if update_batch_stats else "_unused"
    )

    with tf.name_scope(name, "net"):
        x = inputs
        x = _conv("init_conv", x, 3, 1, 16, [1, 1, 1, 1])

    x = inputs

    filters = [16, 32, 32, 64, 64, 128, 128, 256]

    for i in range(len(filters)):

        if i % 3 != 0:
            strides = [1] * 4
        else:
            strides = [1, 2, 2, 1]

        x = _conv('conv_{}'.format(i), x, 3, x.shape[3], filters[i], strides)
        x = tf.contrib.layers.batch_norm(
            x,
            scale=True,
            updates_collections=batch_norm_updates_collections,
            is_training=is_training,
        )
        x = _relu(x, lrelu_leakiness)

    # Global average pooling
    x = tf.reduce_mean(x, [1, 2])

    with tf.variable_scope("logit"):
        w_init = tf.glorot_normal_initializer()
        logits = tf.layers.dense(
            x, num_classes, kernel_initializer=w_init
            )

    return logits
