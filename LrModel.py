#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from BaseModel import BaseModel

def gen_sample(num_examples, num_features, num_tags=2):
    # Random features.
    x = np.random.rand(num_examples, num_features).astype(np.float32)
    #x = np.linspace(-5, 5, num_examples)
    # Random tag indices representing the gold sequence.
    y = np.random.randint(num_tags, size=[num_examples, num_tags]).astype(np.float32)
    return x, y

class LrModel(BaseModel):
    def __init__(self, lr, num_features, num_tags):
        self.num_features = num_features
        self.num_tags = num_tags
        self.lr = lr

    def add_placeholders_op(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.num_features],
                                name="word_ids")
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_tags],
                                name="labels")
    def get_feed(self, x, y):
        feed = {
            self.x : x,
            self.y : y
        }
        return feed

    def add_line_op(self):
        weights = tf.get_variable("weights", [self.num_features, self.num_tags])
        self.line_scores = tf.nn.softmax(tf.matmul(self.x, weights))

    def add_loss_op(self):
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.line_scores), reduction_indices=1))


    def build(self):
        self.add_placeholders_op()
        self.add_line_op()
        self.add_loss_op()

        self.add_train_op("adam", self.lr, self.loss)

        self.initialize_session()

    def train(self):
        x, y = gen_sample(1000, self.num_features, self.num_tags)
        for i in range(10000):
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=self.get_feed(x, y))
            if i % 100 == 0:
                print loss

def main():
    lr = LrModel(0.001, 50, 2)
    lr.build()
    lr.train()

if __name__ == "__main__":
    main()
