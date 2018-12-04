#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
class encoder():
    def __init__(self,inputs,name='encoder'):
        self.inputs=inputs
        self.z_dim=2

        self.build_graph(self.z_dim,name=name)

    def build_graph(self,z_dim,name='encoder'):
        with tf.variable_scope(name):
            self.flat=tf.layers.flatten(inputs=self.inputs,name='flat')

            x=tf.layers.dense(inputs=self.flat,units=512,activation=tf.nn.elu,name='elu1')
            x=tf.layers.dense(inputs=x,units=256,activation=tf.nn.elu,name='elu2')
            x=tf.layers.dense(inputs=x,units=128,activation=tf.nn.elu,name='elu3')

            self.mean=tf.layers.dense(x,units=z_dim,name='mean')
            self.std=tf.layers.dense(x,units=z_dim,activation=tf.nn.sigmoid,name='std')

            self.epsilon=tf.random_normal(tf.shape(self.mean))

            self.z=self.mean+self.std*self.epsilon

            print(self.inputs)
            print(self.mean)
            print(self.epsilon)
            print(self.z)

class decoder():
    def __init__(self,z,name='decoder'):
        self.z=z
        self.build_graph(name=name)

    def build_graph(self,name='encoder'):
        with tf.variable_scope(name):
            x=tf.layers.dense(inputs=self.z,units=128,activation=tf.nn.elu,name='elu1')
            x=tf.layers.dense(inputs=x,units=256,activation=tf.nn.elu,name='elu2')
            x=tf.layers.dense(inputs=x,units=512,activation=tf.nn.elu,name='elu3')

            self.flat=tf.layers.dense(inputs=x,units=28*28,name='flat')

            self.out=tf.reshape(self.flat,[-1,28,28,1])

            print(self.flat)
            print(self.out)

class VAE():
    def __init__(self,inputs,z_dim=2):
        self.enc=encoder(inputs)
        self.dec=decoder(self.enc.z)

        self.latent_loss=self._latent_loss()
        self.h_loss=self._h_loss()

    def _latent_loss(self):
        #return -0.5*tf.reduce_sum(
        #        1.0+2.0*self.enc.std-tf.square(self.enc.mean)-tf.exp(2.0*self.enc.std),1)
        return 0

    def _h_loss(self):
        cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec.out,labels=self.enc.inputs)
        return -tf.reduce_sum(cross_entropy,axis=[1,2,3])

def read_data(batch_size=64):
    mnist=tf.keras.datasets.mnist
    (x_train,y_train),(x_test,y_test)=mnist.load_data()

    x_train=x_train.reshape(x_train.shape[0],28,28,1).astype('float32')/255.
    train_dataset=tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)

    iterator=tf.data.Iterator.from_structure(train_dataset.output_types,
            train_dataset.output_shapes)
    next_element=iterator.get_next()

    init_op=iterator.make_initializer(train_dataset)

    return init_op,next_element

def main(argv):
    print("VAE")
    from numpy import random,arange,ones
    from numpy import array,arange,zeros
    from matplotlib.pyplot import plot,show,figure,close,savefig,xlim,ylim,legend,subplots

    batch_size=32

    x=tf.placeholder(dtype=tf.float32,shape=[None,28,28,1])

    init_op,next_element=read_data(batch_size=batch_size)

    v=VAE(next_element)

    """Checkpoints"""
    try:
        saver=tf.train.Saver()
    except ValueError:
        print("No variables to save")

    with tf.Session() as session:
        print("Session")

        """Init"""
        session.run(init_op)
        tf.global_variables_initializer().run(session=session)

        """Learning"""

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
