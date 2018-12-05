#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
class encoder():
    def __init__(self,inputs,z_dim,name='encoder'):
        self.inputs=inputs
        self.z_dim=z_dim

        self.build_graph(self.z_dim,name=name)

    def build_graph(self,z_dim,name='encoder'):
        with tf.variable_scope(name):
            self.flat=tf.layers.flatten(inputs=self.inputs,name='flat')

            x=tf.layers.dense(inputs=self.flat,units=512,activation=tf.nn.elu,name='elu1')
            x=tf.layers.dense(inputs=x,units=384,activation=tf.nn.elu,name='elu2')
            x=tf.layers.dense(inputs=x,units=256,activation=tf.nn.elu,name='elu3')

            self.mean=tf.layers.dense(x,units=z_dim,name='mean')
            self.log_sigma2=tf.layers.dense(x,units=z_dim,name='log_sigma2')

            eps=tf.random_normal(shape=tf.shape(self.mean),mean=0,stddev=1)
            self.z=self.mean+tf.sqrt(tf.exp(self.log_sigma2))*eps

            print(self.inputs)
            print(self.mean)
            print(eps)
            print(self.z)

class decoder():
    def __init__(self,z,name='decoder'):
        self.z=z
        self.build_graph(name=name)

    def build_graph(self,name='encoder'):
        with tf.variable_scope(name):
            x=tf.layers.dense(inputs=self.z,units=256,activation=tf.nn.elu,name='elu1')
            x=tf.layers.dense(inputs=x,units=384,activation=tf.nn.elu,name='elu2')
            x=tf.layers.dense(inputs=x,units=512,activation=tf.nn.elu,name='elu3')

            self.flat=tf.layers.dense(inputs=x,units=28*28,activation=tf.nn.sigmoid,name='flat')
            self.out=tf.reshape(self.flat,[-1,28,28,1])

            print(self.flat)
            print(self.out)

class VAE():
    def __init__(self,inputs,z_dim=10):
        self.rate=tf.placeholder(tf.float32)

        self.enc=encoder(inputs,z_dim=z_dim)
        self.dec=decoder(self.enc.z)

        self.latent_loss=self._latent_loss()
        self.h_loss=self._h_loss()

        self.loss=self.latent_loss+self.h_loss
        self.optimizer=self._optimizer()
        self.train=self._train()

    def _latent_loss(self):
        return -0.5*tf.reduce_sum(
                1+self.enc.log_sigma2-tf.square(self.enc.mean)-tf.exp(self.enc.log_sigma2)
                )

    def _h_loss(self):
        #cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits(
        #        logits=self.dec.flat,labels=self.enc.flat)
        #cross_entropy=tf.squared_difference(
        #        self.dec.flat,self.enc.flat)
        eps=1e-6
        cross_entropy=self.enc.flat*tf.log(eps+self.dec.flat)+(1.0-self.enc.flat)*tf.log(eps+1.0-self.dec.flat)
        return -tf.reduce_sum(
                cross_entropy)

    def _optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.rate)

    def _train(self):
        return self.optimizer.minimize(self.loss)

def read_data(batch_size=64):
    mnist=tf.keras.datasets.mnist
    (x_train,y_train),(x_test,y_test)=mnist.load_data()

    x_train=x_train.reshape(x_train.shape[0],28,28,1).astype('float32')/255.
    train_dataset=tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(batch_size)

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

    batch_size=128
    steps=1000000
    rate=1e-4

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
        for step in range(steps):
            loss,latent_loss,h_loss,mean,log_sigma2,_=session.run([
                v.loss,v.latent_loss,v.h_loss,
                v.enc.mean,v.enc.log_sigma2,
                v.train
                ],feed_dict={v.rate:rate})
                 
            if step%200 is 0:
                print(step,loss,latent_loss,h_loss)
                print(mean[0])
                print(log_sigma2[0])
            if step%5000 is 0:
                out,inputs=session.run([v.dec.out,v.enc.inputs])
                #print(inputs[0][0])
                print(out[0][0])

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
