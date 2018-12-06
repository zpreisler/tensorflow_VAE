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
        self.shape=tf.shape(inputs)

    def build_graph(self,z_dim,name='encoder'):
        with tf.variable_scope(name):
            self.flat=tf.layers.flatten(inputs=self.inputs,name='flat')

            #x=tf.layers.conv2d(inputs=self.inputs,filters=4,kernel_size=2,strides=1,padding='same',activation=tf.nn.elu,name='elu1')

            x=tf.layers.dense(inputs=self.flat,units=512,activation=tf.nn.elu,name='elu1')
            x=tf.layers.dense(inputs=x,units=384,activation=tf.nn.elu,name='elu2')
            x=tf.layers.dense(inputs=x,units=256,activation=tf.nn.elu,name='elu3')

            self.mean=tf.layers.dense(x,units=z_dim,name='mean')
            self.log_sigma2=tf.layers.dense(x,units=z_dim,name='log_sigma2')

            eps=tf.random_normal(shape=tf.shape(self.mean),mean=0,stddev=1)
            self.z=self.mean+tf.sqrt(tf.exp(self.log_sigma2))*eps

            print(self.z)

class decoder():
    def __init__(self,z,shape,name='decoder'):
        self.z=z
        self.build_graph(shape,name=name)

    def build_graph(self,shape,name='encoder'):
        with tf.variable_scope(name):

            x=tf.layers.dense(inputs=self.z,units=256,activation=tf.nn.elu,name='elu1')
            x=tf.layers.dense(inputs=x,units=384,activation=tf.nn.elu,name='elu2')
            x=tf.layers.dense(inputs=x,units=512,activation=tf.nn.elu,name='elu3')

            self.flat=tf.layers.dense(inputs=x,units=128*128,activation=tf.nn.sigmoid,name='flat')
            self.out=tf.reshape(self.flat,shape)

            #self.input_z=tf.layers.dense(inputs=self.z,units=128*128,activation=tf.nn.elu,name='elu0')
            #self.z2=tf.reshape(self.input_z,shape)
            #x=tf.layers.conv2d_transpose(inputs=self.z2,filters=4,kernel_size=2,strides=1,padding='same',activation=tf.nn.elu,name='elu1')

            print(self.flat)
            print(self.out)

class VAE():
    def __init__(self,inputs,z_dim=10):
        self.rate=tf.placeholder(tf.float32)

        self.enc=encoder(inputs,z_dim=z_dim)
        self.dec=decoder(self.enc.z,self.enc.shape)

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
        eps=1e-6
        cross_entropy=self.enc.flat*tf.log(eps+self.dec.flat)+(1.0-self.enc.flat)*tf.log(eps+1.0-self.dec.flat)
        return -tf.reduce_sum(cross_entropy)

    def _optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.rate)

    def _train(self):
        return self.optimizer.minimize(self.loss)

def read_data(handle,batch_size=64):
    mnist=tf.keras.datasets.mnist
    (x_train,y_train),(x_test,y_test)=mnist.load_data()

    x_train=x_train.reshape(x_train.shape[0],28,28,1).astype('float32')/255.
    train_dataset=tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(batch_size)
    eval_dataset=tf.data.Dataset.from_tensor_slices(x_train[:batch_size]).repeat().batch(batch_size)

    #iterator=tf.data.Iterator.from_structure(train_dataset.output_types,
    #        train_dataset.output_shapes)

    iterator=tf.data.Iterator.from_string_handle(
            handle,
            train_dataset.output_types,
            train_dataset.output_shapes)

    next_element=iterator.get_next()
    train_iterator=train_dataset.make_initializable_iterator()
    eval_iterator=eval_dataset.make_initializable_iterator()

    return train_iterator,eval_iterator,next_element

def plot_img(img,count,shape=(8,8),path='log/'):
    from matplotlib.pyplot import imshow,show,figure,subplots,xlabel,ylabel,subplots_adjust,savefig,close
    from numpy import block,array,prod
    fig,ax=subplots(figsize=(10,10))

    img=img.reshape(-1,128,128)
    ig=img[:prod(shape)]
    t=ig.reshape(*shape,128,128)
    b=block(list(map(list,t)))
    imshow(b)

    ax.set_xticks([])
    ax.set_yticks([])
    subplots_adjust(left=0.0,bottom=0.0,top=1.0,right=1.0)

    savefig(path+'{:03d}.png'.format(count))

    close()
    #show()

def main(argv):
    print("VAE")
    from numpy import random,arange,ones
    from numpy import array,arange,zeros
    from matplotlib.pyplot import plot,show,figure,close,savefig,xlim,ylim,legend,subplots

    from argparse import ArgumentParser
    from pathlib import Path

    from tensorflow_utils import image_pipeline,get_labels_from_filenames
    from glob import glob

    p=ArgumentParser()
    p.add_argument("-a","--learning_rate",type=float,default=1e-4)
    p.add_argument("-s","--steps",type=int,default=100000)
    p.add_argument("-b","--batch_size",type=int,default=128)
    p.add_argument("-n","--suffix",default="_a")
    p.add_argument("-m","--plot_frequency",type=int,default=100)
    p.add_argument("-l","--loss_frequency",type=int,default=100)
    p.add_argument("-q","--quiet",action='store_true',default=False)
    args=p.parse_args()

    dirname='vae_{}_B{}/'.format(args.learning_rate,args.batch_size)
    path='log/'+dirname
    path_figs=path+'figs{}/'.format(args.suffix)
    print(path_figs)

    p=Path(path_figs).mkdir(parents=True,exist_ok=True)
    f_loss=open(path+'loss{}.dat'.format(args.suffix),'w')

    batch_size=args.batch_size
    steps=args.steps
    rate=args.learning_rate

    handle=tf.placeholder(tf.string,shape=[])
    train_iterator,eval_iterator,next_element=read_data(handle,batch_size=batch_size)


    _hex=glob("images/train/hex/rotated/*.png")
    _honeycomb=glob("images/train/honeycomb/rotated/*.png")
    _square=glob("images/train/square/rotated/*.png")
    _fluid=glob("images/train/fluid/original/*.png")
    random.shuffle(_fluid)
    _fluid=_fluid[:300]
    #train_files=_hex+_honeycomb+_square+_fluid
    train_files=_honeycomb+_hex+_square

    train_labels=get_labels_from_filenames(train_files)

    eval_files=glob("images/eval/hex/original/*.png")+glob("images/eval/honeycomb/original/*.png")+glob("images/eval/square/original/*.png")
    eval_files=_honeycomb[:16]+_hex[:16]+_square[:16]+_fluid[:16]
    eval_labels=get_labels_from_filenames(eval_files)

    img_handle,image,train_image_op,eval_image_op=image_pipeline(
            {'images': train_files,'labels': train_labels},
            {'images': eval_files,'labels': eval_labels},
            batch_size=args.batch_size)

    #v=VAE(next_element,z_dim=10)
    v=VAE(image['images'],z_dim=256)

    """Checkpoints"""
    try:
        saver=tf.train.Saver()
    except ValueError:
        print("No variables to save")

    with tf.Session() as session:
        print("Session")

        """Init"""
        tf.global_variables_initializer().run(session=session)

        train_handle=session.run(train_image_op.string_handle())
        eval_handle=session.run(eval_image_op.string_handle())

        session.run(train_image_op.initializer)
        session.run(eval_image_op.initializer)

        #session.run(train_iterator.initializer)
        #session.run(eval_iterator.initializer)
        #train_handle=session.run(train_iterator.string_handle())
        #eval_handle=session.run(eval_iterator.string_handle())

        count=0

        """Learning"""
        for step in range(steps):

            _=session.run(v.train,feed_dict={v.rate:rate,
                img_handle:train_handle})

            if step%args.loss_frequency is 0:

                loss,latent_loss,h_loss,mean,log_sigma2,_=session.run([
                    v.loss/batch_size,
                    v.latent_loss/batch_size,
                    v.h_loss/batch_size,
                    v.enc.mean,v.enc.log_sigma2,
                    v.train],
                    feed_dict={v.rate: rate,
                        img_handle: train_handle})
                 
                print(step,loss,latent_loss,h_loss)
                f_loss.write("{} {} {}\n".format(step,loss,
                    latent_loss,
                    h_loss))

            if step%args.plot_frequency is 0:
                out,inputs=session.run([v.dec.out,v.enc.inputs],
                        feed_dict={img_handle: eval_handle})
                plot_img(out,count,path=path_figs)
                count+=1

        saver.save(session,path+'last{}.ckpt'.format(args.suffix))

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
