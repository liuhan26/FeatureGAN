import tensorflow as tf
import numpy as np
import os
import scipy.io as sio
import argparse

def load(x):
    return sio.loadmat(x)


def variable_init(size):
    in_dim = size[0]
    w_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=w_stddev)


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    E_h1 = tf.nn.relu(tf.matmul(z, E_W1) + E_b1)
    E_h2 = tf.nn.relu(tf.matmul(E_h1, E_W2) + E_b2)
    G_h1 = tf.nn.relu(tf.matmul(E_h2, G_W1) + G_b1)
    G_h2 = tf.matmul(G_h1, G_W2) + G_b2
    G_feature = tf.tanh(G_h2)
    return G_feature


def discriminator(x,y):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W11) + D_b11+tf.matmul(y, D_W12) + D_b12)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

parser = argparse.ArgumentParser()
parser.add_argument("--train_step",default="train")
args = parser.parse_args()

X = tf.placeholder(tf.float32, shape=[None, 256])

D_W11 = tf.Variable(variable_init([256, 128]))
D_b11 = tf.Variable(tf.zeros(shape=[128]))
D_W12 = tf.Variable(variable_init([256, 128]))
D_b12 = tf.Variable(tf.zeros(shape=[128]))
D_W2 = tf.Variable(variable_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W11, D_b11,D_W12, D_b12, D_W2, D_b2]

Z = tf.placeholder(tf.float32, shape=[None, 256])

E_W1 = tf.Variable(variable_init([256, 128]))
E_b1 = tf.Variable(tf.zeros(shape=[128]))
E_W2 = tf.Variable(variable_init([128, 100]))
E_b2 = tf.Variable(tf.zeros(shape=[100]))

G_W1 = tf.Variable(variable_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))
G_W2 = tf.Variable(variable_init([128, 256]))
G_b2 = tf.Variable(tf.zeros(shape=[256]))

theta_G = [E_W1, E_b1, E_W2, E_b2, G_W1, G_b1, G_W2, G_b2]

D_real, D_logit_real = discriminator(Z, X)

G_sample = generator(Z)
D_fake, D_logit_fake = discriminator(Z, G_sample)

D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))

D_loss = D_loss_real + D_loss_fake

L1_loss = tf.reduce_mean(tf.abs(G_sample-X))
g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
G_loss = g_loss + 10 * L1_loss
# g_gradients = tf.gradients(g_loss,G_W1)
# l2_gradients = tf.gradients(L1_loss,G_W1)
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

saver = tf.train.Saver();

if args.train_step=="train":
    for epoch in range(300):
        feature_gallery = load('/home/liuhan/github/FeatureGAN/frgc_exp/data/featuregallery.mat')
        feature_probe = load('/home/liuhan/github/FeatureGAN/frgc_exp/data/featureprobe.mat')
        color = feature_gallery['featuregallery']
        depth = feature_probe['featureprobe']

        for it in range(117):
            Zm = color[it * 128:(it + 1) * 128]/50
            Xm = depth[it * 128:(it + 1) * 128]/50
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: Xm, Z: Zm})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: Xm, Z: Zm})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: Xm, Z: Zm})
            # print('g_gradient: {}'.format(g_diff))
            # print('l2_gradient: {}'.format(l2_diff))
            print('Epoch: {}'.format(epoch))
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()

    saver.save(sess, "out/model.ckpt")
else:
    saver.restore(sess, "out/model.ckpt")
    Zm = load('/home/liuhan/github/FeatureGAN/frgc_exp/test/featuregallery.mat')
    samples = sess.run(G_sample, feed_dict={Z: Zm['featuregallery']/50})
    sio.savemat('/home/liuhan/github/FeatureGAN/frgc_exp/test/feature_transform.mat',{'featuregallery':samples})