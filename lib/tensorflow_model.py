import tensorflow as tf
import numpy as np

image_size=48
image_size_flat=image_size*image_size
image_shape=(image_size,image_size)
num_channels=1
num_classes= 6

x = tf.placeholder("float", shape=[None, image_size_flat], name='x')
image_x = tf.reshape(x, [-1,image_size,image_size,
                     num_channels])
y=tf.placeholder("float", shape=[None,num_classes],
                 name='y')


l_r = tf.placeholder("float",name='l_r')
tf.add_to_collection("var", x)
tf.add_to_collection("var", y)
tf.add_to_collection("var", l_r)

layer_list=np.zeros(16)
layer_list[:3]=64
layer_list[3:7]=128
layer_list[7:13]=256
layer_list[13:16]=512
layer_list


import tensorflow.contrib.slim as slim

def Res_model(input,train=False):
    if train:
        reuse = None
    else:
        reuse = True

    with slim.arg_scope([slim.conv2d],stride=1, padding='SAME',
                      activation_fn=None,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        layer0=slim.conv2d(input,64,[3,3],\
                       normalizer_fn=slim.batch_norm,\
                      scope = 'conv01',reuse=reuse)
        input_layer=layer0
        #print('input_layer =',input_layer)
        j=layer_list[0]
        for i,output in zip(np.arange(1,17),layer_list):

            net = slim.batch_norm(input_layer,activation_fn=tf.nn.relu,\
                                 scope='res_layer'+str(i)+'_b1',reuse=reuse)
            #print('net1',net)
            if output > j:
                j = output;
                net = slim.conv2d(net,output,[3,3],stride=2,\
                                 scope='res_layer'+str(i)+'_conv1',\
                                 reuse=reuse)
            else:
                net = slim.conv2d(net,output,[3,3],stride=1,\
                                 scope='res_layer'+str(i)+'_conv1',\
                                 reuse=reuse)
            #print('net2',net)
            net = slim.batch_norm(net,activation_fn=tf.nn.relu,\
                                 scope='res_layer'+str(i)+'_b2',reuse=reuse)
            #print('net3',net)
            net = slim.conv2d(net,output,[3,3],\
                                 scope='res_layer'+str(i)+'_conv2',\
                                 reuse=reuse)
            #print('net3',net)
            if output == input_layer.get_shape().as_list()[-1]:
                if net.get_shape().as_list()[1] !=input_layer.get_shape().as_list()[1]:
                    input_layer=tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME')
                input_layer = input_layer + net
            else:
                input_channel=input_layer.get_shape().as_list()[-1]
                pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME')
                padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                    input_channel // 2]])
                #print('net1',padded_input)
                input_layer = padded_input + net
            #print('---------------------------')

    net = slim.avg_pool2d(net,[3,3],padding='valid', scope='pool1')
    net = slim.flatten(input_layer, scope='flatten1')
    #print('last =',net)
    #net = slim.fully_connected(net, num_classes,activation_fn=None,scope='fc1',reuse=reuse)
    net = slim.fully_connected(net, num_classes,activation_fn=None,\
                               weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),\
                               weights_regularizer=slim.l2_regularizer(0.0005),\
                               scope='fc1',reuse=reuse)

    return net

logits = Res_model(image_x,True)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
batch_size=64

cost=tf.reduce_mean(cross_entropy)

optimizer=tf.train.RMSPropOptimizer(learning_rate=l_r, decay=0.9).minimize(cost)
y_pred=tf.nn.softmax(logits)


session = tf.Session()
saver = tf.train.Saver(max_to_keep=10000)

session.run(tf.global_variables_initializer())
model_path = './lib/tensor_model/new1_12000_66'
saver.restore(sess=session,save_path=model_path)
