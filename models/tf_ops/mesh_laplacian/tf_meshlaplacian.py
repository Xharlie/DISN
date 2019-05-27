""" Compute Chamfer's Distance.

Original author: Haoqiang Fan.
Modified by Charles R. Qi
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mesh_laplacian_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_meshlaplacian_so.so'))
# print mesh_laplacian_module.mesh_laplacian

def mesh_laplacian(verts,nverts,tris,ntris):
    '''
    .Input("verts: float32")
    .Input("nverts: int32")
    .Input("tris: int32")
    .Input("ntris: int32")
    .Output("laplacian: float32")
    .Output("count: int32")
    .Output("nb: int32");
    '''
    return mesh_laplacian_module.mesh_laplacian(verts,nverts,tris,ntris)

#@tf.RegisterShape('NnDistance')
#def _nn_distance_shape(op):
    #shape1=op.inputs[0].get_shape().with_rank(3)
    #shape2=op.inputs[1].get_shape().with_rank(3)
    #return [tf.TensorShape([shape1.dims[0],shape1.dims[1]]),tf.TensorShape([shape1.dims[0],shape1.dims[1]]),
        #tf.TensorShape([shape2.dims[0],shape2.dims[1]]),tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
@ops.RegisterGradient('MeshLaplacian')
def _mesh_laplacian_grad(op, grad_laplacian, grad_count, grad_nb):
    '''
    .Input("nverts: int32")
    .Input("count: int32")
    .Input("nb: int32")
    .Input("grad_laplacian: float32")
    .Output("grad_verts: float32");
    '''

    nverts = op.inputs[1]
    global count
    count = op.outputs[1]
    global nb
    nb = op.outputs[2]
    global aaa
    aaa = grad_laplacian

    grad_verts = mesh_laplacian_module.mesh_laplacian_grad(nverts,count,nb,grad_laplacian)

    return [grad_verts,None,None,None]


if __name__=='__main__':
    import numpy as np
    import random
    import time
    from tensorflow.python.ops.gradient_checker import compute_gradient
    random.seed(100)
    np.random.seed(100)


    def write_off(fn, verts, faces):
        file = open(fn, 'w')
        file.write('OFF\n')
        file.write('%d %d %d\n' % (len(verts), len(faces), 0))
        for vert in verts:
            file.write('%f %f %f\n' % (vert[0], vert[1], vert[2]))
            # verts.append([float(s) for s in readline().strip().split(' ')])
        for face in faces:
            file.write('3 %f %f %f\n' % (face[0], face[1], face[2]))
            # faces.append([int(s) for s in readline().strip().split(' ')][1:])
        file.close()
        return

    def read_off(fn):
        file = open(fn, 'r')
        if 'OFF' != file.readline().strip():
            print ('Not a valid OFF header')
            return
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = []
        for i_vert in range(n_verts):
            verts.append([float(s) for s in file.readline().strip().split(' ')])
        faces = []
        for i_face in range(n_faces):
            faces.append([int(s) for s in file.readline().strip().split(' ')][1:])
        file.close()
        return np.asarray(verts,dtype=np.float32), np.asarray(faces, dtype=np.int32)


    test1 = True
    test2 = False
    test1 = False
    test2 = True

    with tf.Session('') as sess:
        # xyz1=np.random.randn(32,16384,3).astype('float32')
        # xyz2=np.random.randn(32,1024,3).astype('float32')
        vertices, faces = read_off('bunny.off')

        if test1:
            verts=tf.expand_dims(tf.constant(vertices),0)
            tris=tf.expand_dims(tf.constant(faces),0)
            nverts = tf.constant([[len(vertices)]],dtype=tf.int32)
            ntris = tf.constant([[len(faces)]],dtype=tf.int32)

            laplacian,_,_ = mesh_laplacian(verts, nverts, tris, ntris)
            loss = tf.nn.l2_loss(laplacian - 0)
            verts1_grad = tf.gradients(loss, [verts])[0]
            laplacian_val, old_count_val, old_nb_val = sess.run([laplacian, count, nb])
            for  i in range(10):
                laplacian_val, count_val, nb_val, verts_val = sess.run([laplacian, count, nb, verts])

                verts_val += 0.3* laplacian_val
                verts=tf.constant(verts_val)
                laplacian,_,_ = mesh_laplacian(verts, nverts, tris, ntris)
                # print laplacian_val
                # print count_val-old_count_val, np.sum(count_val-old_count_val)#, nb_val
                # print np.sum(old_nb_val-nb_val)#, nb_val
            # print laplacian_val
            color = np.squeeze(laplacian_val)
            color = 255*(color - np.min(color))/(np.max(color)-np.min(color))
            color[:,0]=0
            np.savetxt('verts_val.xyz', np.concatenate((vertices, color.astype(np.int8)), axis=1))

            write_off('out.off', np.squeeze(verts_val), faces)



        if test2:

            verts1=tf.expand_dims(tf.constant(vertices),0)
            tris1=tf.expand_dims(tf.constant(faces),0)
            nverts1 = tf.constant([[len(vertices)]],dtype=tf.int32)
            ntris1 = tf.constant([[len(faces)]],dtype=tf.int32)

            laplacian1,_,_ = mesh_laplacian(verts1, nverts1, tris1, ntris1)

            vertices[:,1] *= 2
            # vertices[:,0] *= 2
            # vertices[:,2] *= 2
            old_verts2 = tf.expand_dims(tf.constant(vertices),0)
            verts2=tf.Variable(old_verts2)
            tris2=tf.expand_dims(tf.constant(faces),0)
            nverts2 = tf.constant([[len(vertices)]],dtype=tf.int32)
            ntris2 = tf.constant([[len(faces)]],dtype=tf.int32)

            laplacian2,_,_ = mesh_laplacian(verts2, nverts2, tris2, ntris2)

            #
            loss = 10*tf.nn.l2_loss(laplacian2 - laplacian1)
            verts2_grad = tf.gradients(loss, [verts2])[0]
            #
            train=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

            sess.run(tf.initialize_all_variables())
            #
            old_lossval = 10000
            for  i in range(500):
                # feed_dict = {feats_ph: feats_val}
                _, loss_val, verts2_val, verts1_val, verts2_grad_val,count_val,grad_laplasian_val,old_verts2_val =sess.run([train, loss, verts2, verts1, verts2_grad, count, aaa,old_verts2])#, feed_dict=feed_dict)
                # print feats_val
                # feats_val -= feats_grad_val*0.005
                if loss_val<old_lossval:
                    old_lossval = loss_val
                else:
                    break
                # print loss_val#, verts2_val-old_verts2_val
                # print grad_laplasian_val#np.mean(verts2_grad_val,axis=1)#np.argmax(count_val)

            np.savetxt('verts2.xyz', np.squeeze(verts2_val))
            np.savetxt('verts1.xyz', np.squeeze(verts1_val))

            write_off('out2.off', np.squeeze(verts2_val), faces)
            write_off('old_out2.off', np.squeeze(old_verts2_val), faces)

            write_off('out1.off', np.squeeze(verts1_val), faces)