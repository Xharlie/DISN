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
mesh_sampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_meshsampling_so.so'))
# print mesh_sampling_module.mesh_sampling

def mesh_sampling(verts,nverts,tris,ntris,feats, r, r1, r2):
    '''
        Computes the distance of nearest neighbors for a pair of point clouds
        input: verts: (batch_size,max#verts,3)  vertices coordinates
        input: nverts: (batch_size,1)  vertices numbers
        input: tris: (batch_size,max#faces,3)  Triangle vertice indexes
        input: ntris: (batch_size,1)  Triangle numbers
        input: feats: (batch_size,#points,c)  vertice-wise features ----> Require GRAD
        input: R:  (batch_size,n_samples)   random number to sample points
        input: R1: (batch_size,n_samples)   random number 1 to sample points
        input: R2:  (batch_size,n_samples)   random number 2 to sample points
        output: points: (batch_size,n_samples,3)   points sampled from mesh
        output: outfeats:  (batch_size,n_samples,c)   output features for sampled points
        output: correpondingface:  (batch_size,n_samples)   sample points corresponding face indexes
    '''
    return mesh_sampling_module.mesh_sampling(verts,nverts,tris,ntris,feats, r, r1, r2)

#@tf.RegisterShape('NnDistance')
#def _nn_distance_shape(op):
    #shape1=op.inputs[0].get_shape().with_rank(3)
    #shape2=op.inputs[1].get_shape().with_rank(3)
    #return [tf.TensorShape([shape1.dims[0],shape1.dims[1]]),tf.TensorShape([shape1.dims[0],shape1.dims[1]]),
        #tf.TensorShape([shape2.dims[0],shape2.dims[1]]),tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
@ops.RegisterGradient('MeshSampling')
def _mesh_sampling_grad(op, grad_points, grad_outfeats, grad_correpondingface):
    '''
    .Input("verts: float32")
    .Input("tris: int32")
    .Input("r1: float32")
    .Input("r2: float32")
    .Input("correspondingface: int32")
    .Input("grad_outfeats: float32")
    .Output("grad_feats: float32");
    '''
    # print 'opppppp', len(op.inputs)
    verts = op.inputs[0]
    tris = op.inputs[2]
    r1 = op.inputs[6]
    r2 = op.inputs[7]
    correspondingface=op.outputs[2]
    global grad_feat
    grad_feat = mesh_sampling_module.mesh_sampling_grad(verts,tris,r1,r2,correspondingface,grad_outfeats)#[4]
    # print 'grad_feat', len(grad_feat)
    # print grad_feat

    return [None,None,None,None,grad_feat,None,None,None]


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


    test1 = False
    test2 = True

    with tf.Session('') as sess:
        # xyz1=np.random.randn(32,16384,3).astype('float32')
        # xyz2=np.random.randn(32,1024,3).astype('float32')
        vertices, faces = read_off('/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/part_mesh/Models/Chair/1a6f615e8b1b5ae4dbbc9440457e303e.off')

        with tf.device('/gpu:0'):
            if test1:
                # print np.max(vertices, axis=0)
                # print np.min(vertices, axis=0)
                # with tf.device('/gpu:0'):
                r1 = tf.random_uniform([1, 20000])
                r2 = tf.random_uniform([1, 20000])
                r = tf.random_uniform([1, 20000])
                # print r
                verts=tf.expand_dims(tf.constant(vertices),0)
                tris=tf.expand_dims(tf.constant(faces),0)
                feats=tf.expand_dims(tf.constant(vertices),0)
                nverts = tf.constant([[len(vertices)]],dtype=tf.int32)
                ntris = tf.constant([[len(faces)]],dtype=tf.int32)
                # print verts.get_shape(),tris.get_shape(),feats.get_shape(),nverts.get_shape()
                points, outfeats, correspondingfaces = mesh_sampling(verts, nverts, tris, ntris, feats, r, r1, r2)

                # tic = time.time()
                # for i in range(100):
                #     points_val = sess.run([points])
                # print time.time() - tic

                points_val = sess.run([points])

                # print points_val[0].shape
                points_val = np.squeeze(points_val[0])
                np.savetxt('tmp.xyz', points_val)
                # print np.max(points_val, axis=0)
                # print np.min(points_val, axis=0)

            if test2:

                verts=tf.expand_dims(tf.constant(vertices),0)
                tris=tf.expand_dims(tf.constant(faces),0)
                feats=tf.expand_dims(tf.constant(vertices),0)
                nverts = tf.constant([[len(vertices)]],dtype=tf.int32)
                ntris = tf.constant([[len(faces)]],dtype=tf.int32)

                np.random.seed(int(time.time()))
                r1 = tf.constant(np.random.random_sample((1, 40000)),dtype=tf.float32)
                r2 = tf.constant(np.random.random_sample((1, 40000)),dtype=tf.float32)
                r = tf.constant(np.random.random_sample((1, 40000)),dtype=tf.float32)
                # r1 = tf.random_uniform([1, 40000],dtype=tf.float32)
                # r2 = tf.random_uniform([1, 40000],dtype=tf.float32)
                # r = tf.random_uniform([1, 40000],dtype=tf.float32)

                points, outfeats, correspondingfaces = mesh_sampling(verts, nverts, tris, ntris, feats, r, r1, r2)
                for  i in range(3):
                    points_val,correspondingfaces_val,feats_val,verts_val = sess.run([points,correspondingfaces,feats,verts])
                    # print 'correspondingfaces_val',correspondingfaces_val
                points_val = points_val#[0]

                # points_val, feats_val = sess.run([points, feats])
                # np.savetxt('feats_old.xyz', np.squeeze(feats_val))
                # points_val = points_val#[0]

                # print np.max(points_val, axis=1), np.min(points_val, axis=1)

                points_val = np.concatenate((points_val, verts_val), axis =1)
                points_val[:,:,1] *= 2
                newpc = tf.constant(points_val)
                #
                feats = tf.Variable(feats)
                # feats_ph = tf.placeholder(tf.float32, shape=feats.get_shape())
                # # verts = tf.placeholder(tf.float32,shape=verts.get_shape(), )
                points, outfeats, correspondingfaces = mesh_sampling(verts, nverts, tris, ntris, feats, r, r1, r2)
                #
                outfeats = tf.concat([outfeats, feats], axis=1)
                loss = tf.nn.l2_loss(newpc - outfeats)
                feats_grad = tf.gradients(loss, [feats])[0]
                #
                train=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

                sess.run(tf.initialize_all_variables())
                #
                old_lossval = 10000
                # print "start optimization"
                for  i in range(100):
                    # feed_dict = {feats_ph: feats_val}
                    _, loss_val, feats_val, points_val, newpc_val, outfeats_val, feats_grad_val,correspondingfaces_val2 =sess.run([train, loss, feats, points, newpc, outfeats, feats_grad, correspondingfaces])#, feed_dict=feed_dict)
                    # print feats_val
                    # feats_val -= feats_grad_val*0.005
                    # if loss_val<old_lossval:
                    #     old_lossval = loss_val
                    # else:
                    #     break
                    # print loss_val, np.argmax(feats_val[:,:,1]), np.min(feats_val[:,:,1]), np.max(newpc_val[:,:,1]), np.min(newpc_val[:,:,1])
                    # print feats_grad_val[:,np.argmax(feats_val[:,:,1]),1]
                    # print newpc_val[:,:,1]
                    # print outfeats_val[:,:,1]
                    # print feats_val.shape
                    # print feats_val
                    # print np.max(feats_grad_val[:,:,1]),np.min(feats_grad_val[:,:,1])
                #
                # np.savetxt('pts.xyz', points_val)
                np.savetxt('feats.xyz', np.squeeze(feats_val))
                np.savetxt('newpc.xyz', np.squeeze(newpc_val))
                np.savetxt('outfeats.xyz', np.squeeze(outfeats_val))
                write_off('out.off', np.squeeze(feats_val), faces)


