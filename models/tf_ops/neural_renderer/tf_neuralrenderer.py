"""
Reference: https://github.com/hiroharu-kato/neural_renderer
           https://github.com/akanazawa/cmr/blob/master/nnutils/nmr.py
		   https://www.tensorflow.org/extend/adding_an_op

"""

import tensorflow as tf
import chainer
import time
import scipy.misc
import neural_renderer
import numpy as np
import skimage
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
sys.path.append(ROOT_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))
print os.path.join(ROOT_DIR, 'data')
import obj

def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 8: [sc, tx, ty, tz, quaternions]
    Orth preserving the z.
    """
    batch_size = cam.get_shape()[0]
    quat = cam[:, -4:]
    quat = tf.expand_dims(quat, 1)
    X_rot = rotate_vector_by_quaternion(quat, X)

    scale = tf.reshape(cam[:, 0], shape = (-1, 1, 1))
    trans = tf.reshape(cam[:, 1:4], shape = (batch_size, 1, -1))

    proj = scale * X_rot

    proj_xyz = proj + trans
    # proj_xy = proj[:, :, :2] + trans
    # proj_z = proj[:, :, 2, None] + offset_z

    return proj_xyz#tf.concat([proj_xy, proj_z], 2)


def rotate_vector_by_quaternion(q, v, q_ndims=None, v_ndims=None):
    """
    Reference: https://github.com/PhilJd/tf-quaternion/blob/master/tfquaternion/tfquaternion.py
    Rotate a vector (or tensor with last dimension of 3) by q.
    This function computes v' = q * v * conjugate(q) but faster.
    Fast version can be found here:
    https://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/
    Args:
        q: A `Quaternion` or `tf.Tensor` with shape (..., 4)
        v: A `tf.Tensor` with shape (..., 3)
        q_ndims: The number of dimensions of q. Only necessary to specify if
            the shape of q is unknown.
        v_ndims: The number of dimensions of v. Only necessary to specify if
            the shape of v is unknown.
    Returns: A `tf.Tensor` with the broadcasted shape of v and q.
    """

    v = tf.convert_to_tensor(v)
    # normalize
    norm = tf.sqrt(tf.reduce_sum(tf.square(q), axis=-1, keep_dims=True))
    q = tf.divide(q, norm)
    # tf.sqrt(self.norm(keepdims))
    # q = q.normalized()
    w = q[..., 0]
    q_xyz = q[..., 1:]
    if q_xyz.shape.ndims is not None:
        q_ndims = q_xyz.shape.ndims
    if v.shape.ndims is not None:
        v_ndims = v.shape.ndims
    for _ in range(v_ndims - q_ndims):
        q_xyz = tf.expand_dims(q_xyz, axis=0)
    for _ in range(q_ndims - v_ndims):
        v = tf.expand_dims(v, axis=0) + tf.zeros_like(q_xyz)
    q_xyz += tf.zeros_like(v)
    v += tf.zeros_like(q_xyz)
    t = 2 * tf.cross(q_xyz, v)

    return v + tf.expand_dims(w, axis=-1) * t + tf.cross(q_xyz, t)

########################################################################
############ Wrapper class for the chainer Neural Renderer #############
##### All functions must only use numpy arrays as inputs/outputs #######
########################################################################
class NMR(object):
    def __init__(self, renderer, img_size=256):
        # setup renderer
        # renderer = neural_renderer.Renderer()
        # self.renderer = renderer
        self.renderer = renderer
        # self.renderer.image_size = img_size
        self.renderer.perspective = True

        # Set a default camera to be at (0, 0, -2.732)
        self.renderer.eye = [2.732, 0.5, -2.732]#(0, 0.3, -1.5)#[0, 1., -5.]#[2.732, 0.5, -2.732]#neural_renderer.get_points_from_angles(2, 15, -90)#[6, 10, -14]#[2.732, 0, -2.732]#[0, 0, -2.732]#
        # self.renderer.eye = [0, 0, -2.732]#
        # self.renderer.light_direction = [0, 1, -1]
        self.renderer.light_intensity_directional = 0.2
        self.renderer.background_color = [1.,1.,1.]

        # self.renderer.light_intensity_ambient = 0.5
        # self.renderer.light_intensity_directional = 0.5

    def to_gpu(self, device=0):
        # self.renderer.to_gpu(device)
        self.cuda_device = device

    def forward_mask(self, verts, nverts, tris, ntris):#verts, nverts, tris, ntris):
        ''' Renders masks.
        Args:
            verts: B X MaxNverts X 3 numpy array
            nverts: B X 1 numpy array
            tris: B X MaxNtris X 3 numpy array
            ntris: B X 1 numpy array
        Returns:
            masks: B X 256 X 256 numpy array
        '''

        # for i in range(len(nverts)):
        # print vertices.shape, faces.shape
        batch_size = len(nverts)
        masks = np.zeros([batch_size, self.renderer.image_size, self.renderer.image_size], dtype=np.float32)
        self.masks = []
        self.verts_mask = []
        self.nverts = nverts
        self.nverts_mask = nverts
        # print(self.nverts)#, verts.shape, ntris, tris.shape)
        for ib in range(batch_size):
            tris_chainer = chainer.Variable(chainer.cuda.to_gpu(tris[[ib],:ntris[ib, 0], ...], self.cuda_device))
            verts_chainer = chainer.Variable(chainer.cuda.to_gpu(verts[[ib],:nverts[ib, 0], ...], self.cuda_device))
            mask = self.renderer.render_silhouettes(verts_chainer, tris_chainer)
            self.masks += [mask]
            self.verts_mask += [verts_chainer]
            masks[ib,:,:] = mask.data.get()

        return masks

    def backward_mask(self, grad_masks):
        ''' Compute gradient of vertices given mask gradients.
        Args:
            grad_masks: B X 256 X 256 numpy array
        Returns:
            grad_vertices: B X N X 3 numpy array
        '''
        batch_size = len(grad_masks)
        verts_grad = np.zeros(self.verts_size, dtype=np.float32)
        # print ('batch', batch_size, len(self.verts_mask), len(self.masks))
        for ib in range(batch_size):
            mask = self.masks[ib]
            mask.grad = chainer.cuda.to_gpu(grad_masks[[ib],:,:], self.cuda_device)
            mask.backward()
            # print (self.verts_mask[ib].shape, self.nverts[ib])
            verts_grad[ib,:self.nverts[ib,0],:] = self.verts_mask[ib].grad.get()
        # print np.sum(verts_grad)

        return verts_grad

    def forward_img(self, verts, nverts, tris, ntris, textures):
        ''' Renders masks.
        Args:
            verts: B X MaxNverts X 3 numpy array
            nverts: B X 1 numpy array
            tris: B X MaxNtris X 3 numpy array
            ntris: B X 1 numpy array
            textures: B X F X T X T X T X 3 numpy array
        Returns:
            images: B X 3 x 256 X 256 numpy array
        '''
        # print 'vertices', vertices.shape
        # print 'faces', faces.shape
        # print 'textures', textures.shape

        if len(textures.shape) < 6:
            textures = None 
        time1 = time2 = time3 = 0.
        batch_size = len(nverts)
        images = np.zeros([batch_size, 3, self.renderer.image_size, self.renderer.image_size], dtype=np.float32)
        self.images = []
        self.verts_img = []
        self.textures = []
        self.nverts = nverts
        self.ntris = ntris
        ticc = time.time()
        for ib in range(batch_size):
            # print 'ib', ib
            # print tris[[ib],:ntris[ib, 0], ...].shape, verts[[ib],:nverts[ib, 0], ...].shape, textures[[ib],:ntris[ib, 0], ...].shape
            # print tris[[ib],:ntris[ib, 0], ...]
            tic = time.time()
            if textures == None:
                texture_tmp = np.ones([1, ntris[ib, 0], 4, 4, 4, 3], dtype = np.float32)
            else:
                texture_tmp = textures[[ib],:ntris[ib, 0], ...]
            tris_chainer = chainer.Variable(chainer.cuda.to_gpu(tris[[ib],:ntris[ib, 0], ...], self.cuda_device))
            verts_chainer = chainer.Variable(chainer.cuda.to_gpu(verts[[ib],:nverts[ib, 0], ...], self.cuda_device))
            textures_chainer = chainer.Variable(chainer.cuda.to_gpu(texture_tmp, self.cuda_device))
            time1 += (time.time() - tic)
            tic = time.time()
            image = self.renderer.render(verts_chainer, tris_chainer, textures_chainer)
            time2 += (time.time() - tic)
            tic = time.time()
            self.images += [image]
            self.verts_img += [verts_chainer]
            self.textures += [textures_chainer]
            images[ib,:,:] = image.data.get()
            time3 += (time.time() - tic)
        # print (time.time() - ticc,)   

        return images


    def backward_img(self, grad_images):
        ''' Compute gradient of vertices given image gradients.
        Args:
            grad_images: B X 3? X 256 X 256 numpy array
        Returns:
            grad_vertices: B X N X 3 numpy array
            grad_textures: B X F X T X T X T X 3 numpy array
        '''
        batch_size = len(grad_images)
        verts_grad = np.zeros(self.verts_size, dtype=np.float32)
        # print 'verts_grad', verts_grad.shape
        textures_grad = np.zeros(self.textures_size, dtype=np.float32)
        # print ('batch', batch_size, len(self.verts_img), len(self.images))
        for ib in range(batch_size):
            image = self.images[ib]
            image.grad = chainer.cuda.to_gpu(grad_images[[ib],...], self.cuda_device)
            image.backward()
            verts_grad[ib,:self.nverts[ib,0],:] = self.verts_img[ib].grad.get()
            if len(textures_grad.shape) == 6:
                textures_grad[ib,:self.ntris[ib,0],:] = self.textures[ib].grad.get()

        # self.images.grad = chainer.cuda.to_gpu(grad_images, self.cuda_device)
        # self.images.backward()
        return verts_grad, textures_grad#self.vertices.grad.get(), self.textures.grad.get()

    def neural_renderer_mask(self, verts, nverts, tris, ntris, name=None, stateful=True):
        with ops.name_scope(name, "NeuralRenderer") as name:
            rnd_name = 'NeuralRendererGrad' + str(np.random.randint(0, 1E+8))
            tf.RegisterGradient(rnd_name)(self._neural_renderer_mask_grad)  # see _MySquareGrad for grad example
            g = tf.get_default_graph()
            self.to_gpu()
            self.verts_size = verts.shape
            with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
                mask = tf.py_func(self.forward_mask,
                                  [verts, nverts, tris, ntris],
                                  [tf.float32],
                                  stateful=stateful,
                                  name=name)[0]
                mask.set_shape([verts.shape[0], self.renderer.image_size, self.renderer.image_size])
            return mask

    def _neural_renderer_mask_grad(self, op, grad_mask):
        tmp_grad_name = 'NeuralRendererGradPyFunc'+ str(np.random.randint(low=0,high=1e+8))
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": tmp_grad_name, "PyFuncStateless": tmp_grad_name}):
            grad_verts = tf.py_func(self.backward_mask,
                              [grad_mask],
                              [tf.float32],
                              stateful=True,
                              name=tmp_grad_name)[0]
            grad_verts.set_shape(self.verts_size)
        return [grad_verts, None, None, None]


    def neural_renderer_texture(self, verts, nverts, tris, ntris, textures, name=None, stateful=True):
        with ops.name_scope(name, "NeuralRendererTexture") as name:
            rnd_name = 'NeuralRendererTextureGrad' + str(np.random.randint(0, 1E+8))
            tf.RegisterGradient(rnd_name)(self._neural_renderer_texture_grad)  # see _MySquareGrad for grad example
            g = tf.get_default_graph()
            self.to_gpu()
            self.verts_size = verts.shape
            if textures == None:
                textures = np.array([0], dtype = np.float32)
            self.textures_size = textures.shape
            with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
                img = tf.py_func(self.forward_img,
                                  [verts, nverts, tris, ntris, textures],
                                  [tf.float32],
                                  stateful=stateful,
                                  name=name)[0]
                img.set_shape([verts.shape[0], 3, self.renderer.image_size, self.renderer.image_size])
                return img

    def _neural_renderer_texture_grad(self, op, grad_img):
        tmp_grad_name = 'NeuralRendererTextureGradPyFunc'+ str(np.random.randint(low=0,high=1e+8))
        # print ('grad_img', grad_img.get_shape())
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": tmp_grad_name, "PyFuncStateless": tmp_grad_name}):
            grad_verts, grad_texture = tf.py_func(self.backward_img,
                                                  [grad_img],
                                                  [tf.float32, tf.float32],
                                                  stateful=True,
                                                  name=tmp_grad_name)
            grad_verts.set_shape(self.verts_size)  
            grad_texture.set_shape(self.textures_size)  
        # print ('grad_verts', grad_verts.get_shape(), 'grad_texture', grad_texture.get_shape())

        if len(grad_texture.shape) == 6:
            return [grad_verts, None, None, None, grad_texture]
        else:
            return [grad_verts, None, None, None, None]


if __name__ == '__main__':
    ## unitests
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

    def write_off(fn, verts, faces):
        file = open(fn, 'w')
        file.write('OFF\n')
        file.write('%d %d %d\n' % (len(verts), len(faces), 0))
        for vert in verts:
            file.write('%f %f %f\n' % (vert[0], vert[1], vert[2]))
            # verts.append([float(s) for s in readline().strip().split(' ')])
        for face in faces:
            file.write('3 %d %d %d\n' % (face[0], face[1], face[2]))
            # faces.append([int(s) for s in readline().strip().split(' ')][1:])
        file.close()
        return


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    with tf.Session('', config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # with tf.device('/cpu:0'):

        # test1: test mask
        if False:
            verts, tris = read_off('/mnt/ilcompf8d0/user/weiyuewa/sources/pipeline1/pointnet-ffd/utils/iso_sphere.off')
            nmr = NMR()

            verts_pl = tf.Variable(
                tf.constant(np.expand_dims(verts, axis=0)))  # tf.placeholder(tf.float32, shape=(1, len(verts), 3))
            verts_gt = tf.constant(np.expand_dims(verts, axis=0)) * 0.5
            tris_pl = tf.placeholder(tf.int32, shape=(1, len(tris), 3))

            img = nmr.neural_renderer_mask(verts_pl, tris_pl, 'render')
            image_ref = scipy.misc.imread('tmp_ref.png').astype('float32') / 255.
            image_ref = tf.constant(image_ref)
            image_ref = tf.expand_dims(image_ref, axis=0)
            print (image_ref.get_shape())
            loss = tf.nn.l2_loss(img - image_ref)
            print (img.get_shape(), image_ref.get_shape())
            grad_verts = tf.gradients(loss, verts_pl)[0]
            train = tf.train.AdamOptimizer(1e-2).minimize(loss)
            # train=tf.train.GradientDescentOptimizer(learning_rate=0.0005).minimize(loss)
            sess.run(tf.initialize_all_variables())

            for i in range(100):
                _, loss_val, img_val, grad_verts_val, tmp_val = sess.run([train, loss, img, grad_verts, tmp], feed_dict={#verts_pl: np.expand_dims(verts, axis=0),
                                                     tris_pl: np.expand_dims(tris, axis=0),})
                print (i, loss_val, np.sum(grad_verts_val), np.sum(tmp_val))
                print (img_val.shape)

            print (img_val[0][0].shape)
            cv2.imwrite('tmp.png', (img_val[0][0] * 255).astype(np.uint8))

        # test2: test img texture
        if False:
            verts, tris = read_off('/mnt/ilcompf8d0/user/weiyuewa/sources/pipeline1/pointnet-ffd/utils/iso_sphere.off')
            nmr = NMR()
            verts_pl = tf.Variable(
                tf.constant(np.expand_dims(verts, axis=0)))  # tf.placeholder(tf.float32, shape=(1, len(verts), 3))
            verts_gt = tf.constant(np.expand_dims(verts, axis=0)) * 0.5
            tris_pl = tf.placeholder(tf.int32, shape=(1, len(tris), 3))

            img_file = 'example3_ref.png'
            image_ref = scipy.misc.imread(img_file).astype('float32') / 255.
            image_ref = tf.constant(image_ref[None, :, :])
            image_ref = tf.transpose(image_ref, [0, 3, 1, 2])
            texture = tf.Variable(tf.ones([1, tris.shape[0], 2, 2, 2, 3],dtype=tf.float32))
            img = nmr.neural_renderer_texture(verts_pl, tris_pl, texture, 'render')
            print (img.get_shape(), image_ref.get_shape())
            loss = tf.nn.l2_loss(img - image_ref)
            train = tf.train.AdamOptimizer(1e-2).minimize(loss)
            sess.run(tf.initialize_all_variables())

            for i in range(100):
                _, loss_val, img_val, verts_pl_val = sess.run([train, loss, img, verts_pl], feed_dict={#verts_pl: np.expand_dims(verts, axis=0),
                                                     tris_pl: np.expand_dims(tris, axis=0),})
                print (i, loss_val, img_val.shape)

            print (img_val[0].shape)
            img_val = np.transpose(img_val[0], (1,2,0))
            cv2.imwrite('tmp.png', (img_val * 255).astype(np.uint8))
            write_off('tmp.off', np.squeeze(verts_pl_val), tris)

        # test3: test img texture and camera
        if False:
            verts, tris = read_off('/mnt/ilcompf8d0/user/weiyuewa/sources/pipeline1/pointnet-ffd/utils/iso_sphere.off')
            nmr = NMR()
            verts_pl = tf.Variable(
                tf.constant(np.expand_dims(verts, axis=0)))  # tf.placeholder(tf.float32, shape=(1, len(verts), 3))
            verts_gt = tf.constant(np.expand_dims(verts, axis=0)) * 0.5
            tris_pl = tf.placeholder(tf.int32, shape=(1, len(tris), 3))

            img_file = 'example3_ref.png'
            image_ref = scipy.misc.imread(img_file).astype('float32') / 255.
            image_ref = tf.constant(image_ref[None, :, :])
            image_ref = tf.transpose(image_ref, [0, 3, 1, 2])
            texture = tf.Variable(tf.ones([1, tris.shape[0], 2, 2, 2, 3],dtype=tf.float32))

            cams = np.array([[1., 0, 0, 1, 0, 0, 0]], dtype=np.float32)
            cams = tf.Variable(tf.constant(cams))
            verts_pl = orthographic_proj_withz(verts_pl, cams, 5.)

            img = nmr.neural_renderer_texture(verts_pl, tris_pl, texture, 'render')
            print (img.get_shape(), image_ref.get_shape())
            loss = tf.nn.l2_loss(img - image_ref)
            train = tf.train.AdamOptimizer(1e-2).minimize(loss)
            sess.run(tf.initialize_all_variables())

            for i in range(100):
                _, loss_val, img_val, verts_pl_val = sess.run([train, loss, img, verts_pl], feed_dict={#verts_pl: np.expand_dims(verts, axis=0),
                                                     tris_pl: np.expand_dims(tris, axis=0),})
                print (i, loss_val, img_val.shape)

            print (img_val[0].shape)
            img_val = np.transpose(img_val[0], (1,2,0))
            cv2.imwrite('tmp.png', (img_val * 255).astype(np.uint8))
            write_off('tmp.off', np.squeeze(verts_pl_val), tris)

        # test3: test obj loader
        if False:
            verts, tris, texture = neural_renderer.load_obj(
                '/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/ShapeNetCore.v2/03001627/1f65075818c1d832c05575120a46cd3b/models/model_normalized.obj',
                load_texture=True, texture_size=4)

            verts, tris, texture1 = obj.load_obj1(
                '/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/ShapeNetCore.v2/03001627/1f65075818c1d832c05575120a46cd3b/models/model_normalized.obj',#'/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/ShapeNetCore.v2/03001627/1b81441b7e597235d61420a53a0cb96d/models/model_normalized.obj',
                load_texture=True, texture_size=4)

            print (texture.shape, texture1.shape,'equa;')
            # print texture[1000,...]
            # print texture1[1000,...]
            print (np.sum(np.abs(texture - texture1)))
            # print texture
            print (np.mean(np.reshape(texture,[-1, 3]), axis=0))
            print (texture.shape)
            nmr = NMR()
            verts_pl = tf.Variable(
                tf.constant(np.expand_dims(verts, axis=0)))  # tf.placeholder(tf.float32, shape=(1, len(verts), 3))
            verts_gt = tf.constant(np.expand_dims(verts, axis=0)) * 0.5
            tris_pl = tf.placeholder(tf.int32, shape=(1, len(tris), 3))
            texture = tf.tanh(tf.expand_dims(tf.constant(texture), axis=0))
            texture1 = tf.tanh(tf.expand_dims(tf.constant(texture1), axis=0))

            cams = np.array([[1., 0, 0, 1, 0, 0, 0]], dtype=np.float32)
            cams = tf.Variable(tf.constant(cams))
            # verts_pl = orthographic_proj_withz(verts_pl, cams, 5.)
            # texture = tf.Variable(tf.ones([1, tris.shape[0], 2, 2, 2, 3], dtype=tf.float32))
            img = nmr.neural_renderer_texture(verts_pl, tris_pl, texture, 'render')
            img1 = nmr.neural_renderer_texture(verts_pl, tris_pl, texture1, 'render1')
            sess.run(tf.initialize_all_variables())

            img_val, img_val1 = sess.run([img,img1],
                              feed_dict={verts_pl: np.expand_dims(verts, axis=0),
                                  tris_pl: np.expand_dims(tris, axis=0), })

            print (img_val[0].shape)
            img_val = np.transpose(img_val[0], (1, 2, 0))
            img_val1 = np.transpose(img_val1[0], (1, 2, 0))
            cv2.imwrite('tmp.png', (img_val[:,:,::-1] * 255).astype(np.uint8))
            cv2.imwrite('tmp1.png', (img_val1[:,:,::-1] * 255).astype(np.uint8))

        # test4: test batch loader
        if False:
            with open('/home/weiyuewa/Downloads/chairs/list.txt', 'r') as f:
                lines = f.read().splitlines()
                datapath = [line.strip() + '/models/model_normalized.obj' for line in lines]

            batchsize = 4
            maxnverts = 20000
            maxntris = 20000
            texture_size = 2
            batchverts = np.zeros([batchsize, maxnverts, 3]).astype(np.float32)
            batchtris = np.zeros([batchsize, maxntris, 3]).astype(np.int32)
            batchnverts = np.zeros([batchsize, 1]).astype(np.int32)
            batchntris = np.zeros([batchsize, 1]).astype(np.int32)
            batchtextures = np.zeros([batchsize, maxntris, texture_size, texture_size, texture_size, 3]).astype(
                np.float32)

            count = 0
            for id, datafn in enumerate(datapath):

                if count >= batchsize:
                    break

                # verts, tris, texture = obj.load_obj1(
                #     '/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/ShapeNetCore.v2/03001627/1f65075818c1d832c05575120a46cd3b/models/model_normalized.obj',#'/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/ShapeNetCore.v2/03001627/1b81441b7e597235d61420a53a0cb96d/models/model_normalized.obj',
                #     load_texture=True, texture_size=texture_size)

                print (datafn)
                verts, tris, texture = obj.load_obj1(datafn, load_texture=True, texture_size=texture_size)
                print ('len(verts)', len(verts), 'len(tris)', len(tris))

                if (len(verts) > maxnverts or len(tris) > maxntris):
                    continue

                print (verts.shape, tris.shape, texture.shape)
                batchverts[count, :len(verts), :] = verts
                batchtris[count, :len(tris), :] = tris
                batchtextures[count, :len(tris), :, :, :, :] = texture
                batchnverts[count, 0] = len(verts)
                batchntris[count, 0] = len(tris)
                count += 1

            print ('count', count)

            nmr = NMR()
            verts_pl = tf.Variable(tf.constant(batchverts))
            tris_pl = tf.constant(batchtris)  # tf.placeholder(tf.int32, shape=(batchsize, maxntris, 3))
            nverts_pl = tf.constant(batchnverts)
            ntris_pl = tf.constant(batchntris)
            texture = tf.constant(batchtextures)
            # texture = tf.tanh(tf.constant(batchtextures))

            cams = np.array([[1., 0, 0, 1, 0, 0, 0]], dtype=np.float32)
            cams = tf.Variable(tf.constant(cams))

            img = nmr.neural_renderer_texture(verts_pl, nverts_pl, tris_pl, ntris_pl, texture, 'render')

            sess.run(tf.initialize_all_variables())

            img_val = sess.run([img])

            print (img_val[0][0].shape)
            for ib in range(batchsize):
                cv2.imwrite('results/%d.png' % ib,
                            (np.transpose(img_val[0][ib], (1, 2, 0))[:, :, ::-1] * 255).astype(np.uint8))

        # test5: test batch loader and backward
        if False:
            with open('/home/weiyuewa/Downloads/chairs/list.txt', 'r') as f:
                lines = f.read().splitlines()
                datapath = [line.strip()+'/models/model_normalized.obj' for line in lines]

            batchsize = 4
            maxnverts = 20000
            maxntris = 20000
            texture_size = 2
            batchverts = np.zeros([batchsize, maxnverts, 3]).astype(np.float32)
            batchtris = np.zeros([batchsize, maxntris, 3]).astype(np.int32)
            batchnverts = np.zeros([batchsize, 1]).astype(np.int32)
            batchntris = np.zeros([batchsize, 1]).astype(np.int32)
            batchtextures = np.zeros([batchsize, maxntris, texture_size, texture_size, texture_size, 3]).astype(np.float32)

            count = 0
            for id, datafn in enumerate(datapath):

                if count >= batchsize:
                    break

                # verts, tris, texture = obj.load_obj1(
                #     '/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/ShapeNetCore.v2/03001627/1f65075818c1d832c05575120a46cd3b/models/model_normalized.obj',#'/mnt/ilcompf8d0/user/weiyuewa/dataset/shapenet/ShapeNetCore.v2/03001627/1b81441b7e597235d61420a53a0cb96d/models/model_normalized.obj',
                #     load_texture=True, texture_size=texture_size)

                print (datafn)
                verts, tris, texture = obj.load_obj1(datafn, load_texture=True, texture_size=texture_size)
                print ('len(verts)', len(verts), 'len(tris)', len(tris))

                if (len(verts) > maxnverts or len(tris) > maxntris):
                    continue

                print (verts.shape, tris.shape, texture.shape)
                batchverts[count,:len(verts),:] = verts
                batchtris[count,:len(tris),:] = tris
                batchtextures[count,:len(tris),:,:,:,:] = texture
                batchnverts[count, 0] = len(verts)
                batchntris[count, 0] = len(tris)
                count += 1

            print ('count', count)

            img_file = '2.png'
            image_ref = scipy.misc.imread(img_file).astype('float32') / 255.
            image_ref = tf.constant(image_ref[None, :, :])
            image_ref = tf.transpose(image_ref, [0, 3, 1, 2])
            image_ref = tf.tile(image_ref, [batchsize, 1, 1, 1])

            nmr = NMR()
            verts_pl = tf.Variable(tf.constant(batchverts))
            tris_pl = tf.constant(batchtris)#tf.placeholder(tf.int32, shape=(batchsize, maxntris, 3))
            nverts_pl = tf.constant(batchnverts)
            ntris_pl = tf.constant(batchntris)
            texture = tf.constant(batchtextures)
            # texture = tf.tanh(tf.constant(batchtextures))

            cams = np.array([[1., 0, 0, 1, 0, 0, 0]], dtype=np.float32)
            cams = tf.Variable(tf.constant(cams))

            img = nmr.neural_renderer_texture(verts_pl, nverts_pl, tris_pl, ntris_pl, texture, 'render')
            loss = tf.nn.l2_loss(img - image_ref)
            train = tf.train.AdamOptimizer(0.1).minimize(loss)

            sess.run(tf.initialize_all_variables())

            img_val = sess.run([img])

            for i in range(100):
                _, loss_val, img_val, verts_pl_val = sess.run([train, loss, img, verts_pl])
                print (i, loss_val, img_val.shape)

            print (img_val[0].shape)
            for ib in range(batchsize):
                cv2.imwrite('results/%d.png' % ib, (np.transpose(img_val[ib], (1, 2, 0))[:,:,::-1] * 255).astype(np.uint8))

        # test6: test batch loader and backward and mask
        if True:
            with open('/media/hdd2/data/ShapeNet/filelists/ShapeNetCore.v2/03001627_obj.lst', 'r') as f:
                lines = f.read().splitlines()
                datapath = [line+'.obj' for line in lines]

            batchsize = 4
            maxnverts = 40000
            maxntris = 40000
            texture_size = 2
            batchverts = np.zeros([batchsize, maxnverts, 3]).astype(np.float32)
            batchtris = np.zeros([batchsize, maxntris, 3]).astype(np.int32)
            batchnverts = np.zeros([batchsize, 1]).astype(np.int32)
            batchntris = np.zeros([batchsize, 1]).astype(np.int32)
            batchtextures = np.zeros([batchsize, maxntris, texture_size, texture_size, texture_size, 3]).astype(np.float32)

            count = 0
            for id, datafn in enumerate(datapath[0:5]):

                if count >= batchsize:
                    break

                print (datafn)
                verts, tris, texture = obj.load_obj(datafn, load_texture=True, texture_size=texture_size, maxnverts=maxnverts, maxntris=maxntris)
                print ('len(verts)', len(verts), 'len(tris)', len(tris))

                if (len(verts) > maxnverts or len(tris) > maxntris):
                    continue

                print (verts.shape, tris.shape, texture.shape)
                batchverts[count,:len(verts),:] = verts
                batchtris[count,:len(tris),:] = tris
                batchtextures[count,:len(tris),:,:,:,:] = texture
                batchnverts[count, 0] = len(verts)
                batchntris[count, 0] = len(tris)
                count += 1

            print ('count', count)

            img_file = 'ref_mask.png'
            image_ref = scipy.misc.imread(img_file).astype('float32') / 255.
            image_ref = tf.constant(np.expand_dims(image_ref[:, :], axis=0))
            print ('image_ref', image_ref.get_shape())
            # image_ref = tf.transpose(image_ref, [2, 0, 1])
            image_ref = tf.tile(image_ref, [batchsize, 1, 1])

            nmr = NMR()
            verts_pl = tf.Variable(tf.constant(batchverts))
            tris_pl = tf.constant(batchtris)#tf.placeholder(tf.int32, shape=(batchsize, maxntris, 3))
            nverts_pl = tf.constant(batchnverts)
            ntris_pl = tf.constant(batchntris)
            texture = tf.constant(batchtextures)
            # texture = tf.tanh(tf.constant(batchtextures))

            cams = np.array([[1., 0, 0, 1, 0, 0, 0]], dtype=np.float32)
            cams = tf.Variable(tf.constant(cams))

            # verts_pl = orthographic_proj_withz(verts_pl, cams, 5.)

            img = nmr.neural_renderer_mask(verts_pl, nverts_pl, tris_pl, ntris_pl, 'render')
            loss = tf.nn.l2_loss(img - image_ref)
            train = tf.train.AdamOptimizer(0.1).minimize(loss)

            sess.run(tf.initialize_all_variables())

            img_val = sess.run([img])

            for i in range(100):
                _, loss_val, img_val, verts_pl_val = sess.run([train, loss, img, verts_pl])
                print (i, loss_val, img_val.shape)

            print (img_val[0].shape)
            for ib in range(batchsize):
                cv2.imwrite('results/%d.png' % ib, (img_val[ib] * 255).astype(np.uint8))
