import tensorflow as tf
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module

# batch*n
def normalize_vector(v):
    batch=v.shape[0]

    v_mag = tf.sqrt(tf.reduce_sum(tf.square(v), axis=1, keepdims=True))
    v_mag = tf.maximum(v_mag, 1e-8)
    # v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v / v_mag
    return v

#########################
def cross_product(u, v):
    
    # batch = u.shape[0]
    # i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    # j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    # k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    # out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
    return out
       
######################
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3]#batch*3
    y_raw = poses[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = tf.linalg.cross(x,y_raw) #batch*3
    # z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = tf.linalg.cross(z,x) #batch*3
    # y = cross_product(z,x)#batch*3
        
    # x = x.view(-1,3,1)
    # y = y.view(-1,3,1)
    # z = z.view(-1,3,1)
    print('x', x.shape, 'y', y.shape, 'z', z.shape)
    x = tf.reshape(x, [-1, 3, 1])
    y = tf.reshape(y, [-1, 3, 1])
    z = tf.reshape(z, [-1, 3, 1])
    matrix = tf.concat((x,y,z), 2) #batch*3*3
    print('matrix', matrix.shape)

    return matrix

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = tf.matmul(m1, tf.transpose(m2, [0,2,1])) #batch*3*3
    cos = (m[:,0,0] + m[:,1,1] + m[:,2,2] - 1) / 2.
    cos = tf.minimum(cos, 1.)
    cos = tf.maximum(cos, -1.)

    theta = tf.acos(cos)

    return theta

#############
def get_cam_mat(globalfeat, is_training, batch_size, bn, bn_decay, wd=None):

    # with tf.variable_scope("rotation") as scope:  
    #     rotation = tf_util.fully_connected(globalfeat, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    #     rotation = tf_util.fully_connected(rotation, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    #     weights = tf.get_variable('fc3/weights', [256, 3*3],
    #                               initializer=tf.constant_initializer(0.0),
    #                               dtype=tf.float32)
    #     biases = tf.get_variable('fc3/biases', [3*3],
    #                              initializer=tf.constant_initializer(0.0),
    #                              dtype=tf.float32)
    #     biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
    #     rotation = tf.matmul(rotation, weights)
    #     rotation = tf.nn.bias_add(rotation, biases)
    #     pred_rotation = tf.reshape(rotation, [batch_size, 3, 3])

    with tf.variable_scope("scale") as scope:   #
        scale = tf_util.fully_connected(globalfeat, 64, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        scale = tf_util.fully_connected(scale, 32, bn=bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        scale = tf_util.fully_connected(scale, 1, bn=bn, is_training=is_training, scope='fc3', activation_fn=None, bn_decay=bn_decay)
        pred_scale = tf.reshape(scale, [batch_size, 1, 1]) * tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
    with tf.variable_scope("ortho6d") as scope:   #
        rotation = tf_util.fully_connected(globalfeat, 512, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        rotation = tf_util.fully_connected(rotation, 256, bn=bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        rotation = tf_util.fully_connected(rotation, 6, bn=bn, is_training=is_training, scope='fc3', activation_fn=None, bn_decay=bn_decay)
        pred_rotation = tf.reshape(rotation, [batch_size, 6])

    with tf.variable_scope("translation") as scope:  
        translation = tf_util.fully_connected(globalfeat, 128, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        translation = tf_util.fully_connected(translation, 64, bn=bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        # w_trans_init = 
        weights = tf.get_variable('fc3/weights', [64, 3],
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, seed=1),
                                  dtype=tf.float32)
        biases = tf.get_variable('fc3/biases', [3],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        translation = tf.matmul(translation, weights)
        translation = tf.nn.bias_add(translation, biases)
        # translation = tf_util.fully_connected(translation, 3, bn=True, is_training=is_training, scope='fc3', activation_fn=None, bn_decay=bn_decay)
        # translation = tf.tanh(translation) * 0.15
        pred_translation = tf.reshape(translation, [batch_size, 3])
        pred_translation += tf.constant([-0.00193892, 0.00169222, 1.3949631], dtype=tf.float32)

    pred_translation = tf.reshape(pred_translation, [batch_size, 1, 3])
    pred_rotation_mat = compute_rotation_matrix_from_ortho6d(pred_rotation)
    pred_rotation_mat = tf.matmul(pred_scale, pred_rotation_mat)
    pred_RT = tf.concat([pred_rotation_mat, pred_translation], axis = 1)

    # pred_cam = tf_util.fully_connected(camnet, 12, activation_fn=None, scope='fc3')
    return pred_rotation_mat, pred_translation, pred_RT
