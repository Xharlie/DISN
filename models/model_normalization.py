import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import vgg
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'data'))
sys.path.append(os.path.join(BASE_DIR, 'models'))
print(os.path.join(BASE_DIR, 'models'))
import deformnet
import tf_util

def placeholder_inputs(batch_size, num_points, img_size, num_sample_pc = 256, scope='', FLAGS=None):

    with tf.variable_scope(scope) as sc:
        pc_pl = tf.placeholder(tf.float32, shape=(batch_size, num_points, 3))
        sample_pc_pl = tf.placeholder(tf.float32, shape=(batch_size, num_sample_pc, 3))
        sample_pc_rot_pl = tf.placeholder(tf.float32, shape=(batch_size, num_sample_pc, 3))
        if FLAGS.alpha:
            imgs_pl = tf.placeholder(tf.float32, shape=(batch_size, img_size[0], img_size[1], 4))
        else:
            imgs_pl = tf.placeholder(tf.float32, shape=(batch_size, img_size[0], img_size[1], 3))
        sdf_pl = tf.placeholder(tf.float32, shape=(batch_size, num_sample_pc, 1))
        sdf_params_pl = tf.placeholder(tf.float32, shape=(batch_size, 6))
        trans_mat_pl = tf.placeholder(tf.float32, shape=(batch_size, 4, 3))
    sdf = {}
    sdf['pc'] = pc_pl
    sdf['sample_pc'] = sample_pc_pl
    sdf['sample_pc_rot'] = sample_pc_rot_pl
    sdf['imgs'] = imgs_pl
    sdf['sdf'] = sdf_pl
    sdf['sdf_params'] = sdf_params_pl
    sdf['trans_mat'] = trans_mat_pl

    return sdf


def placeholder_features(batch_size, num_points, num_sample_pc = 256, scope='', FLAGS=None):
    with tf.variable_scope(scope) as sc:
        ref_feats_embedding_cnn_pl = tf.placeholder(tf.float32, shape=(batch_size, 1, 1, 1024))
        point_img_feat_pl = tf.placeholder(tf.float32, shape=(batch_size, num_sample_pc, 1, 1472))
    feat = {}
    feat['ref_feats_embedding_cnn'] = ref_feats_embedding_cnn_pl
    feat['point_img_feat'] = point_img_feat_pl
    return feat

def get_model(ref_dict, num_point, is_training, bn=False, bn_decay=None, img_size = 224, wd=1e-5, FLAGS=None):

    ref_img = ref_dict['imgs']
    ref_pc = ref_dict['pc']
    ref_sample_pc = ref_dict['sample_pc']
    ref_sample_pc_rot = ref_dict['sample_pc_rot']
    ref_sdf = ref_dict['sdf']
    # ref_sdf_params = ref_dict['sdf_params']
    ref_trans_mat = ref_dict['trans_mat']

    batch_size = ref_img.get_shape()[0].value

    # endpoints
    end_points = {}
    end_points['ref_pc'] = ref_pc
    end_points['ref_sdf'] = ref_sdf #* 10
    end_points['ref_img'] = ref_img # B*H*W*3|4

    # Image extract features
    if ref_img.shape[1] != img_size or ref_img.shape[2] != img_size:
        if FLAGS.alpha:
            ref_img_rgb = tf.image.resize_bilinear(ref_img[:,:,:,:3], [img_size, img_size])
            # ref_img_alpha = tf.cast(tf.greater(ref_img[:, :, :, 3], tf.constant(0.0, dtype=tf.float32)),dtype=tf.float32)
            # ref_img_alpha = tf.image.resize_nearest_neighbor(
            #     tf.expand_dims(ref_img_alpha, axis=-1), [img_size, img_size])
            ref_img_alpha = tf.image.resize_nearest_neighbor(
                tf.expand_dims(ref_img[:,:,:,3], axis=-1), [img_size, img_size])
            ref_img = tf.concat([ref_img_rgb, ref_img_alpha], axis = -1)
        else:
            ref_img = tf.image.resize_bilinear(ref_img, [img_size, img_size])

    vgg.vgg_16.default_image_size = img_size
    with slim.arg_scope([slim.conv2d],
                         weights_regularizer=slim.l2_regularizer(wd)):
        ref_feats_embedding, vgg_end_points = vgg.vgg_16(ref_img, num_classes=FLAGS.num_classes, is_training=False, scope='vgg_16', spatial_squeeze=False)
        ref_feats_embedding_cnn = tf.squeeze(ref_feats_embedding, axis = [1,2]) 
    end_points['img_embedding'] = ref_feats_embedding_cnn
    point_img_feat=None
    pred_sdf=None
    if FLAGS.binary:
        if FLAGS.threedcnn:
            sample_img_points = tf.zeros((batch_size, FLAGS.num_sample_points, 2), dtype=tf.float32)
            if not FLAGS.multi_view:
                with tf.variable_scope("sdf3dcnn") as scope:
                    pred_sdf = deformnet.get_sdf_3dcnn_binary(None, ref_feats_embedding_cnn,
                                             is_training, batch_size, num_point, bn, bn_decay,wd=wd,FLAGS=FLAGS)
        elif FLAGS.img_feat_far:
            sample_img_points = get_img_points(ref_sample_pc, ref_trans_mat)  # B * N * 2
            vgg_conv3 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv3/conv3_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv3 = tf.contrib.resampler.resampler(vgg_conv3, sample_img_points)
            print('point_vgg_conv3', point_vgg_conv3.shape)
            vgg_conv4 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv4/conv4_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv4 = tf.contrib.resampler.resampler(vgg_conv4, sample_img_points)
            print('point_vgg_conv4', point_vgg_conv4.shape)
            vgg_conv5 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv5/conv5_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv5 = tf.contrib.resampler.resampler(vgg_conv5, sample_img_points)
            print('point_vgg_conv5', point_vgg_conv5.shape)

            point_img_feat = tf.concat(axis=2,
                                       values=[point_vgg_conv3, point_vgg_conv4,
                                               point_vgg_conv5])
            point_img_feat = tf.expand_dims(point_img_feat, axis=2)
            print('point_img_feat', point_img_feat.shape)
            if not FLAGS.multi_view:
                # Predict SDF
                with tf.variable_scope("sdfprediction") as scope:
                    pred_sdf_value_global = deformnet.get_sdf_basic2_binary(ref_sample_pc_rot, ref_feats_embedding_cnn,
                                                                            is_training, batch_size, num_point, bn,
                                                                            bn_decay,
                                                                            wd=wd)

                with tf.variable_scope("sdfprediction_imgfeat") as scope:
                    pred_sdf_value_local = deformnet.get_sdf_basic2_imgfeat_twostream_binary(ref_sample_pc_rot,
                                                                                             point_img_feat,
                                                                                             is_training, batch_size,
                                                                                             num_point,
                                                                                             bn, bn_decay, wd=wd)
        elif FLAGS.img_feat:
            sample_img_points = get_img_points(ref_sample_pc, ref_trans_mat)  # B * N * 2
            if not FLAGS.multi_view:
                with tf.variable_scope("sdfimgfeat") as scope:
                    vgg_conv1 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv1/conv1_2'], (FLAGS.img_h, FLAGS.img_w))
                    point_vgg_conv1 = tf.contrib.resampler.resampler(vgg_conv1, sample_img_points)
                    vgg_conv2 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv2/conv2_2'], (FLAGS.img_h, FLAGS.img_w))
                    point_vgg_conv2 = tf.contrib.resampler.resampler(vgg_conv2, sample_img_points)
                    vgg_conv3 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv3/conv3_3'], (FLAGS.img_h, FLAGS.img_w))
                    point_vgg_conv3 = tf.contrib.resampler.resampler(vgg_conv3, sample_img_points)
                    point_img_feat = tf.concat(axis=2, values=[point_vgg_conv1, point_vgg_conv2, point_vgg_conv3])
                    point_img_feat = tf.expand_dims(point_img_feat, axis=2)
                    pred_sdf = deformnet.get_sdf_basic2_imgfeat_binary(ref_sample_pc_rot, ref_feats_embedding_cnn,
                                                                      point_img_feat, is_training, batch_size, num_point,
                                                                      bn, bn_decay, wd=wd)
        elif FLAGS.img_feat_twostream:
            sample_img_points = get_img_points(ref_sample_pc, ref_trans_mat)  # B * N * 2
            vgg_conv1 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv1/conv1_2'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv1 = tf.contrib.resampler.resampler(vgg_conv1, sample_img_points)
            print('point_vgg_conv1', point_vgg_conv1.shape)
            vgg_conv2 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv2/conv2_2'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv2 = tf.contrib.resampler.resampler(vgg_conv2, sample_img_points)
            print('point_vgg_conv2', point_vgg_conv2.shape)
            vgg_conv3 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv3/conv3_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv3 = tf.contrib.resampler.resampler(vgg_conv3, sample_img_points)
            print('point_vgg_conv3', point_vgg_conv3.shape)
            vgg_conv4 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv4/conv4_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv4 = tf.contrib.resampler.resampler(vgg_conv4, sample_img_points)
            print('point_vgg_conv4', point_vgg_conv4.shape)
            vgg_conv5 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv5/conv5_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv5 = tf.contrib.resampler.resampler(vgg_conv5, sample_img_points)
            print('point_vgg_conv5', point_vgg_conv5.shape)

            point_img_feat = tf.concat(axis=2,
               values=[point_vgg_conv1, point_vgg_conv2, point_vgg_conv3, point_vgg_conv4,
                       point_vgg_conv5])
            point_img_feat = tf.expand_dims(point_img_feat, axis=2)
            print('point_img_feat', point_img_feat.shape)
            if not FLAGS.multi_view:
                # Predict SDF
                with tf.variable_scope("sdfprediction") as scope:
                    pred_sdf_value_global = deformnet.get_sdf_basic2_binary(ref_sample_pc_rot, ref_feats_embedding_cnn,
                                                                     is_training, batch_size, num_point, bn, bn_decay,
                                                                     wd=wd)

                with tf.variable_scope("sdfprediction_imgfeat") as scope:
                    pred_sdf_value_local = deformnet.get_sdf_basic2_imgfeat_twostream_binary(ref_sample_pc_rot, point_img_feat,
                                                                                      is_training, batch_size, num_point,
                                                                                      bn, bn_decay, wd=wd)
            # # batch_size, -1, 2
            # pred_sdf = tf.reshape(tf.concat([pred_sdf_value_global, pred_sdf_value_local], axis=-1), (-1,4))
            # pred_sdf = tf_util.fully_connected(pred_sdf, 4, scope='twostream/fc_1')
            # pred_sdf = tf_util.fully_connected(pred_sdf, 2, activation_fn=None, scope='twostream/fc_2')
            # pred_sdf = tf.reshape(pred_sdf, (FLAGS.batch_size, -1, 2))
                pred_sdf = pred_sdf_value_global + pred_sdf_value_local

        else:
            sample_img_points = tf.zeros((batch_size, FLAGS.num_sample_points, 2), dtype=tf.float32)
            if not FLAGS.multi_view:
                with tf.variable_scope("sdfprediction") as scope:
                    pred_sdf = deformnet.get_sdf_basic2_binary(ref_sample_pc, ref_feats_embedding_cnn,
                                is_training, batch_size, num_point, bn, bn_decay,wd=wd)
    else:
        if FLAGS.threedcnn:
            sample_img_points = tf.zeros((batch_size, FLAGS.num_sample_points, 2), dtype=tf.float32)
            if not FLAGS.multi_view:
                with tf.variable_scope("sdf3dcnn") as scope:
                    pred_sdf = deformnet.get_sdf_3dcnn(None, ref_feats_embedding_cnn,
                                             is_training, batch_size, num_point, bn, bn_decay,wd=wd,FLAGS=FLAGS)

        elif FLAGS.img_feat_far:
            sample_img_points = get_img_points(ref_sample_pc, ref_trans_mat)  # B * N * 2
            vgg_conv3 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv3/conv3_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv3 = tf.contrib.resampler.resampler(vgg_conv3, sample_img_points)
            print('point_vgg_conv3', point_vgg_conv3.shape)
            vgg_conv4 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv4/conv4_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv4 = tf.contrib.resampler.resampler(vgg_conv4, sample_img_points)
            print('point_vgg_conv4', point_vgg_conv4.shape)
            vgg_conv5 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv5/conv5_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv5 = tf.contrib.resampler.resampler(vgg_conv5, sample_img_points)
            print('point_vgg_conv5', point_vgg_conv5.shape)

            point_img_feat = tf.concat(axis=2,
                                       values=[point_vgg_conv3, point_vgg_conv4,
                                               point_vgg_conv5])
            point_img_feat = tf.expand_dims(point_img_feat, axis=2)
            print('point_img_feat', point_img_feat.shape)
            if not FLAGS.multi_view:
                # Predict SDF
                with tf.variable_scope("sdfprediction") as scope:
                    pred_sdf_value_global = deformnet.get_sdf_basic2(ref_sample_pc_rot, ref_feats_embedding_cnn,
                                                                            is_training, batch_size, num_point, bn,
                                                                            bn_decay,
                                                                            wd=wd)

                with tf.variable_scope("sdfprediction_imgfeat") as scope:
                    pred_sdf_value_local = deformnet.get_sdf_basic2_imgfeat_twostream(ref_sample_pc_rot,
                                                                                             point_img_feat,
                                                                                             is_training, batch_size,
                                                                                             num_point,
                                                                                             bn, bn_decay, wd=wd)
        elif FLAGS.img_feat:
            sample_img_points = get_img_points(ref_sample_pc, ref_trans_mat)  # B * N * 2
            if not FLAGS.multi_view:
                with tf.variable_scope("sdfimgfeat") as scope:
                    vgg_conv1 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv1/conv1_2'], (FLAGS.img_h, FLAGS.img_w))
                    point_vgg_conv1 = tf.contrib.resampler.resampler(vgg_conv1, sample_img_points)
                    vgg_conv2 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv2/conv2_2'], (FLAGS.img_h, FLAGS.img_w))
                    point_vgg_conv2 = tf.contrib.resampler.resampler(vgg_conv2, sample_img_points)
                    vgg_conv3 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv3/conv3_3'], (FLAGS.img_h, FLAGS.img_w))
                    point_vgg_conv3 = tf.contrib.resampler.resampler(vgg_conv3, sample_img_points)
                    point_img_feat = tf.concat(axis=2, values=[point_vgg_conv1, point_vgg_conv2, point_vgg_conv3])
                    print("point_img_feat.shape", point_img_feat.get_shape())
                    point_img_feat = tf.expand_dims(point_img_feat, axis=2)
                    pred_sdf = deformnet.get_sdf_basic2_imgfeat(ref_sample_pc_rot, ref_feats_embedding_cnn,
                                                                      point_img_feat, is_training, batch_size, num_point,
                                                                      bn, bn_decay, wd=wd)
        elif FLAGS.img_feat_twostream:
            sample_img_points = get_img_points(ref_sample_pc, ref_trans_mat)  # B * N * 2
            vgg_conv1 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv1/conv1_2'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv1 = tf.contrib.resampler.resampler(vgg_conv1, sample_img_points)
            print('point_vgg_conv1', point_vgg_conv1.shape)
            vgg_conv2 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv2/conv2_2'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv2 = tf.contrib.resampler.resampler(vgg_conv2, sample_img_points)
            print('point_vgg_conv2', point_vgg_conv2.shape)
            vgg_conv3 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv3/conv3_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv3 = tf.contrib.resampler.resampler(vgg_conv3, sample_img_points)
            print('point_vgg_conv3', point_vgg_conv3.shape)
            vgg_conv4 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv4/conv4_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv4 = tf.contrib.resampler.resampler(vgg_conv4, sample_img_points)
            print('point_vgg_conv4', point_vgg_conv4.shape)
            vgg_conv5 = tf.image.resize_bilinear(vgg_end_points['vgg_16/conv5/conv5_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv5 = tf.contrib.resampler.resampler(vgg_conv5, sample_img_points)
            print('point_vgg_conv5', point_vgg_conv5.shape)

            point_img_feat = tf.concat(axis=2,
                                       values=[point_vgg_conv1, point_vgg_conv2, point_vgg_conv3, point_vgg_conv4,
                                               point_vgg_conv5])
            point_img_feat = tf.expand_dims(point_img_feat, axis=2)
            print('point_img_feat', point_img_feat.shape)
            if not FLAGS.multi_view:
                # Predict SDF
                with tf.variable_scope("sdfprediction") as scope:
                    pred_sdf_value_global = deformnet.get_sdf_basic2(ref_sample_pc_rot, ref_feats_embedding_cnn,
                                                                     is_training, batch_size, num_point, bn, bn_decay,
                                                                     wd=wd)

                with tf.variable_scope("sdfprediction_imgfeat") as scope:
                    pred_sdf_value_local = deformnet.get_sdf_basic2_imgfeat_twostream(ref_sample_pc_rot, point_img_feat,
                                                                                      is_training, batch_size, num_point,
                                                                                      bn, bn_decay, wd=wd)

                pred_sdf = pred_sdf_value_global + pred_sdf_value_local
        else:
            sample_img_points = tf.zeros((batch_size, FLAGS.num_sample_points, 2), dtype=tf.float32)
            if not FLAGS.multi_view:
                with tf.variable_scope("sdfprediction") as scope:
                    pred_sdf = deformnet.get_sdf_basic2(ref_sample_pc, ref_feats_embedding_cnn,
                                is_training, batch_size, num_point, bn, bn_decay,wd=wd)
        if FLAGS.tanh:
            pred_sdf = tf.tanh(pred_sdf)
    end_points['pred_sdf'] = pred_sdf
    end_points["sample_img_points"] = sample_img_points
    end_points["ref_feats_embedding_cnn"] = ref_feats_embedding_cnn
    end_points["point_img_feat"] = point_img_feat
    # with tf.variable_scope("refpc_reconstruction") as scope:
    #     reconst_pc = pointnet.get_decoder(ref_feats_embedding_cnn, is_training,wd=wd)
    #     end_points['reconst_pc'] = reconst_pc

    return end_points

def get_decoder(num_point, input_pls, feature_pls, bn=False, bn_decay=None,wd=None):
    ref_feats_embedding_cnn = feature_pls["ref_feats_embedding_cnn"]
    point_img_feat = feature_pls["point_img_feat"]
    ref_sample_pc_rot = input_pls['sample_pc_rot']

    with tf.variable_scope("sdfprediction") as scope:
        pred_sdf_value_global = deformnet.get_sdf_basic2(ref_sample_pc_rot, ref_feats_embedding_cnn,
                                                         False, 1, num_point, bn, bn_decay,
                                                         wd=wd)

    with tf.variable_scope("sdfprediction_imgfeat") as scope:
        pred_sdf_value_local = deformnet.get_sdf_basic2_imgfeat_twostream(ref_sample_pc_rot, point_img_feat,
                                                                          False, 1, num_point,
                                                                          bn, bn_decay, wd=wd)
    multi_pred_sdf = pred_sdf_value_global + pred_sdf_value_local
    return multi_pred_sdf


def get_img_points(sample_pc, trans_mat_right):
    # sample_pc B*N*3
    size_lst = sample_pc.get_shape().as_list()
    homo_pc = tf.concat((sample_pc, tf.ones((size_lst[0], size_lst[1], 1),dtype=np.float32)),axis= -1)
    print("homo_pc.get_shape()", homo_pc.get_shape())
    pc_xyz = tf.matmul(homo_pc, trans_mat_right)
    print("pc_xyz.get_shape()", pc_xyz.get_shape()) # B * N * 3
    pc_xy = tf.divide(pc_xyz[:,:,:2], tf.expand_dims(pc_xyz[:,:,2], axis = 2))
    mintensor = tf.constant([0.0,0.0], dtype=tf.float32)
    maxtensor = tf.constant([136.0,136.0], dtype=tf.float32)
    return tf.minimum(maxtensor, tf.maximum(mintensor, pc_xy))


def get_subemb_3dcnn(src_pc, globalfeats, is_training, batch_size, bn, bn_decay, wd=None, FLAGS=None):
    # src_pc -> batch_size, num_sample_pc, 3
    globalfeats_expand = tf.reshape(globalfeats, [batch_size, 1, 1, 1, -1])
    print('globalfeats_expand', globalfeats_expand.get_shape())
    net2 = tf_util.conv3d_transpose(globalfeats_expand, 128, [2, 2, 2], stride=[2, 2, 2],
                                    bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, scope='sub3deconv1')  # 2

    emb = tf_util.conv3d_transpose(net2, 64, [3, 3, 3], stride=[2, 2, 2], bn_decay=bn_decay, bn=bn,
                                    is_training=is_training, weight_decay=wd, scope='sub3deconv2')  # 4
    x_first_indx, x_shift = get_axis_index(src_pc[:,:,0])
    y_first_indx, y_shift = get_axis_index(src_pc[:,:,1])
    z_first_indx, z_shift = get_axis_index(src_pc[:,:,2])
    print("x_first_indx.get_shape() ", x_first_indx.get_shape())
    print("z_shift.get_shape() ", z_shift.get_shape())
    pred = get_emb(x_first_indx, y_first_indx, z_first_indx, emb, batch_size, FLAGS)
    print("pred.get_shape() ", pred.get_shape())
    return pred, tf.concat((x_shift, y_shift, z_shift),axis=2)

def get_emb(x_first_indx, y_first_indx, z_first_indx, emb, batch_size, FLAGS):
    basic_grid_0 = tf.tile(tf.reshape(tf.range(0, batch_size),
         [batch_size, 1, 1]), [1, FLAGS.num_sample_points, 1])
    min_min_min= tf.concat((basic_grid_0, x_first_indx, y_first_indx, z_first_indx), axis=2)
    min_min_max= tf.concat((basic_grid_0, x_first_indx, y_first_indx, z_first_indx + 1), axis=2)
    min_max_min= tf.concat((basic_grid_0, x_first_indx, y_first_indx + 1, z_first_indx), axis=2)
    min_max_max= tf.concat((basic_grid_0, x_first_indx, y_first_indx + 1, z_first_indx+1), axis=2)
    max_min_min= tf.concat((basic_grid_0, x_first_indx + 1, y_first_indx, z_first_indx), axis=2)
    max_min_max= tf.concat((basic_grid_0, x_first_indx + 1, y_first_indx, z_first_indx + 1), axis=2)
    max_max_min= tf.concat((basic_grid_0, x_first_indx + 1, y_first_indx + 1, z_first_indx), axis=2)
    max_max_max= tf.concat((basic_grid_0, x_first_indx + 1, y_first_indx + 1, z_first_indx + 1), axis=2)
    min_min_min_emb = tf.gather_nd(emb, min_min_min)
    min_min_max_emb = tf.gather_nd(emb, min_min_max)
    min_max_min_emb = tf.gather_nd(emb, min_max_min)
    min_max_max_emb = tf.gather_nd(emb, min_max_max)
    max_min_min_emb = tf.gather_nd(emb, max_min_min)
    max_min_max_emb = tf.gather_nd(emb, max_min_max)
    max_max_min_emb = tf.gather_nd(emb, max_max_min)
    max_max_max_emb = tf.gather_nd(emb, max_max_max)
    return tf.concat((min_min_min_emb, min_min_max_emb, min_max_min_emb, min_max_max_emb,
                      max_min_min_emb, max_min_max_emb, max_max_min_emb, max_max_max_emb), axis=2)

def get_axis_index(src_axis):
    upper = tf.cast(tf.less(src_axis, tf.constant(-0.75, dtype=tf.float32)), dtype=tf.float32)
    bottom = tf.cast(tf.greater(src_axis, tf.constant(0.25, dtype=tf.float32)), dtype=tf.float32)
    mid = tf.subtract(tf.constant(1.0, dtype=tf.float32), upper+bottom)
    index = tf.cast(upper * 0 + mid + bottom * 2.0, dtype=tf.int32)
    center = -0.5 * upper + 0.5 * bottom
    return tf.expand_dims(index, axis = 2), tf.expand_dims(src_axis - center, axis = 2)


def get_loss(end_points, sdf_weight=10., regularization=True, mask_tp = "neg",
             mask_rt=40000, num_sample_points = 2048, FLAGS=None, batch_size=None):
    """ pred: BxNx3,
        label: BxNx3, """

    pred_sdf = end_points['pred_sdf']
    gt_sdf = end_points['ref_sdf']
    # reconst_pc = end_points['reconst_pc']
    # ref_pc = end_points['ref_pc']

    # N = src_pc.get_shape()[1].value
    # B = src_pc.get_shape()[0].value

    ################
    # Compute loss #
    ################
    end_points['losses'] = {}

    #############reconstruction
    # recon_cf_loss, end_points = losses.get_chamfer_loss(reconst_pc, ref_pc, end_points)
    # recon_cf_loss = 10000 * recon_cf_loss
    # end_points['losses']['recon_cf_loss'] = recon_cf_loss
    # recon_em_loss, end_points = losses.get_em_loss(reconst_pc, ref_pc, end_points)
    # end_points['losses']['recon_em_loss'] = recon_em_loss

    ###########sdf
    # sdf_loss = tf.nn.l2_loss(gt_sdf - pred_sdf)
    # sdf_loss = tf.reduce_mean(sdf_loss)
    print ('gt_sdf', gt_sdf.shape)
    print ('pred_sdf', pred_sdf.shape)
    if batch_size is None:
        batch_size = FLAGS.batch_size
    if FLAGS.binary:
        label_sdf = tf.reshape(tf.cast(tf.math.greater(gt_sdf, tf.constant(0.0)), dtype=tf.int32),
                               (batch_size, num_sample_points))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(label_sdf, tf.argmax(pred_sdf, axis=2, output_type=tf.int32)), dtype=tf.float32))
        end_points['losses']['accuracy'] = accuracy
        sdf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_sdf, logits=pred_sdf)
        sdf_loss = tf.reduce_mean(sdf_loss) * 100
        end_points['losses']['sdf_loss'] = sdf_loss
    else:
        ############### accuracy
        zero = tf.constant(0, dtype=tf.float32)
        sides_product = tf.multiply(pred_sdf, gt_sdf)
        # accuracy = tf.reduce_mean(tf.to_float(tf.less_equal(tf.constant(0.0), sides_product)))

        gt_sign = tf.greater(gt_sdf,zero)
        pred_sign = tf.greater(pred_sdf,zero)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(gt_sign, pred_sign), dtype=tf.float32))
        end_points['losses']['accuracy'] = accuracy

        if mask_tp == "neg":
            print("hand_craft weight_maskt")
            weight_mask = tf.to_float(tf.less_equal(gt_sdf, tf.constant(0.01))) * 4. \
                          + tf.to_float(tf.greater(gt_sdf, tf.constant(0.01)))
        elif mask_tp == "two_sides":
            weight_mask = 2 * tf.math.sigmoid(-1. * sides_product * mask_rt)
        elif mask_tp == "neg_two_sides":
            weight_mask = (tf.to_float(tf.less_equal(gt_sdf, tf.constant(0.01))) * 4.
                           + tf.to_float(tf.greater(gt_sdf, tf.constant(0.01)))) \
                          * 2 * tf.math.sigmoid(-1. * sides_product * mask_rt)
        elif mask_tp == "exp":
            weight_mask = tf.math.exp(tf.minimum(-1. * sides_product*mask_rt, 1.6))
        else:
            weight_mask = 1
        end_points['weighed_mask'] = weight_mask
        sdf_loss = tf.reduce_mean(tf.abs(gt_sdf * sdf_weight - pred_sdf) * weight_mask)
        end_points['losses']['sdf_loss_realvalue'] = tf.reduce_mean(tf.abs(gt_sdf - pred_sdf / sdf_weight))
        sdf_loss = sdf_loss * 1000
    end_points['losses']['sdf_loss'] = sdf_loss
    loss = sdf_loss
    ############### weight decay
    if regularization:
        vgg_regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        decoder_regularization_loss = tf.add_n(tf.get_collection('regularizer'))
        end_points['losses']['regularization'] = (vgg_regularization_loss + decoder_regularization_loss)
        loss += (vgg_regularization_loss + decoder_regularization_loss)
    end_points['losses']['overall_loss'] = loss
    return loss, end_points