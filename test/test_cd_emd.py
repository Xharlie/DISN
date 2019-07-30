import argparse
import numpy as np
import random
import tensorflow as tf
import socket
import pymesh
import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
from tensorflow.contrib.framework.python.framework import checkpoint_utils

import models.tf_ops.approxmatch.tf_approxmatch as tf_approxmatch
import models.tf_ops.nn_distance.tf_nndistance as tf_nndistance
import create_file_lst
slim = tf.contrib.slim

parser = argparse.ArgumentParser()

parser.add_argument('--store', action='store_true')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 201]')
parser.add_argument('--img_h', type=int, default=137, help='Image Height')
parser.add_argument('--img_w', type=int, default=137, help='Image Width')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg global embedding dimensions')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument('--mask_tp', type=str, default="neg_two_sides")
parser.add_argument('--mask_rt', type=int, default=40000)
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--rot', action='store_true')
parser.add_argument('--tanh', action='store_true')
parser.add_argument('--cat_limit', type=int, default=168000, help="balance each category, 1500 * 24 = 36000")
parser.add_argument('--sdf_res', type=int, default=64, help='sdf grid')
parser.add_argument('--binary', action='store_true')

parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 32]')
parser.add_argument('--log_dir', default='checkpoint/exp_200', help='Log dir [default: log]')
parser.add_argument('--test_lst_dir', default='/ssd1/datasets/ShapeNet/filelists', help='test mesh data list')
parser.add_argument('--num_sample_points', type=int, default=2048, help='Sample Point Number for each obj to test[default: 2048]')
parser.add_argument('--threedcnn', action='store_true')
parser.add_argument('--img_feat_onestream', action='store_true')
parser.add_argument('--img_feat_far', action='store_true')
parser.add_argument('--img_feat_twostream', action='store_true')
parser.add_argument('--category', default="all", help='Which single class to train on [default: None]')
parser.add_argument('--view_num', type=int, default=24, help="how many views do you want to create for each obj")
parser.add_argument('--cam_est', action='store_true')
parser.add_argument('--cal_dir', type=str, default="", help="target obj directory that needs to be tested")

FLAGS = parser.parse_args()
print('pid: %s'%(str(os.getpid())))
print(FLAGS)

EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu
PRETRAINED_MODEL_PATH = FLAGS.log_dir
LOG_DIR = FLAGS.log_dir
SDF_WEIGHT = 10.

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

RESULT_PATH = os.path.join(LOG_DIR, 'test_results_allpts')
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

IMG_SIZE = FLAGS.img_h
VV =False
HOSTNAME = socket.gethostname()

TEST_LISTINFO = []
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

if VV:
    if FLAGS.threedcnn:
        info = {'rendered_dir': '/media/ssd/projects/Deformation/ShapeNet/ShapeNetRenderingh5',
                'sdf_dir': '/media/ssd/projects/Deformation/ShapeNet/SDF_full'}
    elif FLAGS.img_feat_onestream or FLAGS.img_feat_far or FLAGS.img_feat_twostream:
        info = {'rendered_dir': '/media/ssd/projects/Deformation/ShapeNet/ShapeNetRenderingh5_v1',
                'sdf_dir': '/media/ssd/projects/Deformation/ShapeNet/SDF_v1'}
    else:
        info = {'rendered_dir': '/media/ssd/projects/Deformation/ShapeNet/ShapeNetRenderingh5',
                'sdf_dir': '/media/ssd/projects/Deformation/ShapeNet/SDF_simp'}
else:
    if FLAGS.threedcnn:
        info = {'rendered_dir': '/ssd1/datasets/ShapeNet/ShapeNetRenderingh5_v2',
                'sdf_dir': '/ssd1/datasets/ShapeNet/SDF_full/64_expr_1.2',
                'gt_marching_cube': "/hdd_extra1/datasets/ShapeNet/march_cube_objs/"}
    elif FLAGS.img_feat_onestream or FLAGS.img_feat_far or FLAGS.img_feat_twostream:
        info = {'rendered_dir': '/ssd1/datasets/ShapeNet/ShapeNetRenderingh5_v1',
                'sdf_dir': '/ssd1/datasets/ShapeNet/SDF_v1/256_expr_1.2_bw_0.1',
                'gt_marching_cube':"/ssd1/datasets/ShapeNet/march_cube_objs_v1"}
        if FLAGS.cam_est:
            info = {'rendered_dir': '/ssd1/datasets/ShapeNet/ShapeNetRenderingh5_v1_pred_3d',
                    'sdf_dir': '/ssd1/datasets/ShapeNet/SDF_v1/256_expr_1.2_bw_0.1',
                    'gt_marching_cube': "/ssd1/datasets/ShapeNet/march_cube_objs_v1/"}
    else:
        info = {'rendered_dir': '/ssd1/datasets/ShapeNet/ShapeNetRenderingh5_v2',
                'sdf_dir': '/ssd1/datasets/ShapeNet/SDF_neg/simp_256_expr_1.2_bw_0.1',
                'gt_marching_cube': "/hdd_extra1/datasets/ShapeNet/march_cube_objs"}
print(info)

def load_model(sess, LOAD_MODEL_FILE, prefixs, strict=False):

    vars_in_pretrained_model = dict(checkpoint_utils.list_variables(LOAD_MODEL_FILE))
    vars_in_defined_model = []

    for var in tf.trainable_variables():
        if isinstance(prefixs, list):
            for prefix in prefixs:
                if (var.op.name.startswith(prefix)) and (var.op.name in vars_in_pretrained_model.keys()) and ('logits' not in var.op.name):
                    if (list(var.shape) == vars_in_pretrained_model[var.op.name]):
                        vars_in_defined_model.append(var)
        else:
            if (var.op.name.startswith(prefixs)) and (var.op.name in vars_in_pretrained_model.keys()) and ('logits' not in var.op.name):
                if (list(var.shape) == vars_in_pretrained_model[var.op.name]):
                    vars_in_defined_model.append(var)
    saver = tf.train.Saver(vars_in_defined_model)
    saver.restore(sess, LOAD_MODEL_FILE)
    print( "Model loaded in file: %s" % (LOAD_MODEL_FILE))

    return True

def build_file_dict(dir):
    file_dict = {}
    for file in os.listdir(dir):
        full_path = os.path.join(dir, file)
        if os.path.isfile(full_path):
            obj_id = file.split("_")[1]
            if obj_id in file_dict.keys():
                file_dict[obj_id].append(full_path)
            else:
                file_dict[obj_id] = [full_path]
    return file_dict

class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

def cd_emd_all(cats, pred_dir, gt_dir, test_lst_dir):
    for cat_nm, cat_id in cats.items():
        pred_dir_cat = os.path.join(pred_dir, cat_id)
        gt_dir_cat = os.path.join(gt_dir, cat_id)
        test_lst_f = os.path.join(test_lst_dir, cat_id+"_test.lst")
        cd_emd_cat(cat_id, cat_nm, pred_dir_cat, gt_dir_cat, test_lst_f)
    print("done!")

def save_all_cat_gt_pnt(cats, gt_dir, test_lst_dir):
    for cat_nm, cat_id in cats.items():
        gt_dir_cat = os.path.join(gt_dir, cat_id)
        test_lst_f = os.path.join(test_lst_dir, cat_id+"_test.lst")
        sample_save_gt_pnt(cat_id, cat_nm, gt_dir_cat, test_lst_f)
    print("done!")

def sample_save_gt_pnt(cat_id, cat_nm, gt_dir_cat, test_lst_f):
    count = 0
    with open(test_lst_f, "r") as f:
        test_objs = f.readlines()
        count += 1
        for obj_id in test_objs:
            obj_id = obj_id.rstrip('\r\n')
            obj_path = os.path.join(gt_dir_cat, obj_id, "isosurf.obj")
            # pred_path_lst = pred_dict[obj_id]
            verts_batch = np.zeros((FLAGS.num_sample_points, 3), dtype=np.float32)
            mesh1 = pymesh.load_mesh(obj_path)
            if mesh1.vertices.shape[0] > 0:
                choice = np.random.randint(mesh1.vertices.shape[0], size=FLAGS.num_sample_points)
                verts_batch = mesh1.vertices[choice, ...]
            savefn = os.path.join(gt_dir_cat, obj_id, "pnt_{}.txt".format(FLAGS.num_sample_points))
            np.savetxt(savefn, verts_batch, delimiter=',')
            print("saved gt pnt of {} at {}".format(obj_id, savefn))


def save_all_cat_pred_pnt(cats, pred_dir, test_lst_dir):
    for cat_nm, cat_id in cats.items():
        pred_dir_cat = os.path.join(pred_dir, cat_id)
        test_lst_f = os.path.join(test_lst_dir, cat_id+"_test.lst")
        sample_save_pred_pnt(cat_id, cat_nm, pred_dir_cat, test_lst_f)
    print("done!")

def sample_save_pred_pnt(cat_id, cat_nm, pred_dir, test_lst_f):
    pred_dict = build_file_dict(pred_dir)
    with open(test_lst_f, "r") as f:
        test_objs = f.readlines()
        for obj_id in test_objs:
            obj_id = obj_id.rstrip('\r\n')
            pred_path_lst = pred_dict[obj_id]
            verts_batch = np.zeros((FLAGS.view_num, FLAGS.num_sample_points, 3), dtype=np.float32)
            for i in range(len(pred_path_lst)):
                pred_mesh_fl = pred_path_lst[i]
                mesh1 = pymesh.load_mesh(pred_mesh_fl)
                if mesh1.vertices.shape[0] > 0:
                    choice = np.random.randint(mesh1.vertices.shape[0], size=FLAGS.num_sample_points)
                    verts_batch[i, ...] = mesh1.vertices[choice, ...]
                savedir = os.path.join(os.path.dirname(pred_dir),"pnt_{}_{}".format(FLAGS.num_sample_points, cat_id))
                os.makedirs(savedir,exist_ok=True)
                view_id = pred_mesh_fl[-6:-4]
                savefn = os.path.join(savedir, "pnt_{}_{}.txt".format(obj_id, view_id))
                print(savefn)
                np.savetxt(savefn, verts_batch[i, ...], delimiter=',')
                print("saved gt pnt of {} at {}".format(obj_id, savefn))

def cal_f_score_all_cat(cats, pred_dir, gt_dir, test_lst_dir,threshold_lst, side_len):
    precision_lst = []
    recall_lst = []
    cnt_lst = []
    for cat_nm, cat_id in cats.items():
        pred_dir_cat = os.path.join(pred_dir, cat_id)
        gt_dir_cat = os.path.join(gt_dir, cat_id)
        test_lst_f = os.path.join(test_lst_dir, cat_id + "_test.lst")
        thresholds = np.asarray(threshold_lst, dtype=np.float32) * 0.01 * side_len
        precision_avg, recall_avg, cnt \
            = f_score_cat(cat_id, cat_nm, pred_dir_cat, gt_dir_cat, test_lst_f, thresholds)
        precision_lst.append(precision_avg)
        recall_lst.append(recall_avg)
        cnt_lst.append(cnt)
        print("{}, {}, precision_avg {}, recal_avg{}, count {}"
              .format(cat_nm, cat_id, precision_avg, recall_avg, cnt))
    print("done!")
    precision = np.asarray(precision_lst) # 13 * 5
    recall = np.asarray(recall_lst)
    pre_w_avg = np.average(precision, axis=0, weights=cnt_lst)
    rec_w_avg = np.average(recall, axis=0, weights=cnt_lst)
    f_score = 2 * (pre_w_avg * rec_w_avg) / (pre_w_avg + rec_w_avg)
    print("pre_w_avg {}, rec_w_avg {}, f_score {}".format(pre_w_avg, rec_w_avg, f_score))

def f_score_cat(cat_id, cat_nm, pred_dir, gt_dir, test_lst_f, thresholds):
    pred_dict = build_file_dict(pred_dir)
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            sampled_pc = tf.placeholder(tf.float32, shape=(FLAGS.batch_size + 1, FLAGS.num_sample_points, 3))
            #
            dists_forward_sqrt, dists_backward_sqrt \
                = get_points_distance(sampled_pc)
            count = 0
            precision_sum = 0
            recall_sum = 0
            with open(test_lst_f, "r") as f:
                test_objs = f.readlines()
                for obj_id in test_objs:
                    obj_id = obj_id.rstrip('\r\n')
                    pred_pnt_dir = os.path.join(os.path.dirname(pred_dir),
                        "pnt_{}_{}".format(FLAGS.num_sample_points, cat_id))
                    forfl = os.path.join(pred_pnt_dir, "for_dist_{}.txt".format(obj_id))
                    backfl = os.path.join(pred_pnt_dir, "bac_dist_{}.txt".format(obj_id))
                    if not os.path.exists(forfl):
                        gt_pnt_path = os.path.join(gt_dir, obj_id, "pnt_{}.txt".format(FLAGS.num_sample_points))
                        # npnts, 3
                        gt_pnts = np.loadtxt(gt_pnt_path,dtype=float, delimiter=',')
                        pred_path_lst = pred_dict[obj_id]
                        verts_batch = np.zeros((FLAGS.view_num + 1, FLAGS.num_sample_points, 3), dtype=np.float32)
                        verts_batch[0, ...] = gt_pnts
                        for i in range(len(pred_path_lst)):
                            pred_mesh_fl = pred_path_lst[i]
                            view_id = pred_mesh_fl[-6:-4]
                            pred_pnt_path = os.path.join(pred_pnt_dir, "pnt_{}_{}.txt".format(obj_id, view_id))
                            pred_pnts = np.loadtxt(pred_pnt_path,dtype=float, delimiter=',')
                            verts_batch[i + 1, ...] = pred_pnts
                        if FLAGS.batch_size == FLAGS.view_num:
                            feed_dict = {sampled_pc: verts_batch}
                            # view * npnt
                            dists_forward_sqrt_val, dists_backward_sqrt_val\
                                = sess.run([dists_forward_sqrt, dists_backward_sqrt], feed_dict=feed_dict)
                        else:
                            raise NotImplementedError
                        np.savetxt(forfl, dists_forward_sqrt_val)
                        np.savetxt(backfl, dists_backward_sqrt_val)
                    else:
                        dists_forward_sqrt_val = np.loadtxt(forfl)
                        dists_backward_sqrt_val = np.loadtxt(backfl)
                    dists_forward_sqrt_val = np.tile(dists_forward_sqrt_val, [thresholds.shape[0], 1])
                    dists_backward_sqrt_val = np.tile(dists_backward_sqrt_val, [thresholds.shape[0], 1])
                    pre_sum_val = np.sum(np.less(dists_forward_sqrt_val, thresholds), axis=1)
                    rec_sum_val = np.sum(np.less(dists_backward_sqrt_val, thresholds), axis=1)
                    precision = pre_sum_val / (dists_forward_sqrt_val.shape[1])
                    recall = rec_sum_val / (dists_backward_sqrt_val.shape[1])
                    print("cat_id {}, obj_id {}: pre_sum {}, rec_sum {}, precision {}, recall {}"
                          .format(cat_id, obj_id, pre_sum_val, rec_sum_val, precision, recall))
                    precision_sum += precision
                    recall_sum += recall
                    count += 1
    return precision_sum/count, recall_sum/count, count

def get_points_distance(sampled_pc):
    src_pc = tf.tile(tf.expand_dims(sampled_pc[0, :, :], axis=0), (FLAGS.batch_size, 1, 1))
    if sampled_pc.get_shape().as_list()[0] == 2:
        pred = tf.expand_dims(sampled_pc[1, :, :], axis=0)
    else:
        pred = sampled_pc[1:, :, :]
    print(src_pc.get_shape())
    print(pred.get_shape())
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pred, src_pc)
    dists_forward_sqrt = tf.math.sqrt(dists_forward)
    dists_backward_sqrt = tf.math.sqrt(dists_backward)
    dists_forward_sqrt = tf.reshape(dists_forward_sqrt, [-1])
    dists_backward_sqrt = tf.reshape(dists_backward_sqrt, [-1])
    return dists_forward_sqrt, dists_backward_sqrt

def cd_emd_cat(cat_id, cat_nm, pred_dir, gt_dir, test_lst_f):
    pred_dict = build_file_dict(pred_dir)
    sum_cf_loss = 0.
    sum_em_loss = 0.
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            sampled_pc = tf.placeholder(tf.float32, shape=(FLAGS.batch_size+1, FLAGS.num_sample_points, 3))
            avg_cf_loss, min_cf_loss, arg_min_cf, avg_em_loss, min_em_loss, arg_min_em = get_points_loss(sampled_pc)
            count = 0
            with open(test_lst_f, "r") as f:
                test_objs = f.readlines()
                count+=1
                for obj_id in test_objs:
                    obj_id = obj_id.rstrip('\r\n')
                    src_path = os.path.join(gt_dir, obj_id, "isosurf.obj")
                    pred_path_lst = pred_dict[obj_id]
                    verts_batch = np.zeros((FLAGS.view_num+1, FLAGS.num_sample_points, 3), dtype=np.float32)
                    mesh1 = pymesh.load_mesh(src_path)
                    if mesh1.vertices.shape[0] > 0:
                        choice = np.random.randint(mesh1.vertices.shape[0], size=FLAGS.num_sample_points)
                        verts_batch[0, ...] = mesh1.vertices[choice,...]
                    pred_path_lst = random.sample(pred_path_lst, FLAGS.view_num)
                    for i in range(len(pred_path_lst)):
                        pred_mesh_fl = pred_path_lst[i]
                        mesh1 = pymesh.load_mesh(pred_mesh_fl)
                        if mesh1.vertices.shape[0] > 0:
                            choice = np.random.randint(mesh1.vertices.shape[0], size=FLAGS.num_sample_points)
                            verts_batch[i+1, ...] = mesh1.vertices[choice, ...]
                    if FLAGS.batch_size == FLAGS.view_num:
                        feed_dict = {sampled_pc: verts_batch}
                        avg_cf_loss_val, min_cf_loss_val, arg_min_cf_val, avg_em_loss_val, min_em_loss_val, arg_min_em_val \
                            = sess.run([avg_cf_loss, min_cf_loss, arg_min_cf, avg_em_loss, min_em_loss, arg_min_em],
                                       feed_dict=feed_dict)
                    else:
                        sum_avg_cf_loss_val = 0.
                        min_cf_loss_val = 9999.
                        arg_min_cf_val = 0
                        sum_avg_em_loss_val = 0.
                        min_em_loss_val = 9999.
                        arg_min_em_val = 0
                        for b in range(FLAGS.view_num//FLAGS.batch_size):
                            verts_batch_b = np.stack([verts_batch[0, ...], verts_batch[b, ...]])
                            feed_dict = {sampled_pc: verts_batch_b}
                            avg_cf_loss_val, _, _, avg_em_loss_val, _, _ \
                                = sess.run([avg_cf_loss, min_cf_loss, arg_min_cf, avg_em_loss, min_em_loss, arg_min_em],
                                           feed_dict=feed_dict)
                            sum_avg_cf_loss_val +=avg_cf_loss_val
                            sum_avg_em_loss_val +=avg_em_loss_val
                            if min_cf_loss_val > avg_cf_loss_val:
                                min_cf_loss_val = avg_cf_loss_val
                                arg_min_cf_val = b
                            if min_em_loss_val > avg_em_loss_val:
                                min_em_loss_val = avg_em_loss_val
                                arg_min_em_val = b
                        avg_cf_loss_val = sum_avg_cf_loss_val / (FLAGS.view_num//FLAGS.batch_size)
                        avg_em_loss_val = sum_avg_em_loss_val / (FLAGS.view_num//FLAGS.batch_size)
                    sum_cf_loss += avg_cf_loss_val
                    sum_em_loss += avg_em_loss_val
                    print(str(count) +  " ",src_path, "avg cf:{}, min_cf:{}, arg_cf view:{}, avg emd:{}, min_emd:{}, arg_em view:{}".
                          format(str(avg_cf_loss_val), str(min_cf_loss_val), str(arg_min_cf_val),
                                 str(avg_em_loss_val), str(min_em_loss_val), str(arg_min_em_val)))
            print("cat_nm:{}, cat_id:{}, avg_cf:{}, avg_emd:{}".
                  format(cat_nm, cat_id, sum_cf_loss/len(test_objs), sum_em_loss/len(test_objs)))


def get_points_loss(sampled_pc):
    src_pc = tf.tile(tf.expand_dims(sampled_pc[0,:,:], axis=0), (FLAGS.batch_size, 1, 1))
    if sampled_pc.get_shape().as_list()[0] == 2:
        pred = tf.expand_dims(sampled_pc[1,:,:], axis=0)
    else:
        pred = sampled_pc[1:, :, :]
    print(src_pc.get_shape())
    print(pred.get_shape())

    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pred, src_pc)
    cf_loss_views = (tf.reduce_mean(dists_forward, axis=1) + tf.reduce_mean(dists_backward, axis=1)) * 1000
    print("cf_loss_views.get_shape()", cf_loss_views.get_shape())
    avg_cf_loss = tf.reduce_mean(cf_loss_views)
    min_cf_loss = tf.reduce_min(cf_loss_views)
    arg_min_cf = tf.argmin(cf_loss_views, axis=0)

    match = tf_approxmatch.approx_match(src_pc, pred)
    match_cost = tf_approxmatch.match_cost(src_pc, pred, match) * 0.01
    print("match_cost.get_shape()", match_cost.get_shape())

    avg_em_loss = tf.reduce_mean(match_cost)
    min_em_loss = tf.reduce_min(match_cost)
    arg_min_em = tf.argmin(match_cost)

    return avg_cf_loss, min_cf_loss, arg_min_cf, avg_em_loss, min_em_loss, arg_min_em


if __name__ == "__main__":
    cats_all = {
        "watercraft": "04530566",
        "rifle": "04090263",
        "display": "03211117",
        "lamp": "03636649",
        "speaker": "03691459",
        "chair": "03001627",
        "bench": "02828884",
        "cabinet": "02933112",
        "car": "02958343",
        "airplane": "02691156",
        "sofa": "04256520",
        "table": "04379243",
        "phone": "04401088"
    }
    if FLAGS.category == "all":
        cats=cats_all
    elif FLAGS.category == "clean":
        cats ={ "cabinet": "02933112",
                "display": "03211117",
                "lamp": "03636649",
                "speaker": "03691459",
                "rifle": "04090263",
                "watercraft": "04530566"
        }
    else:
        cats={FLAGS.category: cats_all[FLAGS.category]}

    # cd_emd_all(cats, FLAGS.cal_dir,
    #            info["gt_marching_cube"], FLAGS.test_lst_dir)

    # 1. test cd_emd for all categories / some of the categories:

    # lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info(version=1)s
    # cd_emd_all(cats,
    #            "checkpoint/all_best/sdf_2d_sdfproj_twostream_wd_2048_weight10_ftprev_inweight/test_objs/65_0.0",
    #            "/ssd1/datasets/ShapeNet/march_cube_objs_v1/", "/ssd1/datasets/ShapeNet/filelists/",
    #            num_points=FLAGS.num_points, maxnverts=1000000, maxntris=1000000, num_view=4)

# nohup python -u test_cd_emd.py --gpu 1 --batch_size 24 --img_feat_twostream  --num_points 2048 --category chair &> cd_emd_all_all_chair_woweight_2048.log &


    # lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info(version=2)
    # cd_emd_all(cats,
    #            "checkpoint/all/binary/test_objs/65_0.0_sep",
    #            "/hdd_extra1/datasets/ShapeNet/march_cube_objs/", "/ssd1/datasets/ShapeNet/filelists/",
    #            num_points=FLAGS.num_points, maxnverts=1000000, maxntris=1000000, num_view=24)

    # nohup python -u test_allpts.py --gpu 3 --batch_size 24 --binary  --test_lst_dir /ssd1/datasets/ShapeNet/filelists/ --num_points 2048 --category all &> cd_emd_all_all_binary.log &


    # 2.
    ## save all gt pnt
    # save_all_cat_gt_pnt(cats, info["gt_marching_cube"], FLAGS.test_lst_dir)
# nohup python -u test/test_cd_emd.py --gpu 1 --batch_size 1 --img_feat_twostream  --num_points 2048 --category all &> log/save_gtpnt_v1.log &
# nohup python -u test/test_cd_emd.py --gpu 1 --batch_size 1  --num_points 2048 --category all &> log/save_gtpnt_v2.log &

    # save all pred pnt
    # save_all_cat_pred_pnt(cats, FLAGS.cal_dir, FLAGS.test_lst_dir)

#
# nohup python -u test/test_cd_emd.py --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/IM-SVR/test_objs/65_0.0 &> log/save_IM-SVR_pnt.log &
# nohup python -u test/test_cd_emd.py --num_points 2048 --category clean --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/IM-SVR/test_objs/65_0.0_sep &> log/save_IM-SVR_sep_pnt.log &
#
# nohup python -u test/test_cd_emd.py --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/3dcnn/test_objs/65_0.0 &> log/save_3dcnn_pnt.log &
#
# nohup python -u test/test_cd_emd.py --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/65_0.0 &> log/save_DISN_pnt.log &
# nohup python -u test/test_cd_emd.py --num_points 2048 --category clean --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/65_0.0_sep &> log/save_DISN_sep_pnt.log &
#
# nohup python -u test/test_cd_emd.py --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/camest_65_0.0 &> log/save_DISN_est_pnt.log &
# nohup python -u test/test_cd_emd.py --num_points 2048 --category clean --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/camest_65_0.0_sep &> log/save_DISN_est_sep_pnt.log &

#
# nohup python -u test/test_cd_emd.py --gpu 0 --binary --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/loctwobin/test_objs/65_0.0 &> log/save_binary_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 0 --binary --cam_est --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/loctwobin/test_objs/camest_65_0.0 &> log/save_binarycamest_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 1  --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/noloco/test_objs/65_0.0 &> log/save_global_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --img_feat_onestream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/onestream/test_objs/65_0.0 &> log/save_onestream_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --cam_est --img_feat_onestream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/onestream/test_objs/camest_65_0.0 &> log/save_onestreamcamest_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/DISNChair/test_objs/65_0.0 &> log/save_DISN_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --cam_est --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/DISNChair/test_objs/camest_65_0.0 &> log/save_DISNcamest_chair_pnt_1.6.log &



    # calculate distance
    cal_f_score_all_cat(cats, FLAGS.cal_dir, info["gt_marching_cube"],
        FLAGS.test_lst_dir, [[0.5], [1], [2], [5], [10], [20]], 2.5)

# nohup python -u test/test_cd_emd.py --gpu 0 --threedcnn --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/3dcnn/test_objs/65_0.0 &> log/f_3dcnn_pnt.log &
# nohup python -u test/test_cd_emd.py --gpu 1 --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/IM-SVR/test_objs/65_0.0_comb &> log/f_IM-SVR_pnt.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --img_feat_twostream --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/65_0.0_comb &> log/f_DISN_pnt.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --cam_est --img_feat_twostream --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/camest_65_0.0_comb &> log/f_DISNcamest_pnt.log &


# nohup python -u test/test_cd_emd.py --gpu 0 --threedcnn --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/3dcnn/test_objs/65_0.0 &> log/f_3dcnn_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 1 --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/IM-SVR/test_objs/65_0.0_comb &> log/f_IM-SVR_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --img_feat_twostream --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/65_0.0_comb &> log/f_DISN_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --cam_est --img_feat_twostream --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/camest_65_0.0_comb &> log/f_DISNcamest_pnt_1.6.log &




# nohup python -u test/test_cd_emd.py --gpu 0 --binary --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/loctwobin/test_objs/65_0.0 &> log/f_binary_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --binary --cam_est --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/loctwobin/test_objs/camest_65_0.0 &> log/f_binarycamest_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 1  --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/noloco/test_objs/65_0.0 &> log/f_global_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --img_feat_onestream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/onestream/test_objs/65_0.0 &> log/f_onestream_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --cam_est --img_feat_onestream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/onestream/test_objs/camest_65_0.0 &> log/f_onestreamcamest_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/DISNChair/test_objs/65_0.0 &> log/f_DISN_chair_pnt_1.6.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --cam_est --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/DISNChair/test_objs/camest_65_0.0 &> log/f_DISNcamest_chair_pnt_1.6.log &



# 2.5
# #
# nohup python -u test/test_cd_emd.py --gpu 2 --threedcnn --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/3dcnn/test_objs/65_0.0 &> log/f_3dcnn_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/IM-SVR/test_objs/65_0.0_comb &> log/f_IM-SVR_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --img_feat_twostream --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/65_0.0_comb &> log/f_DISN_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --cam_est --img_feat_twostream --num_points 2048 --category all --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/main/DISN/test_objs/camest_65_0.0_comb &> log/f_DISNcamest_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --binary --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/loctwobin/test_objs/65_0.0 &> log/f_binary_chair_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --binary --cam_est --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/loctwobin/test_objs/camest_65_0.0 &> log/f_binarycamest_chair_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 2  --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/noloco/test_objs/65_0.0 &> log/f_global_chair_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --img_feat_onestream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/onestream/test_objs/65_0.0 &> log/f_onestream_chair_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --cam_est --img_feat_onestream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/onestream/test_objs/camest_65_0.0 &> log/f_onestreamcamest_chair_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 3 --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/DISNChair/test_objs/65_0.0 &> log/f_DISN_chair_pnt_2.5.log &
# nohup python -u test/test_cd_emd.py --gpu 2 --cam_est --img_feat_twostream --num_points 2048 --category chair --cal_dir /home/xharlie/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/ablation/DISNChair/test_objs/camest_65_0.0 &> log/f_DISNcamest_chair_pnt_2.5.log &
# #
