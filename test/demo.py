import argparse
from datetime import datetime
import numpy as np
import random
import tensorflow as tf
import socket
import os
import cv2
import sys
import h5py
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(os.path.join(os.path.dirname(BASE_DIR), 'data'))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(os.path.dirname(BASE_DIR), 'data'))
print(os.path.join(ROOT_DIR, 'data'))
import model_sdf_2d_proj_twostream as model
import output_utils

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 0]')
parser.add_argument('--res', type=int, default='64', help='RESOLUTION')
parser.add_argument('--category', default=None, help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='checkpoint/main/DISN_all_w', help='Log dir [default: log]')
parser.add_argument('--restore_dir', default='/home/laughtervv/mnt/laughtergg/dev/ProgressivePointSetGeneration/shapenet/sdf/checkpoint/all_best/w', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--valid_lst', default='/media/ssd/projects/Deformation/Sources/DF/shapenet/filelists/single_obj.lst', help='test mesh data list')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

cat = "03001627"

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
RESOLUTION = FLAGS.res
if RESOLUTION > 100:
    NUM_POINTS = RESOLUTION * RESOLUTION# * RESOLUTION
else:
    NUM_POINTS = RESOLUTION * RESOLUTION * RESOLUTION
GPU_INDEX = FLAGS.gpu
PRETRAINED_MODEL_PATH = FLAGS.restore_dir
VALID_LST = FLAGS.valid_lst
LOG_DIR = FLAGS.log_dir


os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

RESULT_PATH = os.path.join(LOG_DIR, 'test_results_allpts', os.path.basename(VALID_LST)[:-4])
if not os.path.exists(RESULT_PATH): os.makedirs(RESULT_PATH)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

IMG_SIZE = 137

HOSTNAME = socket.gethostname()

VALID_DATASET = []
# with open(VALID_LST, 'r') as f:
#     lines = f.read().splitlines()
#     for line in lines:
#         line = line.strip().split(' ')
#         VALID_DATASET += [(line[0], line[1], line[2])]

CAT_LIST = ["02691156","02828884","02933112","02958343","03001627","03211117","03636649","03691459","04090263","04256520","04379243","04401088","04530566"]
for cat in CAT_LIST:
    VALID_LST = '/media/ssd/projects/Deformation/Sources/DF/shapenet/filelists/%s_test.lst' % cat
    with open(VALID_LST, 'r') as f:
        lines = f.read().splitlines()
        random.shuffle(lines)
        for line in lines[:10]:
            # for i in range(24):
            VALID_DATASET += [(cat, line.strip(), '%02d' % random.randint(0,23))]

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def train():
    log_string(LOG_DIR)
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            input_pls = model.placeholder_inputs(BATCH_SIZE, 2048, (IMG_SIZE, IMG_SIZE), num_sample_pc=NUM_POINTS, scope='inputs_pl')
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.

            print("--- Get model and loss")
            # Get model and loss

            end_points = model.get_model(input_pls, NUM_POINTS, is_training_pl, bn=False)
            # loss, end_points = model.get_loss(end_points)
            # tf.summary.scalar('loss', loss)

            # Create a session
            config = tf.ConfigProto()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
            config=tf.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init)

            ######### Loading Checkpoint ###############

            saver = tf.train.Saver([v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if('lr' not in v.name) and ('batch' not in v.name)])  
            ckptstate = tf.train.get_checkpoint_state(PRETRAINED_MODEL_PATH)

            if ckptstate is not None:
                LOAD_MODEL_FILE = os.path.join(PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
                try:
                    saver.restore(sess, LOAD_MODEL_FILE)
                    print( "Model loaded in file: %s" % LOAD_MODEL_FILE)    
                except:
                    print( "Fail to load overall modelfile: %s" % PRETRAINED_MODEL_PATH)
                    return

            ###########################################

            ops = {'input_pls': input_pls,
                   'is_training_pl': is_training_pl,
                   # 'loss': loss,
                   'end_points': end_points}

            test_one_epoch(sess, ops, saver)

def test_one_epoch(sess, ops, saver):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    # Shuffle train samples
    num_batches = int(len(VALID_DATASET) / BATCH_SIZE)

    # random.shuffle(VALID_DATASET)

    print('num_batches', num_batches)

    log_string(str(datetime.now()))

# info = {'rendered_dir': '/media/ssd/projects/Deformation/ShapeNet/ShapeNetRendering',
#         'sdf_dir': '/media/ssd/projects/Deformation/ShapeNet/SDF'}
    ref_pc = np.zeros([BATCH_SIZE, 2048, 3]).astype(np.float32)
    ref_samplepc = np.zeros([BATCH_SIZE, NUM_POINTS*8, 3]).astype(np.float32)
    ref_sdf = np.zeros([BATCH_SIZE, NUM_POINTS*8, 1]).astype(np.float32)
    imgs = np.zeros([BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]).astype(np.float32)

    x_ = np.linspace(-1., 1., RESOLUTION)
    y_ = np.linspace(-1., 1., RESOLUTION)
    z_ = np.linspace(-1., 1., RESOLUTION)
    z, y, x = np.meshgrid(x_, y_, z_, indexing='ij')
    x = np.expand_dims(x, 3)
    y = np.expand_dims(y, 3)
    z = np.expand_dims(z, 3)
    all_pts = np.concatenate((x, y, z), axis=3)
    all_pts = np.reshape(all_pts, (-1, 3))
    pred_sdf_val = np.zeros([BATCH_SIZE, all_pts.shape[0], 1])
    all_pts = np.expand_dims(all_pts.astype(np.float32), 0)
    print('all_pts', all_pts.shape)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        ib = 0

        h5_folder = '/media/ssd/projects/Deformation/ShapeNet/SDF_v1/' + VALID_DATASET[start_idx + ib][0]
        img_folder = '/media/ssd/projects/Deformation/ShapeNet/ShapeNetRenderingh5_v1/' + VALID_DATASET[start_idx + ib][0]
        bin_fn = os.path.join(os.path.join(RESULT_PATH, '%s_%s.bin' % (VALID_DATASET[start_idx + ib][1], VALID_DATASET[start_idx + ib][2])))

        # if start_idx == 57 or start_idx == 309:
        #     continue
        # if os.path.exists(bin_fn):
        #     continue
        print('%d/%d: %s' % (start_idx, num_batches, VALID_DATASET[start_idx + ib][1]))

        # src_batch_data, ref_batch_data = TRAIN_DATASET.get_batch(start_idx)
        ##load data

        # for ib in range(BATCH_SIZE):
            # ib = random.randint(0, BATCH_SIZE)

        # h5_fn = os.path.join(h5_folder, VALID_DATASET[start_idx + ib][0]+'/ori_sample.h5')
        # h5_f = h5py.File(h5_fn) 
        # pcsample_sdf = h5_f['pc_sdf_sample'][:]
        # choice = np.random.randint(pcsample_sdf.shape[0], size=NUM_POINTS*8)
        # pcsample_sdf = pcsample_sdf[choice, :]
        # ref_samplepc[ib,:,:] = pcsample_sdf[:,:3]
        # ref_sdf[ib,:,0] = pcsample_sdf[:,3]

        # pc_sdf = h5_f['pc_sdf_original'][:]
        # choice = np.random.randint(pc_sdf.shape[0], size=2048)
        # pc_sdf = pc_sdf[choice, :3]
        # ref_pc[ib,:,:] = pc_sdf#[:,:3]
        img_fn = os.path.join(img_folder, VALID_DATASET[start_idx + ib][1], 'rendering', VALID_DATASET[start_idx + ib][2]+'.png')
        # imgs[ib,:,:,:] = cv2.imread(img_fn, cv2.IMREAD_UNCHANGED)[:,:,:3].astype(np.float32) / 255.
        imgh5_f = h5py.File(os.path.join(img_folder, VALID_DATASET[start_idx + ib][1], "%s.h5"%VALID_DATASET[start_idx + ib][2]))
        img_raw = imgh5_f["img_arr"][:]
        img_arr = img_raw[:,:,:3]
        img_arr[img_raw[:,:,3] == 0] = [255, 255, 255]
        imgs[ib,:,:,:] = img_arr.astype(np.float32) / 255.
        trans_mat = np.expand_dims(imgh5_f["trans_mat"][:].astype(np.float32), 0)

        tic = time.time()
        if RESOLUTION > 100:
            for icur in range(RESOLUTION):
                cur_pts = all_pts[:, icur*RESOLUTION*RESOLUTION:(icur+1)*RESOLUTION*RESOLUTION, :]

                feed_dict = {ops['is_training_pl']: is_training,
                             ops['input_pls']['trans_mat']: trans_mat,
                             ops['input_pls']['sample_pc']: cur_pts,
                             ops['input_pls']['sample_pc_rot']: cur_pts,
                             ops['input_pls']['imgs']: imgs}

                cur_pred_sdf_val, cur_ref_img_val = sess.run([ops['end_points']['pred_sdf'], ops['end_points']['ref_img']], feed_dict=feed_dict)
                pred_sdf_val[:, icur*RESOLUTION*RESOLUTION:(icur+1)*RESOLUTION*RESOLUTION, :] = cur_pred_sdf_val / 10.
        else:

            feed_dict = {ops['is_training_pl']: is_training,
                         ops['input_pls']['trans_mat']: trans_mat,
                         ops['input_pls']['sample_pc']: all_pts,
                         ops['input_pls']['sample_pc_rot']: all_pts,
                         ops['input_pls']['imgs']: imgs}

            cur_pred_sdf_val, cur_ref_img_val = sess.run([ops['end_points']['pred_sdf'], ops['end_points']['ref_img']], feed_dict=feed_dict)
            pred_sdf_val[:, :, :] = cur_pred_sdf_val / 10. + 0.003
        print("time", time.time() - tic)
        bid = 0

        # np.savetxt(os.path.join(RESULT_PATH, '%d_sdf_pred.txt' % batch_idx), np.concatenate((ref_samplepc[bid,:,:], np.expand_dims(pred_sdf_val[bid,:,0],1)), axis=1))
        # vmin = -0.1
        # vmax = 0.1
        # pred_sdf_display = pred_sdf_val[bid,:,0]
        # pred_sdf_display = (pred_sdf_display - vmin) / (vmax - vmin)            

        # output_utils.output_scale_point_cloud(ref_samplepc[bid,:,:], pred_sdf_display, os.path.join(RESULT_PATH, '%s_pred.obj' % VALID_DATASET[start_idx + ib][0]))
        # output_utils.output_scale_point_cloud(ref_samplepc[bid,:,:], ref_sdf_display, os.path.join(RESULT_PATH, '%s_gt.obj' % VALID_DATASET[start_idx + ib][0]))
        cv2.imwrite(os.path.join(RESULT_PATH, '%s_%s_ref_img_resized.png' % (VALID_DATASET[start_idx + ib][1], VALID_DATASET[start_idx + ib][2])), (cur_ref_img_val[bid,:,:,:] * 255).astype(np.uint8))
        # print(pred_sdf_val[0,...].shape, pred_sdf_val[0,...].dtype)
        output_utils.save_sdf_bin(bin_fn, pred_sdf_val[0,:,0], RESOLUTION)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
