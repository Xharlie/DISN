# DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction
Please cite our NeurIPS 2019 paper[https://arxiv.org/abs/1905.10711]
``` 
@inProceedings{xu2019disn,
  title={DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction},
  author={Xu, Qiangeng and Wang, Weiyue and Ceylan, Duygu and Mech, Radomir and Neumann, Ulrich},
  booktitle={NeurIPS},
  year={2019}
}
``` 

Also our data preparation used this paper 'Vega: non-linear fem deformable object simulator'[http://run.usc.edu/vega/SinSchroederBarbic2012.pdf] 
Please also cite it if you use our code to generate sdf files
``` 
@inproceedings{sin2013vega,
  title={Vega: non-linear FEM deformable object simulator},
  author={Sin, Fun Shing and Schroeder, Daniel and Barbi{\v{c}}, Jernej},
  booktitle={Computer Graphics Forum},
  volume={32},
  number={1},
  pages={36--48},
  year={2013},
  organization={Wiley Online Library}
}
``` 
## Data Preparation

* ### file location setup:
  * #### under preprocessing/info.json, you can change the locations of your data: An example can be: 
   ```  
        "raw_dirs_v1": {
        "mesh_dir": "/ssd1/datasets/ShapeNet/ShapeNetCore.v1/",
        "norm_mesh_dir": "/ssd1/datasets/ShapeNet/march_cube_objs_v1/",
        "rendered_dir": "/ssd1/datasets/ShapeNet/ShapeNetRendering/",
        "renderedh5_dir": "/ssd1/datasets/ShapeNet/ShapeNetRenderingh5_v1/",
        "sdf_dir": "/ssd1/datasets/ShapeNet/SDF_v1/"
        }
   ```
  
* ### Download ShapeNetCore.v1 
  download the dataset following the instruction of https://www.shapenet.org/account/  (about 30GB)
  
  ```
  cd {your download dir}
  wget http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1.zip 
  unzip ShapeNetCore.v1.zip -d {your mesh_dir}
  ```
  
* ### Generate sdf files and the reconstructed models from the sdf file 
  ```
  mkdir log
  pip install trimesh==2.37.20
  cd {DISN}
  source isosurface/LIB_PATH
  nohup python -u preprocessing/create_point_sdf_grid.py --thread_num {recommend 9} --category {default 'all', but can be single category like 'chair'} &> log/create_sdf.log &
  
  ## SDF folder takes about 9.0G, marching cube obj folder takes about 245G
  
  ```
* ### Download and generate 2d image h5 files:
  * #### download 2d image following 3DR2N2[https://github.com/chrischoy/3D-R2N2], please cite their paper if you use this image tar file:
  
  ```
  wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
  untar it to {your rendered_dir}
  ```
  * #### run h5 file generation:
  
  ```
  cd {DISN}
  nohup python -u preprocessing/create_img_h5.py &> log/create_imgh5.log &
  ```

## Training
* ### train the sdf generation with provided camera parameters:

  if train from scratch, you can load official pretrained vgg_16 by setting --restore_modelcnn; or you can  --restore_model to your checkpoint to continue the training):

  we also support flip the background color from black to white since most online images have white background(by using --backcolorwhite)
  ```
  nohup python -u train/train_sdf.py --gpu 0 --img_feat_twostream --restore_modelcnn ./models/CNN/pretrained_model/vgg_16.ckpt --log_dir checkpoint/main/DISN --category all --num_sample_points 2048 --batch_size 20 --learning_rate 0.0001 --cat_limit 36000 &> log/DISN_train_all.log &
  ```
* ### train the sdf generation with provided camera parameters:



  
* ### Segmentation 
  * #### ScanNet
  Please refer to pointnet++ for downloading ScanNet use link: 
  ```
  cd segmentation
 
  nohup python -u train/train_ggcn_scannet.py &> log & 
  ```
