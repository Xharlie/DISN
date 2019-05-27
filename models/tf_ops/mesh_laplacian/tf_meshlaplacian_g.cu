//Author: Weiyue Wang
//Reference: https://github.com/charlesq34/pointnet-autoencoder/blob/master/tf_ops/nn_distance/tf_nndistance_g.cu
//           https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <stdio.h>
#include <assert.h>


__device__ void getPoint(const float *vertices, int v_id, float *p){
    p[0] = vertices[3* v_id];
    p[1] = vertices[3* v_id+1];
    p[2] = vertices[3* v_id+2];
}

__device__ void getTriangle(const int *triangles, int t_id, int &v1, int &v2, int &v3){
    v1 = triangles[3 * t_id];
    v2 = triangles[3 * t_id + 1];
    v3 = triangles[3 * t_id + 2];
}

__host__ void getTriangle_cpu(const int *triangles, int t_id, int &v1, int &v2, int &v3){
    v1 = triangles[3 * t_id];
    v2 = triangles[3 * t_id + 1];
    v3 = triangles[3 * t_id + 2];
}

__host__ bool findnb_cpu(const int *nb, int v_id){
    bool flag = false;
    for (int i=0; i<20; i++)
        if (nb[i] == v_id){
            flag = true;
            break;
        }
    return flag;
}
__device__  bool findnb(const int *nb, int v_id){
    bool flag = false;
    for (int i=0; i<20; i++)
        if (nb[i] == v_id){
            flag = true;
            break;
        }
//    printf("%d\n",flag);
    return flag;
}

__device__ void setLaplacian(float *laplacian, int *count, int v_id1, int v_id2, float * p1, float * p2){

    atomicAdd(&laplacian[3 * v_id1], p2[0] - p1[0]);
    atomicAdd(&laplacian[3 * v_id1+1], p2[1] - p1[1]);
    atomicAdd(&laplacian[3 * v_id1+2], p2[2] - p1[2]);

    atomicAdd(&count[v_id1], 1);


    atomicAdd(&laplacian[3 * v_id2], p1[0] - p2[0]);
    atomicAdd(&laplacian[3 * v_id2+1], p1[1] - p2[1]);
    atomicAdd(&laplacian[3 * v_id2+2], p1[2] - p2[2]);

    atomicAdd(&count[v_id2], 1);
    // now releasing the lock
//    atomicExch( &nb[(i*maxnverts+v1)*20+count[i*maxnverts+v1]], 0 );
//    atomicExch( &nb[(i*maxnverts+v1)*20+count[i*maxnverts+v1]], 0 );
//    atomicExch( &nb[(i*maxnverts+v1)*20+count[i*maxnverts+v1]], 0 );
}


__global__ void InitMeshLaplacianKernel(const int b, const int maxnverts, float* laplacian, int* count){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int v_id=threadIdx.x+blockIdx.y*blockDim.x; v_id<maxnverts; v_id+=blockDim.x*gridDim.y){
            for (int i_c = 0; i_c < 3; i_c++){
                laplacian[i*maxnverts*3+v_id*3+i_c] = 0;
                count[i*maxnverts+v_id] = 0;
            }
        }
    }
}

__global__ void AvgMeshLaplacianKernel(const int b, const int maxnverts, float* laplacian, int* count){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int v_id=threadIdx.x+blockIdx.y*blockDim.x; v_id<maxnverts; v_id+=blockDim.x*gridDim.y){
            for (int i_c = 0; i_c < 3; i_c++)
                if (count[i*maxnverts+v_id]!=0)
                    laplacian[i*maxnverts*3+v_id*3+i_c] /= count[i*maxnverts+v_id];
        }
    }
}
__global__ void MeshLaplacianKernel(const int b, const int * nverts, const int maxnverts, const float * vertices, const int * ntriangles, const int maxntriangles, const int * triangles,
                                    float * laplacian, int* nb, int* count){

    for (int i=blockIdx.x;i<b;i+=gridDim.x){
//        int n_verts = nverts[i];
//
//        for (int v1=threadIdx.x+blockIdx.y*blockDim.x; v1 < n_verts; v1+=blockDim.x*gridDim.y){
//            for (int v2=0; v2<20; v2++){
//                if (nb[(i*maxnverts+v1)*20 + v2] == -1)
//                    break;
//                else{
//                    float p1[3], p2[3];
//                    getPoint(&vertices[i*maxnverts*3], v1, p1);
//                    getPoint(&vertices[i*maxnverts*3], nb[(i*maxnverts+v1)*20 + v2], p2);
//                    setLaplacian(&laplacian[i*maxnverts*3], &count[i*maxnverts], v1, nb[(i*maxnverts+v1)*20 + v2], p1, p2);
//                }
//            }
//        }
        int n_triangles = ntriangles[i];
        for (int triangle_id=threadIdx.x+blockIdx.y*blockDim.x; triangle_id < n_triangles; triangle_id+=blockDim.x*gridDim.y){

            int v1, v2, v3;
            float p1[3], p2[3], p3[3];
            getTriangle(&triangles[i*maxntriangles*3], triangle_id, v1, v2, v3);

            getPoint(&vertices[i*maxnverts*3], v1, p1);
            getPoint(&vertices[i*maxnverts*3], v2, p2);
            getPoint(&vertices[i*maxnverts*3], v3, p3);

            if (!findnb(&nb[(i*maxnverts+v1)*20], v2)){
                nb[(i*maxnverts+v1)*20+count[i*maxnverts+v1]] = v2;
                nb[(i*maxnverts+v2)*20+count[i*maxnverts+v2]] = v1;
                setLaplacian(&laplacian[i*maxnverts*3], &count[i*maxnverts], v1, v2, p1, p2);
            }
            if (!findnb(&nb[(i*maxnverts+v1)*20], v3)){
                nb[(i*maxnverts+v1)*20+count[i*maxnverts+v1]] = v3;
                nb[(i*maxnverts+v3)*20+count[i*maxnverts+v3]] = v1;
                setLaplacian(&laplacian[i*maxnverts*3], &count[i*maxnverts], v1, v3, p1, p3);
            }
            if (!findnb(&nb[(i*maxnverts+v2)*20], v3)){
                nb[(i*maxnverts+v2)*20+count[i*maxnverts+v2]] = v3;
                nb[(i*maxnverts+v3)*20+count[i*maxnverts+v3]] = v2;
//                printf("%d\n",nb[(i*maxnverts+v3)*20+count[i*maxnverts+v3]]);
                setLaplacian(&laplacian[i*maxnverts*3], &count[i*maxnverts], v2, v3, p2, p3);
            }
//            else
//            printf("%d\n",nb[(i*maxnverts+v3)*20+count[i*maxnverts+v3]]);

        }
        __syncthreads();
    }
}


void MeshLaplacianKernelLauncher( \
    /*inputs*/  const int b, const int * n_verts, const int maxn_verts, const float * vertices, const int * n_triangles, const int maxn_triangles, const int * triangles, \
    /*outputs*/ float * laplacian, int * count, int * nb){

    InitMeshLaplacianKernel<<<dim3(2,8,1),512>>>(b, maxn_verts, laplacian, count);
//    cudaMalloc((void**)&nb, b*maxn_verts*20*sizeof(int));
    cudaMemset(nb, -1, b*maxn_verts*20*sizeof(int));
    MeshLaplacianKernel<<<8,8>>>(b, n_verts, maxn_verts, vertices, n_triangles, maxn_triangles, triangles, laplacian, nb, count);
    AvgMeshLaplacianKernel<<<dim3(2,8,1),512>>>(b, maxn_verts, laplacian, count);


}

/****************** Gradient ******************/

__device__ void setLaplaciangrad (const float * grad_laplacian, float *grad_verts, int v_id1, int v_id2){

    atomicAdd(&grad_verts[3 * v_id1], - 1 * grad_laplacian[3 * v_id1]);
    atomicAdd(&grad_verts[3 * v_id1+1], - 1 * grad_laplacian[3 * v_id1+1]);
    atomicAdd(&grad_verts[3 * v_id1+2], - 1 * grad_laplacian[3 * v_id1+2]);

    atomicAdd(&grad_verts[3 * v_id2], grad_laplacian[3 * v_id1]);
    atomicAdd(&grad_verts[3 * v_id2+1], grad_laplacian[3 * v_id1+1]);
    atomicAdd(&grad_verts[3 * v_id2+2], grad_laplacian[3 * v_id1+2]);
}


__global__ void MeshLaplacianGradKernel(const int b, const int maxnverts, const int *nverts, const int * nb, const float * grad_laplacian, float* grad_verts){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        int n_verts = nverts[i];
        for (int v1=threadIdx.x+blockIdx.y*blockDim.x; v1 < n_verts; v1+=blockDim.x*gridDim.y){
            for (int v2=0; v2<20; v2++){
                if (nb[(i*maxnverts+v1)*20 + v2] == -1)
                    break;
                else{
                    setLaplaciangrad(&grad_laplacian[i*maxnverts*3], &grad_verts[i*maxnverts*3], v1, nb[(i*maxnverts+v1)*20 + v2]);
                }
            }
        }
    }
}

__global__ void InitGradKernel(const int b, const int maxnverts, float* grad_verts){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int v_id=threadIdx.x+blockIdx.y*blockDim.x; v_id<maxnverts; v_id+=blockDim.x*gridDim.y){
            for (int i_c = 0; i_c < 3; i_c++){
                grad_verts[i*maxnverts*3+v_id*3+i_c] = 0.;
            }
        }
    }
}

__global__ void AvgGradKernel(const int b, const int maxnverts, float* grad_verts, const int* cumulativeCounts){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int v_id=threadIdx.x+blockIdx.y*blockDim.x; v_id<maxnverts; v_id+=blockDim.x*gridDim.y){
            for (int i_c = 0; i_c < 3; i_c++)
                if (cumulativeCounts[i*maxnverts+v_id]!=0)
                    grad_verts[i*maxnverts*3+v_id*3+i_c] /= cumulativeCounts[i*maxnverts+v_id];
        }
    }
}

void MeshLaplacianGradKernelLauncher(const int b, const int maxnverts, const int * nverts, const float * grad_laplacian, const int * count, const int * nb,
                                    float* grad_verts){
    InitGradKernel<<<dim3(2,8,1),512>>>(b, maxnverts, grad_verts);
    MeshLaplacianGradKernel<<<dim3(2,8,1),512>>>(b, maxnverts, nverts, nb, grad_laplacian, grad_verts);
    AvgGradKernel<<<dim3(2,8,1),512>>>(b, maxnverts, grad_verts, count);
}

#endif

