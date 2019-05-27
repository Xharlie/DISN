//Author: Weiyue Wang
//Reference: https://github.com/charlesq34/pointnet-autoencoder/blob/master/tf_ops/nn_distance/tf_nndistance_g.cu
//           https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <stdio.h>
#include <assert.h>

__device__ float TriangleArea(float *a, float *b, float *c){
//	float s=(a+b+c)/2;
//	float Area=sqrt(s*(s-a)*(s-b)*(s-c));

//    int x1,x2,x3,y1,y2,y3;
    float side1 = 10 * sqrt ( (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]) );
    float side2 = 10 * sqrt ( (a[0]-c[0])*(a[0]-c[0]) + (a[1]-c[1])*(a[1]-c[1]) + (a[2]-c[2])*(a[2]-c[2]) );
    float side3 = 10 * sqrt ( (c[0]-b[0])*(c[0]-b[0]) + (c[1]-b[1])*(c[1]-b[1]) + (c[2]-b[2])*(c[2]-b[2]) );
//    float side2 = sqrt ( (x2-x3)*(x2-x3) + (y2-y3)*(y2-y3) + (y2-y3)*(y2-y3) );
//    float side3 = sqrt ( (x3-x1)*(x3-x1) + (y3-y1)*(y3-y1) + (y3-y1)*(y3-y1) );
    float s = (side1 + side2 + side3)/2;
    float area =  sqrt( s * (s - side1) * (s - side2) * (s - side3));

//    printf("area:%f, side:%f %f %f %f\n", area, side1, side2, side3, s);

	return area;
}

__device__ void getPoint(const float *vertices, int v_id, float *p){
    p[0] = vertices[3* v_id];
    p[1] = vertices[3* v_id+1];
    p[2] = vertices[3* v_id+2];
}

__device__ void getFeat(const float *feats, int v_id, int n_c, float *p){

    for (int i = 0; i < n_c; i++)
        p[i] = feats[v_id + i];
}

__device__ void getTriangle(const int *triangles, int t_id, int &v1, int &v2, int &v3){
    v1 = triangles[3 * t_id];
    v2 = triangles[3 * t_id + 1];
    v3 = triangles[3 * t_id + 2];
}

__device__  int lower_bound (const float * array, int n, const float& val)
{
  int it, first=0;
  int step;
  int count = n-1;
  while (count>0)
  {
    it = first; step=count/2; it += step;
//      printf("%f: %f \n", array[it], val);
    if (array[it]<val) {                 // or: if (comp(*it,val)), for version (2)

      first=++it;
      count-=step+1;
    }
    else count=step;
  }
  return first;
}


__device__ void randomPointTriangle (float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3,
                          float r1, float r2, float p[3])
{
    float r1sqr = std::sqrt (r1);
    float OneMinR1Sqr = (1 - r1sqr);
    float OneMinR2 = (1 - r2);
    a1 *= OneMinR1Sqr;
    a2 *= OneMinR1Sqr;
    a3 *= OneMinR1Sqr;
    b1 *= OneMinR2;
    b2 *= OneMinR2;
    b3 *= OneMinR2;
    c1 = r1sqr * (r2 * c1 + b1) + a1;
    c2 = r1sqr * (r2 * c2 + b2) + a2;
    c3 = r1sqr * (r2 * c3 + b3) + a3;
    p[0] = c1;
    p[1] = c2;
    p[2] = c3;
}

__device__ void randomPointTriangle_array (const float * A, const float * B, const float * C, const float r1, const float r2, float * p, int n_c){
    float r1sqr = std::sqrt (r1);
    float OneMinR1Sqr = (1 - r1sqr);
    float OneMinR2 = (1 - r2);

    for (int i = 0; i < n_c; i++)
        p[i] = r1sqr * r2 * C[i] + r1sqr * OneMinR2 * B[i] + OneMinR1Sqr * A[i];
//    float *AA = new float[n_c], *BB = new float[n_c];
//
//    for (int i = 0; i < n_c; i++)
//        AA[i] = A[i] * OneMinR1Sqr;
//
//    for (int i = 0; i < n_c; i++)
//        BB[i] = B[i] * OneMinR2;
//
//    for (int i = 0; i < n_c; i++)
//        p[i] = r1sqr * (r2 * C[i] + BB[i]) + AA[i];
//
//    delete [] AA;
//    delete [] BB;
}

__device__  int randPSurface (const int *triangles, const float *vertices, const float * feats, const float * cumulativeAreas, int n_triangles, const float totalArea, const int n_c, float *p, float * outfeats, const float r, const float r1, const float r2){

    int el = (lower_bound(cumulativeAreas, n_triangles, r * totalArea));
    int v1, v2, v3;

    getTriangle(triangles, el, v1, v2, v3);

    randomPointTriangle_array (&vertices[3*v1], &vertices[3*v2], &vertices[3*v3], r1, r2, p, 3);
    randomPointTriangle_array (&feats[n_c*v1], &feats[n_c*v2], &feats[n_c*v3], r1, r2, outfeats, n_c);
    return el;
}

__global__ void MeshSamplingKernel(const int b, const int * nverts, const int maxnverts, const float * vertices, const int * ntriangles, const int maxntriangles, const int * triangles, const int n_c, const float * feats, const float * r, const float * r1, const float * r2, const int n_samples, const float * cumulativeAreas, float * points, float* outfeats, int * correspondingface){

    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        int n_triangles = ntriangles[i];
        for (int sample_id=threadIdx.x+blockIdx.y*blockDim.x; sample_id < n_samples; sample_id+=blockDim.x*gridDim.y){//for (int sample_id = blockIdx.x * blockDim.x + threadIdx.x; sample_id< n_samples; sample_id += blockDim.x * gridDim.x)
            correspondingface[(i*n_samples+sample_id)] = randPSurface (&triangles[i*maxntriangles*3], &vertices[i*maxnverts*3], &feats[i*maxnverts*n_c],
                                                                       &cumulativeAreas[i*maxntriangles], n_triangles, cumulativeAreas[n_triangles-1], n_c,
                                                                       &points[(i*n_samples+sample_id)*3], &outfeats[(i*n_samples+sample_id)*n_c], r[(i*n_samples+sample_id)], r1[(i*n_samples+sample_id)], r2[(i*n_samples+sample_id)]);

        }
        __syncthreads();
    }
}

__device__  int randPSurfaceNoFeat (const int *triangles, const float *vertices, const float * cumulativeAreas, int n_triangles, const float totalArea,  float *p, const float r, const float r1, const float r2){

    int el = (lower_bound(cumulativeAreas, n_triangles, r * totalArea));
    int v1, v2, v3;

    getTriangle(triangles, el, v1, v2, v3);

    randomPointTriangle_array (&vertices[3*v1], &vertices[3*v2], &vertices[3*v3], r1, r2, p, 3);
    return el;
}

__global__ void MeshSamplingNoFeatKernel(const int b, const int * nverts, const int maxnverts, const float * vertices, const int * ntriangles, const int maxntriangles, const int * triangles,  const float * r, const float * r1, const float * r2, const int n_samples, const float * cumulativeAreas, float * points, int * correspondingface){

    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        int n_triangles = ntriangles[i];
        for (int sample_id=threadIdx.x+blockIdx.y*blockDim.x; sample_id < n_samples; sample_id+=blockDim.x*gridDim.y){//for (int sample_id = blockIdx.x * blockDim.x + threadIdx.x; sample_id< n_samples; sample_id += blockDim.x * gridDim.x)
            correspondingface[(i*n_samples+sample_id)] = randPSurfaceNoFeat (&triangles[i*maxntriangles*3], &vertices[i*maxnverts*3],
                                                                       &cumulativeAreas[i*maxntriangles], n_triangles, cumulativeAreas[n_triangles-1],
                                                                       &points[(i*n_samples+sample_id)*3],  r[(i*n_samples+sample_id)], r1[(i*n_samples+sample_id)], r2[(i*n_samples+sample_id)]);

        }
        __syncthreads();
    }
}

__global__ void CumulativeAreaKernel(const int b, const int * nverts, const int maxnverts, const float * vertices, const int * ntriangles, const int maxntriangles, const int * triangles, float * cumulativeAreas){

    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        int n_triangles = ntriangles[i];
        int n_verts = nverts[i];

        assert (n_triangles <= maxntriangles);
        assert (n_verts <= maxnverts);

        float p1[3], p2[3], p3[3], totalArea = 0;
        int v1,v2,v3;
        for (int triangle_id=0; triangle_id < n_triangles; triangle_id++){//for (int triangle_id = 0; triangle_id < n_triangles; ++triangle_id)

            getTriangle(&triangles[i*maxntriangles*3], triangle_id, v1, v2, v3);
            getPoint(&vertices[i*maxnverts*3], v1, p1);
            getPoint(&vertices[i*maxnverts*3], v2, p2);
            getPoint(&vertices[i*maxnverts*3], v3, p3);

            float area = TriangleArea(p1, p2, p3);

            if (!(isnan(area)))
                totalArea += area;

            cumulativeAreas[i*maxntriangles+triangle_id] = totalArea;
//            printf("%f %f %f, %f %f %f, %f %f %f....", p1[0], p1[1],p1[2], p2[0], p2[1],p2[2], p3[0], p3[1],p3[2]);
//            printf("tid:%d , %d %d %d, %f, %f\n", triangle_id, v1, v2, v3, area, totalArea);
//            printf("tid:%d %d\n", sample_id, maxntriangles);
//            printf("%d:%f, ", triangle_id, area);
        }
    }
}


void MeshSamplingKernelLauncher( \
    /*inputs*/  const int b, const int * n_verts, const int maxn_verts, const float * vertices, const int * n_triangles, const int maxn_triangles, const int * triangles, const int n_c, const float * feats, const  float * r, const float * r1, const float * r2, const int n_samples, \
    /*outputs*/ float * points, float* outfeats, int * correspondingface){

    float *cumulativeAreas;
    cudaMalloc((void**)&cumulativeAreas, b*maxn_triangles*sizeof(float));

    CumulativeAreaKernel<<<32,1>>>(b, n_verts, maxn_verts, vertices, n_triangles, maxn_triangles, triangles,cumulativeAreas);
    MeshSamplingKernel<<<dim3(2,8,1),512>>>(b, n_verts, maxn_verts, vertices, n_triangles, maxn_triangles, triangles, n_c, feats, r, r1, r2, n_samples, cumulativeAreas, points, outfeats, correspondingface);
    cudaFree(cumulativeAreas);
}

void MeshSamplingNoFeatKernelLauncher( \
    /*inputs*/  const int b, const int * n_verts, const int maxn_verts, const float * vertices, const int * n_triangles, const int maxn_triangles, const int * triangles, const  float * r, const float * r1, const float * r2, const int n_samples, \
    /*outputs*/ float * points, int * correspondingface){

    float *cumulativeAreas;
    cudaMalloc((void**)&cumulativeAreas, b*maxn_triangles*sizeof(float));

    CumulativeAreaKernel<<<32,1>>>(b, n_verts, maxn_verts, vertices, n_triangles, maxn_triangles, triangles,cumulativeAreas);
    MeshSamplingNoFeatKernel<<<dim3(2,8,1),512>>>(b, n_verts, maxn_verts, vertices, n_triangles, maxn_triangles, triangles,  r, r1, r2, n_samples, cumulativeAreas, points, correspondingface);
    cudaFree(cumulativeAreas);
}

__device__ void gradrandomPointTriangle_array (float * A, float * B, float * C,
                                                int * count_A, int * count_B, int * count_C,
                                                const float r1, const float r2, const float * gp, const int n_c){
    float r1sqr = std::sqrt (r1);
    float OneMinR1Sqr = (1 - r1sqr);
    float OneMinR2 = (1 - r2);

    for (int i = 0; i < n_c; i++){
        atomicAdd(&A[i], gp[i] * r1sqr * r2);
        atomicAdd(&B[i], gp[i] * r1sqr * OneMinR2);
        atomicAdd(&C[i], gp[i] * OneMinR1Sqr);
        atomicAdd(&count_A[i], 1);
        atomicAdd(&count_B[i], 1);
        atomicAdd(&count_C[i], 1);
    }

}


__global__ void MeshSamplingGradKernel(const int b, const int maxnverts, const int maxntriangles, const int * triangles,
                                        const int n_c, const float * r1, const float * r2, const int n_samples, const float * grad_outfeat,
                                        const int * correspondingface, int* cumulativeCounts, float* grad_feats){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int sample_id=threadIdx.x+blockIdx.y*blockDim.x; sample_id<n_samples; sample_id+=blockDim.x*gridDim.y){
            // index outfeat: (i * n_samples+sample_id)*n_c
            // index infeat: (i * maxnverts+sample_id)*n_c
            int v1, v2, v3;

            getTriangle(&triangles[i*maxntriangles*3], correspondingface[(i*n_samples+sample_id)], v1, v2, v3);

            float * grad_feats_tmp = &grad_feats[i*maxnverts*n_c];
            int * cumulativeCounts_tmp = &cumulativeCounts[i*maxnverts*n_c];
            gradrandomPointTriangle_array(&grad_feats_tmp[n_c*v1], &grad_feats_tmp[n_c*v2], &grad_feats_tmp[n_c*v3],
                                          &cumulativeCounts_tmp[n_c*v1], &cumulativeCounts_tmp[n_c*v2], &cumulativeCounts_tmp[n_c*v3],
                                          r1[(i*n_samples+sample_id)], r2[(i*n_samples+sample_id)],
                                          &grad_outfeat[(i*n_samples+sample_id)*n_c], n_c);

        }
    }
}

__global__ void InitGradKernel(const int b, const int maxnverts, const int n_c, float* grad_feats, int* cumulativeCounts){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int v_id=threadIdx.x+blockIdx.y*blockDim.x; v_id<maxnverts; v_id+=blockDim.x*gridDim.y){
            for (int i_c = 0; i_c < n_c; i_c++){
                grad_feats[i*maxnverts*n_c+v_id*n_c+i_c] = 0.;
                cumulativeCounts[i*maxnverts*n_c+v_id*n_c+i_c] = 0;
            }
        }
    }
}

__global__ void AvgGradKernel(const int b, const int maxnverts, const int n_c, float* grad_feats, int* cumulativeCounts){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int v_id=threadIdx.x+blockIdx.y*blockDim.x; v_id<maxnverts; v_id+=blockDim.x*gridDim.y){
            for (int i_c = 0; i_c < n_c; i_c++)
                if (cumulativeCounts[i*maxnverts*n_c+v_id*n_c+i_c]!=0)
                    grad_feats[i*maxnverts*n_c+v_id*n_c+i_c] /= cumulativeCounts[i*maxnverts*n_c+v_id*n_c+i_c];
        }
    }
}

void MeshSamplingGradKernelLauncher(const int b, const int maxnverts, const int maxntriangles, const int * triangles,
                                    const int n_c, const float * r1, const float * r2, const int n_samples, const float * grad_outfeat,
                                    const int * correspondingface, float* grad_feats){
    int *cumulativeCounts;
    cudaMalloc((void**)&cumulativeCounts, b*maxnverts*sizeof(int));
    InitGradKernel<<<dim3(2,8,1),512>>>(b, maxnverts, n_c, grad_feats, cumulativeCounts);
    MeshSamplingGradKernel<<<dim3(2,8,1),512>>>(b, maxnverts, maxntriangles, triangles, n_c, r1, r2, n_samples, grad_outfeat,
                                                 correspondingface, cumulativeCounts, grad_feats);
    AvgGradKernel<<<dim3(2,8,1),512>>>(b, maxnverts, n_c, grad_feats, cumulativeCounts);
    cudaFree(cumulativeCounts);
}

void MeshSamplingGradNoFeatKernelLauncher(const int b, const int maxnverts, const int maxntriangles, const int * triangles,
                                    const float * r1, const float * r2, const int n_samples, const float * grad_outfeat,
                                    const int * correspondingface, float* grad_feats){
    int *cumulativeCounts;
    cudaMalloc((void**)&cumulativeCounts, b*maxnverts*sizeof(int));
    InitGradKernel<<<dim3(2,8,1),512>>>(b, maxnverts, grad_feats, cumulativeCounts);
    MeshSamplingGradKernel<<<dim3(2,8,1),512>>>(b, maxnverts, maxntriangles, triangles, 3, r1, r2, n_samples, grad_points,
                                                 correspondingface, cumulativeCounts, grad_verts);
    AvgGradKernel<<<dim3(2,8,1),512>>>(b, maxnverts, grad_feats, cumulativeCounts);
    cudaFree(cumulativeCounts);
}

#endif

