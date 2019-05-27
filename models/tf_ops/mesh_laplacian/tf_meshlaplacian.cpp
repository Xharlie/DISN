#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <stdio.h>
REGISTER_OP("MeshLaplacian")
    .Input("verts: float32")
    .Input("nverts: int32")
    .Input("tris: int32")
    .Input("ntris: int32")
    .Output("laplacian: float32")
    .Output("count: int32")
    .Output("nb: int32");
REGISTER_OP("MeshLaplacianGrad")
    .Input("nverts: int32")
    .Input("count: int32")
    .Input("nb: int32")
    .Input("grad_laplacian: float32")
    .Output("grad_verts: float32");
using namespace tensorflow;


//void MeshLaplacianKernelLauncher(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i);

void MeshLaplacianKernelLauncher( \
    /*inputs*/  const int b, const int * n_verts, const int maxn_verts, const float * vertices, const int * n_triangles, const int maxn_triangles, const int * triangles, \
    /*outputs*/ float * laplacian, int * count, int * nb);

class MeshLaplacianGpuOp : public OpKernel{
    public:
        explicit MeshLaplacianGpuOp(OpKernelConstruction* context):OpKernel(context){}
        void Compute(OpKernelContext * context)override{

            /*****inputs*****/
            const Tensor& verts_tensor=context->input(0);
            const Tensor& nverts_tensor=context->input(1);
            const Tensor& tris_tensor=context->input(2);
            const Tensor& ntris_tensor=context->input(3);

            //verts
            OP_REQUIRES(context,verts_tensor.dims()==3,errors::InvalidArgument("MeshLaplacian requires verts be of shape (batch,#vertss,3)"));
            OP_REQUIRES(context,verts_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MeshLaplacian only accepts 3d point set vertss"));
            int b=verts_tensor.shape().dim_size(0);
            int maxn_verts=verts_tensor.shape().dim_size(1);

            //nverts
            OP_REQUIRES(context,nverts_tensor.dims()==2,errors::InvalidArgument("MeshLaplacian requires nverts be of shape (batch,1)"));
            OP_REQUIRES(context,nverts_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshLaplacian expects verts and nverts have same batch size"));

            //tris
            OP_REQUIRES(context,tris_tensor.dims()==3,errors::InvalidArgument("MeshLaplacian requires tris_tensor be of shape (batch,#tris,3)"));
            OP_REQUIRES(context,tris_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MeshLaplacian only accepts 3d point set tris"));
            int maxn_tris=tris_tensor.shape().dim_size(1);
            OP_REQUIRES(context,tris_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshLaplacian expects vertss and tris have same batch size"));

            //ntris
            OP_REQUIRES(context,ntris_tensor.dims()==2,errors::InvalidArgument("MeshLaplacian requires ntris be of shape (batch,1)"));
            OP_REQUIRES(context,ntris_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshLaplacian expects ntris and nverts have same batch size"));

            auto verts_flat=verts_tensor.flat<float>();
            const float * verts=&verts_flat(0);

            auto nverts_flat=nverts_tensor.flat<int>();
            const int * nverts=&nverts_flat(0);

            auto tris_flat=tris_tensor.flat<int>();
            const int * tris=&tris_flat(0);

            auto ntris_flat=ntris_tensor.flat<int>();
            const int * ntris=&ntris_flat(0);

            // outputs
            Tensor * laplacian_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,maxn_verts,3},&laplacian_tensor));
            auto laplacian_flat=laplacian_tensor->flat<float>();
            float * laplacian=&(laplacian_flat(0));

            Tensor * count_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,maxn_verts},&count_tensor));
            auto count_flat=count_tensor->flat<int>();
            int * count=&(count_flat(0));

            Tensor * nb_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{b,maxn_verts,20},&nb_tensor));
            auto nb_flat=nb_tensor->flat<int>();
            int * nb=&(nb_flat(0));

            MeshLaplacianKernelLauncher(
                            /*inputs*/  b, nverts, maxn_verts, verts, ntris, maxn_tris, tris, \
                            /*outputs*/ laplacian, count, nb);
        }
};
REGISTER_KERNEL_BUILDER(Name("MeshLaplacian").Device(DEVICE_GPU), MeshLaplacianGpuOp);


void MeshLaplacianGradKernelLauncher(const int b, const int maxnverts, const int * nverts, const float * grad_laplacian, const int * count, const int * nb,
                                    float* grad_verts);
class MeshLaplacianGradGpuOp : public OpKernel{
    public:
        explicit MeshLaplacianGradGpuOp(OpKernelConstruction* context):OpKernel(context){}
        void Compute(OpKernelContext * context)override{
            // inputs
            const Tensor& nverts_tensor=context->input(0); // b x maxnverts x 3
            const Tensor& count_tensor=context->input(1); // b x maxnverts
            const Tensor& nb_tensor=context->input(2); // b x maxnverts x 20
            const Tensor& grad_laplacian_tensor=context->input(3); // b x maxnverts x 3

            // nverts
            OP_REQUIRES(context,nverts_tensor.dims()==2,errors::InvalidArgument("MeshLaplacian requires nverts be of shape (batch,1)"));
            int b=nverts_tensor.shape().dim_size(0);

            // count
            OP_REQUIRES(context,count_tensor.dims()==2,errors::InvalidArgument("MeshLaplacian requires count_tensor be of shape (batch,#tris,3)"));
            OP_REQUIRES(context,count_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects count_tensor and verts_tensor have same batch size"));
            int maxnverts=count_tensor.shape().dim_size(1);

            // nb
            OP_REQUIRES(context,nb_tensor.dims()==3,errors::InvalidArgument("MeshLaplacian requires nb_tensor be of shape (batch,#tris,3)"));
            OP_REQUIRES(context,nb_tensor.shape().dim_size(2)==20,errors::InvalidArgument("MeshLaplacian requires max verts neighbourhood 20"));
            OP_REQUIRES(context,nb_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects nb_tensor and verts_tensor have same batch size"));
            OP_REQUIRES(context,nb_tensor.shape().dim_size(1)==maxnverts,errors::InvalidArgument("MeshSampling expects nb_tensor and verts_tensor have same maxn_verts"));

            // grad_laplacian
            OP_REQUIRES(context,grad_laplacian_tensor.dims()==3,errors::InvalidArgument("MeshLaplacian requires grad_laplacian_tensor be of shape (batch,#tris,3)"));
            OP_REQUIRES(context,grad_laplacian_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MeshLaplacian only accepts 3d point set tris"));
            OP_REQUIRES(context,grad_laplacian_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects grad_laplacian_tensor and verts_tensor have same batch size"));
            OP_REQUIRES(context,grad_laplacian_tensor.shape().dim_size(1)==maxnverts,errors::InvalidArgument("MeshSampling expects nb_tensor and verts_tensor have same maxn_verts"));

            auto nverts_flat=nverts_tensor.flat<int>();
            const int * nverts=&nverts_flat(0);
            auto count_flat=count_tensor.flat<int>();
            const int * count=&count_flat(0);
            auto nb_flat=nb_tensor.flat<int>();
            const int * nb=&nb_flat(0);
            auto grad_laplacian_flat=grad_laplacian_tensor.flat<float>();
            const float * grad_laplacian=&grad_laplacian_flat(0);



            // outputs
            Tensor * grad_verts_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,maxnverts,3},&grad_verts_tensor));

            auto grad_verts_flat=grad_verts_tensor->flat<float>();
            float * grad_verts=&grad_verts_flat(0);

            MeshLaplacianGradKernelLauncher(b, maxnverts, nverts, grad_laplacian, count, nb, grad_verts);
        }
};
REGISTER_KERNEL_BUILDER(Name("MeshLaplacianGrad").Device(DEVICE_GPU), MeshLaplacianGradGpuOp);
