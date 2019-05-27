#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;
#include <stdio.h>
REGISTER_OP("MeshSampling")
    .Attr("infeats: bool")
    .Input("verts: float32")
    .Input("nverts: int32")
    .Input("tris: int32")
    .Input("ntris: int32")
    .Input("feats: float32")
    .Input("r: float32")
    .Input("r1: float32")
    .Input("r2: float32")
    .Output("points: float32")
    .Output("outfeats: float32")
    .Output("correspondingface: int32") // batchsize * n_sample
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle vertsshape; // batch_size * maxnverts * 3
        c->WithRank(c->input(0), 3, &vertsshape);
        ::tensorflow::shape_inference::ShapeHandle trisshape; // batch_size * maxntris * 3
        c->WithRank(c->input(2), 3, &trisshape);
        ::tensorflow::shape_inference::ShapeHandle featsshape; // batch_size * maxnverts * n_c
        c->WithRank(c->input(4), 3, &featsshape);
        ::tensorflow::shape_inference::ShapeHandle rshape; // batch_size * n_sample
        c->WithRank(c->input(5), 2, &rshape);
        // batch_size * npoints
        ::tensorflow::shape_inference::ShapeHandle pointsshape = c->MakeShape({c->Dim(vertsshape, 0), c->Dim(rshape, 1), c->Dim(vertsshape, 2)});
        c->set_output(0, pointsshape);
        ::tensorflow::shape_inference::ShapeHandle outfeatsshape = c->MakeShape({c->Dim(vertsshape, 0), c->Dim(rshape, 1), c->Dim(featsshape, 2)});
        c->set_output(1, outfeatsshape);
        ::tensorflow::shape_inference::ShapeHandle correspondingfaceshape = c->MakeShape({c->Dim(vertsshape, 0), c->Dim(rshape, 1)});
        c->set_output(2, correspondingfaceshape);
        return Status::OK();
    });
REGISTER_OP("MeshSamplingGrad")
    .Input("verts: float32")
    .Input("tris: int32")
    .Input("r1: float32")
    .Input("r2: float32")
    .Input("correspondingface: int32")
    .Input("grad_outfeats: float32")
//    .Output("grad_verts: float32")
//    .Output("grad_nverts: int32")
//    .Output("grad_tris: int32")
//    .Output("grad_ntris: int32")
    .Output("grad_verts: float32")
    .Output("grad_feats: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle vertsshape; // batch_size * maxnverts * 3
        c->WithRank(c->input(0), 3, &vertsshape);
        ::tensorflow::shape_inference::ShapeHandle gradoutfeatsshape; // batch_size * n_sample * n_c
        c->WithRank(c->input(5), 3, &gradoutfeatsshape);
        ::tensorflow::shape_inference::ShapeHandle grad_featsshape = c->MakeShape({c->Dim(vertsshape, 0), c->Dim(vertsshape, 1), c->Dim(gradoutfeatsshape, 2)});
        c->set_output(0, grad_featsshape);
        ::tensorflow::shape_inference::ShapeHandle grad_featsshape = c->MakeShape({c->Dim(vertsshape, 0), c->Dim(vertsshape, 1), c->Dim(vertsshape, 2)});
        c->set_output(1, grad_featsshape);
        return Status::OK();
    });
//    .Output("grad_r: float32")
//    .Output("grad_r1: float32")
//    .Output("grad_r2: float32");
using namespace tensorflow;


//void MeshSamplingKernelLauncher(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i);

void MeshSamplingKernelLauncher( \
    /*inputs*/  const int b, const int * n_verts, const int maxn_verts, const float * vertices, const int * n_triangles, const int maxn_triangles, const int * triangles, const int n_c, const float * feats, const  float * r, const float * r1, const float * r2, const int n_samples, \
    /*outputs*/ float * points, float* outfeats, int * correspondingface);

void MeshSamplingNoFeatKernelLauncher( \
    /*inputs*/  const int b, const int * n_verts, const int maxn_verts, const float * vertices, const int * n_triangles, const int maxn_triangles, const int * triangles, const  float * r, const float * r1, const float * r2, const int n_samples, \
    /*outputs*/ float * points, int * correspondingface);

class MeshSamplingGpuOp : public OpKernel{
    public:
        explicit MeshSamplingGpuOp(OpKernelConstruction* context):OpKernel(context){
            OP_REQUIRES_OK(context, context->GetAttr("infeats", &infeats_));
        }
        void Compute(OpKernelContext * context)override{

            /*****inputs*****/
            const Tensor& verts_tensor=context->input(0);
            const Tensor& nverts_tensor=context->input(1);
            const Tensor& tris_tensor=context->input(2);
            const Tensor& ntris_tensor=context->input(3);
            const Tensor& feats_tensor=context->input(4);
            const Tensor& r_tensor=context->input(5);
            const Tensor& r1_tensor=context->input(6);
            const Tensor& r2_tensor=context->input(7);

            //verts
            OP_REQUIRES(context,verts_tensor.dims()==3,errors::InvalidArgument("MeshSampling requires verts be of shape (batch,#vertss,3)"));
            OP_REQUIRES(context,verts_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MeshSampling only accepts 3d point set vertss"));
            int b=verts_tensor.shape().dim_size(0);
            int maxn_verts=verts_tensor.shape().dim_size(1);

            //nverts
            OP_REQUIRES(context,nverts_tensor.dims()==2,errors::InvalidArgument("MeshSampling requires nverts be of shape (batch,1)"));
            OP_REQUIRES(context,nverts_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects verts and nverts have same batch size"));

            //tris
            OP_REQUIRES(context,tris_tensor.dims()==3,errors::InvalidArgument("MeshSampling requires tris_tensor be of shape (batch,#tris,3)"));
            OP_REQUIRES(context,tris_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MeshSampling only accepts 3d point set tris"));
            int maxn_tris=tris_tensor.shape().dim_size(1);
            OP_REQUIRES(context,tris_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects vertss and tris have same batch size"));

            //ntris
            OP_REQUIRES(context,ntris_tensor.dims()==2,errors::InvalidArgument("MeshSampling requires ntris be of shape (batch,1)"));
            OP_REQUIRES(context,ntris_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects ntris and nverts have same batch size"));

            //feats
            if (infeats_){
                OP_REQUIRES(context,feats_tensor.dims()==3,errors::InvalidArgument("MeshSampling requires tris_tensor be of shape (batch,#tris,#featurechannels)"));
                OP_REQUIRES(context,feats_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects feats and verts have same batch size"));
                OP_REQUIRES(context,feats_tensor.shape().dim_size(1)==maxn_verts,errors::InvalidArgument("MeshSampling expects feats and verts have same number of points"));
                int nc_feats = feats_tensor.shape().dim_size(2);
            }

            //R
            OP_REQUIRES(context,r_tensor.dims()==2,errors::InvalidArgument("MeshSampling requires r_tensor be of shape (batch, n_samples)"));
            OP_REQUIRES(context,r_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects r and verts have same batch size"));
            const int n_samples=r_tensor.shape().dim_size(1);

            //R1
            OP_REQUIRES(context,r1_tensor.dims()==2,errors::InvalidArgument("MeshSampling requires r_tensor be of shape (batch, n_samples)"));
            OP_REQUIRES(context,r1_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects r and verts have same batch size"));
            OP_REQUIRES(context,r1_tensor.shape().dim_size(1)==n_samples,errors::InvalidArgument("MeshSampling expects r1.shape[1] and n_samples are equal"));

            //R2
            OP_REQUIRES(context,r2_tensor.dims()==2,errors::InvalidArgument("MeshSampling requires r_tensor be of shape (batch, n_samples)"));
            OP_REQUIRES(context,r2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects r and verts have same batch size"));
            OP_REQUIRES(context,r2_tensor.shape().dim_size(1)==n_samples,errors::InvalidArgument("MeshSampling expects r2.shape[1] and n_samples are equal"));

            auto verts_flat=verts_tensor.flat<float>();
            const float * verts=&verts_flat(0);

            auto nverts_flat=nverts_tensor.flat<int>();
            const int * nverts=&nverts_flat(0);

            auto tris_flat=tris_tensor.flat<int>();
            const int * tris=&tris_flat(0);

            auto ntris_flat=ntris_tensor.flat<int>();
            const int * ntris=&ntris_flat(0);

            auto feats_flat=feats_tensor.flat<float>();
            const float * feats=&feats_flat(0);

            auto r_flat=r_tensor.flat<float>();
            const float * r=&r_flat(0);
            auto r1_flat=r1_tensor.flat<float>();
            const float * r1=&r1_flat(0);
            auto r2_flat=r2_tensor.flat<float>();
            const float * r2=&r2_flat(0);

            // outputs
            Tensor * points_tensor=NULL;
            Tensor * outfeats_tensor=NULL;
            Tensor * correspondingface_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n_samples,3},&points_tensor));
            OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n_samples,nc_feats},&outfeats_tensor));
            OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{b,n_samples},&correspondingface_tensor));
            auto points_flat=points_tensor->flat<float>();
            auto outfeats_flat=outfeats_tensor->flat<float>();
            auto correspondingface_flat=correspondingface_tensor->flat<int>();

            float * points=&(points_flat(0));
            float * outfeats=&(outfeats_flat(0));
            int * correspondingface=&(correspondingface_flat(0));

            if (infeats_)
                MeshSamplingKernelLauncher(
                                /*inputs*/  b, nverts, maxn_verts, verts, ntris, maxn_tris, tris, nc_feats, feats, r, r1, r2, n_samples, \
                                /*outputs*/ points, outfeats, correspondingface);
            else
                MeshSamplingNoFeatKernelLauncher(
                                /*inputs*/  b, nverts, maxn_verts, verts, ntris, maxn_tris, tris, r, r1, r2, n_samples, \
                                /*outputs*/ points, correspondingface);
        }
    private:
        bool infeats_;
};
REGISTER_KERNEL_BUILDER(Name("MeshSampling").Device(DEVICE_GPU), MeshSamplingGpuOp);


void MeshSamplingGradKernelLauncher(const int b, const int maxnverts, const int maxntriangles, const int * triangles,
                                    const int n_c, const float * r1, const float * r2, const int n_samples, const float * grad_outfeats,
                                    const int * correspondingface,
                        /*output:*/ float* grad_feats);
class MeshSamplingGradGpuOp : public OpKernel{
    public:
        explicit MeshSamplingGradGpuOp(OpKernelConstruction* context):OpKernel(context){}
        void Compute(OpKernelContext * context)override{
            // inputs
            const Tensor& verts_tensor=context->input(0); // b x maxntris x 3
            const Tensor& tris_tensor=context->input(1); // b x maxntris x 3
            const Tensor& r1_tensor=context->input(2); // b x n_samples
            const Tensor& r2_tensor=context->input(3); // b x n_samples
            const Tensor& correspondingface_tensor=context->input(4); // b x n_samples
            const Tensor& grad_outfeats_tensor=context->input(5); // b x n_samples x n_c

            // verts
            OP_REQUIRES(context,verts_tensor.dims()==3,errors::InvalidArgument("MeshSampling requires verts_tensor be of shape (batch,#verts,3)"));
            OP_REQUIRES(context,verts_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MeshSampling only accepts 3d point set verts"));
            int b=verts_tensor.shape().dim_size(0);
            int maxnverts = verts_tensor.shape().dim_size(1);

            // triangles
            OP_REQUIRES(context,tris_tensor.dims()==3,errors::InvalidArgument("MeshSampling requires tris_tensor be of shape (batch,#tris,3)"));
            OP_REQUIRES(context,tris_tensor.shape().dim_size(2)==3,errors::InvalidArgument("MeshSampling only accepts 3d point set tris"));
            OP_REQUIRES(context,tris_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects tris_tensor and verts_tensor have same batch size"));
            int maxntriangles = tris_tensor.shape().dim_size(1);

            //grad_outfeats
            OP_REQUIRES(context,grad_outfeats_tensor.dims()==3,errors::InvalidArgument("MeshSampling requires tris_tensor be of shape (batch,#n_samples,n_c)"));
            int n_c=grad_outfeats_tensor.shape().dim_size(2);
            int n_samples=grad_outfeats_tensor.shape().dim_size(1);
            OP_REQUIRES(context,grad_outfeats_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects grad_outfeats_tensor and tris_tensor have same batch size"));

            //R1
            OP_REQUIRES(context,r1_tensor.dims()==2,errors::InvalidArgument("MeshSampling requires r_tensor be of shape (batch, n_samples)"));
            OP_REQUIRES(context,r1_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects r and verts have same batch size"));
            OP_REQUIRES(context,r1_tensor.shape().dim_size(1)==n_samples,errors::InvalidArgument("MeshSampling expects r1.shape[1] and n_samples are equal"));

            //R2
            OP_REQUIRES(context,r2_tensor.dims()==2,errors::InvalidArgument("MeshSampling requires r_tensor be of shape (batch, n_samples)"));
            OP_REQUIRES(context,r2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects r and verts have same batch size"));
            OP_REQUIRES(context,r2_tensor.shape().dim_size(1)==n_samples,errors::InvalidArgument("MeshSampling expects r2.shape[1] and n_samples are equal"));

            //correpondingface
            OP_REQUIRES(context,correspondingface_tensor.dims()==2,errors::InvalidArgument("MeshSampling requires r_tensor be of shape (batch, n_samples)"));
            OP_REQUIRES(context,correspondingface_tensor.shape().dim_size(0)==b,errors::InvalidArgument("MeshSampling expects r and verts have same batch size"));
            OP_REQUIRES(context,correspondingface_tensor.shape().dim_size(1)==n_samples,errors::InvalidArgument("MeshSampling expects r2.shape[1] and n_samples are equal"));

            auto tris_flat= tris_tensor.flat<int>();
            const int * tris=&tris_flat(0);
            auto grad_outfeats_flat=grad_outfeats_tensor.flat<float>();
            const float * grad_outfeats=&grad_outfeats_flat(0);
            auto r1_flat=r1_tensor.flat<float>();
            const float * r1=&r1_flat(0);
            auto r2_flat=r2_tensor.flat<float>();
            const float * r2=&r2_flat(0);
            auto correspondingface_flat=correspondingface_tensor.flat<int>();
            const int * correspondingface=&correspondingface_flat(0);

            // outputs
            Tensor * grad_feats_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,maxnverts,3},&grad_feats_tensor));

            auto grad_feats_flat=grad_feats_tensor->flat<float>();
            float * grad_feats=&grad_feats_flat(0);

            Tensor * grad_verts_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,maxnverts,3},&grad_verts_tensor));

            auto grad_feats_flat=grad_feats_tensor->flat<float>();
            float * grad_feats=&grad_feats_flat(0);
//            Tensor * grad_verts=NULL;
//            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,0},&grad_verts));
//            Tensor * grad_nverts=NULL;
//            OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,0},&grad_nverts));
//            Tensor * grad_tris=NULL;
//            OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{b,0},&grad_tris));
//            Tensor * grad_ntris=NULL;
//            OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape{b,0},&grad_ntris));
//            Tensor * grad_r=NULL;
//            OP_REQUIRES_OK(context,context->allocate_output(5,TensorShape{b,0},&grad_r));
//            Tensor * grad_r1=NULL;
//            OP_REQUIRES_OK(context,context->allocate_output(6,TensorShape{b,0},&grad_r1));
//            Tensor * grad_r2=NULL;
//            OP_REQUIRES_OK(context,context->allocate_output(7,TensorShape{b,0},&grad_r2));

            MeshSamplingGradKernelLauncher(b, maxnverts, maxntriangles, tris, n_c, r1, r2, n_samples, grad_outfeats, correspondingface,
                                /*output:*/grad_feats);
        }
};
REGISTER_KERNEL_BUILDER(Name("MeshSamplingGrad").Device(DEVICE_GPU), MeshSamplingGradGpuOp);
