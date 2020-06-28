// torch library headers
#include <torch/extension.h>
#include <THC/THC.h>

// C++ standard header
#include <algorithm>
#include <vector>
#include <math.h>
#include <omp.h>
#include <cstdio>
#include <iostream>
#include <memory>

// CUDA and/or cuBLAS header
#include "cuda_helper.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

// Mark: CUDA EMD primal form (through sinkhorn iteration) from Optas github
void approxmatchLauncher(int b,
    int n,
    int m,
    const float * xyz1,
    const float * xyz2,
    float * match,
    float * temp);

void matchcostLauncher(int b,
    int n,
    int m,
    const float * xyz1,
    const float * xyz2,
    const float * match,
    float * out);

void matchcostgradLauncher(int b,
    int n,
    int m,
    const float * xyz1,
    const float * xyz2,
    const float * match,
    float * grad1,
    float * grad2);

// Mark: CUDA Chamfer distance
int ChamferDistanceKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const int m,
    const float* xyz2,
    float* result,
    int* result_i,
    float* result2,
    int* result2_i);

int ChamferDistanceGradKernelLauncher(
    const int b, const int n,
    const float* xyz1,
    const int m,
    const float* xyz2,
    const float* grad_dist1,
    const int* idx1,
    const float* grad_dist2,
    const int* idx2,
    float* grad_xyz1,
    float* grad_xyz2);

std::vector<at::Tensor> emd_distance_forward_cuda(const at::Tensor xyz1, 
    const at::Tensor xyz2) 
{
	// Allocate necessary data structures
	at::Tensor match = at::zeros({xyz1.size(0), xyz1.size(1), xyz2.size(1)}, 
		xyz1.options());
	at::Tensor cost = at::zeros({xyz1.size(0)}, xyz1.options());
	at::Tensor temp = at::zeros({xyz1.size(0), 2 * (xyz1.size(1) + xyz2.size(1))}, 
		xyz1.options());

	// Find the approximate matching 
	approxmatchLauncher(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1),
    xyz1.data<float>(),
    xyz2.data<float>(),
	match.data<float>(),
    temp.data<float>());

	// Compute the matching cost
	matchcostLauncher(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1),
    xyz1.data<float>(),
    xyz2.data<float>(),
	match.data<float>(),
	cost.data<float>());

    return {cost, match};
}

// CUDA 

std::vector<at::Tensor> emd_distance_backward_cuda(const at::Tensor xyz1,
    const at::Tensor xyz2,
    const at::Tensor match)
{
	// Allocate necessary data structures
	at::Tensor gradxyz1 = at::zeros_like(xyz1);
	at::Tensor gradxyz2 = at::zeros_like(xyz2);

    matchcostgradLauncher(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1), 
    xyz1.data<float>(),
    xyz2.data<float>(),
    match.data<float>(), 
    gradxyz1.data<float>(), 
    gradxyz2.data<float>());
    
    // return gradients
    return {gradxyz1, gradxyz2};
}

void chamfer_distance_forward_cuda(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    const at::Tensor dist1, 
    const at::Tensor dist2, 
    const at::Tensor idx1, 
    const at::Tensor idx2) 
{
    ChamferDistanceKernelLauncher(xyz1.size(0),
    xyz1.size(1),
    xyz1.data<float>(),
    xyz2.size(1),
    xyz2.data<float>(),
    dist1.data<float>(), 
    idx1.data<int>(),
    dist2.data<float>(), 
    idx2.data<int>());
}

void chamfer_distance_backward_cuda(
    const at::Tensor xyz1,
    const at::Tensor xyz2, 
    at::Tensor gradxyz1, 
    at::Tensor gradxyz2, 
    at::Tensor graddist1, 
    at::Tensor graddist2, 
    at::Tensor idx1, 
    at::Tensor idx2)
{
    ChamferDistanceGradKernelLauncher(xyz1.size(0),
    xyz1.size(1),
    xyz1.data<float>(),
    xyz2.size(1),
    xyz2.data<float>(),
    graddist1.data<float>(), 
    idx1.data<int>(),
    graddist2.data<float>(), 
    idx2.data<int>(),
    gradxyz1.data<float>(), gradxyz2.data<float>());
}



//'''
//
//CPU function wrappers!!!
//'''


// Mark: Wasserstein (EMD) cpu function

void approxmatch_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,float * match){
	for (int i=0;i<b;i++){
		int factorl=std::max(n,m)/n;
		int factorr=std::max(n,m)/m;
		std::vector<double> saturatedl(n,double(factorl)),saturatedr(m,double(factorr));
		std::vector<double> weight(n*m);
		for (int j=0;j<n*m;j++)
			match[j]=0;
		for (int j=8;j>=-2;j--){
			//printf("i=%d j=%d\n",i,j);
			double level=-powf(4.0,j);
			if (j==-2)
				level=0;
			for (int k=0;k<n;k++){
				double x1=xyz1[k*3+0];
				double y1=xyz1[k*3+1];
				double z1=xyz1[k*3+2];
				for (int l=0;l<m;l++){
					double x2=xyz2[l*3+0];
					double y2=xyz2[l*3+1];
					double z2=xyz2[l*3+2];
					weight[k*m+l]=expf(level*((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)))*saturatedr[l];
				}
			}
			std::vector<double> ss(m,1e-9);
			for (int k=0;k<n;k++){
				double s=1e-9;
				for (int l=0;l<m;l++){
					s+=weight[k*m+l];
				}
				for (int l=0;l<m;l++){
					weight[k*m+l]=weight[k*m+l]/s*saturatedl[k];
				}
				for (int l=0;l<m;l++)
					ss[l]+=weight[k*m+l];
			}
			for (int l=0;l<m;l++){
				double s=ss[l];
				double r=std::min(saturatedr[l]/s,1.0);
				ss[l]=r;
			}
			std::vector<double> ss2(m,0);
			for (int k=0;k<n;k++){
				double s=0;
				for (int l=0;l<m;l++){
					weight[k*m+l]*=ss[l];
					s+=weight[k*m+l];
					ss2[l]+=weight[k*m+l];
				}
				saturatedl[k]=std::max(saturatedl[k]-s,0.0);
			}
			for (int k=0;k<n*m;k++)
				match[k]+=weight[k];
			for (int l=0;l<m;l++){
				saturatedr[l]=std::max(saturatedr[l]-ss2[l],0.0);
			}
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
	}
}

void matchcost_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * cost){
	for (int i=0;i<b;i++){
		double s=0;
		for (int j=0;j<n;j++)
			for (int k=0;k<m;k++){
				float x1=xyz1[j*3+0];
				float y1=xyz1[j*3+1];
				float z1=xyz1[j*3+2];
				float x2=xyz2[k*3+0];
				float y2=xyz2[k*3+1];
				float z2=xyz2[k*3+2];
				float d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1))*match[j*m+k];
				s+=d;
			}
		cost[0]=s;
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		cost+=1;
	}
}

void matchcostgrad_cpu(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad1,float * grad2){
	for (int i=0;i<b;i++){
		for (int j=0;j<n;j++)
			grad1[j*3+0]=0;
		for (int j=0;j<m;j++){
			float sx=0,sy=0,sz=0;
			for (int k=0;k<n;k++){
				float x2=xyz2[j*3+0];
				float y2=xyz2[j*3+1];
				float z2=xyz2[j*3+2];
				float x1=xyz1[k*3+0];
				float y1=xyz1[k*3+1];
				float z1=xyz1[k*3+2];
				float d=std::max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
				float dx=match[k*m+j]*((x2-x1)/d);
				float dy=match[k*m+j]*((y2-y1)/d);
				float dz=match[k*m+j]*((z2-z1)/d);
				grad1[k*3+0]-=dx;
				grad1[k*3+1]-=dy;
				grad1[k*3+2]-=dz;
				sx+=dx;
				sy+=dy;
				sz+=dz;
			}
			grad2[j*3+0]=sx;
			grad2[j*3+1]=sy;
			grad2[j*3+2]=sz;
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		grad1+=n*3;
		grad2+=m*3;
	}
}

// Mark: Chamfer distance cpu function

void nnsearch(
    const int b, const int n, const int m,
    const float* xyz1,
    const float* xyz2,
    float* dist,
    int* idx)
{
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < n; j++) {
            const float x1 = xyz1[(i*n+j)*3+0];
            const float y1 = xyz1[(i*n+j)*3+1];
            const float z1 = xyz1[(i*n+j)*3+2];
            double best = 0;
            int besti = 0;
            for (int k = 0; k < m; k++) {
                const float x2 = xyz2[(i*m+k)*3+0] - x1;
                const float y2 = xyz2[(i*m+k)*3+1] - y1;
                const float z2 = xyz2[(i*m+k)*3+2] - z1;
                const double d=x2*x2+y2*y2+z2*z2;
                if (k==0 || d < best){
                    best = d;
                    besti = k;
                }
            }
            dist[i*n+j] = best;
            idx[i*n+j] = besti;
        }
    }
}



//* batch Euclidean grid cpu
void varifold_kernel_cpu(int b,
	int n,
	int m,
	const float * xyz1,
	const float * xyz2, 
	const float * nor1, 
	const float * nor2,
	float * match){
	for (int i=0;i<b;i++){
		std::vector<double> weight(n*m);
		for (int k=0;k<n;k++){
			double x1 =xyz1[k*3+0];
			double y1 =xyz1[k*3+1];
			double z1 =xyz1[k*3+2];
			double nx1=nor1[k*3+0];
			double ny1=nor1[k*3+1];
			double nz1=nor1[k*3+2];
			for (int l=0;l<m;l++){
				double x2 =xyz2[l*3+0];
				double y2 =xyz2[l*3+1];
				double z2 =xyz2[l*3+2];
				double nx2=nor2[l*3+0];
				double ny2=nor2[l*3+1];
				double nz2=nor2[l*3+2];
				match[k*m+l]=std::max(expf(-1.0*((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)))*powf(nx1*nx2+ny1*ny2+nz1*nz2,2),1e-10f);
			}
		}
		xyz1+=n*3;
		xyz2+=m*3;
		nor1+=n*3;
		nor2+=m*3;
		match+=n*m;
	}
}

std::vector<at::Tensor>  emd_distance_forward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2){

	// Allocate necessary data structures
	at::Tensor match = at::zeros({xyz1.size(0), xyz1.size(1), xyz2.size(1)}, 
		xyz1.options());
	at::Tensor cost = at::zeros({xyz1.size(0)}, xyz1.options());
	// Find the approximate matching 

	approxmatch_cpu(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1),
    xyz1.data<float>(),
    xyz2.data<float>(),
	match.data<float>());

	// Compute the matching cost
	matchcost_cpu(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1),
    xyz1.data<float>(),
    xyz2.data<float>(),
	match.data<float>(),
	cost.data<float>());

    // return output
	return {cost, match};
}

std::vector<at::Tensor>  emd_distance_backward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2,
    const at::Tensor match){

	// Allocate necessary data structures
	at::Tensor gradxyz1 = at::zeros_like(xyz1);
	at::Tensor gradxyz2 = at::zeros_like(xyz2);

    matchcostgrad_cpu(xyz1.size(0), 
    xyz1.size(1), 
    xyz2.size(1), 
    xyz1.data<float>(),
    xyz2.data<float>(),
    match.data<float>(), 
    gradxyz1.data<float>(), 
    gradxyz2.data<float>());    

    // return gradients
    return {gradxyz1, gradxyz2};

}

void chamfer_distance_forward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    const at::Tensor dist1, 
    const at::Tensor dist2, 
    const at::Tensor idx1, 
    const at::Tensor idx2) 
{
    const int batchsize = xyz1.size(0);
    const int n = xyz1.size(1);
    const int m = xyz2.size(1);

    const float* xyz1_data = xyz1.data<float>();
    const float* xyz2_data = xyz2.data<float>();
    float* dist1_data = dist1.data<float>();
    float* dist2_data = dist2.data<float>();
    int* idx1_data = idx1.data<int>();
    int* idx2_data = idx2.data<int>();

    nnsearch(batchsize, n, m, xyz1_data, xyz2_data, dist1_data, idx1_data);
    nnsearch(batchsize, m, n, xyz2_data, xyz1_data, dist2_data, idx2_data);
}


void chamfer_distance_backward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    at::Tensor gradxyz1, 
    at::Tensor gradxyz2, 
    at::Tensor graddist1, 
    at::Tensor graddist2, 
    at::Tensor idx1, 
    at::Tensor idx2) 
{
    const int b = xyz1.size(0);
    const int n = xyz1.size(1);
    const int m = xyz2.size(1);

    const float* xyz1_data = xyz1.data<float>();
    const float* xyz2_data = xyz2.data<float>();
    float* gradxyz1_data = gradxyz1.data<float>();
    float* gradxyz2_data = gradxyz2.data<float>();
    float* graddist1_data = graddist1.data<float>();
    float* graddist2_data = graddist2.data<float>();
    const int* idx1_data = idx1.data<int>();
    const int* idx2_data = idx2.data<int>();

    for (int i = 0; i < b*n*3; i++)
        gradxyz1_data[i] = 0;
    for (int i = 0; i < b*m*3; i++)
        gradxyz2_data[i] = 0;
    for (int i = 0;i < b; i++) {
        for (int j = 0; j < n; j++) {
            const float x1 = xyz1_data[(i*n+j)*3+0];
            const float y1 = xyz1_data[(i*n+j)*3+1];
            const float z1 = xyz1_data[(i*n+j)*3+2];
            const int j2 = idx1_data[i*n+j];

            const float x2 = xyz2_data[(i*m+j2)*3+0];
            const float y2 = xyz2_data[(i*m+j2)*3+1];
            const float z2 = xyz2_data[(i*m+j2)*3+2];
            const float g = graddist1_data[i*n+j]*2;

            gradxyz1_data[(i*n+j)*3+0] += g*(x1-x2);
            gradxyz1_data[(i*n+j)*3+1] += g*(y1-y2);
            gradxyz1_data[(i*n+j)*3+2] += g*(z1-z2);
            gradxyz2_data[(i*m+j2)*3+0] -= (g*(x1-x2));
            gradxyz2_data[(i*m+j2)*3+1] -= (g*(y1-y2));
            gradxyz2_data[(i*m+j2)*3+2] -= (g*(z1-z2));
        }
        for (int j = 0; j < m; j++) {
            const float x1 = xyz2_data[(i*m+j)*3+0];
            const float y1 = xyz2_data[(i*m+j)*3+1];
            const float z1 = xyz2_data[(i*m+j)*3+2];
            const int j2 = idx2_data[i*m+j];
            const float x2 = xyz1_data[(i*n+j2)*3+0];
            const float y2 = xyz1_data[(i*n+j2)*3+1];
            const float z2 = xyz1_data[(i*n+j2)*3+2];
            const float g = graddist2_data[i*m+j]*2;
            gradxyz2_data[(i*m+j)*3+0] += g*(x1-x2);
            gradxyz2_data[(i*m+j)*3+1] += g*(y1-y2);
            gradxyz2_data[(i*m+j)*3+2] += g*(z1-z2);
            gradxyz1_data[(i*n+j2)*3+0] -= (g*(x1-x2));
            gradxyz1_data[(i*n+j2)*3+1] -= (g*(y1-y2));
            gradxyz1_data[(i*n+j2)*3+2] -= (g*(z1-z2));
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cd_forward", &chamfer_distance_forward, "Chamfer Distance forward");
    m.def("cd_forward_cuda", &chamfer_distance_forward_cuda, "ChamferDistance forward (CUDA)");
    m.def("cd_backward", &chamfer_distance_backward, "Chamfer Distance backward");
    m.def("cd_backward_cuda", &chamfer_distance_backward_cuda, "ChamferDistance backward (CUDA)");
    m.def("emd_distance_forward", &emd_distance_forward, "Wasserstein (Earth Mover's) Distance forward");
    m.def("emd_distance_forward_cuda", &emd_distance_forward_cuda, "Wasserstein (Earth Mover's) Distance forward (CUDA)");
    m.def("emd_distance_backward", &emd_distance_backward, "Wasserstein (Earth Mover's) Distance backward");
    m.def("emd_distance_backward_cuda", &emd_distance_backward_cuda, "Wasserstein (Earth Mover's) Distance backward (CUDA)");
}
