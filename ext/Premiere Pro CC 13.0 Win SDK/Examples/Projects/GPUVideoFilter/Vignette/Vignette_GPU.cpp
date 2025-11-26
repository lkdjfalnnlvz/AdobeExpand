/************************************************************************************/
/* The MIT License (MIT)															*/
/* Copyright (c) 2016 Bartlomiej Walczak											*/
/*																					*/
/* Permission is hereby granted, free of charge, to any person obtaining a copy		*/
/* of this software and associated documentation files (the "Software"), to deal	*/
/* in the Software without restriction, including without limitation the rights		*/
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell		*/
/* copies of the Software, and to permit persons to whom the Software is			*/
/* furnished to do so, subject to the following conditions:							*/
/*																					*/
/* The above copyright notice and this permission notice shall be included			*/
/* in all copies or substantial portions of the Software.							*/
/*																					*/
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS			*/
/* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,		*/
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		*/
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			*/
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING			*/
/* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS		*/
/* IN THE SOFTWARE.																	*/
/*																					*/
/************************************************************************************/

/*

	Quick Vignette is an effect plugin supporting all typical pixel formats
	and bit depths in After Effects and Premiere Pro, including CUDA/OpenCL 
	acceleration.
	
	For more information see fxphd course 
		SYS204 - Plugin development for Premiere Pro and After Effects
		https://www.fxphd.com/details/?idCourse=526
		
	Revision history:
		date			developer	version		notes
		2016 Apr 11		bwal		1.0			version to be included in the SDK
        2017 May 06     zal         1.0.1       fixed custom build steps for CUDA on Windows

*/
		
/* Vignette_GPU.h */

#include "Vignette.h"
#include "VignetteGPU.h"
#include "PrGPUFilterModule.h"
#include "PrSDKVideoSegmentProperties.h"

#if _WIN32
#include <CL/cl.h>
#else
#include <OpenCL/cl.h>
#endif
#include <cuda_runtime.h>
#include <math.h>

//  CUDA KERNEL 
//  * See Vignette.cu
extern void Vignette_CUDA (float *destBuf, int destPitch, int is16f, int width, int height, VigInfoGPU* viP);

//  OPENCL KERNEL include hack
static const char* kVignetteKernel =
#include "Vignette.cl"
;

static cl_kernel sKernelCache[4];

/*
**
*/
class fxphdVignette :
	public PrGPUFilterBase
{
public:
	prSuiteError InitializeCUDA ()
	{
		// Nothing to do here. CUDA Kernel statically linked

		return suiteError_NoError;
		}

	prSuiteError InitializeOpenCL ()
		{
		if (mDeviceIndex > sizeof(sKernelCache) / sizeof(cl_kernel))  	{			
			return suiteError_Fail;		// Exceeded max device count
		}

		mCommandQueue = (cl_command_queue)mDeviceInfo.outCommandQueueHandle;

		// Load and compile the kernel - a real plugin would cache binaries to disk
		mKernel = sKernelCache[mDeviceIndex];
		if (!mKernel)
		{
			cl_int result = CL_SUCCESS;
			size_t size = strlen(kVignetteKernel);
			cl_context context = (cl_context)mDeviceInfo.outContextHandle;
			cl_device_id device = (cl_device_id)mDeviceInfo.outDeviceHandle;
			cl_program program = clCreateProgramWithSource(context, 1, &kVignetteKernel, &size, &result);
			if (result != CL_SUCCESS)
			{
				return suiteError_Fail;
			}

			result = clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0);
			if (result != CL_SUCCESS)
			{
				return suiteError_Fail;
			}

			mKernel = clCreateKernel(program, "kVignetteCL", &result);
			if (result != CL_SUCCESS)
			{
				return suiteError_Fail;
			}

			sKernelCache[mDeviceIndex] = mKernel;
		}

		return suiteError_NoError;
	}


	virtual prSuiteError Initialize( PrGPUFilterInstance* ioInstanceData )
	{
		PrGPUFilterBase::Initialize(ioInstanceData);

		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA)	
			return InitializeCUDA();			

		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL)
			return InitializeOpenCL();			

		return suiteError_Fail;			// GPUDeviceFramework unknown
	}

	prSuiteError Render(
		const PrGPUFilterRenderParams* inRenderParams,
		const PPixHand* inFrames,
		csSDK_size_t inFrameCount,
		PPixHand* outFrame)
	{

		// read the parameters

		VigInfoGPU	viP;

		void* frameData = 0;
		mGPUDeviceSuite->GetGPUPPixData(*outFrame, &frameData);

		PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
		mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);

		prRect bounds = {};
		mPPixSuite->GetBounds(*outFrame, &bounds);
		int width = bounds.right - bounds.left;
		int height = bounds.bottom - bounds.top;

		// read params

		double featherF = GetParam(VIG_FEATHER, inRenderParams->inClipTime).mFloat64 / 100.0;
		featherF = (featherF == 1.0) ? 0.99999 : featherF;

		viP.amountF = static_cast<float>( GetParam(VIG_AMOUNT, inRenderParams->inClipTime).mFloat64 / 100.0 );
		double ellipse_aaF = GetParam(VIG_SIZE, inRenderParams->inClipTime).mFloat64 * width / 200.0;
		double ellipse_bbF = GetParam(VIG_SIZE, inRenderParams->inClipTime).mFloat64 * height / 200.0;
		ellipse_bbF = ellipse_aaF * GetParam(VIG_ROUNDNESS, inRenderParams->inClipTime).mFloat64 / 100.0 +
			ellipse_bbF * (100.0 - GetParam(VIG_ROUNDNESS, inRenderParams->inClipTime).mFloat64) / 100.0;

		viP.outer_aaF = static_cast<float>(ellipse_aaF * (1.0 + featherF));
		viP.outer_bbF = static_cast<float>(ellipse_bbF * (1.0 + featherF));

		viP.inner_aaF = static_cast<float>(ellipse_aaF * (1.0 - featherF));
		viP.inner_bbF = static_cast<float>(ellipse_bbF * (1.0 - featherF));

		viP.outer_abF = viP.outer_aaF * viP.outer_bbF;
		viP.outer_aaF *= viP.outer_aaF;
		viP.outer_bbF *= viP.outer_bbF;

		viP.inner_abF = viP.inner_aaF * viP.inner_bbF;
		viP.inner_aaF *= viP.inner_aaF;
		viP.inner_bbF *= viP.inner_bbF;
		viP.inner_aabbF = viP.inner_aaF * viP.inner_bbF;

		viP.x_t = static_cast<float>(GetParam(VIG_CENTER, inRenderParams->inClipTime).mPoint.x * width);
		viP.y_t = static_cast<float>(GetParam(VIG_CENTER, inRenderParams->inClipTime).mPoint.y * height);

		csSDK_int32 rowBytes = 0;
		mPPixSuite->GetRowBytes(*outFrame, &rowBytes);
		int is16f = pixelFormat != PrPixelFormat_GPU_BGRA_4444_32f;

		// Get dest data
		void* destFrameData = 0;
		csSDK_int32 destRowBytes = 0;
		mGPUDeviceSuite->GetGPUPPixData(*outFrame, &destFrameData);
		mPPixSuite->GetRowBytes(*outFrame, &destRowBytes);
		int destPitch = destRowBytes / GetGPUBytesPerPixel(pixelFormat);


		// Start CUDA or OpenCL specific code

		if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA) {
			
			// CUDA device pointers
			float* destBuffer = reinterpret_cast<float*> (destFrameData);	

			// Launch CUDA kernel
			Vignette_CUDA ( destBuffer, 
							destPitch,
							is16f, 
							width, 
							height, 
							&viP );
	
			if ( cudaPeekAtLastError() != cudaSuccess) 			
			{
				return suiteError_Fail;
			}

		} else {
			// OpenCL device pointers
			cl_mem destBuffer = reinterpret_cast<cl_mem>(destFrameData);

			// Set the arguments
			clSetKernelArg(mKernel, 0, sizeof(cl_mem), &destBuffer);
			clSetKernelArg(mKernel, 1, sizeof(int), &destPitch);
			clSetKernelArg(mKernel, 2, sizeof(int), &is16f);
			clSetKernelArg(mKernel, 3, sizeof(int), &width);
			clSetKernelArg(mKernel, 4, sizeof(int), &height);
			clSetKernelArg(mKernel, 5, sizeof(float), &viP.amountF);
			clSetKernelArg(mKernel, 6, sizeof(float), &viP.outer_aaF);
			clSetKernelArg(mKernel, 7, sizeof(float), &viP.outer_bbF);
			clSetKernelArg(mKernel, 8, sizeof(float), &viP.outer_abF);
			clSetKernelArg(mKernel, 9, sizeof(float), &viP.inner_aaF);
			clSetKernelArg(mKernel, 10, sizeof(float), &viP.inner_bbF);
			clSetKernelArg(mKernel, 11, sizeof(float), &viP.inner_aabbF);
			clSetKernelArg(mKernel, 12, sizeof(float), &viP.inner_abF);
			clSetKernelArg(mKernel, 13, sizeof(float), &viP.x_t);
			clSetKernelArg(mKernel, 14, sizeof(float), &viP.y_t);

			// Launch the kernel
			size_t threadBlock[2] = { 16, 16 };
			size_t grid[2] = { RoundUp(width, threadBlock[0]), RoundUp(height, threadBlock[1] )};

			cl_int result = clEnqueueNDRangeKernel(
				mCommandQueue,
				mKernel,
				2,
				0,
				grid,
				threadBlock,
				0,
				0,
				0);

			if ( result != CL_SUCCESS )	
				return suiteError_Fail;
		}
		return suiteError_NoError;
	}

private:
	// CUDA


	// OpenCL
	cl_command_queue mCommandQueue;
	cl_kernel mKernel;
};


DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<fxphdVignette>)
