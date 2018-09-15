// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------
#include <iomanip>

#include "caffe/FRCNN/frcnn_proposal_layer.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"  
#include "caffe/FRCNN/util/frcnn_gpu_nms.hpp"  

namespace caffe {

namespace Frcnn {

using std::vector;

__global__ void GetIndex(const int n,int *indices){
  CUDA_KERNEL_LOOP(index , n){
    indices[index] = index;
  }
}

template <typename Dtype>
__global__ void BBoxTransformInv(const int nthreads, const Dtype* const bottom_rpn_bbox,
                                 const int height, const int width, const int feat_stride,
                                 const int im_height, const int im_width,
                                 const int* sorted_indices, const float* anchors,
                                 float* const transform_bbox) {
  CUDA_KERNEL_LOOP(index , nthreads) {
    const int score_idx = sorted_indices[index];
    const int i = score_idx % width;  // width
    const int j = (score_idx % (width * height)) / width;  // height
    const int k = score_idx / (width * height); // channel
    float *box = transform_bbox + index * 4;
    box[0] = anchors[k * 4 + 0] + i * feat_stride;
    box[1] = anchors[k * 4 + 1] + j * feat_stride;
    box[2] = anchors[k * 4 + 2] + i * feat_stride;
    box[3] = anchors[k * 4 + 3] + j * feat_stride;
    const Dtype det[4] = { bottom_rpn_bbox[(k * 4 + 0) * height * width + j * width + i],
                           bottom_rpn_bbox[(k * 4 + 1) * height * width + j * width + i],
                           bottom_rpn_bbox[(k * 4 + 2) * height * width + j * width + i],
                           bottom_rpn_bbox[(k * 4 + 3) * height * width + j * width + i] };
    float src_w = box[2] - box[0] + 1;
    float src_h = box[3] - box[1] + 1;
    float src_ctr_x = box[0] + 0.5 * src_w;
    float src_ctr_y = box[1] + 0.5 * src_h;
    float pred_ctr_x = det[0] * src_w + src_ctr_x;
    float pred_ctr_y = det[1] * src_h + src_ctr_y;
    float pred_w = exp(det[2]) * src_w;
    float pred_h = exp(det[3]) * src_h;
    box[0] = pred_ctr_x - 0.5 * pred_w;
    box[1] = pred_ctr_y - 0.5 * pred_h;
    box[2] = pred_ctr_x + 0.5 * pred_w;
    box[3] = pred_ctr_y + 0.5 * pred_h;
    box[0] = max(0.0f, min(box[0], im_width - 1.0));
    box[1] = max(0.0f, min(box[1], im_height - 1.0));
    box[2] = max(0.0f, min(box[2], im_width - 1.0));
    box[3] = max(0.0f, min(box[3], im_height - 1.0));
  }
}

__global__ void SelectBox(const int nthreads, const float *box, float min_size,
                          int *flags) {
  CUDA_KERNEL_LOOP(index , nthreads) {
    if ((box[index * 4 + 2] - box[index * 4 + 0] < min_size) ||
        (box[index * 4 + 3] - box[index * 4 + 1] < min_size)) {
      flags[index] = 0;
    } else {
      flags[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SelectBoxByIndices(const int nthreads, const float *in_box, int *selected_indices,
                          float *out_box, const Dtype *in_score, Dtype *out_score) {
  CUDA_KERNEL_LOOP(index , nthreads) {
    if ((index == 0 && selected_indices[index] == 1) ||
        (index > 0 && selected_indices[index] == selected_indices[index - 1] + 1)) {
      out_box[(selected_indices[index] - 1) * 4 + 0] = in_box[index * 4 + 0];
      out_box[(selected_indices[index] - 1) * 4 + 1] = in_box[index * 4 + 1];
      out_box[(selected_indices[index] - 1) * 4 + 2] = in_box[index * 4 + 2];
      out_box[(selected_indices[index] - 1) * 4 + 3] = in_box[index * 4 + 3];
      if (in_score!=NULL && out_score!=NULL) {
        out_score[selected_indices[index] - 1] = in_score[index];
      }
    }
  }
}

template <typename Dtype>
__global__ void SelectBoxAftNMS(const int nthreads, const float *in_box, int *keep_indices,
                          Dtype *top_data, const Dtype *in_score, Dtype* top_score) {
  CUDA_KERNEL_LOOP(index , nthreads) {
    top_data[index * 5] = 0;
    int keep_idx = keep_indices[index];
    for (int j = 1; j < 5; ++j) {
      top_data[index * 5 + j] = in_box[keep_idx * 4 + j - 1];
    }
    if (top_score != NULL && in_score != NULL) {
      top_score[index] = in_score[keep_idx];
    }
  }
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  Forward_cpu(bottom, top);
  return ;
#if 0
  DLOG(ERROR) << "========== enter proposal layer";
  const Dtype *bottom_rpn_score = bottom[0]->gpu_data();
  const Dtype *bottom_rpn_bbox = bottom[1]->gpu_data();
  // bottom data comes from host memory
  Dtype bottom_im_info[3];
  CHECK_EQ(bottom[2]->count(), 3);
  CUDA_CHECK(cudaMemcpy(bottom_im_info, bottom[2]->gpu_data(), sizeof(Dtype) * 3, cudaMemcpyDeviceToHost));

  const int num = bottom[1]->num();
  const int channes = bottom[1]->channels();
  const int height = bottom[1]->height();
  const int width = bottom[1]->width();
  CHECK(num == 1) << "only single item batches are supported";
  CHECK(channes % 4 == 0) << "rpn bbox pred channels should be divided by 4";

  const float im_height = bottom_im_info[0];
  const float im_width = bottom_im_info[1];

  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float rpn_nms_thresh;
  int rpn_min_size;
  if (this->phase_ == TRAIN) {
    rpn_pre_nms_top_n = FrcnnParam::rpn_pre_nms_top_n;
    rpn_post_nms_top_n = FrcnnParam::rpn_post_nms_top_n;
    rpn_nms_thresh = FrcnnParam::rpn_nms_thresh;
    rpn_min_size = FrcnnParam::rpn_min_size;
  } else {
    rpn_pre_nms_top_n = FrcnnParam::test_rpn_pre_nms_top_n;
    rpn_post_nms_top_n = FrcnnParam::test_rpn_post_nms_top_n;
    rpn_nms_thresh = FrcnnParam::test_rpn_nms_thresh;
    rpn_min_size = FrcnnParam::test_rpn_min_size;
  }
  LOG_IF(ERROR, rpn_pre_nms_top_n <= 0 ) << "rpn_pre_nms_top_n : " << rpn_pre_nms_top_n;
  LOG_IF(ERROR, rpn_post_nms_top_n <= 0 ) << "rpn_post_nms_top_n : " << rpn_post_nms_top_n;
  if (rpn_pre_nms_top_n <= 0 || rpn_post_nms_top_n <= 0 ) return;

  const int config_n_anchors = FrcnnParam::anchors.size() / 4;
  const int total_anchor_num = config_n_anchors * height * width;

  //Step 1. -------------------------------Sort the rpn result----------------------
  // the first half of rpn_score is the bg score
  // Note that the sorting operator will change the order fg_scores (bottom_rpn_score)
  Dtype *fg_scores = (Dtype*)(&bottom_rpn_score[total_anchor_num]);

  Dtype *sorted_scores = NULL;
  CUDA_CHECK(cudaMalloc((void**)&sorted_scores, sizeof(Dtype) * total_anchor_num));
  cub::DoubleBuffer<Dtype> d_keys(fg_scores, sorted_scores);

  int *indices = NULL;
  CUDA_CHECK(cudaMalloc((void**)&indices, sizeof(int) * total_anchor_num));
  GetIndex<<<caffe::CAFFE_GET_BLOCKS(total_anchor_num), caffe::CAFFE_CUDA_NUM_THREADS>>>(
      total_anchor_num, indices);
  cudaDeviceSynchronize();

  int *sorted_indices = NULL;
  CUDA_CHECK(cudaMalloc((void**)&sorted_indices, sizeof(int) * total_anchor_num));
  cub::DoubleBuffer<int> d_values(indices, sorted_indices);

  void *sort_temp_storage_ = NULL;
  size_t sort_temp_storage_bytes_ = 0;
  // calculate the temp_storage_bytes
  cub::DeviceRadixSort::SortPairsDescending(sort_temp_storage_, sort_temp_storage_bytes_,
                                             d_keys, d_values, total_anchor_num);
  DLOG(ERROR) << "sort_temp_storage_bytes_ : " << sort_temp_storage_bytes_;
  CUDA_CHECK(cudaMalloc(&sort_temp_storage_, sort_temp_storage_bytes_));
  // sorting
  cub::DeviceRadixSort::SortPairsDescending(sort_temp_storage_, sort_temp_storage_bytes_,
                                            d_keys, d_values, total_anchor_num);
  cudaDeviceSynchronize();

  //Step 2. ---------------------------bbox transform----------------------------
  const int retained_anchor_num = std::min(total_anchor_num, rpn_pre_nms_top_n);
  // float *transform_bbox = NULL;
  // CUDA_CHECK(cudaMalloc(&transform_bbox, sizeof(float) * retained_anchor_num * 4));
  BBoxTransformInv<Dtype><<<caffe::CAFFE_GET_BLOCKS(retained_anchor_num), caffe::CAFFE_CUDA_NUM_THREADS>>>(
      retained_anchor_num, bottom_rpn_bbox, height, width, FrcnnParam::feat_stride,
      im_height, im_width, sorted_indices, anchors_, transform_bbox_);
  cudaDeviceSynchronize();

  //Step 3. -------------------------filter out small box-----------------------

  // select the box larger than min size
  // int *selected_flags = NULL;
  // CUDA_CHECK(cudaMalloc(&selected_flags, sizeof(int) * retained_anchor_num));
  SelectBox<<<caffe::CAFFE_GET_BLOCKS(retained_anchor_num), caffe::CAFFE_CUDA_NUM_THREADS>>>(
      retained_anchor_num, transform_bbox_, bottom_im_info[2] * rpn_min_size, selected_flags_);
  cudaDeviceSynchronize();

  // cumulative sum up the flags to get the copy index
  int *selected_indices_ = NULL;
  CUDA_CHECK(cudaMalloc((void**)&selected_indices_, sizeof(int) * retained_anchor_num));
  void *cumsum_temp_storage_ = NULL;
  size_t cumsum_temp_storage_bytes_ = 0;
  cub::DeviceScan::InclusiveSum(cumsum_temp_storage_, cumsum_temp_storage_bytes_,
                                 selected_flags_, selected_indices_, retained_anchor_num);
  DLOG(ERROR) << "cumsum_temp_storage_bytes : " << cumsum_temp_storage_bytes_;
  CUDA_CHECK(cudaMalloc(&cumsum_temp_storage_, cumsum_temp_storage_bytes_));
  cub::DeviceScan::InclusiveSum(sort_temp_storage_, cumsum_temp_storage_bytes_,
                                selected_flags_, selected_indices_, retained_anchor_num);

  // CUDA_CHECK(cudaFree(cumsum_temp_storage));

  int selected_num = -1;
  cudaMemcpy(&selected_num, &selected_indices_[retained_anchor_num - 1], sizeof(int), cudaMemcpyDeviceToHost);
  CHECK_GT(selected_num, 0);

  Dtype *bbox_score_ = NULL;
  if (top.size() == 2) CUDA_CHECK(cudaMalloc(&bbox_score_, sizeof(Dtype) * retained_anchor_num));
  SelectBoxByIndices<<<caffe::CAFFE_GET_BLOCKS(selected_num), caffe::CAFFE_CUDA_NUM_THREADS>>>(
      selected_num, transform_bbox_, selected_indices_, transform_bbox_, sorted_scores, bbox_score_);
  cudaDeviceSynchronize();
  
  //Step 4. -----------------------------apply nms-------------------------------
  DLOG(ERROR) << "========== apply nms with rpn_nms_thresh : " << rpn_nms_thresh;
  vector<int> keep_indices(selected_num);
  int keep_num = -1;
  gpu_nms(&keep_indices[0], &keep_num, transform_bbox_, selected_num, 4, rpn_nms_thresh);
  DLOG(ERROR) << "rpn num after gpu nms: " << keep_num;

  keep_num = std::min(keep_num, rpn_post_nms_top_n);
  DLOG(ERROR) << "========== copy to top";
  cudaMemcpy(gpu_keep_indices_, &keep_indices[0], sizeof(int) * keep_num, cudaMemcpyHostToDevice);

  top[0]->Reshape(keep_num, 5, 1, 1);
  Dtype *top_data = top[0]->mutable_gpu_data();
  Dtype *top_score = NULL;
  if (top.size() == 2) {
    top[1]->Reshape(keep_num, 1, 1, 1);
    top_score = top[1]->mutable_gpu_data();
  }
  SelectBoxAftNMS<<<caffe::CAFFE_GET_BLOCKS(keep_num), caffe::CAFFE_CUDA_NUM_THREADS>>>(
      keep_num, transform_bbox_, gpu_keep_indices_, top_data, bbox_score_, top_score);

  DLOG(ERROR) << "========== exit proposal layer";
  ////////////////////////////////////
  // do not forget to free the malloc memory
  CUDA_CHECK(cudaFree(sorted_scores));
  CUDA_CHECK(cudaFree(indices));
  CUDA_CHECK(cudaFree(sorted_indices));
  CUDA_CHECK(cudaFree(sort_temp_storage_));
  CUDA_CHECK(cudaFree(cumsum_temp_storage_));
  CUDA_CHECK(cudaFree(selected_indices_));
  if (bbox_score_!=NULL)  CUDA_CHECK(cudaFree(bbox_score_));

#endif

}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FrcnnProposalLayer);

} // namespace frcnn

} // namespace caffe
