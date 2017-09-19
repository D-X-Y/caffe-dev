// ------------------------------------------------------------------
// Xuanyi . Refer to Dong Jian
// 2016/03/31
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_PROPOSAL_TARGET_LAYER_HPP_
#define CAFFE_FRCNN_PROPOSAL_TARGET_LAYER_HPP_

#include <vector>
#include <boost/lexical_cast.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"

namespace caffe {

namespace Frcnn {

/*************************************************
FRCNN_PROPOSAL_TARGET
Assign object detection proposals to ground-truth targets. Produces proposal
classification labels and bounding-box regression targets.
bottom: 'rpn_rois'
bottom: 'gt_boxes'
top: 'rois'
top: 'labels'
top: 'bbox_targets'
top: 'bbox_inside_weights'
top: 'bbox_outside_weights'
**************************************************/
template <typename Dtype>
class FrcnnProposalTargetLayer : public Layer<Dtype> {
 public:
  explicit FrcnnProposalTargetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){}

  virtual inline const char* type() const { return "FrcnnProposalTarget"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 5; }
  virtual inline int MaxTopBlobs() const { return 5; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // For FRCNN
  void _sample_rois(const vector<Point4f<Dtype> > &all_rois, const vector<Point4f<Dtype> > &gt_boxes,
        const vector<int> &gt_label, const int fg_rois_per_image, const int rois_per_image, vector<int> &labels,
        vector<Point4f<Dtype> > &rois, vector<vector<Point4f<Dtype> > > &bbox_targets, vector<vector<Point4f<Dtype> > > &bbox_inside_weights);
  int config_n_classes_;
  bool agnostic;
  shared_ptr<Caffe::RNG> rng_;
  int _count_;
  int _fg_num_;
  int _bg_num_;
 
};

}  // namespace frcnn

}  // namespace caffe

#endif  // CAFFE_FRCNN_PROPOSAL_TARGET_LAYER_HPP_
