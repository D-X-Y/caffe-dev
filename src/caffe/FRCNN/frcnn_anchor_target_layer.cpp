// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <iomanip>
#include "caffe/FRCNN/frcnn_anchor_target_layer.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"

namespace caffe {

namespace Frcnn {

using std::vector;

template <typename Dtype>
void FrcnnAnchorTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                               const vector<Blob<Dtype> *> &top) {
  anchors_ = FrcnnParam::anchors;
  config_n_anchors_ = FrcnnParam::anchors.size() / 4;
  feat_stride_ = FrcnnParam::feat_stride;
  border_ = FrcnnParam::rpn_allowed_border;

  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  // labels
  top[0]->Reshape(1, 1, config_n_anchors_ * height, width);
  // bbox_targets
  top[1]->Reshape(1, config_n_anchors_ * 4, height, width);
  // bbox_inside_weights
  top[2]->Reshape(1, config_n_anchors_ * 4, height, width);
  // bbox_outside_weights
  top[3]->Reshape(1, config_n_anchors_ * 4, height, width);

  LOG(INFO) << "FrcnnAnchorTargetLayer : " << config_n_anchors_ << " anchors , " << feat_stride_ << " feat_stride , " << border_ << " allowed_border";
  LOG(INFO) << "FrcnnAnchorTargetLayer : FrcnnParam::rpn_negative_overlap : " << FrcnnParam::rpn_negative_overlap;
  LOG(INFO) << "FrcnnAnchorTargetLayer : FrcnnParam::rpn_positive_overlap : " << FrcnnParam::rpn_positive_overlap;
  LOG(INFO) << "FrcnnAnchorTargetLayer : FrcnnParam::rpn_bbox_inside_weights : " << FrcnnParam::rpn_bbox_inside_weights[0] << ", " << FrcnnParam::rpn_bbox_inside_weights[1] << ", " << FrcnnParam::rpn_bbox_inside_weights[2] << ", " << FrcnnParam::rpn_bbox_inside_weights[3];
  LOG(INFO) << "FrcnnAnchorTargetLayer : " << this->layer_param_.name() << " SetUp";

  //DEBUG SET VALUE
  _squared_sum =  _sum = Point4f<Dtype>(0, 0, 0, 0);
  _counts = 0;
  _count = 0;
  _fg_sum = _bg_sum = 0;

}

template <typename Dtype>
void FrcnnAnchorTargetLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  DLOG(ERROR) << "========== enter anchor target layer {} " << bottom[2]->cpu_data()[0] << ", " << bottom[2]->cpu_data()[1] << " , scales : " << bottom[2]->cpu_data()[2] << " {} height : " << bottom[0]->height() << " width : " << bottom[0]->width();

  //const Dtype *bottom_gt_bbox = bottom[1]->cpu_data();
  const Dtype *bottom_im_info = bottom[2]->cpu_data();

  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  CHECK(num == 1) << "only single item batches are supported";

  const Dtype im_height = bottom_im_info[0];
  const Dtype im_width = bottom_im_info[1];

  DLOG(ERROR) << "========== get gt boxes : " << bottom[1]->num();
  // gt boxes (x1, y1, x2, y2, label)
  vector<Point4f<Dtype> > gt_boxes;
  for (int i = 0; i < bottom[1]->num(); i++) {
    //const Dtype * base_address = &bottom_gt_bbox[(i * bottom[1]->offset(1))];
    //gt_boxes.push_back(Point4f<Dtype>(base_address[0], base_address[1], base_address[2],base_address[3]));
    gt_boxes.push_back(Point4f<Dtype>(
        bottom[1]->data_at(i, 0, 0, 0),
        bottom[1]->data_at(i, 1, 0, 0),
        bottom[1]->data_at(i, 2, 0, 0),
        bottom[1]->data_at(i, 3, 0, 0)));
    CHECK(gt_boxes[i][0]>=0 && gt_boxes[i][1]>=0);
    CHECK(gt_boxes[i][2]<=im_width && gt_boxes[i][3]<=im_height);
    //DLOG(ERROR) << "============= " << i << "  : " << gt_boxes[i][0] << ", " << gt_boxes[i][1] << ", " << gt_boxes[i][2] << ", " << gt_boxes[i][3];
  }


  // Generate anchors
  DLOG(ERROR) << "========== generate anchors";
  vector<int> inds_inside;
  vector<Point4f<Dtype> > anchors;

  Dtype bounds[4] = {-border_, -border_, im_width + border_, im_height + border_};

  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (int k = 0; k < config_n_anchors_; k++) {
        float x1 = w * feat_stride_ + anchors_[k * 4 + 0];  // shift_x[i][j];
        float y1 = h * feat_stride_ + anchors_[k * 4 + 1];  // shift_y[i][j];
        float x2 = w * feat_stride_ + anchors_[k * 4 + 2];  // shift_x[i][j];
        float y2 = h * feat_stride_ + anchors_[k * 4 + 3];  // shift_y[i][j];
        if (x1 >= bounds[0] && y1 >= bounds[1] && x2 < bounds[2] &&
            y2 < bounds[3]) {
          inds_inside.push_back((h * width + w) * config_n_anchors_ + k);
          anchors.push_back(Point4f<Dtype>(x1, y1, x2, y2));
        }
      }
    }
  }

  DLOG(ERROR) << "========= total_anchors  : " << config_n_anchors_ * height * width;
  DLOG(ERROR) << "========= inside_anchors : " << inds_inside.size();

  const int n_anchors = anchors.size();

  // label: 1 is positive, 0 is negative, -1 is dont care
  vector<int> labels(n_anchors, -1);

  vector<Dtype> max_overlaps(anchors.size(), -1);
  vector<int> argmax_overlaps(anchors.size(), -1);
  vector<Dtype> gt_max_overlaps(gt_boxes.size(), -1);
  vector<int> gt_argmax_overlaps(gt_boxes.size(), -1);

  vector<vector<Dtype> > ious = get_ious(anchors, gt_boxes);

  for (int ia = 0; ia < n_anchors; ia++) {
    for (size_t igt = 0; igt < gt_boxes.size(); igt++) {
      if (ious[ia][igt] > max_overlaps[ia]) {
        max_overlaps[ia] = ious[ia][igt];
        argmax_overlaps[ia] = igt;
      }
      if (ious[ia][igt] > gt_max_overlaps[igt]) {
        gt_max_overlaps[igt] = ious[ia][igt];
        gt_argmax_overlaps[igt] = ia;
      }
    }
  }

/*
  for (size_t i = 0; i < gt_max_overlaps.size(); i ++) {
    if (gt_max_overlaps[i] < FrcnnParam::rpn_positive_overlap) {
      DLOG(ERROR) << gt_max_overlaps[i] << ":gt--" << gt_boxes[i].to_string()
          << "  anchor--" << anchors[gt_argmax_overlaps[i]].to_string();
    }
  }
*/

  if (FrcnnParam::rpn_clobber_positives==false) {
    //assign bg labels first so that positive labels can clobber them
    for (int i = 0; i < max_overlaps.size(); ++i) {
      if (max_overlaps[i] < FrcnnParam::rpn_negative_overlap)
        labels[i] = 0;
    }
  }
  
  // fg label: for each gt, anchor with highest overlap
  int debug_for_highest_over = 0;
  for (int j = 0; j < gt_max_overlaps.size(); j ++) {
    for (int i = 0; i < max_overlaps.size(); ++i) {
      if ( std::abs(gt_max_overlaps[j] - ious[i][j]) <= FrcnnParam::eps ) {
        labels[i] = 1;
        debug_for_highest_over ++;
      }
    }
  }

  // fg label: above threshold IOU
  for (int i = 0; i < max_overlaps.size(); ++i) {
    if (max_overlaps[i] >= FrcnnParam::rpn_positive_overlap) {
      labels[i] = 1;
    }
  }

  if (FrcnnParam::rpn_clobber_positives) {
    // assign bg labels last so that negative labels can clobber positives
    for (int i = 0; i < max_overlaps.size(); ++i) {
      if (max_overlaps[i] < FrcnnParam::rpn_negative_overlap)
        labels[i] = 0;
    }
  }

  DLOG(ERROR) << "label == 1  : " << std::count(labels.begin(), labels.end(), 1);
  DLOG(ERROR) << "label == 0  : " << std::count(labels.begin(), labels.end(), 0);
  DLOG(ERROR) << "label == -1 : " << std::count(labels.begin(), labels.end(),-1);
  DLOG(ERROR) << "debug_for_highest_over : " << debug_for_highest_over;

  // subsample positive labels if we have too many
  int num_fg = float(FrcnnParam::rpn_fg_fraction) * FrcnnParam::rpn_batchsize;
  const int fg_inds_size = std::count(labels.begin(), labels.end(), 1);
  DLOG(ERROR) << "========== supress_positive labels";
  if (fg_inds_size > num_fg) {
    vector<int> fg_inds ;
    for (size_t index = 0; index < labels.size(); index++ ) 
      if (labels[index] == 1) fg_inds.push_back(index);

    std::set<int> ind_set;
    while (ind_set.size() < fg_inds.size() - num_fg) {
      int tmp_idx = caffe::caffe_rng_rand() % fg_inds.size();
      ind_set.insert(fg_inds[tmp_idx]);
    }
    for (std::set<int>::iterator it = ind_set.begin(); it != ind_set.end(); it++) {
      labels[*it] = -1;
    }
    
  }

  DLOG(ERROR) << "========== supress negative labels";
  // subsample negative labels if we have too many
  int num_bg = FrcnnParam::rpn_batchsize - std::count(labels.begin(), labels.end(), 1);
  const int bg_inds_size = std::count(labels.begin(), labels.end(), 0);
  if (bg_inds_size > num_bg) {
    vector<int> bg_inds ;
    for (size_t index = 0; index < labels.size(); index++ )
      if (labels[index] == 0) bg_inds.push_back(index);

    std::set<int> ind_set;
    while (ind_set.size() < bg_inds.size() - num_bg) {
      int tmp_idx = caffe::caffe_rng_rand() % bg_inds.size();
      ind_set.insert(bg_inds[tmp_idx]);
    }
    for (std::set<int>::iterator it = ind_set.begin(); it != ind_set.end(); it++) {
      labels[*it] = -1;
    }
  }

  DLOG(ERROR) << "label == 1  : " << std::count(labels.begin(), labels.end(), 1);
  DLOG(ERROR) << "label == 0  : " << std::count(labels.begin(), labels.end(), 0);
  DLOG(ERROR) << "label == -1 : " << std::count(labels.begin(), labels.end(),-1);

  DLOG(ERROR) << "========== transfer bbox";
  vector<Point4f<Dtype> > bbox_targets;
  Point4f<Dtype> bbox_targets_data;

  bool rpn_normalize_targets = FrcnnParam::rpn_normalize_targets;
  Point4f<Dtype> rpn_normalize_means = FrcnnParam::rpn_normalize_means;
  Point4f<Dtype> rpn_normalize_stds = FrcnnParam::rpn_normalize_stds;
  if (!rpn_normalize_targets){
    Point4f<Dtype> rpn_normalize_means = {0.0 , 0.0 , 0.0 , 0.0};
    Point4f<Dtype> rpn_normalize_stds = {1.0, 1.0, 1.0, 1.0};}

  for (int i =0; i < argmax_overlaps.size(); i++) {
    if (argmax_overlaps[i] < 0 )
      bbox_targets.push_back(Point4f<Dtype>());
    else 
      bbox_targets_data = bbox_transform(anchors[i], gt_boxes[argmax_overlaps[i]] );
      for (int j = 0; j < 4; j ++) {
        bbox_targets_data[j] = (bbox_targets_data[j]-rpn_normalize_means[j]) / rpn_normalize_stds[j];
      }
      bbox_targets.push_back(bbox_targets_data);
  }

  vector<Point4f<Dtype> > bbox_inside_weights(n_anchors, Point4f<Dtype>());
  for (int i = 0; i < n_anchors; i++) {
    if (labels[i] == 1) {
      bbox_inside_weights[i].Point[0] = FrcnnParam::rpn_bbox_inside_weights[0];
      bbox_inside_weights[i].Point[1] = FrcnnParam::rpn_bbox_inside_weights[1];
      bbox_inside_weights[i].Point[2] = FrcnnParam::rpn_bbox_inside_weights[2];
      bbox_inside_weights[i].Point[3] = FrcnnParam::rpn_bbox_inside_weights[3];
    } else {
      CHECK_EQ(bbox_inside_weights[i].Point[0], 0);
      CHECK_EQ(bbox_inside_weights[i].Point[1], 0);
      CHECK_EQ(bbox_inside_weights[i].Point[2], 0);
      CHECK_EQ(bbox_inside_weights[i].Point[3], 0);
    }
  }

  Dtype positive_weights, negative_weights;
  if (FrcnnParam::rpn_positive_weight < 0) {
    //uniform weighting of examples (given non-uniform sampling)
    int num_examples = labels.size() - std::count(labels.begin(), labels.end(), -1);
    positive_weights = Dtype(1) / num_examples;
    negative_weights = Dtype(1) / num_examples;
    CHECK_GT(num_examples, 0);
  } else {
    CHECK_LT(FrcnnParam::rpn_positive_weight, 1) << "ilegal rpn_positive_weight";
    CHECK_GT(FrcnnParam::rpn_positive_weight, 0) << "ilegal rpn_positive_weight";
    positive_weights = Dtype(FrcnnParam::rpn_positive_weight) /
            std::count(labels.begin(), labels.end(), 1);
    negative_weights = Dtype(1-FrcnnParam::rpn_positive_weight) /
            std::count(labels.begin(), labels.end(), 0);
  }
  DLOG(ERROR) << "positive_weights:" << positive_weights << std::endl;
  DLOG(ERROR) << "negative_weights:" << negative_weights << std::endl;

  vector<Point4f<Dtype> > bbox_outside_weights(n_anchors, Point4f<Dtype>());
  for (int i = 0; i < n_anchors; i++) {
    if (labels[i] == 1) {
      bbox_outside_weights[i].Point[0] = positive_weights;
      bbox_outside_weights[i].Point[1] = positive_weights;
      bbox_outside_weights[i].Point[2] = positive_weights;
      bbox_outside_weights[i].Point[3] = positive_weights;
    } else if (labels[i] == 0) {
      bbox_outside_weights[i].Point[0] = negative_weights;
      bbox_outside_weights[i].Point[1] = negative_weights;
      bbox_outside_weights[i].Point[2] = negative_weights;
      bbox_outside_weights[i].Point[3] = negative_weights;
    } else {
      CHECK_EQ(bbox_outside_weights[i].Point[0], 0);
      CHECK_EQ(bbox_outside_weights[i].Point[1], 0);
      CHECK_EQ(bbox_outside_weights[i].Point[2], 0);
      CHECK_EQ(bbox_outside_weights[i].Point[3], 0);
    }
  }

  Info_Stds_Means_AvePos(bbox_targets, labels);

  DLOG(ERROR) << "========== copy to top";
  // labels
  top[0]->Reshape(1, 1, config_n_anchors_ * height, width);
  // bbox_targets
  top[1]->Reshape(1, config_n_anchors_ * 4, height, width);
  // bbox_inside_weights
  top[2]->Reshape(1, config_n_anchors_ * 4, height, width);
  // bbox_outside_weights
  top[3]->Reshape(1, config_n_anchors_ * 4, height, width);
  
  Dtype* top_labels = top[0]->mutable_cpu_data();
  Dtype* top_bbox_targets = top[1]->mutable_cpu_data();
  Dtype* top_bbox_inside_weights = top[2]->mutable_cpu_data();
  Dtype* top_bbox_outside_weights = top[3]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(-1), top[0]->mutable_cpu_data());
  caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  caffe_set(top[2]->count(), Dtype(0), top[2]->mutable_cpu_data());
  caffe_set(top[3]->count(), Dtype(0), top[3]->mutable_cpu_data());
  

  CHECK_EQ(inds_inside.size(), bbox_targets.size());
  CHECK_EQ(inds_inside.size(), bbox_outside_weights.size());
  CHECK_EQ(bbox_inside_weights.size(), bbox_outside_weights.size());

  for (size_t index = 0; index < inds_inside.size(); index++) {
    const int _anchor = inds_inside[index] % config_n_anchors_;
    const int _height = (inds_inside[index] / config_n_anchors_) / width;
    const int _width  = (inds_inside[index] / config_n_anchors_) % width;
    top_labels[ top[0]->offset(0,0,_anchor*height+_height,_width) ] = labels[index];
    for (int cor = 0; cor < 4; cor++) {
      top_bbox_targets        [ top[1]->offset(0,_anchor*4+cor,_height,_width) ] = bbox_targets[index][cor];
      top_bbox_inside_weights [ top[2]->offset(0,_anchor*4+cor,_height,_width) ] = bbox_inside_weights[index][cor];
      top_bbox_outside_weights[ top[3]->offset(0,_anchor*4+cor,_height,_width) ] = bbox_outside_weights[index][cor];
    }
  }
  
}

template <typename Dtype>
void FrcnnAnchorTargetLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FrcnnAnchorTargetLayer);
#endif

INSTANTIATE_CLASS(FrcnnAnchorTargetLayer);
REGISTER_LAYER_CLASS(FrcnnAnchorTarget);

} // namespace frcnn

} // namespace caffe
