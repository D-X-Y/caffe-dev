// ------------------------------------------------------------------
//Jiang Jia'nan . Refer to Dong Jian
// 2017/09/19
// ------------------------------------------------------------------
#ifndef CAFFE_FRCNN_PRARM_HPP_
#define CAFFE_FRCNN_PRARM_HPP_

#include <vector>
#include <string>

namespace caffe{

namespace Frcnn {

class FrcnnParam {
public:
  // ======================================== Train
  // Scales to use during training (can list multiple scales)
  // Each scale is the pixel size of an image's shortest side
  static std::vector<float> scales;
  static float max_size;
  static float batch_size;

  static float fg_fraction;
  static float fg_thresh;
  // Overlap threshold for a ROI to be considered background (class = 0
  // ifoverlap in [LO, HI))
  static float bg_thresh_hi;
  static float bg_thresh_lo;
  static bool use_flipped;

  // Train bounding-box regressors
  static bool bbox_reg;
  static float bbox_thresh;
  static std::string snapshot_infix;
  // Normalize the targets (subtract empirical mean, divide by empirical stddev)
  static bool bbox_normalize_targets;
  static float bbox_inside_weights[4];
  static float bbox_normalize_means[4];
  static float bbox_normalize_stds[4];

  static bool rpn_normalize_targets;
  static float rpn_normalize_means[4];
  static float rpn_normalize_stds[4];

  static bool agnostic;

  // RPN to detect objects
  static float rpn_positive_overlap;
  static float rpn_negative_overlap;
  // If an anchor statisfied by positive and negative conditions set to negative
  static bool rpn_clobber_positives;
  static float rpn_fg_fraction;
  static int rpn_batchsize;
  static float rpn_nms_thresh;
  static int rpn_pre_nms_top_n;
  static int rpn_post_nms_top_n;
  // Proposal height and width both need to be greater than RPN_MIN_SIZE (at
  // orig image scale)
  static float rpn_min_size;
  // Deprecated (outside weights)
  static float rpn_bbox_inside_weights[4];
  // Give the positive RPN examples weight of p * 1 / {num positives}
  // and give negatives a weight of (1 - p)
  // Set to -1.0 to use uniform example weighting
  static float rpn_positive_weight;
  // allowed_border, when compute anchors targets, extend the border_
  static float rpn_allowed_border;

  // ======================================== Test
  static std::vector<float> test_scales;
  static float test_max_size;
  static float test_nms;

  static bool test_bbox_reg;
  // RPN to detect objects
  static float test_rpn_nms_thresh;
  static int test_rpn_pre_nms_top_n;
  static int test_rpn_post_nms_top_n;
  // Proposal height and width both need to be greater than RPN_MIN_SIZE (at
  // orig image scale)
  static float test_rpn_min_size;

  // ========================================
  // Means PIXEL
  static float pixel_means[3]; // BGR
  static int rng_seed;
  static float eps;
  static float inf;

  // ========================================
  static int feat_stride;
  static std::vector<float> anchors;
  static float test_score_thresh;
  static int n_classes;
  static int iter_test;
  // ========================================
  static void load_param(const std::string default_config_path);
  static void print_param();
};

}  // namespace detection

}

#endif // CAFFE_FRCNN_PRARM_HPP_
