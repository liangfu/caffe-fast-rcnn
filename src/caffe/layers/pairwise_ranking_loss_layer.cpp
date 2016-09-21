#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pairwise_ranking_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void PairwiseRankingLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* pos_dst = bottom[0]->cpu_data();
  const Dtype* neg_dst = bottom[1]->cpu_data();
  Dtype* per_triplet_loss = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();

  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = 0;
  for (int i=0; i<count; ++i) {
    per_triplet_loss[i] = std::max(Dtype(0),
        this->layer_param_.pairwise_ranking_loss_param().margin() + pos_dst[i] -
        neg_dst[i]);
    loss[0] += per_triplet_loss[i];
  }
  loss[0] /= count;
}

template <typename Dtype>
void PairwiseRankingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    bool discard_hard_samples = this->layer_param_.pairwise_ranking_loss_param().discard_hard_samples();
    Dtype* pos_diff = bottom[0]->mutable_cpu_diff();
    Dtype* neg_diff = bottom[1]->mutable_cpu_diff();
    const Dtype* pos_dst = bottom[0]->cpu_data();
    const Dtype* neg_dst = bottom[1]->cpu_data();
    int count = bottom[0]->count();
    for (int i=0; i<count; ++i) {
      if (pos_diff[i] > Dtype(0) && (!discard_hard_samples || pos_dst[i] < neg_dst[i])) {
        pos_diff[i] = Dtype(+1);
        neg_diff[i] = Dtype(-1);
      } else {
        pos_diff[i] = Dtype(0);
        neg_diff[i] = Dtype(0);
      }
    }
  }
}

INSTANTIATE_CLASS(PairwiseRankingLossLayer);
REGISTER_LAYER_CLASS(PairwiseRankingLoss);

}  // namespace caffe
