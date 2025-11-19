#ifndef TOPK_TEST_H
#define TOPK_TEST_H

#include <stdint.h>

void entry(const float tensor_pf_points[8][3][10], const float tensor_pf_features[8][4][10], float tensor_output[8][8][10]);

#endif /* TOPK_TEST_H */