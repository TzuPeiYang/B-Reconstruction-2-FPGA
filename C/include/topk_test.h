#ifndef TOPK_TEST_H
#define TOPK_TEST_H

#include <stdint.h>

void entry(const float tensor_input[1][5][10], float tensor_topk_values[1][3][10], int64_t tensor_topk_indices[1][3][10]);

#endif /* TOPK_TEST_H */