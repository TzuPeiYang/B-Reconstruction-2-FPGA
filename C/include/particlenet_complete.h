
#ifndef NETWORK_H
#define NETWORK_H

#include <stdint.h>

void entry(const float tensor_pf_points[1][3][35], const float tensor_pf_features[1][4][35], const float tensor_pf_mask[1][1][35], float tensor_MSE[1][4]);

#endif /* NETWORK_H */
    
