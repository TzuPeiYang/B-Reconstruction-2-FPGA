#ifndef PF_POINTS_H
    #define PF_POINTS_H

    #include <stdint.h>

    #define PF_POINTS_NDIMS 3
    #define PF_POINTS_TOTAL_SIZE 48

    static const float pf_points[1][3][16] = {
        { { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f },
        { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f },
        { 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f } }
    };

    #endif // PF_POINTS_H
    