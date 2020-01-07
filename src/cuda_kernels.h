// -*-c++-*-

#pragma once

namespace saga {

void catclassifier_pred_float_i32(int n, const float *p, int32_t *y, unsigned int c);

void catclassifier_backprop_float_i32(int n, const float *p, float *dx,
                                      const int32_t *dy, float *loss,
                                      unsigned int c, float scale);

void adam_float(int n, float alpha, float *weights, const float *dweights, float *t,
                float b1t, float b2t);

}
