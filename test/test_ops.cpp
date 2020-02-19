#include <signal.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <algorithm>
#include <numeric>
#include <string.h>

#include "saga.h"
#include "cli.h"

using namespace saga;




struct TensorData {
  Dims dims;
  std::vector<float> data;
};

// -----------------------------------------------
static const TensorData relu_input = {
  {2, 4},
  {-100, 0, 100, 200,
   -1000, 0, 1000, 2000}
};

static const TensorData relu_output = {
  {2, 4},
  {0, 0, 100, 200,
   0, 0, 1000, 2000}
};

// -----------------------------------------------
static const TensorData conv_input_x = {
  {1, 16, 8, 8},
  {
 0.300, -0.028, -0.045,  0.086, -0.495,  0.126, -0.372, -0.235,
-0.195, -0.207, -0.160, -0.043, -0.377, -0.339, -0.080, -0.271,
-0.357,  0.404,  0.228,  0.114,  0.234, -0.050, -0.403, -0.383,
-0.019, -0.481, -0.038, -0.023, -0.358, -0.267, -0.490, -0.357,
-0.010, -0.112,  0.141, -0.339,  0.255, -0.251,  0.366,  0.391,
-0.152, -0.357,  0.155,  0.410, -0.208, -0.199,  0.099, -0.297,
-0.155, -0.252,  0.426,  0.195, -0.360, -0.388,  0.354,  0.293,
-0.331, -0.482, -0.352,  0.431, -0.386,  0.203, -0.443,  0.152,
-0.004, -0.073, -0.063,  0.449,  0.434, -0.210, -0.134, -0.383,
-0.099,  0.392, -0.344, -0.353,  0.269,  0.203,  0.352, -0.094,
 0.466,  0.082, -0.391, -0.022, -0.415, -0.313, -0.095, -0.183,
 0.186,  0.321, -0.067, -0.242,  0.365,  0.482, -0.314,  0.216,
 0.424, -0.436,  0.311, -0.389,  0.497,  0.012,  0.432, -0.040,
 0.223, -0.482,  0.032, -0.261, -0.263, -0.371, -0.431, -0.130,
 0.224,  0.457,  0.043, -0.406,  0.207, -0.185, -0.153,  0.428,
 0.331,  0.429,  0.370,  0.473, -0.175,  0.466,  0.216, -0.061,
 0.001,  0.080,  0.191, -0.493,  0.413,  0.373, -0.290,  0.009,
-0.255,  0.070,  0.413,  0.022,  0.483,  0.305,  0.411,  0.304,
 0.491, -0.397, -0.204,  0.255, -0.144,  0.158, -0.030, -0.252,
-0.054,  0.417, -0.423, -0.014,  0.357,  0.488, -0.272,  0.488,
 0.014, -0.288,  0.037, -0.285, -0.226,  0.351,  0.227,  0.196,
-0.413, -0.135, -0.334, -0.359,  0.174, -0.189, -0.225, -0.463,
-0.211,  0.452,  0.400, -0.153,  0.437, -0.369, -0.211, -0.081,
-0.196,  0.071, -0.065, -0.216, -0.426, -0.330, -0.178, -0.049,
 0.004, -0.319,  0.221,  0.333, -0.014,  0.210, -0.452,  0.420,
-0.036, -0.316,  0.114, -0.498,  0.333,  0.035,  0.491, -0.069,
 0.322, -0.209, -0.425,  0.067,  0.076,  0.328, -0.054, -0.168,
 0.160, -0.224, -0.041,  0.275,  0.103, -0.401,  0.377,  0.428,
-0.256, -0.405, -0.291,  0.063,  0.045, -0.295, -0.456,  0.018,
-0.009, -0.018,  0.185,  0.132,  0.023, -0.306, -0.239, -0.234,
 0.013, -0.448,  0.488, -0.415, -0.376, -0.364, -0.028, -0.035,
 0.437, -0.339, -0.158, -0.211, -0.183, -0.176, -0.203, -0.166,
-0.028, -0.263, -0.461, -0.414,  0.391, -0.264,  0.273, -0.418,
-0.121, -0.411,  0.479, -0.048,  0.243, -0.080, -0.168,  0.097,
-0.432,  0.215,  0.078, -0.251,  0.493, -0.460, -0.087,  0.307,
-0.442, -0.008, -0.325,  0.248,  0.010, -0.345, -0.262, -0.061,
 0.407, -0.273, -0.397, -0.282,  0.346,  0.432, -0.025,  0.124,
 0.033, -0.072, -0.233,  0.456, -0.481,  0.416,  0.412,  0.085,
 0.059, -0.146, -0.076,  0.321,  0.320, -0.372,  0.314, -0.110,
 0.427,  0.330, -0.148, -0.013,  0.044, -0.207, -0.168,  0.114,
-0.478,  0.065,  0.232, -0.145, -0.366,  0.188, -0.321,  0.090,
-0.468,  0.226,  0.414, -0.194,  0.220, -0.328,  0.320,  0.302,
-0.475,  0.306,  0.164, -0.248, -0.035,  0.273,  0.080,  0.407,
 0.045, -0.071,  0.134,  0.237, -0.056,  0.273, -0.105,  0.316,
-0.401,  0.053,  0.075,  0.195,  0.485,  0.132, -0.075, -0.278,
-0.483,  0.234, -0.270,  0.332,  0.063, -0.244, -0.122, -0.497,
 0.091, -0.030, -0.035,  0.487,  0.345, -0.480, -0.317,  0.441,
-0.216, -0.100, -0.094, -0.229,  0.450,  0.213,  0.123, -0.220,
-0.458, -0.403, -0.216,  0.256, -0.184, -0.454,  0.102,  0.263,
-0.037, -0.044, -0.260,  0.445,  0.339,  0.164, -0.461,  0.379,
 0.111, -0.270,  0.048, -0.194, -0.406,  0.154,  0.248, -0.222,
 0.290, -0.379,  0.096,  0.209,  0.031, -0.105, -0.222, -0.051,
-0.032,  0.384,  0.000,  0.172, -0.002, -0.284, -0.111, -0.483,
 0.354, -0.134, -0.112,  0.457,  0.334,  0.240,  0.252, -0.225,
-0.172, -0.480, -0.245, -0.291,  0.056,  0.467, -0.013,  0.333,
-0.454, -0.094,  0.281,  0.485,  0.351, -0.349, -0.140,  0.112,
 0.063, -0.060,  0.138,  0.275, -0.294, -0.266,  0.322,  0.252,
-0.256,  0.151,  0.161, -0.181,  0.471, -0.339,  0.072, -0.114,
 0.052,  0.001,  0.180,  0.321, -0.385,  0.212, -0.487,  0.156,
-0.434,  0.423,  0.458, -0.034, -0.304,  0.072,  0.272,  0.277,
-0.291,  0.071,  0.471,  0.465,  0.345, -0.269, -0.034, -0.052,
-0.282,  0.437,  0.144,  0.256, -0.356,  0.490,  0.498, -0.078,
 0.359, -0.120, -0.167,  0.347, -0.147,  0.252, -0.498,  0.052,
 0.245, -0.129,  0.003, -0.257, -0.144,  0.028, -0.032, -0.242,
 0.382,  0.177,  0.018, -0.138, -0.432,  0.063, -0.297,  0.403,
-0.252, -0.469,  0.493,  0.327,  0.269,  0.294,  0.056, -0.253,
 0.313,  0.352,  0.376,  0.050, -0.464,  0.394,  0.228,  0.287,
-0.155,  0.408,  0.292, -0.426, -0.038,  0.007, -0.411, -0.009,
 0.314, -0.056, -0.267,  0.245,  0.363, -0.073, -0.366, -0.328,
 0.443, -0.334,  0.343, -0.116, -0.459, -0.170, -0.377, -0.007,
 0.199,  0.171, -0.111,  0.273, -0.055, -0.079,  0.459, -0.102,
-0.374, -0.167, -0.491, -0.011, -0.216,  0.276,  0.094,  0.067,
 0.433, -0.076,  0.491, -0.222,  0.235, -0.072, -0.408, -0.133,
 0.188, -0.254,  0.134, -0.057,  0.394, -0.208, -0.020,  0.143,
-0.213, -0.147,  0.201, -0.009, -0.478,  0.315,  0.210,  0.100,
 0.132,  0.021,  0.126,  0.058, -0.014,  0.345, -0.127,  0.491,
-0.434,  0.483,  0.382, -0.488, -0.037,  0.080, -0.229,  0.298,
-0.316,  0.095,  0.367,  0.474, -0.330,  0.134,  0.111,  0.088,
-0.398, -0.497, -0.235, -0.097,  0.364, -0.335, -0.127,  0.286,
-0.393,  0.115, -0.444,  0.313, -0.166,  0.260, -0.151, -0.336,
 0.412, -0.363, -0.208,  0.187,  0.437, -0.245, -0.036,  0.460,
-0.116,  0.216, -0.212, -0.411, -0.001,  0.453, -0.263, -0.344,
 0.227, -0.223, -0.409, -0.451,  0.035,  0.275,  0.195,  0.131,
 0.064, -0.029,  0.486,  0.338, -0.138,  0.209,  0.072, -0.413,
-0.212,  0.259, -0.196, -0.193,  0.085,  0.066, -0.079,  0.375,
-0.185,  0.046,  0.487, -0.355, -0.123, -0.065, -0.408, -0.382,
 0.116,  0.468, -0.285,  0.479,  0.224, -0.045, -0.370,  0.291,
 0.499,  0.353,  0.384, -0.489,  0.002, -0.061,  0.296,  0.283,
-0.356,  0.168,  0.383, -0.495,  0.372,  0.197,  0.314, -0.227,
-0.197,  0.220,  0.440, -0.239,  0.462,  0.378, -0.070, -0.343,
 0.431,  0.222, -0.260, -0.060, -0.279, -0.489, -0.450,  0.196,
-0.256,  0.387, -0.321,  0.416, -0.341,  0.497,  0.048,  0.454,
 0.341, -0.396, -0.130,  0.191, -0.108,  0.176, -0.182, -0.374,
-0.415, -0.163, -0.041,  0.242, -0.433, -0.229, -0.224, -0.176,
 0.397, -0.116, -0.093, -0.394,  0.237, -0.077, -0.182,  0.449,
 0.470, -0.232,  0.126,  0.403,  0.253, -0.218, -0.313, -0.156,
-0.291, -0.432,  0.276,  0.489, -0.467, -0.419, -0.193, -0.469,
-0.173,  0.230, -0.420,  0.410,  0.366, -0.055,  0.418, -0.049,
-0.324, -0.025,  0.229, -0.152, -0.452, -0.288,  0.230,  0.312,
-0.166,  0.216, -0.169, -0.181, -0.284,  0.416, -0.057, -0.162,
 0.367, -0.448, -0.125,  0.456,  0.227, -0.304,  0.158,  0.434,
-0.480,  0.241, -0.127,  0.350,  0.308, -0.209, -0.375, -0.453,
 0.266, -0.210,  0.175,  0.028,  0.475,  0.001,  0.310, -0.198,
 0.265, -0.306,  0.470,  0.392,  0.439,  0.432,  0.456, -0.220,
-0.186,  0.144, -0.401, -0.413, -0.216, -0.397, -0.327,  0.207,
-0.212,  0.498, -0.325,  0.063, -0.270, -0.462,  0.121, -0.334,
 0.406, -0.366,  0.401,  0.336,  0.193, -0.228, -0.016, -0.128,
-0.101, -0.130,  0.169,  0.281,  0.188,  0.276, -0.208, -0.492,
-0.210,  0.012, -0.267, -0.398,  0.307, -0.281, -0.386, -0.426,
-0.424,  0.199,  0.253, -0.468, -0.312, -0.394, -0.220,  0.240,
 0.020, -0.317,  0.032, -0.338,  0.379, -0.199, -0.403,  0.199,
 0.163,  0.346, -0.431,  0.280,  0.466, -0.376,  0.456, -0.380,
 0.411, -0.234,  0.412, -0.472, -0.315,  0.272, -0.152, -0.121,
-0.256, -0.106, -0.108,  0.198,  0.489, -0.478,  0.072,  0.396,
 0.161, -0.027, -0.002, -0.001, -0.319, -0.198, -0.185,  0.067,
-0.008,  0.127, -0.438, -0.313, -0.162, -0.475,  0.388, -0.041,
-0.437,  0.202,  0.418, -0.448, -0.323,  0.222,  0.111,  0.205,
 0.045, -0.219,  0.405,  0.341,  0.427,  0.396,  0.240,  0.236,
-0.010,  0.018, -0.084, -0.117,  0.028,  0.247, -0.128, -0.158,
 0.202, -0.148,  0.204,  0.331, -0.287,  0.347,  0.450, -0.182,
 0.076,  0.420, -0.223,  0.171,  0.067,  0.115, -0.124,  0.342,
-0.493, -0.007,  0.074, -0.123,  0.231,  0.475, -0.282, -0.407,
-0.235, -0.439,  0.033,  0.373,  0.271,  0.389, -0.311, -0.443,
 0.500,  0.280,  0.305,  0.260,  0.080,  0.047,  0.398,  0.371,
 0.162,  0.329, -0.342,  0.123,  0.409,  0.388,  0.347,  0.099,
-0.296,  0.400, -0.485,  0.064, -0.278,  0.254,  0.288,  0.265,
 0.353, -0.052, -0.299,  0.054, -0.352, -0.035,  0.360,  0.081,
-0.359, -0.477,  0.193, -0.414, -0.295,  0.260, -0.152, -0.172,
  }
};

static const TensorData conv_input_w = {
  {2, 16, 3, 3},
  {
 0.499,  0.411, -0.156,  0.035, -0.397, -0.103,  0.387, -0.147,
-0.064, -0.197, -0.428,  0.316,  0.028,  0.096, -0.474,  0.389,
-0.279, -0.427, -0.127,  0.367, -0.002,  0.202, -0.126,  0.018,
-0.403, -0.099, -0.276,  0.317, -0.459, -0.394, -0.375,  0.265,
-0.158,  0.282, -0.230, -0.443, -0.065, -0.181, -0.359, -0.404,
 0.305,  0.122,  0.473,  0.243, -0.431, -0.074, -0.447, -0.499,
-0.201, -0.299,  0.198,  0.217,  0.219,  0.067,  0.221, -0.446,
-0.068,  0.453,  0.053,  0.288, -0.028, -0.295,  0.026, -0.490,
 0.483,  0.272, -0.102, -0.466,  0.115, -0.352,  0.303,  0.323,
-0.150, -0.436,  0.264, -0.483,  0.345, -0.138,  0.247, -0.123,
-0.051,  0.058, -0.099,  0.090, -0.471, -0.109, -0.205,  0.125,
 0.161, -0.031,  0.275, -0.220,  0.090,  0.334,  0.302,  0.279,
-0.426, -0.020, -0.373, -0.448, -0.189,  0.154,  0.331, -0.326,
-0.168, -0.377,  0.139,  0.147,  0.455, -0.186,  0.480, -0.064,
 0.346,  0.100,  0.334,  0.076,  0.183,  0.106,  0.005,  0.125,
 0.495,  0.249,  0.102, -0.217,  0.149,  0.322, -0.085, -0.402,
-0.294,  0.375, -0.376,  0.462,  0.018, -0.134,  0.338,  0.421,
-0.001, -0.073,  0.045,  0.476,  0.415, -0.281,  0.208,  0.335,
-0.373, -0.473, -0.040,  0.405,  0.466,  0.223, -0.237,  0.391,
-0.007, -0.272, -0.371,  0.242, -0.353,  0.494, -0.105, -0.388,
-0.160, -0.053,  0.402,  0.391,  0.166,  0.396,  0.132,  0.244,
 0.097,  0.433, -0.017, -0.285, -0.065,  0.433, -0.063,  0.418,
 0.399, -0.199,  0.022,  0.049,  0.025,  0.074,  0.023,  0.340,
 0.464, -0.386,  0.024,  0.242,  0.144,  0.025, -0.394,  0.454,
 0.004, -0.273, -0.300,  0.051,  0.312, -0.298, -0.131,  0.351,
-0.219, -0.373, -0.281, -0.100,  0.391, -0.321,  0.417, -0.379,
-0.245, -0.194,  0.260, -0.154, -0.456,  0.228,  0.174,  0.207,
 0.009,  0.278, -0.099, -0.149,  0.029,  0.124,  0.005,  0.327,
 0.269, -0.300,  0.161, -0.180, -0.084, -0.201,  0.327,  0.450,
 0.418,  0.228,  0.437, -0.251, -0.488, -0.427,  0.083,  0.360,
 0.326,  0.129,  0.178, -0.324,  0.245,  0.112, -0.192, -0.129,
-0.291,  0.394, -0.134,  0.123, -0.256, -0.485, -0.116,  0.319,
 0.486,  0.087, -0.187,  0.120, -0.307,  0.407, -0.222, -0.434,
 0.286, -0.103, -0.194, -0.062, -0.224,  0.209, -0.272, -0.110,
 0.336,  0.469,  0.361,  0.157, -0.279,  0.315,  0.343, -0.213,
-0.040,  0.341, -0.263, -0.234, -0.027,  0.254, -0.398,  0.427,

  }
};

static const TensorData conv_input_b = {
  {1, 2},
  {   0,1  }
};

static const TensorData conv_output = {
  {1, 2, 8, 8},
  {
    0.037164, -0.639819, -0.655080, -0.402218, -0.337608, -0.837830, -0.687602,  0.867310,
    -0.268539,  0.345598,  2.401416, -0.701783, -0.058161, -0.212514, -0.238752, -1.089485,
    0.317468,  0.430713, -0.985866, -1.264723, -0.256637,  0.229643, -0.948437, -1.084298,
    0.310575,  0.695393,  0.382057, -0.132019,  0.577687,  1.796881, -0.764575, -0.214408,
    1.545978,  1.512623, -2.244003,  0.590055,  2.024252,  0.130671, -0.123187,  0.107463,
    0.275534, -0.767234,  0.259399,  0.283982, -0.125084,  0.779120, -1.101588,  1.558083,
    -0.661247,  0.158162, -0.129463, -1.152336,  0.955141,  0.540990,  1.712773, -1.123315,
    -0.016402,  0.602601, -0.433587,  1.400666,  0.536972,  0.518417, -0.537091,  0.744992,
    0.632153,  0.409998,  1.807859,  2.547388,  1.307041, -1.003249,  1.856459, -0.209686,
    0.014870,  1.462278,  0.767560, -0.766491,  2.693764,  1.383324,  0.914823,  2.273910,
    0.595120,  1.265597,  1.585944,  2.258425,  2.407766,  1.479852,  0.355708,  1.160102,
    1.316131, -0.409843,  0.538945, -0.203907,  0.729188,  0.848432,  0.497852,  1.328918,
    1.510841, -0.932780,  0.628429,  0.605102,  1.725408,  1.629320,  3.937838,  2.028712,
    -0.059642,  0.220429,  0.962347,  0.972278,  1.340777,  0.417079,  0.771337, -0.055068,
    1.391066,  0.066745,  1.035366,  0.905360, -1.085903,  2.057608,  0.241396,  0.310593,
    1.373153,  1.098142,  1.133877, -0.318406,  0.860474,  2.233053,  1.411554,  0.630138,
 }
};

// -----------------------------------------------
static const TensorData maxpool_input = {
  {2, 2, 8, 8},
  {
-0.331, -0.066,  0.420,  0.041, -0.213, -0.424,  0.443,  0.329,
-0.162, -0.460, -0.442,  0.498,  0.281,  0.444, -0.181, -0.441,
 0.059, -0.101,  0.046,  0.030, -0.322,  0.429, -0.411,  0.397,
-0.372, -0.158, -0.047, -0.213, -0.438,  0.161,  0.141, -0.374,
-0.276,  0.013,  0.128,  0.286,  0.436, -0.340,  0.086,  0.464,
 0.353,  0.254,  0.494, -0.057,  0.002,  0.066,  0.020,  0.175,
-0.355,  0.341,  0.299,  0.023, -0.163,  0.405, -0.359, -0.335,
-0.404,  0.051,  0.110, -0.042, -0.277,  0.133, -0.302,  0.106,
 0.339,  0.247, -0.026, -0.329, -0.053,  0.445,  0.076,  0.489,
-0.485, -0.292, -0.499, -0.167, -0.091,  0.367,  0.362, -0.116,
 0.064, -0.272, -0.440, -0.306,  0.299,  0.456,  0.381,  0.109,
 0.082,  0.056, -0.061,  0.499, -0.120, -0.441, -0.468,  0.454,
-0.025, -0.342, -0.445,  0.284, -0.300,  0.406,  0.227,  0.076,
 0.207, -0.115, -0.218,  0.243,  0.126,  0.022,  0.383, -0.303,
 0.420,  0.067,  0.002, -0.242, -0.342, -0.158, -0.304,  0.279,
-0.456,  0.266, -0.212, -0.410, -0.067,  0.162,  0.393,  0.416,
-0.222, -0.108, -0.014, -0.275, -0.345,  0.028,  0.253,  0.281,
 0.044,  0.147, -0.453, -0.171,  0.445, -0.012, -0.045,  0.121,
-0.465, -0.147,  0.161,  0.324, -0.215,  0.157,  0.227,  0.422,
-0.466, -0.021, -0.364, -0.384,  0.284, -0.093, -0.147,  0.109,
-0.485,  0.358,  0.478,  0.125,  0.454,  0.125,  0.317,  0.316,
-0.108,  0.439,  0.304,  0.366,  0.099,  0.387, -0.250, -0.408,
 0.445, -0.400, -0.397,  0.487,  0.167, -0.439, -0.095, -0.386,
-0.395, -0.487, -0.212,  0.350, -0.359,  0.364,  0.069, -0.115,
 0.273,  0.264, -0.286,  0.449, -0.383,  0.114,  0.179,  0.173,
-0.225,  0.483, -0.419, -0.294,  0.241,  0.460, -0.066, -0.107,
-0.074,  0.491, -0.361,  0.467, -0.425,  0.204,  0.349, -0.003,
-0.147, -0.400,  0.241, -0.200, -0.206,  0.383,  0.038,  0.420,
 0.427,  0.018, -0.463, -0.425,  0.316, -0.287,  0.468, -0.408,
-0.320,  0.484, -0.431, -0.369, -0.374,  0.091, -0.393,  0.036,
-0.459, -0.341, -0.405, -0.233, -0.431,  0.342, -0.370, -0.393,
-0.397, -0.474,  0.236,  0.211, -0.163, -0.418, -0.158, -0.480
  }
};

static const TensorData maxpool_output = {
  {2, 2, 4, 4},
  {
    -0.066000,  0.498000,  0.444000,  0.443000,
    0.059000,  0.046000,  0.429000,  0.397000,
    0.353000,  0.494000,  0.436000,  0.464000,
    0.341000,  0.299000,  0.405000,  0.106000,
    0.339000, -0.026000,  0.445000,  0.489000,
    0.082000,  0.499000,  0.456000,  0.454000,
    0.207000,  0.284000,  0.406000,  0.383000,
    0.420000,  0.002000,  0.162000,  0.416000,
    0.147000, -0.014000,  0.445000,  0.281000,
    -0.021000,  0.324000,  0.284000,  0.422000,
    0.439000,  0.478000,  0.454000,  0.317000,
    0.445000,  0.487000,  0.364000,  0.069000,
    0.483000,  0.449000,  0.460000,  0.179000,
    0.491000,  0.467000,  0.383000,  0.420000,
    0.484000, -0.369000,  0.316000,  0.468000,
    -0.341000,  0.236000,  0.342000, -0.158000
  }
};




static const TensorData maxpool_output_global = {
  {2, 2, 1, 1},
  { 0.498000, 0.499000, 0.487000,  0.491000
  }
};



// -----------------------------------------------
static const TensorData fc1_input_x = {
  {2, 2},
  {
    1, 100,
    -1, -100
  }
};

static const TensorData fc1_input_w = {
  {2, 3},
  {
    3, 5,
    7, 9,
    11, 13,
  }
};

static const TensorData fc1_input_b = {
  {3},
  {
    -1, 0, -1
  }
};

static const TensorData fc1_output_no_bias = {
  {2, 3},
  {
    903, 1105, 1307,
    -903, -1105, -1307,
  }
};

static const TensorData fc1_output_bias = {
  {2, 3},
  {
    902, 1105, 1306,
    -904, -1105, -1308
  }
};


// -----------------------------------------------
static const TensorData fc2_input_x = {
  {2, 32},
  {
    0.375, -0.084, -0.082,  0.146, -0.372,  0.452, -0.348,  0.059,
    0.054, -0.025,  0.087, -0.414, -0.298, -0.365,  0.094, -0.119,
    0.265,  0.224,  0.169, -0.144,  0.415, -0.398,  0.167,  0.059,
    0.260, -0.115,  0.110, -0.219, -0.302,  0.372,  0.306, -0.010,

    0.081,  0.078,  0.217,  0.165, -0.462, -0.236,  0.346,  0.312,
   -0.378,  0.254,  0.070, -0.008,  0.053, -0.270, -0.474, -0.250,
    0.444, -0.045, -0.190,  0.026,  0.180, -0.282, -0.211,  0.311,
    0.121,  0.394, -0.420, -0.326, -0.057,  0.422,  0.498, -0.417,
  }
};

static const TensorData fc2_input_w = {
  {8, 32},
  {
-0.467,  0.417,  0.164,  0.388,  0.368,  0.204, -0.449, -0.264,
 0.339, -0.360,  0.000,  0.417, -0.498,  0.321,  0.339,  0.170,
-0.463,  0.321,  0.171,  0.120,  0.073, -0.177,  0.317,  0.077,
 0.270,  0.273, -0.376,  0.256,  0.281,  0.429,  0.266,  0.429,
 0.367,  0.439,  0.184, -0.224, -0.302,  0.395, -0.450,  0.412,
-0.177,  0.184, -0.376, -0.490, -0.433, -0.352,  0.158, -0.308,
-0.374, -0.063, -0.298,  0.280, -0.481, -0.043,  0.108, -0.331,
 0.026,  0.163, -0.071,  0.361, -0.073, -0.111, -0.332, -0.245,
 0.343,  0.355, -0.480, -0.022,  0.243, -0.480, -0.270, -0.415,
-0.134, -0.224,  0.252, -0.298, -0.367, -0.070, -0.346, -0.069,
 0.178, -0.048, -0.057,  0.452,  0.184,  0.317,  0.087,  0.033,
-0.460, -0.472, -0.106,  0.058, -0.272,  0.157, -0.423, -0.030,
-0.420, -0.067, -0.119,  0.217,  0.074,  0.204,  0.297, -0.281,
-0.430,  0.455,  0.260,  0.015,  0.074,  0.308, -0.183, -0.495,
 0.080, -0.258,  0.304,  0.114,  0.479, -0.060, -0.343,  0.137,
-0.385,  0.459,  0.166,  0.099, -0.037,  0.042,  0.324,  0.048,
 0.132, -0.191, -0.258,  0.479,  0.088,  0.331, -0.389, -0.090,
 0.020, -0.109,  0.282,  0.302,  0.186,  0.450,  0.226, -0.228,
 0.186,  0.424, -0.193,  0.382,  0.337,  0.443,  0.229, -0.108,
 0.399,  0.187,  0.273, -0.311, -0.342, -0.152,  0.311, -0.414,
-0.076, -0.360, -0.075, -0.471,  0.118,  0.368, -0.277, -0.338,
-0.196, -0.415,  0.499, -0.214, -0.045,  0.442, -0.493,  0.272,
-0.183,  0.379, -0.246, -0.386, -0.440,  0.263, -0.250,  0.291,
 0.291,  0.211,  0.318, -0.324,  0.400, -0.266, -0.226, -0.336,
 0.451,  0.127, -0.023,  0.216,  0.123, -0.370, -0.461, -0.493,
-0.084,  0.064,  0.364,  0.015, -0.466, -0.262,  0.032,  0.141,
 0.281, -0.092,  0.443, -0.463,  0.485,  0.074,  0.107, -0.359,
 0.481, -0.306,  0.348,  0.390,  0.105,  0.385,  0.041, -0.065,
-0.252,  0.421, -0.149, -0.444, -0.034, -0.012,  0.079, -0.022,
-0.437, -0.407,  0.242, -0.280, -0.069, -0.316,  0.356, -0.003,
-0.076, -0.484,  0.437,  0.415,  0.483, -0.090, -0.112,  0.396,
-0.359, -0.395, -0.098, -0.354,  0.245, -0.491, -0.226,  0.436,
  }
};

static const TensorData fc2_input_b = {
  {8},
  {
    0.499,  0.053,  0.072,  0.112,  0.274, -0.069, -0.486, -0.291,
  }
};


static const TensorData fc2_output_no_bias = {
  {2, 8},
  {

    0.038842, 0.384617,  0.001430, -0.135650,
    0.514054, -0.346473,  0.952006, -0.095627,
    -0.796724, -0.174328, -0.372716,  0.734828,
    -0.086809, -0.510408, -0.303503, -0.442628
  }
};



static const TensorData fc2_output_bias = {
  {2, 8},
  {
    0.537842,  0.437617, 0.073430, -0.023650,
    0.788054, -0.415473, 0.466006, -0.386627,
    -0.297724, -0.121328, -0.300716,  0.846828,
    0.187191, -0.579408, -0.789503, -0.733628
  }
};



static std::shared_ptr<Tensor>
load_tensor(Tensor::DataType dt, const TensorData &td)
{
  auto t = makeCPUTensor(dt, td.dims);
  auto dst = t->access();
  const size_t elements = td.dims.elements();

  Dims e(td.dims.size(), 0);
  for(size_t i = 0; i < elements; i++) {
    dst->set(e, td.data[i]);

    for(ssize_t j = e.size() - 1; j >= 0; j--) {
      e[j]++;
      if(e[j] == td.dims[j]) {
        e[j] = 0;
      } else {
        break;
      }
    }
  }
  return t;
}



static int
test_op(std::shared_ptr<Context> ctx,
        const char *op,
        const Tensors &inputs,
        const Attributes &attributes,
        std::shared_ptr<Tensor> ref_output)
{
  Graph g;

  auto xit = inputs.find("x");
  if(xit == inputs.end()) {
    printf("Test of %s - No input tensor\n", op);
    return 1;
  }
  auto x = xit->second;
  int batch_size = x->dims_[0];
  auto n = g.addNode(op, inputs, attributes);
  auto p = ctx->createProgram(g, {
      .inference = true,
      .training = false,
      .batch_size = batch_size,
      .initial_learning_rate = 1e-3,
      .tensor_layout = TensorLayout::Auto
    });
  auto y = p->resolveTensor(n->y());
  p->infer();

  if(!ref_output) {
    printf("Test of %s - No reference\n", op);
    for(auto it : inputs) {
      it.second->print(it.first.c_str());
    }
    y->print("Y");
    return 1;

  }

  double sse = y->sse(*ref_output);

  if(sse > 1e-4) {
    printf("Test of %s FAILED sse:%e\n", op, sse);
    for(auto it : inputs) {
      it.second->print(it.first.c_str());
    }
    y->print("  Y");
    ref_output->print("REF");
    return 1;
  } else {
    printf("Test of %s OK SSE:%e\n", op, sse);
  }
  return 0;
}



extern int
ops_main(int argc, char **argv)
{
  int opt;

  auto dt = Tensor::DataType::FLOAT;

  while((opt = getopt(argc, argv, "h")) != -1) {
    switch(opt) {
    case 'h':
      dt = Tensor::DataType::HALF;
      break;
    }
  }

  argc -= optind;
  argv += optind;

  auto ctx = createContext();

  int r = 0;
  r |= test_op(ctx, "relu", {{"x", load_tensor(dt, relu_input)}}, {},
          load_tensor(dt, relu_output));

  r |= test_op(ctx, "conv", {
      {"x", load_tensor(dt, conv_input_x)},
      {"w", load_tensor(dt, conv_input_w)},
      {"b", load_tensor(dt, conv_input_b)},
    }, {{"size", 3}, {"activations", 2}, {"pad", 1}, {"bias", true}},
    load_tensor(dt, conv_output));

  r |= test_op(ctx, "maxpool", {
      {"x", load_tensor(dt, maxpool_input)},
        }, {{"size", 2}, {"stride", 2}},
    load_tensor(dt, maxpool_output));

  r |= test_op(ctx, "maxpool", {
      {"x", load_tensor(dt, maxpool_input)},
        }, {{"global", true}},
    load_tensor(dt, maxpool_output_global));

  r |= test_op(ctx, "fc", {
      {"x", load_tensor(dt, fc1_input_x)},
      {"w", load_tensor(dt, fc1_input_w)},
        }, {{"transW", false}},
    load_tensor(dt, fc1_output_no_bias));

  r |= test_op(ctx, "fc", {
      {"x", load_tensor(dt, fc1_input_x)},
      {"w", load_tensor(dt, fc1_input_w)},
      {"b", load_tensor(dt, fc1_input_b)},
        }, {{"transW", false}},
    load_tensor(dt, fc1_output_bias));

  r |= test_op(ctx, "fc", {
      {"x", load_tensor(dt, fc2_input_x)},
      {"w", load_tensor(dt, fc2_input_w)},
        }, {{"transW", true}},
    load_tensor(dt, fc2_output_no_bias));

  r |= test_op(ctx, "fc", {
      {"x", load_tensor(dt, fc2_input_x)},
      {"w", load_tensor(dt, fc2_input_w)},
      {"b", load_tensor(dt, fc2_input_b)},
        }, {{"transW", true}},
    load_tensor(dt, fc2_output_bias));
  return r;
}


SAGA_CLI_CMD("ops",
             "ops [OPTIONS ...]",
             "Run test of operations",
             ops_main);
