extern "C" __global__
__launch_bounds__(256) void k_gelu_pointwise(const float* gelu_inv_sqrt2, const float* gelu_one, const float* gelu_half, const float* x, float* gelu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 750) {
        int a0 = _gid;
        float in1__u0 = gelu_inv_sqrt2[0];
        float in3__u0 = x[a0 * 4];
        float in2__u0 = gelu_one[0];
        float in0__u0 = gelu_half[0];
        float v4__u0 = (in2__u0 + erff((in1__u0 * in3__u0))) * (in0__u0 * in3__u0);
        float in1__u1 = gelu_inv_sqrt2[0];
        float in3__u1 = x[a0 * 4 + 1];
        float in2__u1 = gelu_one[0];
        float in0__u1 = gelu_half[0];
        float v4__u1 = (in2__u1 + erff((in1__u1 * in3__u1))) * (in0__u1 * in3__u1);
        float in1__u2 = gelu_inv_sqrt2[0];
        float in3__u2 = x[a0 * 4 + 2];
        float in2__u2 = gelu_one[0];
        float in0__u2 = gelu_half[0];
        float v4__u2 = (in2__u2 + erff((in1__u2 * in3__u2))) * (in0__u2 * in3__u2);
        float in1__u3 = gelu_inv_sqrt2[0];
        float in3__u3 = x[a0 * 4 + 3];
        float in2__u3 = gelu_one[0];
        float in0__u3 = gelu_half[0];
        float v4__u3 = (in2__u3 + erff((in1__u3 * in3__u3))) * (in0__u3 * in3__u3);
        float4 _vs_v4__u0_dee486d2 = make_float4(v4__u0, v4__u1, v4__u2, v4__u3);
        *reinterpret_cast<float4*>(&gelu[a0 * 4]) = _vs_v4__u0_dee486d2;
    }
}
