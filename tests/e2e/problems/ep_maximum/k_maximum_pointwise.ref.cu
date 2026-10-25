extern "C" __global__
__launch_bounds__(256) void k_maximum_pointwise(const float* x0, const float* x1, float* maximum) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 750) {
        int a0 = _gid;
        float in0__u0 = x0[a0 * 4];
        float in1__u0 = x1[a0 * 4];
        float v0__u0 = fmaxf(in0__u0, in1__u0);
        float in0__u1 = x0[a0 * 4 + 1];
        float in1__u1 = x1[a0 * 4 + 1];
        float v0__u1 = fmaxf(in0__u1, in1__u1);
        float in0__u2 = x0[a0 * 4 + 2];
        float in1__u2 = x1[a0 * 4 + 2];
        float v0__u2 = fmaxf(in0__u2, in1__u2);
        float in0__u3 = x0[a0 * 4 + 3];
        float in1__u3 = x1[a0 * 4 + 3];
        float v0__u3 = fmaxf(in0__u3, in1__u3);
        float4 _vs_v0__u0_52df4eb8 = make_float4(v0__u0, v0__u1, v0__u2, v0__u3);
        *reinterpret_cast<float4*>(&maximum[a0 * 4]) = _vs_v0__u0_52df4eb8;
    }
}
