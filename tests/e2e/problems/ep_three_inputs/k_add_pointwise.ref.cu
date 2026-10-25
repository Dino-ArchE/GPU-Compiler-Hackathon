extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* b, const float* a, const float* c, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        float in0__u0 = a[a0 * 4];
        float in1__u0 = b[a0 * 4];
        float in2__u0 = c[a0 * 4];
        float v1__u0 = in2__u0 + (in0__u0 * in1__u0);
        float in0__u1 = a[a0 * 4 + 1];
        float in1__u1 = b[a0 * 4 + 1];
        float in2__u1 = c[a0 * 4 + 1];
        float v1__u1 = in2__u1 + (in0__u1 * in1__u1);
        float in0__u2 = a[a0 * 4 + 2];
        float in1__u2 = b[a0 * 4 + 2];
        float in2__u2 = c[a0 * 4 + 2];
        float v1__u2 = in2__u2 + (in0__u2 * in1__u2);
        float in0__u3 = a[a0 * 4 + 3];
        float in1__u3 = b[a0 * 4 + 3];
        float in2__u3 = c[a0 * 4 + 3];
        float v1__u3 = in2__u3 + (in0__u3 * in1__u3);
        float4 _vs_v1__u0_93c81c5a = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&add[a0 * 4]) = _vs_v1__u0_93c81c5a;
}
