extern "C" __global__
__launch_bounds__(256) void k_matmul_aadd3d(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 576) {
        int a0_b = _gid / 24;
        int a1_b = (_gid) % 24;
        int a0_u = (_gid) % 1;
        int a1_u = (_gid) % 1;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c0_2 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;
        float acc0__c1_2 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c2_1 = 0.0f;
        float acc0__c2_2 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c3_1 = 0.0f;
        float acc0__c3_2 = 0.0f;
        for (int a2 = 0; a2 < 200; a2++) {
            float in0__ar0 = x0[(a0_b * 4 + a0_u * 4) * 200 + a2];
            float in1__bc0 = x1[a2 * 72 + (a1_b * 3 + a1_u * 3)];
            float v0__c0_0 = in0__ar0 * in1__bc0;
            acc0__c0_0 += v0__c0_0;
            float in1__bc1 = x1[a2 * 72 + (a1_b * 3 + a1_u * 3 + 1)];
            float v0__c0_1 = in0__ar0 * in1__bc1;
            acc0__c0_1 += v0__c0_1;
            float in1__bc2 = x1[a2 * 72 + (a1_b * 3 + a1_u * 3 + 2)];
            float v0__c0_2 = in0__ar0 * in1__bc2;
            acc0__c0_2 += v0__c0_2;
            float in0__ar1 = x0[(a0_b * 4 + a0_u * 4 + 1) * 200 + a2];
            float v0__c1_0 = in0__ar1 * in1__bc0;
            acc0__c1_0 += v0__c1_0;
            float v0__c1_1 = in0__ar1 * in1__bc1;
            acc0__c1_1 += v0__c1_1;
            float v0__c1_2 = in0__ar1 * in1__bc2;
            acc0__c1_2 += v0__c1_2;
            float in0__ar2 = x0[(a0_b * 4 + a0_u * 4 + 2) * 200 + a2];
            float v0__c2_0 = in0__ar2 * in1__bc0;
            acc0__c2_0 += v0__c2_0;
            float v0__c2_1 = in0__ar2 * in1__bc1;
            acc0__c2_1 += v0__c2_1;
            float v0__c2_2 = in0__ar2 * in1__bc2;
            acc0__c2_2 += v0__c2_2;
            float in0__ar3 = x0[(a0_b * 4 + a0_u * 4 + 3) * 200 + a2];
            float v0__c3_0 = in0__ar3 * in1__bc0;
            acc0__c3_0 += v0__c3_0;
            float v0__c3_1 = in0__ar3 * in1__bc1;
            acc0__c3_1 += v0__c3_1;
            float v0__c3_2 = in0__ar3 * in1__bc2;
            acc0__c3_2 += v0__c3_2;
        }
        matmul[(a0_b * 4 + a0_u * 4) * 72 + (a1_b * 3 + a1_u * 3)] = acc0__c0_0;
        matmul[(a0_b * 4 + a0_u * 4) * 72 + (a1_b * 3 + a1_u * 3 + 1)] = acc0__c0_1;
        matmul[(a0_b * 4 + a0_u * 4) * 72 + (a1_b * 3 + a1_u * 3 + 2)] = acc0__c0_2;
        matmul[(a0_b * 4 + a0_u * 4 + 1) * 72 + (a1_b * 3 + a1_u * 3)] = acc0__c1_0;
        matmul[(a0_b * 4 + a0_u * 4 + 1) * 72 + (a1_b * 3 + a1_u * 3 + 1)] = acc0__c1_1;
        matmul[(a0_b * 4 + a0_u * 4 + 1) * 72 + (a1_b * 3 + a1_u * 3 + 2)] = acc0__c1_2;
        matmul[(a0_b * 4 + a0_u * 4 + 2) * 72 + (a1_b * 3 + a1_u * 3)] = acc0__c2_0;
        matmul[(a0_b * 4 + a0_u * 4 + 2) * 72 + (a1_b * 3 + a1_u * 3 + 1)] = acc0__c2_1;
        matmul[(a0_b * 4 + a0_u * 4 + 2) * 72 + (a1_b * 3 + a1_u * 3 + 2)] = acc0__c2_2;
        matmul[(a0_b * 4 + a0_u * 4 + 3) * 72 + (a1_b * 3 + a1_u * 3)] = acc0__c3_0;
        matmul[(a0_b * 4 + a0_u * 4 + 3) * 72 + (a1_b * 3 + a1_u * 3 + 1)] = acc0__c3_1;
        matmul[(a0_b * 4 + a0_u * 4 + 3) * 72 + (a1_b * 3 + a1_u * 3 + 2)] = acc0__c3_2;
    }
}
