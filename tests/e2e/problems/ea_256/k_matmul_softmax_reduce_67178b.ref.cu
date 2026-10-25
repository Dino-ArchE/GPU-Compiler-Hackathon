extern "C" __global__
__launch_bounds__(128) void k_matmul_softmax_reduce_67178b(const float* div_c1, const float* q, const float* kt, const float* v, float* matmul_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 256;
        int a5_b = (_gid / 128) % 2;
        int a0_u = (_gid / 16) % 8;
        int a5_u = (_gid) % 16;
        float in0 = div_c1[0];
        float v1 = 1.0f / in0;
        float acc3__c0_0 = 0.0f;
        float acc5__sum__c0_0 = 0.0f;
        float acc1__c0_0 = -1e+30f;
        float acc3__c0_1 = 0.0f;
        float acc5__sum__c0_1 = 0.0f;
        float acc1__c0_1 = -1e+30f;
        float acc3__c1_0 = 0.0f;
        float acc5__sum__c1_0 = 0.0f;
        float acc1__c1_0 = -1e+30f;
        float acc3__c1_1 = 0.0f;
        float acc5__sum__c1_1 = 0.0f;
        float acc1__c1_1 = -1e+30f;
        float acc3__c2_0 = 0.0f;
        float acc5__sum__c2_0 = 0.0f;
        float acc1__c2_0 = -1e+30f;
        float acc3__c2_1 = 0.0f;
        float acc5__sum__c2_1 = 0.0f;
        float acc1__c2_1 = -1e+30f;
        float acc3__c3_0 = 0.0f;
        float acc5__sum__c3_0 = 0.0f;
        float acc1__c3_0 = -1e+30f;
        float acc3__c3_1 = 0.0f;
        float acc5__sum__c3_1 = 0.0f;
        float acc1__c3_1 = -1e+30f;
        float acc3__c4_0 = 0.0f;
        float acc5__sum__c4_0 = 0.0f;
        float acc1__c4_0 = -1e+30f;
        float acc3__c4_1 = 0.0f;
        float acc5__sum__c4_1 = 0.0f;
        float acc1__c4_1 = -1e+30f;
        float acc3__c5_0 = 0.0f;
        float acc5__sum__c5_0 = 0.0f;
        float acc1__c5_0 = -1e+30f;
        float acc3__c5_1 = 0.0f;
        float acc5__sum__c5_1 = 0.0f;
        float acc1__c5_1 = -1e+30f;
        for (int a1 = 0; a1 < 256; a1++) {
            float acc0__ar0 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar0 = q[(a0_b * 48 + a0_u * 6) % 256 * 64 + a2];
                float in1__ar0 = kt[a2 * 256 + a1];
                float v3__ar0 = in1__ar0 * in2__ar0;
                acc0__ar0 += v3__ar0;
            }
            float acc0__ar1 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar1 = q[(a0_b * 48 + a0_u * 6 + 1) % 256 * 64 + a2];
                float in1__ar1 = kt[a2 * 256 + a1];
                float v3__ar1 = in1__ar1 * in2__ar1;
                acc0__ar1 += v3__ar1;
            }
            float acc0__ar2 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar2 = q[(a0_b * 48 + a0_u * 6 + 2) % 256 * 64 + a2];
                float in1__ar2 = kt[a2 * 256 + a1];
                float v3__ar2 = in1__ar2 * in2__ar2;
                acc0__ar2 += v3__ar2;
            }
            float acc0__ar3 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar3 = q[(a0_b * 48 + a0_u * 6 + 3) % 256 * 64 + a2];
                float in1__ar3 = kt[a2 * 256 + a1];
                float v3__ar3 = in1__ar3 * in2__ar3;
                acc0__ar3 += v3__ar3;
            }
            float acc0__ar4 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar4 = q[(a0_b * 48 + a0_u * 6 + 4) % 256 * 64 + a2];
                float in1__ar4 = kt[a2 * 256 + a1];
                float v3__ar4 = in1__ar4 * in2__ar4;
                acc0__ar4 += v3__ar4;
            }
            float acc0__ar5 = 0.0f;
            for (int a2 = 0; a2 < 64; a2++) {
                float in2__ar5 = q[(a0_b * 48 + a0_u * 6 + 5) % 256 * 64 + a2];
                float in1__ar5 = kt[a2 * 256 + a1];
                float v3__ar5 = in1__ar5 * in2__ar5;
                acc0__ar5 += v3__ar5;
            }
            float v4__c0_0 = acc0__ar0 * v1;
            float one__c0_0 = 1.0f;
            float v4__gn__c0_0 = fmaxf(acc1__c0_0, v4__c0_0);
            float v4__alpha__c0_0 = expf((acc1__c0_0 - v4__gn__c0_0));
            float v4__beta__c0_0 = expf((v4__c0_0 - v4__gn__c0_0));
            float v4__acc3_sa__c0_0 = acc3__c0_0 * v4__alpha__c0_0;
            float v4__acc3_sb__c0_0 = one__c0_0 * v4__beta__c0_0;
            acc3__c0_0 = v4__acc3_sa__c0_0 + v4__acc3_sb__c0_0;
            float v4__acc5__sum_sa__c0_0 = acc5__sum__c0_0 * v4__alpha__c0_0;
            float2 _v_in7__bc0 = *reinterpret_cast<const float2*>(&v[a1 * 64 + (a5_b * 32 + a5_u * 2)]);
            float in7__bc0 = _v_in7__bc0.x;
            float in7__bc1 = _v_in7__bc0.y;
            float v4__acc5__sum_sb__c0_0 = in7__bc0 * v4__beta__c0_0;
            acc5__sum__c0_0 = v4__acc5__sum_sa__c0_0 + v4__acc5__sum_sb__c0_0;
            acc1__c0_0 = fmaxf(acc1__c0_0, v4__c0_0);
            float v4__c0_1 = acc0__ar0 * v1;
            float one__c0_1 = 1.0f;
            float v4__gn__c0_1 = fmaxf(acc1__c0_1, v4__c0_1);
            float v4__alpha__c0_1 = expf((acc1__c0_1 - v4__gn__c0_1));
            float v4__beta__c0_1 = expf((v4__c0_1 - v4__gn__c0_1));
            float v4__acc3_sa__c0_1 = acc3__c0_1 * v4__alpha__c0_1;
            float v4__acc3_sb__c0_1 = one__c0_1 * v4__beta__c0_1;
            acc3__c0_1 = v4__acc3_sa__c0_1 + v4__acc3_sb__c0_1;
            float v4__acc5__sum_sa__c0_1 = acc5__sum__c0_1 * v4__alpha__c0_1;
            float v4__acc5__sum_sb__c0_1 = in7__bc1 * v4__beta__c0_1;
            acc5__sum__c0_1 = v4__acc5__sum_sa__c0_1 + v4__acc5__sum_sb__c0_1;
            acc1__c0_1 = fmaxf(acc1__c0_1, v4__c0_1);
            float v4__c1_0 = acc0__ar1 * v1;
            float one__c1_0 = 1.0f;
            float v4__gn__c1_0 = fmaxf(acc1__c1_0, v4__c1_0);
            float v4__alpha__c1_0 = expf((acc1__c1_0 - v4__gn__c1_0));
            float v4__beta__c1_0 = expf((v4__c1_0 - v4__gn__c1_0));
            float v4__acc3_sa__c1_0 = acc3__c1_0 * v4__alpha__c1_0;
            float v4__acc3_sb__c1_0 = one__c1_0 * v4__beta__c1_0;
            acc3__c1_0 = v4__acc3_sa__c1_0 + v4__acc3_sb__c1_0;
            float v4__acc5__sum_sa__c1_0 = acc5__sum__c1_0 * v4__alpha__c1_0;
            float v4__acc5__sum_sb__c1_0 = in7__bc0 * v4__beta__c1_0;
            acc5__sum__c1_0 = v4__acc5__sum_sa__c1_0 + v4__acc5__sum_sb__c1_0;
            acc1__c1_0 = fmaxf(acc1__c1_0, v4__c1_0);
            float v4__c1_1 = acc0__ar1 * v1;
            float one__c1_1 = 1.0f;
            float v4__gn__c1_1 = fmaxf(acc1__c1_1, v4__c1_1);
            float v4__alpha__c1_1 = expf((acc1__c1_1 - v4__gn__c1_1));
            float v4__beta__c1_1 = expf((v4__c1_1 - v4__gn__c1_1));
            float v4__acc3_sa__c1_1 = acc3__c1_1 * v4__alpha__c1_1;
            float v4__acc3_sb__c1_1 = one__c1_1 * v4__beta__c1_1;
            acc3__c1_1 = v4__acc3_sa__c1_1 + v4__acc3_sb__c1_1;
            float v4__acc5__sum_sa__c1_1 = acc5__sum__c1_1 * v4__alpha__c1_1;
            float v4__acc5__sum_sb__c1_1 = in7__bc1 * v4__beta__c1_1;
            acc5__sum__c1_1 = v4__acc5__sum_sa__c1_1 + v4__acc5__sum_sb__c1_1;
            acc1__c1_1 = fmaxf(acc1__c1_1, v4__c1_1);
            float v4__c2_0 = acc0__ar2 * v1;
            float one__c2_0 = 1.0f;
            float v4__gn__c2_0 = fmaxf(acc1__c2_0, v4__c2_0);
            float v4__alpha__c2_0 = expf((acc1__c2_0 - v4__gn__c2_0));
            float v4__beta__c2_0 = expf((v4__c2_0 - v4__gn__c2_0));
            float v4__acc3_sa__c2_0 = acc3__c2_0 * v4__alpha__c2_0;
            float v4__acc3_sb__c2_0 = one__c2_0 * v4__beta__c2_0;
            acc3__c2_0 = v4__acc3_sa__c2_0 + v4__acc3_sb__c2_0;
            float v4__acc5__sum_sa__c2_0 = acc5__sum__c2_0 * v4__alpha__c2_0;
            float v4__acc5__sum_sb__c2_0 = in7__bc0 * v4__beta__c2_0;
            acc5__sum__c2_0 = v4__acc5__sum_sa__c2_0 + v4__acc5__sum_sb__c2_0;
            acc1__c2_0 = fmaxf(acc1__c2_0, v4__c2_0);
            float v4__c2_1 = acc0__ar2 * v1;
            float one__c2_1 = 1.0f;
            float v4__gn__c2_1 = fmaxf(acc1__c2_1, v4__c2_1);
            float v4__alpha__c2_1 = expf((acc1__c2_1 - v4__gn__c2_1));
            float v4__beta__c2_1 = expf((v4__c2_1 - v4__gn__c2_1));
            float v4__acc3_sa__c2_1 = acc3__c2_1 * v4__alpha__c2_1;
            float v4__acc3_sb__c2_1 = one__c2_1 * v4__beta__c2_1;
            acc3__c2_1 = v4__acc3_sa__c2_1 + v4__acc3_sb__c2_1;
            float v4__acc5__sum_sa__c2_1 = acc5__sum__c2_1 * v4__alpha__c2_1;
            float v4__acc5__sum_sb__c2_1 = in7__bc1 * v4__beta__c2_1;
            acc5__sum__c2_1 = v4__acc5__sum_sa__c2_1 + v4__acc5__sum_sb__c2_1;
            acc1__c2_1 = fmaxf(acc1__c2_1, v4__c2_1);
            float v4__c3_0 = acc0__ar3 * v1;
            float one__c3_0 = 1.0f;
            float v4__gn__c3_0 = fmaxf(acc1__c3_0, v4__c3_0);
            float v4__alpha__c3_0 = expf((acc1__c3_0 - v4__gn__c3_0));
            float v4__beta__c3_0 = expf((v4__c3_0 - v4__gn__c3_0));
            float v4__acc3_sa__c3_0 = acc3__c3_0 * v4__alpha__c3_0;
            float v4__acc3_sb__c3_0 = one__c3_0 * v4__beta__c3_0;
            acc3__c3_0 = v4__acc3_sa__c3_0 + v4__acc3_sb__c3_0;
            float v4__acc5__sum_sa__c3_0 = acc5__sum__c3_0 * v4__alpha__c3_0;
            float v4__acc5__sum_sb__c3_0 = in7__bc0 * v4__beta__c3_0;
            acc5__sum__c3_0 = v4__acc5__sum_sa__c3_0 + v4__acc5__sum_sb__c3_0;
            acc1__c3_0 = fmaxf(acc1__c3_0, v4__c3_0);
            float v4__c3_1 = acc0__ar3 * v1;
            float one__c3_1 = 1.0f;
            float v4__gn__c3_1 = fmaxf(acc1__c3_1, v4__c3_1);
            float v4__alpha__c3_1 = expf((acc1__c3_1 - v4__gn__c3_1));
            float v4__beta__c3_1 = expf((v4__c3_1 - v4__gn__c3_1));
            float v4__acc3_sa__c3_1 = acc3__c3_1 * v4__alpha__c3_1;
            float v4__acc3_sb__c3_1 = one__c3_1 * v4__beta__c3_1;
            acc3__c3_1 = v4__acc3_sa__c3_1 + v4__acc3_sb__c3_1;
            float v4__acc5__sum_sa__c3_1 = acc5__sum__c3_1 * v4__alpha__c3_1;
            float v4__acc5__sum_sb__c3_1 = in7__bc1 * v4__beta__c3_1;
            acc5__sum__c3_1 = v4__acc5__sum_sa__c3_1 + v4__acc5__sum_sb__c3_1;
            acc1__c3_1 = fmaxf(acc1__c3_1, v4__c3_1);
            float v4__c4_0 = acc0__ar4 * v1;
            float one__c4_0 = 1.0f;
            float v4__gn__c4_0 = fmaxf(acc1__c4_0, v4__c4_0);
            float v4__alpha__c4_0 = expf((acc1__c4_0 - v4__gn__c4_0));
            float v4__beta__c4_0 = expf((v4__c4_0 - v4__gn__c4_0));
            float v4__acc3_sa__c4_0 = acc3__c4_0 * v4__alpha__c4_0;
            float v4__acc3_sb__c4_0 = one__c4_0 * v4__beta__c4_0;
            acc3__c4_0 = v4__acc3_sa__c4_0 + v4__acc3_sb__c4_0;
            float v4__acc5__sum_sa__c4_0 = acc5__sum__c4_0 * v4__alpha__c4_0;
            float v4__acc5__sum_sb__c4_0 = in7__bc0 * v4__beta__c4_0;
            acc5__sum__c4_0 = v4__acc5__sum_sa__c4_0 + v4__acc5__sum_sb__c4_0;
            acc1__c4_0 = fmaxf(acc1__c4_0, v4__c4_0);
            float v4__c4_1 = acc0__ar4 * v1;
            float one__c4_1 = 1.0f;
            float v4__gn__c4_1 = fmaxf(acc1__c4_1, v4__c4_1);
            float v4__alpha__c4_1 = expf((acc1__c4_1 - v4__gn__c4_1));
            float v4__beta__c4_1 = expf((v4__c4_1 - v4__gn__c4_1));
            float v4__acc3_sa__c4_1 = acc3__c4_1 * v4__alpha__c4_1;
            float v4__acc3_sb__c4_1 = one__c4_1 * v4__beta__c4_1;
            acc3__c4_1 = v4__acc3_sa__c4_1 + v4__acc3_sb__c4_1;
            float v4__acc5__sum_sa__c4_1 = acc5__sum__c4_1 * v4__alpha__c4_1;
            float v4__acc5__sum_sb__c4_1 = in7__bc1 * v4__beta__c4_1;
            acc5__sum__c4_1 = v4__acc5__sum_sa__c4_1 + v4__acc5__sum_sb__c4_1;
            acc1__c4_1 = fmaxf(acc1__c4_1, v4__c4_1);
            float v4__c5_0 = acc0__ar5 * v1;
            float one__c5_0 = 1.0f;
            float v4__gn__c5_0 = fmaxf(acc1__c5_0, v4__c5_0);
            float v4__alpha__c5_0 = expf((acc1__c5_0 - v4__gn__c5_0));
            float v4__beta__c5_0 = expf((v4__c5_0 - v4__gn__c5_0));
            float v4__acc3_sa__c5_0 = acc3__c5_0 * v4__alpha__c5_0;
            float v4__acc3_sb__c5_0 = one__c5_0 * v4__beta__c5_0;
            acc3__c5_0 = v4__acc3_sa__c5_0 + v4__acc3_sb__c5_0;
            float v4__acc5__sum_sa__c5_0 = acc5__sum__c5_0 * v4__alpha__c5_0;
            float v4__acc5__sum_sb__c5_0 = in7__bc0 * v4__beta__c5_0;
            acc5__sum__c5_0 = v4__acc5__sum_sa__c5_0 + v4__acc5__sum_sb__c5_0;
            acc1__c5_0 = fmaxf(acc1__c5_0, v4__c5_0);
            float v4__c5_1 = acc0__ar5 * v1;
            float one__c5_1 = 1.0f;
            float v4__gn__c5_1 = fmaxf(acc1__c5_1, v4__c5_1);
            float v4__alpha__c5_1 = expf((acc1__c5_1 - v4__gn__c5_1));
            float v4__beta__c5_1 = expf((v4__c5_1 - v4__gn__c5_1));
            float v4__acc3_sa__c5_1 = acc3__c5_1 * v4__alpha__c5_1;
            float v4__acc3_sb__c5_1 = one__c5_1 * v4__beta__c5_1;
            acc3__c5_1 = v4__acc3_sa__c5_1 + v4__acc3_sb__c5_1;
            float v4__acc5__sum_sa__c5_1 = acc5__sum__c5_1 * v4__alpha__c5_1;
            float v4__acc5__sum_sb__c5_1 = in7__bc1 * v4__beta__c5_1;
            acc5__sum__c5_1 = v4__acc5__sum_sa__c5_1 + v4__acc5__sum_sb__c5_1;
            acc1__c5_1 = fmaxf(acc1__c5_1, v4__c5_1);
        }
        float acc5__c0_0 = acc5__sum__c0_0 * (1.0f / acc3__c0_0);
        if (a0_b * 48 + a0_u * 6 + 0 < 256) {
            matmul_1[(a0_b * 48 + a0_u * 6) % 256 * 64 + (a5_b * 32 + a5_u * 2)] = acc5__c0_0;
        }
        float acc5__c0_1 = acc5__sum__c0_1 * (1.0f / acc3__c0_1);
        if (a0_b * 48 + a0_u * 6 + 0 < 256) {
            matmul_1[(a0_b * 48 + a0_u * 6) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)] = acc5__c0_1;
        }
        float acc5__c1_0 = acc5__sum__c1_0 * (1.0f / acc3__c1_0);
        if (a0_b * 48 + a0_u * 6 + 1 < 256) {
            matmul_1[(a0_b * 48 + a0_u * 6 + 1) % 256 * 64 + (a5_b * 32 + a5_u * 2)] = acc5__c1_0;
        }
        float acc5__c1_1 = acc5__sum__c1_1 * (1.0f / acc3__c1_1);
        if (a0_b * 48 + a0_u * 6 + 1 < 256) {
            matmul_1[(a0_b * 48 + a0_u * 6 + 1) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)] = acc5__c1_1;
        }
        float acc5__c2_0 = acc5__sum__c2_0 * (1.0f / acc3__c2_0);
        if (a0_b * 48 + a0_u * 6 + 2 < 256) {
            matmul_1[(a0_b * 48 + a0_u * 6 + 2) % 256 * 64 + (a5_b * 32 + a5_u * 2)] = acc5__c2_0;
        }
        float acc5__c2_1 = acc5__sum__c2_1 * (1.0f / acc3__c2_1);
        if (a0_b * 48 + a0_u * 6 + 2 < 256) {
            matmul_1[(a0_b * 48 + a0_u * 6 + 2) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)] = acc5__c2_1;
        }
        float acc5__c3_0 = acc5__sum__c3_0 * (1.0f / acc3__c3_0);
        if (a0_b * 48 + a0_u * 6 + 3 < 256) {
            matmul_1[(a0_b * 48 + a0_u * 6 + 3) % 256 * 64 + (a5_b * 32 + a5_u * 2)] = acc5__c3_0;
        }
        float acc5__c3_1 = acc5__sum__c3_1 * (1.0f / acc3__c3_1);
        if (a0_b * 48 + a0_u * 6 + 3 < 256) {
            matmul_1[(a0_b * 48 + a0_u * 6 + 3) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)] = acc5__c3_1;
        }
        float acc5__c4_0 = acc5__sum__c4_0 * (1.0f / acc3__c4_0);
        if (a0_b * 48 + a0_u * 6 + 4 < 256) {
            matmul_1[(a0_b * 48 + a0_u * 6 + 4) % 256 * 64 + (a5_b * 32 + a5_u * 2)] = acc5__c4_0;
        }
        float acc5__c4_1 = acc5__sum__c4_1 * (1.0f / acc3__c4_1);
        if (a0_b * 48 + a0_u * 6 + 4 < 256) {
            matmul_1[(a0_b * 48 + a0_u * 6 + 4) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)] = acc5__c4_1;
        }
        float acc5__c5_0 = acc5__sum__c5_0 * (1.0f / acc3__c5_0);
        if (a0_b * 48 + a0_u * 6 + 5 < 256) {
            matmul_1[(a0_b * 48 + a0_u * 6 + 5) % 256 * 64 + (a5_b * 32 + a5_u * 2)] = acc5__c5_0;
        }
        float acc5__c5_1 = acc5__sum__c5_1 * (1.0f / acc3__c5_1);
        if (a0_b * 48 + a0_u * 6 + 5 < 256) {
            matmul_1[(a0_b * 48 + a0_u * 6 + 5) % 256 * 64 + (a5_b * 32 + a5_u * 2 + 1)] = acc5__c5_1;
        }
}
