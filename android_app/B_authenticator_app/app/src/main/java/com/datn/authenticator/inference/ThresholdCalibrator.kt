package com.datn.authenticator.inference

import android.util.Log

/**
 * Hiệu chuẩn ngưỡng quyết định riêng cho từng owner lúc enroll: neo theo trung vị
 * điểm của pool người lạ, nhích về phía genuine một phần [LENIENCY], kẹp [FLOOR, CEIL]:
 *   thr = impMedian + LENIENCY · (genMedian − impMedian).
 */
object ThresholdCalibrator {
    private const val TAG = "ThresholdCalibrator"

    private const val LENIENCY    = 0.30f   // 0 = sát impostor (lỏng nhất) … 1 = sát genuine (chặt nhất)
    private const val FLOOR       = 0.10f   // sàn an toàn
    private const val CEIL        = 0.45f   // trần (tránh từ chối owner do anchor quá sạch)
    private const val AGG_WINDOW  = 5       // gộp ~5 cửa sổ trước khi hiệu chuẩn (khớp EWMA)
    private const val N_BOOTSTRAP = 128     // số nhóm gộp lấy mẫu (ổn định trung vị)
    private const val SEED        = 42L

    /** Ngưỡng dùng khi không hiệu chuẩn được (sentinel = "chưa calibrate"). */
    const val UNCALIBRATED = Float.NaN

    fun isCalibrated(thr: Float): Boolean = !thr.isNaN() && thr > 0f

    /**
     * Trả về ngưỡng per-owner ∈ [FLOOR, CEIL], hoặc [fallback] nếu thiếu dữ liệu.
     */
    fun calibrate(
        anchors: List<FloatArray>,
        impostorPool: Array<FloatArray>,
        cohortMean: Float,
        cohortStd: Float,
        fallback: Float,
    ): Float {
        if (anchors.size < 3) return fallback

        val rng = java.util.Random(SEED)
        val genuine = leaveOneOutScores(anchors, cohortMean, cohortStd)
        val gAgg = bootstrapAggregate(genuine, rng)
        if (gAgg.isEmpty()) return fallback
        val genMed = median(gAgg)

        val thr: Float = if (impostorPool.size >= AGG_WINDOW) {
            val impostor = impostorScores(impostorPool, anchors, cohortMean, cohortStd)
            val impMed = median(bootstrapAggregate(impostor, rng))
            impMed + LENIENCY * (genMed - impMed)
        } else {
            LENIENCY * genMed
        }

        val clamped = thr.coerceIn(FLOOR, CEIL)
        Log.i(TAG, "Per-owner threshold = ${"%.3f".format(clamped)} " +
                "(genMed=${"%.3f".format(genMed)}, raw=${"%.3f".format(thr)}, " +
                "anchors=${anchors.size}, pool=${impostorPool.size})")
        return clamped
    }

    /** Điểm genuine: mỗi anchor chấm với các anchor còn lại (leave-one-out). */
    private fun leaveOneOutScores(
        anchors: List<FloatArray>, cohortMean: Float, cohortStd: Float,
    ): FloatArray = FloatArray(anchors.size) { i ->
        val others = ArrayList<FloatArray>(anchors.size - 1)
        for (j in anchors.indices) if (j != i) others.add(anchors[j])
        InferenceEngine.scoreAgainstAnchors(anchors[i], others, cohortMean, cohortStd)
    }

    /** Điểm impostor: mỗi mẫu pool chấm với toàn bộ anchor. */
    private fun impostorScores(
        pool: Array<FloatArray>, anchors: List<FloatArray>,
        cohortMean: Float, cohortStd: Float,
    ): FloatArray = FloatArray(pool.size) { i ->
        InferenceEngine.scoreAgainstAnchors(pool[i], anchors, cohortMean, cohortStd)
    }

    /**
     * Gộp điểm thành [N_BOOTSTRAP] nhóm, mỗi nhóm là trung bình của [AGG_WINDOW]
     * mẫu rút ngẫu nhiên có hoàn lại — xấp xỉ phân bố điểm sau khi EWMA gộp.
     */
    private fun bootstrapAggregate(scores: FloatArray, rng: java.util.Random): FloatArray {
        if (scores.isEmpty()) return FloatArray(0)
        return FloatArray(N_BOOTSTRAP) {
            var sum = 0f
            repeat(AGG_WINDOW) { sum += scores[rng.nextInt(scores.size)] }
            sum / AGG_WINDOW
        }
    }

    private fun median(a: FloatArray): Float {
        if (a.isEmpty()) return 0f
        val s = a.sortedArray()
        val m = s.size / 2
        return if (s.size % 2 == 1) s[m] else (s[m - 1] + s[m]) / 2f
    }
}
