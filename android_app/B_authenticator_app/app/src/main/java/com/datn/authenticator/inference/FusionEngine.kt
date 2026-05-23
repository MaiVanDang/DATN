package com.datn.authenticator.inference

import android.util.Log

/**
 * Score-level fusion for inertial + touch modalities.
 *
 * Mirrors verifier.py (_tune_fusion_w):
 *   fused = w * p_inertial + (1 - w) * p_touch
 *   w tuned via grid search (51 steps) maximising AUC on held-out val set.
 *   Tie-break: prefer w closest to 0.5.
 *
 * When touch RF is unavailable, falls back to pure-inertial (w=1.0).
 */
object FusionEngine {

    private const val TAG = "FusionEngine"
    const val DEFAULT_W = 0.5f
    private const val FUSION_STEPS = 51

    /**
     * Compute fused score given per-window inertial probabilities and a
     * session-level touch probability.
     *
     *   p_inertial : P(owner | window) from RF_inertial, per window
     *   p_touch    : P(owner | touch_session) from RF_touch, broadcast to all windows
     *   w          : inertial weight in [0,1]
     */
    fun fuse(pInertialPerWindow: FloatArray, pTouch: Float?, w: Float): Float {
        if (pTouch == null || pInertialPerWindow.isEmpty()) {
            return if (pInertialPerWindow.isEmpty()) 0.5f
            else pInertialPerWindow.average().toFloat()
        }
        val fused = pInertialPerWindow.map { w * it + (1f - w) * pTouch }
        return fused.average().toFloat()
    }

    /**
     * Grid-search fusion weight on a small validation set.
     *
     * @param ownerInertialProba  per-window p_inertial for the owner's val session
     * @param ownerTouchProba     session-level p_touch for the owner's val session
     * @param impostorData        list of (p_inertial_windows, p_touch) for impostors
     * @return best w in [0,1]
     */
    fun tuneWeight(
        ownerInertialProba: FloatArray,
        ownerTouchProba: Float,
        impostorData: List<Pair<FloatArray, Float>>,
    ): Float {
        if (impostorData.isEmpty() || ownerInertialProba.isEmpty()) {
            Log.w(TAG, "tuneWeight: insufficient data, using default w=${DEFAULT_W}")
            return DEFAULT_W
        }

        // Build flat arrays: scores + labels
        // Owner windows → label 1, impostor windows → label 0
        val scores  = mutableListOf<Pair<Float, Int>>()  // (fused_score, label)

        val ws = FloatArray(FUSION_STEPS) { it / (FUSION_STEPS - 1f) }
        var bestW = DEFAULT_W
        var bestAuc = -1.0
        var bestDist = 1.0

        for (w in ws) {
            val fusedScores = mutableListOf<Pair<Float, Int>>()

            // Owner
            for (p in ownerInertialProba) {
                fusedScores.add((w * p + (1f - w) * ownerTouchProba) to 1)
            }
            // Impostors
            for ((pImI, pImT) in impostorData) {
                for (p in pImI) {
                    fusedScores.add((w * p + (1f - w) * pImT) to 0)
                }
            }

            val auc = rocAuc(fusedScores)
            val dist = Math.abs(w - 0.5f).toDouble()
            if (auc > bestAuc + 1e-6) {
                bestAuc = auc; bestW = w; bestDist = dist
            } else if (Math.abs(auc - bestAuc) <= 1e-6 && dist < bestDist) {
                bestW = w; bestDist = dist
            }
        }

        Log.i(TAG, "tuneWeight: best_w=$bestW auc=${"%.4f".format(bestAuc)}")
        return bestW
    }

    /** Trapezoidal AUC from (score, label) pairs. */
    private fun rocAuc(data: List<Pair<Float, Int>>): Double {
        if (data.isEmpty()) return 0.5
        val pos = data.count { it.second == 1 }
        val neg = data.size - pos
        if (pos == 0 || neg == 0) return 0.5

        val sorted = data.sortedByDescending { it.first }
        var tp = 0; var fp = 0
        var auc = 0.0
        var prevTp = 0; var prevFp = 0

        for ((_, label) in sorted) {
            if (label == 1) tp++ else fp++
            auc += (fp - prevFp).toDouble() * (tp + prevTp) / 2.0
            prevTp = tp; prevFp = fp
        }
        return auc / (pos.toLong() * neg.toLong())
    }
}
