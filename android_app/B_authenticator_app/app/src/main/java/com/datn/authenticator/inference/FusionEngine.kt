package com.datn.authenticator.inference

import android.util.Log
import kotlin.math.abs

object FusionEngine {
    private const val TAG = "FusionEngine"
    const val DEFAULT_W = 0.5f
    private const val FUSION_STEPS = 51

    fun fuse(pInertialPerWindow: FloatArray, pTouch: Float?, w: Float): Float {
        if (pTouch == null || pInertialPerWindow.isEmpty()) {
            return if (pInertialPerWindow.isEmpty()) 0.5f
            else pInertialPerWindow.average().toFloat()
        }
        return w * pInertialPerWindow.average().toFloat() + (1f - w) * pTouch
    }

    fun tuneWeight(
        ownerInertialProba: FloatArray,
        ownerTouchProba: Float,
        impostorData: List<Pair<FloatArray, Float>>,
    ): Float {
        if (impostorData.isEmpty() || ownerInertialProba.isEmpty()) {
            Log.w(TAG, "tuneWeight: insufficient data, using default w=${DEFAULT_W}")
            return DEFAULT_W
        }

        val ws = FloatArray(FUSION_STEPS) { it / (FUSION_STEPS - 1f) }
        var bestW = DEFAULT_W
        var bestAuc = -1.0
        var bestDist = 1.0

        for (w in ws) {
            val fusedScores = mutableListOf<Pair<Float, Int>>()

            for (p in ownerInertialProba) {
                fusedScores.add((w * p + (1f - w) * ownerTouchProba) to 1)
            }

            for ((pImI, pImT) in impostorData) {
                for (p in pImI) {
                    fusedScores.add((w * p + (1f - w) * pImT) to 0)
                }
            }

            val auc = rocAuc(fusedScores)
            val dist = abs(w - 0.5f).toDouble()
            if (auc > bestAuc + 1e-6) {
                bestAuc = auc; bestW = w; bestDist = dist
            } else if (abs(auc - bestAuc) <= 1e-6 && dist < bestDist) {
                bestW = w; bestDist = dist
            }
        }

        Log.i(TAG, "tuneWeight: best_w=$bestW auc=${"%.4f".format(bestAuc)}")
        return bestW
    }

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
