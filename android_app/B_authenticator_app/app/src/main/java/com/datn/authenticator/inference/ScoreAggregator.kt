package com.datn.authenticator.inference

import com.datn.authenticator.model.AuthState
import java.util.ArrayDeque
import kotlin.math.pow

/**
 * Maintains a sliding window of the last N=5 session scores and computes
 * a weighted exponential moving average:
 *
 *      S_t = Σ(w_i × p_legit_i) / Σ(w_i)
 *      w_i = α^(N − i),  α = 0.8
 *
 * This implements Mục 3.5.1 / 5.3.3 of the thesis report.
 *
 * Thread-safe: all public methods are synchronized so the AuthenticationService
 * can push from one coroutine while the UI samples [currentScore]/[currentState]
 * from the main thread.
 */
class ScoreAggregator(
    val windowSize: Int = 5,
    val alpha: Float = 0.8f,
    val initialScore: Float = 0.5f,
    val trustedThreshold: Float = AuthState.DEFAULT_TRUSTED_THRESHOLD,
    val warningThreshold: Float = AuthState.DEFAULT_WARNING_THRESHOLD,
) {
    init {
        require(windowSize in 1..50) { "windowSize must be 1..50, got $windowSize" }
        require(alpha in 0.01f..1.0f) { "alpha must be 0.01..1.0, got $alpha" }
        require(initialScore in 0f..1f) { "initialScore must be 0..1, got $initialScore" }
    }

    private val recentScores = ArrayDeque<Float>(windowSize)

    /**
     * Append the latest sigmoid probability and recompute the aggregate.
     * @return the new aggregate score.
     */
    @Synchronized
    fun push(probabilityLegit: Float): Float {
        require(probabilityLegit in 0f..1f) { "probability must be 0..1, got $probabilityLegit" }
        if (recentScores.size >= windowSize) recentScores.removeFirst()
        recentScores.addLast(probabilityLegit)
        return computeAggregate()
    }

    /** Read-only — does not mutate. */
    @Synchronized
    fun currentScore(): Float = computeAggregate()

    @Synchronized
    fun currentState(): AuthState =
        AuthState.fromScore(currentScore(), trustedThreshold, warningThreshold)

    @Synchronized
    fun reset() = recentScores.clear()

    @Synchronized
    fun snapshot(): List<Float> = recentScores.toList()

    private fun computeAggregate(): Float {
        if (recentScores.isEmpty()) return initialScore

        // Weights: w_i = α^(N − i) — newer samples (higher i) get heavier weight
        // when 0 < α < 1 because the exponent is smaller.
        // To match the formula in §5.3.3 with i=1..N and the LATEST sample at i=N,
        // we iterate from the back (latest first) using k = 0,1,2,...
        // weight_for_kth_back = α^k  (so latest k=0 gets weight 1.0)
        var weightedSum = 0f
        var weightTotal = 0f
        var k = 0
        // descendingIterator returns the most recent first
        val it = recentScores.descendingIterator()
        while (it.hasNext()) {
            val w = alpha.pow(k.toFloat())
            weightedSum += w * it.next()
            weightTotal += w
            k++
        }
        return weightedSum / weightTotal
    }
}
