package com.datn.authenticator.inference

import com.datn.authenticator.model.AuthState
import java.util.ArrayDeque
import kotlin.math.pow

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

    @Synchronized
    fun push(probabilityLegit: Float): Float {
        require(probabilityLegit in 0f..1f) { "probability must be 0..1, got $probabilityLegit" }
        if (recentScores.size >= windowSize) recentScores.removeFirst()
        recentScores.addLast(probabilityLegit)
        return computeAggregate()
    }

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

        var weightedSum = 0f
        var weightTotal = 0f
        var k = 0

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
