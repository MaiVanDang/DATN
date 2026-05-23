package com.datn.authenticator.inference

import android.util.Log
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Binary Random Forest classifier — CART trees, Gini impurity.
 *
 * Mirrors the sklearn hyperparameters used in verifier.py:
 *   n_estimators  = 200
 *   max_features  = "sqrt"   → floor(sqrt(n_features))
 *   class_weight  = "balanced" → handled by class-frequency weighting in Gini
 *   min_samples_leaf = 2 (inertial) or 1 (touch) — caller sets this
 *
 * Training input:
 *   X : Array<FloatArray>   — feature matrix (n_samples × n_features)
 *   y : IntArray            — binary labels {0, 1}
 *
 * Inference:
 *   predictProba(x: FloatArray): Float  — P(class=1)
 *
 * Serialization: compact binary (magic + n_trees + per-tree node arrays).
 */
class RandomForestClassifier(
    private val nEstimators: Int = 200,
    private val minSamplesLeaf: Int = 2,
    private val seed: Long = 42L,
) {

    private val trees = mutableListOf<DecisionTree>()
    var isTrained: Boolean = false
        private set

    // ── Training ──────────────────────────────────────────────────────────

    fun fit(X: Array<FloatArray>, y: IntArray) {
        require(X.size == y.size && X.isNotEmpty()) { "X and y must have same non-zero length" }
        trees.clear()

        val nSamples = X.size
        val nFeatures = X[0].size
        val maxFeatures = maxOf(1, sqrt(nFeatures.toDouble()).toInt())

        // Class weights for balanced mode
        val n1 = y.count { it == 1 }
        val n0 = nSamples - n1
        val w1 = if (n1 > 0) nSamples.toDouble() / (2 * n1) else 1.0
        val w0 = if (n0 > 0) nSamples.toDouble() / (2 * n0) else 1.0
        val weights = FloatArray(nSamples) { if (y[it] == 1) w1.toFloat() else w0.toFloat() }

        val rng = java.util.Random(seed)
        repeat(nEstimators) { treeIdx ->
            // Bootstrap: sample nSamples indices with replacement
            val indices = IntArray(nSamples) { rng.nextInt(nSamples) }
            val tree = DecisionTree(maxFeatures, minSamplesLeaf, rng.nextLong())
            tree.build(X, y, weights, indices)
            trees.add(tree)
        }
        isTrained = true
        Log.i(TAG, "RF trained: $nEstimators trees, n=$nSamples, features=$nFeatures")
    }

    /** P(y=1) averaged over all trees. */
    fun predictProba(x: FloatArray): Float {
        if (trees.isEmpty()) return 0.5f
        var sum = 0f
        for (t in trees) sum += t.predictProba(x)
        return sum / trees.size
    }

    // ── Serialization ─────────────────────────────────────────────────────

    fun writeTo(out: OutputStream) {
        DataOutputStream(out).use { dos ->
            dos.writeInt(MAGIC)
            dos.writeInt(trees.size)
            for (t in trees) t.writeTo(dos)
        }
    }

    fun readFrom(inp: InputStream) {
        DataInputStream(inp).use { dis ->
            val magic = dis.readInt()
            require(magic == MAGIC) { "Bad RF magic 0x${magic.toString(16)}" }
            val n = dis.readInt()
            trees.clear()
            repeat(n) {
                val t = DecisionTree(1, 1, 0L)
                t.readFrom(dis)
                trees.add(t)
            }
            isTrained = true
        }
    }

    companion object {
        private const val TAG = "RandomForest"
        const val MAGIC = 0xBF0001.toInt()
    }

    // ── Decision tree ─────────────────────────────────────────────────────

    internal class DecisionTree(
        private val maxFeatures: Int,
        private val minSamplesLeaf: Int,
        seed: Long,
    ) {
        // Compact node storage (arrays of parallel arrays for cache efficiency)
        private var feature    = IntArray(0)   // split feature index (-1 = leaf)
        private var threshold  = FloatArray(0) // split threshold
        private var leftChild  = IntArray(0)   // index of left child
        private var rightChild = IntArray(0)   // index of right child
        private var leafProba  = FloatArray(0) // P(y=1) at leaf nodes

        private val rng = java.util.Random(seed)
        private var nodeCount = 0

        fun build(X: Array<FloatArray>, y: IntArray, weights: FloatArray, indices: IntArray) {
            // Pre-allocate a generous node buffer (2 × samples is sufficient for balanced trees)
            val maxNodes = 2 * indices.size + 1
            feature    = IntArray(maxNodes) { -1 }
            threshold  = FloatArray(maxNodes)
            leftChild  = IntArray(maxNodes) { -1 }
            rightChild = IntArray(maxNodes) { -1 }
            leafProba  = FloatArray(maxNodes)
            nodeCount  = 0

            buildNode(X, y, weights, indices.toMutableList(), depth = 0)

            // Trim to actual size
            feature    = feature.copyOf(nodeCount)
            threshold  = threshold.copyOf(nodeCount)
            leftChild  = leftChild.copyOf(nodeCount)
            rightChild = rightChild.copyOf(nodeCount)
            leafProba  = leafProba.copyOf(nodeCount)
        }

        private fun buildNode(
            X: Array<FloatArray>, y: IntArray, weights: FloatArray,
            indices: MutableList<Int>, depth: Int,
        ): Int {
            val nodeIdx = nodeCount++

            val w1 = indices.sumOf { if (y[it] == 1) weights[it].toDouble() else 0.0 }
            val w0 = indices.sumOf { if (y[it] == 0) weights[it].toDouble() else 0.0 }
            val wTotal = w1 + w0
            val proba = if (wTotal > 0) (w1 / wTotal).toFloat() else 0.5f

            // Leaf conditions: too small, pure, or max depth
            if (indices.size < 2 * minSamplesLeaf || w1 == 0.0 || w0 == 0.0 || depth >= MAX_DEPTH) {
                feature[nodeIdx] = -1
                leafProba[nodeIdx] = proba
                return nodeIdx
            }

            // Find best split among random subset of features
            val nFeatures = X[0].size
            val featSubset = (0 until nFeatures).shuffled(rng).take(maxFeatures)

            var bestFeat = -1
            var bestThresh = 0f
            var bestGini = Double.MAX_VALUE

            for (f in featSubset) {
                // Collect values and sort
                val vals = indices.map { X[it][f] }.toFloatArray().also { it.sort() }
                // Try midpoints between consecutive distinct values
                var prevVal = vals[0]
                for (k in 1 until vals.size) {
                    val v = vals[k]
                    if (v == prevVal) { prevVal = v; continue }
                    val mid = (prevVal + v) / 2f
                    prevVal = v

                    var lw1 = 0.0; var lw0 = 0.0; var rw1 = 0.0; var rw0 = 0.0
                    for (i in indices) {
                        val w = weights[i].toDouble()
                        val isPos = y[i] == 1
                        if (X[i][f] <= mid) { if (isPos) lw1 += w else lw0 += w }
                        else                { if (isPos) rw1 += w else rw0 += w }
                    }
                    val lw = lw1 + lw0; val rw = rw1 + rw0
                    if (lw < minSamplesLeaf || rw < minSamplesLeaf) continue

                    val giniL = if (lw > 0) 1 - (lw1/lw).let { it*it } - (lw0/lw).let { it*it } else 0.0
                    val giniR = if (rw > 0) 1 - (rw1/rw).let { it*it } - (rw0/rw).let { it*it } else 0.0
                    val gini = (lw * giniL + rw * giniR) / wTotal

                    if (gini < bestGini) { bestGini = gini; bestFeat = f; bestThresh = mid }
                }
            }

            if (bestFeat == -1) {
                feature[nodeIdx] = -1
                leafProba[nodeIdx] = proba
                return nodeIdx
            }

            val left  = indices.filter { X[it][bestFeat] <= bestThresh }.toMutableList()
            val right = indices.filter { X[it][bestFeat] >  bestThresh }.toMutableList()

            if (left.size < minSamplesLeaf || right.size < minSamplesLeaf) {
                feature[nodeIdx] = -1
                leafProba[nodeIdx] = proba
                return nodeIdx
            }

            feature[nodeIdx]   = bestFeat
            threshold[nodeIdx] = bestThresh
            leftChild[nodeIdx]  = buildNode(X, y, weights, left,  depth + 1)
            rightChild[nodeIdx] = buildNode(X, y, weights, right, depth + 1)
            return nodeIdx
        }

        fun predictProba(x: FloatArray): Float {
            var node = 0
            while (feature[node] != -1) {
                node = if (x[feature[node]] <= threshold[node]) leftChild[node] else rightChild[node]
            }
            return leafProba[node]
        }

        fun writeTo(dos: DataOutputStream) {
            dos.writeInt(nodeCount)
            for (i in 0 until nodeCount) {
                dos.writeInt(feature[i])
                dos.writeFloat(threshold[i])
                dos.writeInt(leftChild[i])
                dos.writeInt(rightChild[i])
                dos.writeFloat(leafProba[i])
            }
        }

        fun readFrom(dis: DataInputStream) {
            nodeCount = dis.readInt()
            feature    = IntArray(nodeCount)
            threshold  = FloatArray(nodeCount)
            leftChild  = IntArray(nodeCount)
            rightChild = IntArray(nodeCount)
            leafProba  = FloatArray(nodeCount)
            for (i in 0 until nodeCount) {
                feature[i]    = dis.readInt()
                threshold[i]  = dis.readFloat()
                leftChild[i]  = dis.readInt()
                rightChild[i] = dis.readInt()
                leafProba[i]  = dis.readFloat()
            }
        }

        companion object { private const val MAX_DEPTH = 20 }
    }
}
