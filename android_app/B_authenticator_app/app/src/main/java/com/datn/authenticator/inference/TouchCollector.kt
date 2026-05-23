package com.datn.authenticator.inference

import android.view.MotionEvent
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Singleton touch event collector + 48-D feature extractor.
 *
 * Feature schema (identical to touch_features.py in the training pipeline):
 *   TAP    (16): tap_n, hold_mean/std/median/p25/p75,
 *                tap_displacement_mean/std/median/p25/p75,
 *                tap_iti_mean/std/median/p25/p75
 *   SCROLL (23): scroll_n, duration_mean/std, trajectory_mean/std,
 *                straight_dist_mean/std, v_mean_mean/std, v_max_mean/std,
 *                v_last5_mean/std, mrl_mean/std, a_first5_mean/std,
 *                dir_circmean, dir_circstd,
 *                frac_up, frac_down, frac_left, frac_right
 *   KEY     (9): key_n, inter_mean/std/median/p25/p75,
 *                delete_rate, typing_speed_per_sec, burst_rate
 *
 * Usage:
 *   // In any Activity:
 *   override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
 *       TouchCollector.onTouchEvent(ev)
 *       return super.dispatchTouchEvent(ev)
 *   }
 *
 *   // Register a TextWatcher on any EditText for key timing:
 *   editText.addTextChangedListener(TouchCollector.makeKeyWatcher())
 */
object TouchCollector {

    const val FEAT_DIM = 48

    // ── Raw event storage ─────────────────────────────────────────────────

    private val taps    = mutableListOf<TapEvent>()
    private val scrolls = mutableListOf<ScrollGesture>()
    private val keyEvents = mutableListOf<KeyEvent>()   // inter-key timing

    // Current gesture being tracked
    private var activeScroll: MutableList<MotionPoint>? = null
    private var downX = 0f; private var downY = 0f; private var downT = 0L
    private var isScrollCandidate = false

    private const val SCROLL_MOVE_THRESHOLD_PX = 10f  // min movement to count as scroll

    // ── Public API ────────────────────────────────────────────────────────

    /** Feed every MotionEvent from dispatchTouchEvent(). Thread-safe via synchronized. */
    @Synchronized
    fun onTouchEvent(event: MotionEvent) {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                downX = event.x; downY = event.y; downT = event.eventTime
                isScrollCandidate = false
                activeScroll = mutableListOf(MotionPoint(event.x, event.y, event.eventTime))
            }
            MotionEvent.ACTION_MOVE -> {
                val pts = activeScroll ?: return
                pts.add(MotionPoint(event.x, event.y, event.eventTime))
                if (hypot((event.x - downX).toDouble(), (event.y - downY).toDouble()) > SCROLL_MOVE_THRESHOLD_PX) {
                    isScrollCandidate = true
                }
            }
            MotionEvent.ACTION_UP -> {
                val upX = event.x; val upY = event.y; val upT = event.eventTime
                val duration = upT - downT
                if (isScrollCandidate) {
                    val pts = activeScroll
                    if (pts != null && pts.size >= 3) {
                        pts.add(MotionPoint(upX, upY, upT))
                        scrolls.add(ScrollGesture(pts.toList()))
                    }
                } else if (duration > 20 && duration < 2000) {
                    // Tap: short press, no significant movement
                    taps.add(TapEvent(downT, upT, downX, downY, upX, upY))
                }
                activeScroll = null
            }
            MotionEvent.ACTION_CANCEL -> activeScroll = null
        }
    }

    /** Feed key character insertions from a TextWatcher.afterTextChanged(). */
    @Synchronized
    fun onKeyInserted(isDelete: Boolean) {
        keyEvents.add(KeyEvent(System.currentTimeMillis(), isDelete))
    }

    /** Build the 48-D feature vector from accumulated events. Returns null if too few events. */
    @Synchronized
    fun buildFeatureVector(): FloatArray? {
        if (taps.size < 3 && scrolls.size < 2 && keyEvents.size < 5) return null
        val v = FloatArray(FEAT_DIM)
        var idx = 0

        // ── TAP features (16) ─────────────────────────────────────────────
        val holdMs = taps.map { (it.upT - it.downT).toDouble() }
        val displacement = taps.map { hypot((it.upX - it.downX).toDouble(), (it.upY - it.downY).toDouble()) }
        // ITI: time between consecutive DOWN events, valid range 100ms–8000ms
        val itiMs = (1 until taps.size)
            .map { (taps[it].downT - taps[it - 1].downT).toDouble() }
            .filter { it in 100.0..8000.0 }

        v[idx++] = taps.size.toFloat()
        if (taps.isNotEmpty()) {
            val hStats = stats5(holdMs); val dStats = stats5(displacement); val iStats = stats5(itiMs)
            v[idx++] = hStats.mean.toFloat(); v[idx++] = hStats.std.toFloat()
            v[idx++] = hStats.median.toFloat(); v[idx++] = hStats.p25.toFloat(); v[idx++] = hStats.p75.toFloat()
            v[idx++] = dStats.mean.toFloat(); v[idx++] = dStats.std.toFloat()
            v[idx++] = dStats.median.toFloat(); v[idx++] = dStats.p25.toFloat(); v[idx++] = dStats.p75.toFloat()
            v[idx++] = iStats.mean.toFloat(); v[idx++] = iStats.std.toFloat()
            v[idx++] = iStats.median.toFloat(); v[idx++] = iStats.p25.toFloat(); v[idx++] = iStats.p75.toFloat()
        } else { idx += 15 }

        // ── SCROLL features (23) ──────────────────────────────────────────
        v[idx++] = scrolls.size.toFloat()
        if (scrolls.isNotEmpty()) {
            val durations   = mutableListOf<Double>()
            val trajectories= mutableListOf<Double>()
            val straightDists=mutableListOf<Double>()
            val vMeans      = mutableListOf<Double>()
            val vMaxes      = mutableListOf<Double>()
            val vLast5s     = mutableListOf<Double>()
            val mrls        = mutableListOf<Double>()
            val aFirst5s    = mutableListOf<Double>()
            val directions  = mutableListOf<Double>()   // angles in radians

            for (g in scrolls) {
                val pts = g.points
                val dur = (pts.last().t - pts.first().t).toDouble().coerceAtLeast(1.0)
                durations.add(dur)

                // Segment-level velocities
                val segV = mutableListOf<Double>()
                var traj = 0.0
                for (i in 1 until pts.size) {
                    val dt = (pts[i].t - pts[i-1].t).toDouble().coerceAtLeast(1.0)
                    val dx = pts[i].x - pts[i-1].x; val dy = pts[i].y - pts[i-1].y
                    val dist = hypot(dx.toDouble(), dy.toDouble())
                    traj += dist
                    segV.add(dist / dt * 1000)   // px/s
                }
                trajectories.add(traj)
                straightDists.add(hypot((pts.last().x - pts.first().x).toDouble(),
                    (pts.last().y - pts.first().y).toDouble()))
                vMeans.add(segV.average())
                vMaxes.add(segV.maxOrNull() ?: 0.0)
                vLast5s.add(segV.takeLast(5).average())

                // Acceleration of first 5 segments
                val first5 = segV.take(5)
                val aFirst5 = if (first5.size >= 2) {
                    var acc = 0.0
                    for (i in 1 until first5.size) acc += (first5[i] - first5[i-1])
                    acc / (first5.size - 1)
                } else 0.0
                aFirst5s.add(aFirst5)

                // Mean resultant length + circular stats for direction
                val dx = pts.last().x - pts.first().x
                val dy = pts.last().y - pts.first().y
                val angle = atan2(dy.toDouble(), dx.toDouble())  // -PI..PI
                directions.add(angle)

                // Per-gesture MRL: use segment angles
                var sinSum = 0.0; var cosSum = 0.0
                for (i in 1 until pts.size) {
                    val a = atan2((pts[i].y - pts[i-1].y).toDouble(), (pts[i].x - pts[i-1].x).toDouble())
                    sinSum += sin(a); cosSum += cos(a)
                }
                val n = (pts.size - 1).toDouble()
                mrls.add(if (n > 0) sqrt(sinSum*sinSum + cosSum*cosSum) / n else 0.0)
            }

            // Aggregate
            val durS = statsPair(durations); val trajS = statsPair(trajectories)
            val sdS = statsPair(straightDists); val vmS = statsPair(vMeans)
            val vxS = statsPair(vMaxes); val vlS = statsPair(vLast5s)
            val mrlS = statsPair(mrls); val aS = statsPair(aFirst5s)

            v[idx++] = durS.first.toFloat();  v[idx++] = durS.second.toFloat()
            v[idx++] = trajS.first.toFloat(); v[idx++] = trajS.second.toFloat()
            v[idx++] = sdS.first.toFloat();   v[idx++] = sdS.second.toFloat()
            v[idx++] = vmS.first.toFloat();   v[idx++] = vmS.second.toFloat()
            v[idx++] = vxS.first.toFloat();   v[idx++] = vxS.second.toFloat()
            v[idx++] = vlS.first.toFloat();   v[idx++] = vlS.second.toFloat()
            v[idx++] = mrlS.first.toFloat();  v[idx++] = mrlS.second.toFloat()
            v[idx++] = aS.first.toFloat();    v[idx++] = aS.second.toFloat()

            // Circular mean/std of direction
            val sinMean = directions.map { sin(it) }.average()
            val cosMean = directions.map { cos(it) }.average()
            v[idx++] = atan2(sinMean, cosMean).toFloat()  // circmean
            val R = sqrt(sinMean*sinMean + cosMean*cosMean)
            v[idx++] = if (R < 1.0 && R > 0.0) sqrt(-2.0 * ln(R)).toFloat() else 0f // circstd

            // Directional fractions (based on dominant axis of each gesture)
            var up = 0; var down = 0; var left = 0; var right = 0
            for (g in scrolls) {
                val dx = g.points.last().x - g.points.first().x
                val dy = g.points.last().y - g.points.first().y
                if (Math.abs(dy) >= Math.abs(dx)) { if (dy < 0) up++ else down++ }
                else { if (dx < 0) left++ else right++ }
            }
            val total = scrolls.size.toFloat().coerceAtLeast(1f)
            v[idx++] = up / total; v[idx++] = down / total
            v[idx++] = left / total; v[idx++] = right / total
        } else { idx += 22 }

        // ── KEYSTROKE features (9) ────────────────────────────────────────
        val allKeys = keyEvents.sortedBy { it.t }
        val interMs = allKeys.zipWithNext { a, b -> (b.t - a.t).toDouble() }
            .filter { it in 20.0..3000.0 }

        v[idx++] = keyEvents.size.toFloat()
        if (interMs.size >= 2) {
            val kStats = stats5(interMs)
            v[idx++] = kStats.mean.toFloat(); v[idx++] = kStats.std.toFloat()
            v[idx++] = kStats.median.toFloat(); v[idx++] = kStats.p25.toFloat(); v[idx++] = kStats.p75.toFloat()
        } else { idx += 5 }
        val deleteRate = if (keyEvents.isNotEmpty()) keyEvents.count { it.isDelete } / keyEvents.size.toFloat() else 0f
        v[idx++] = deleteRate
        val typingSpeed = if (interMs.isNotEmpty()) (1000.0 / interMs.median()).toFloat() else 0f
        v[idx++] = typingSpeed
        val burstRate = if (interMs.isNotEmpty()) interMs.count { it < 200 } / interMs.size.toFloat() else 0f
        v[idx++] = burstRate

        check(idx == FEAT_DIM) { "Feature dim mismatch: got $idx, expected $FEAT_DIM" }
        return v
    }

    /** Reset for new session. */
    @Synchronized
    fun resetSession() {
        taps.clear(); scrolls.clear(); keyEvents.clear()
        activeScroll = null
    }

    fun tapCount(): Int = taps.size
    fun scrollCount(): Int = scrolls.size
    fun keyCount(): Int = keyEvents.size

    // ── Data classes ──────────────────────────────────────────────────────

    private data class TapEvent(val downT: Long, val upT: Long,
        val downX: Float, val downY: Float, val upX: Float, val upY: Float)
    private data class ScrollGesture(val points: List<MotionPoint>)
    private data class MotionPoint(val x: Float, val y: Float, val t: Long)
    private data class KeyEvent(val t: Long, val isDelete: Boolean)

    // ── Statistics helpers ────────────────────────────────────────────────

    private data class Stats5(val mean: Double, val std: Double, val median: Double, val p25: Double, val p75: Double)

    private fun stats5(data: List<Double>): Stats5 {
        if (data.isEmpty()) return Stats5(0.0, 0.0, 0.0, 0.0, 0.0)
        val sorted = data.sorted()
        val mean = sorted.average()
        val std = sqrt(sorted.map { (it - mean) * (it - mean) }.average())
        return Stats5(mean, std, sorted.percentile(50.0), sorted.percentile(25.0), sorted.percentile(75.0))
    }

    /** Returns (mean, std). */
    private fun statsPair(data: List<Double>): Pair<Double, Double> {
        if (data.isEmpty()) return 0.0 to 0.0
        val m = data.average()
        val s = sqrt(data.map { (it - m) * (it - m) }.average())
        return m to s
    }

    private fun List<Double>.percentile(p: Double): Double {
        if (isEmpty()) return 0.0
        val sorted = sorted()
        val idx = p / 100.0 * (sorted.size - 1)
        val lo = sorted[idx.toInt()]
        val hi = sorted.getOrElse(idx.toInt() + 1) { lo }
        return lo + (hi - lo) * (idx - idx.toInt())
    }

    private fun List<Double>.median() = percentile(50.0)
}
