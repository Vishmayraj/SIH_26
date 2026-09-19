package org.sih26.deadreckoning.sensors

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import kotlin.math.exp
import org.sih26.deadreckoning.fusion.Stage12Prediction
import org.sih26.deadreckoning.fusion.Stage12SpeedModel

private const val TAG = "MotionSpeedNetOnnx"
private const val MODEL_ASSET_PATH = "models/motionspeednet.onnx"

/**
 * ONNX Runtime-backed [Stage12SpeedModel]: the only Android-specific piece of the
 * Stage 12 bridge (see fusion/Stage12Channel.kt's class doc for everything else).
 * Loads `models/motionspeednet.onnx` - the float32 export of
 * models/stage12/checkpoints/best_motionspeednet.pth produced by
 * models/stage12/export.py, verified bit-exact against the PyTorch checkpoint at
 * export time - from the app's assets.
 *
 * Every failure mode here (missing asset, corrupt session, a single bad inference
 * call) surfaces as [predict] returning null rather than a thrown exception reaching
 * [org.sih26.deadreckoning.fusion.FusionPipeline], because
 * [org.sih26.deadreckoning.fusion.Stage12Channel] treats null as "fall back to
 * Channel P" - the whole point of injecting this behind an interface is that the
 * model failing to load must degrade to the pre-Stage-12 pipeline, not take the
 * fusion pipeline down with it.
 */
class MotionSpeedNetOnnx private constructor(private val session: OrtSession, private val env: OrtEnvironment) :
    Stage12SpeedModel {

    override fun predict(accWindow: Array<FloatArray>, gyroWindow: Array<FloatArray>): Stage12Prediction? {
        return try {
            val accFlat = FloatArray(accWindow.size * 4)
            val gyFlat = FloatArray(gyroWindow.size * 4)
            for (i in accWindow.indices) {
                System.arraycopy(accWindow[i], 0, accFlat, i * 4, 4)
                System.arraycopy(gyroWindow[i], 0, gyFlat, i * 4, 4)
            }
            val shape = longArrayOf(1, accWindow.size.toLong(), 4)

            OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(accFlat), shape).use { accTensor ->
                OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(gyFlat), shape).use { gyTensor ->
                    session.run(mapOf("x_acc" to accTensor, "x_gy" to gyTensor)).use { result ->
                        val speedMean = (result.get("speed_mean").get().value as FloatArray)[0]
                        val speedLogVar = (result.get("speed_log_var").get().value as FloatArray)[0]
                        val yawCorrection = (result.get("yaw_rate_correction").get().value as FloatArray)[0]
                        Stage12Prediction(
                            speedMps = speedMean.toDouble(),
                            speedVariance = exp(speedLogVar.toDouble()),
                            yawRateCorrectionRadS = yawCorrection.toDouble()
                        )
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Stage 12 inference failed; caller falls back to Channel P.", e)
            null
        }
    }

    companion object {
        /** Returns null (never throws) if the asset is missing or the ONNX session
         * fails to construct - callers pass that straight through to
         * [org.sih26.deadreckoning.fusion.FusionPipeline] as `stage12Model = null`,
         * which disables the channel for the session exactly as if it had never
         * been requested. */
        fun loadFromAssets(context: Context): MotionSpeedNetOnnx? {
            return try {
                val modelBytes = context.assets.open(MODEL_ASSET_PATH).use { it.readBytes() }
                val env = OrtEnvironment.getEnvironment()
                val session = env.createSession(modelBytes, OrtSession.SessionOptions())
                Log.i(TAG, "Stage 12 MotionSpeedNet ONNX session ready ($MODEL_ASSET_PATH).")
                MotionSpeedNetOnnx(session, env)
            } catch (e: Exception) {
                Log.e(TAG, "Could not load $MODEL_ASSET_PATH; Stage 12 disabled for this session.", e)
                null
            }
        }
    }
}
