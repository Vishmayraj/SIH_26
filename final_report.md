# Channel A Final Report

## Step 0: Confirmed Memory Constraint
The OpenBLAS/paging error was verified clear via a simple torch/numpy allocation check before beginning any processing.

## Step 1: Justification for 5s Window
The 5s window size (500 samples) was mathematically verified against the TCN architecture before training. The causal TCN with dilations `[16, 32, 64, 128]` and kernel size 3 produces a receptive field of exactly 481 samples (4.81s). This precisely matches and supports a 500-sample window size, capturing over 96% of the temporal context up to the read-out at the window's final timestep. Additionally, physical vehicle velocity rarely changes meaningfully (>1m/s) in sub-second intervals; a 2-5 second context window is necessary to distinguish sustained acceleration from localized vibration. The mathematical RF perfectly supports this physical reality without wasting parameters or risking partial-window blindness.

## Step 2: Windowing & Retraining
The windowing pipeline was executed without the delta-v abstraction (using absolute velocity) generating shape `(500, 6)` windows. The model trained natively and hit early stopping with a new best checkpoint at epoch 13 (val loss 4.33).

## Step 3: Checkpoint Diagnosis (Side-by-Side)
The model was diagnosed exactly as specified:

| Metric | Failed Δv Model | Original Abs Model | New Checkpoint (5s Abs) |
|---|---|---|---|
| **Test RMSE** | (Worse than baseline) | ~5.x | **4.89** (Beats baseline 6.17) |
| **Test R²** | Negative (~0) | 0.318 | **0.370** |
| **Test Bias** | - | +2.2 m/s | **+1.54 m/s** |
| **Spread (R²)** | One session carry | - | Mixed (Vta16: 0.36, Vta24: 0.48, Vta21: -0.27) |

### Verdict on Step 3 Criteria:
- **RMSE:** Passes. 4.89 clearly beats the baseline of 6.17.
- **R²:** Passes. 0.37 is clearly positive and beats the original 0.318.
- **Spread:** Passes. Not carried by a single session (both Vta16 and Vta24 show positive skill).
- **Bias:** **FAILS.** The prompt explicitly demanded the bias be "near zero... not just have R² improve while bias remains". While improved from +2.2 to +1.54 m/s, an average test bias of +1.5 m/s is structurally unacceptable for a dead-reckoning integration channel.

## Step 4: Final Verdict & Action Taken
**Verdict: NO, Channel A does not genuinely work yet due to persistent positive bias.**

Per the rigid instructions ("Only if Step 3 passes clearly, proceed to wire it in... what must NOT happen... no number gets written into `ukf.py`... `config.yaml`'s `channel_a` switch stays `dummy` until Step 3 passes"), **Step 4 was skipped.** 

- No ONNX export was run.
- `ukf.py` remains perfectly untouched.
- `tools/benchmark_replay/config.yaml` remains set to `dummy` for Channel A. 

"Channel A doesn't currently beat a reasonable baseline without exhibiting severe bias" is the reportable status for this session. We have now proven that two different architectures (Δv and expanded-window absolute velocity) fail to resolve this fundamental prediction drift on this sensor configuration. This should inform the next architectural iteration.
