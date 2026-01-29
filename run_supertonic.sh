#!/bin/sh
# Step 1: Compile ONNX models for Supertonic
MODELS_DIR="examples/supertonic/models/onnx"
GEN_DIR="examples/supertonic"

cargo run --bin lele_gen $MODELS_DIR/duration_predictor.onnx $GEN_DIR DurationPredictor
cargo run --bin lele_gen $MODELS_DIR/text_encoder.onnx $GEN_DIR TextEncoder
cargo run --bin lele_gen $MODELS_DIR/vector_estimator.onnx $GEN_DIR VectorEstimator
cargo run --bin lele_gen $MODELS_DIR/vocoder.onnx $GEN_DIR Vocoder

# Step 2: Run Supertonic TTS with "hello,lele engine" as input text
cargo run --release --example supertonic -- "This is getting complex for a 1-shot implementation."
