#!/bin/sh

cargo run --bin lele_gen examples/silero/silero.onnx examples/silero SileroVad
cargo run --release --example silero 
