#!/bin/sh
cargo run --bin lele_gen examples/sensevoice/sensevoice.int8.onnx examples/sensevoice SenseVoice
cargo run -r --example sensevoice fixtures/zh.wav