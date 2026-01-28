fn main() -> std::io::Result<()> {
    prost_build::compile_protos(&["src/onnx.proto"], &["src/"])?;
    Ok(())
}
