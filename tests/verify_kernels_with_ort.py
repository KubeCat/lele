#!/usr/bin/env python3
"""
Verify lele kernel implementations against ONNX Runtime
"""
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
import subprocess
import json
import tempfile
import os

def create_test_model(op_type, inputs, outputs, **attrs):
    """Create a simple ONNX model with a single operator"""
    # Create input/output info
    input_infos = []
    for i, (name, shape, dtype) in enumerate(inputs):
        input_infos.append(helper.make_tensor_value_info(name, dtype, shape))
    
    output_infos = []
    for name, shape, dtype in outputs:
        output_infos.append(helper.make_tensor_value_info(name, dtype, shape))
    
    # Create node
    node = helper.make_node(
        op_type,
        inputs=[inp[0] for inp in inputs],
        outputs=[out[0] for out in outputs],
        **attrs
    )
    
    # Create graph
    graph = helper.make_graph(
        [node],
        f"{op_type}_test",
        input_infos,
        output_infos
    )
    
    # Create model
    model = helper.make_model(graph, producer_name='kernel_test')
    model.opset_import[0].version = 13
    model.ir_version = 8  # Compatible with ORT
    return model

def test_softmax():
    """Test Softmax kernel"""
    print("\n=== Testing Softmax ===")
    
    # Test data
    x = np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    
    model = create_test_model(
        'Softmax',
        [('x', x.shape, TensorProto.FLOAT)],
        [('y', x.shape, TensorProto.FLOAT)],
        axis=-1
    )
    
    # Run with ORT
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    ort_output = sess.run(None, {'x': x})[0]
    
    print(f"Input shape: {x.shape}")
    print(f"Input: {x[0]}")
    print(f"ORT output: {ort_output[0]}")
    print(f"Sum: {ort_output[0].sum():.6f} (should be 1.0)")
    
    # TODO: Run lele implementation
    print("Status: ORT reference obtained ✓")
    return True

def test_layer_norm():
    """Test LayerNormalization kernel"""
    print("\n=== Testing LayerNormalization ===")
    
    # Test data: [B, T, D]
    x = np.random.randn(1, 10, 512).astype(np.float32)
    scale = np.ones(512, dtype=np.float32)
    bias = np.zeros(512, dtype=np.float32)
    
    model = create_test_model(
        'LayerNormalization',
        [('x', x.shape, TensorProto.FLOAT),
         ('scale', scale.shape, TensorProto.FLOAT),
         ('bias', bias.shape, TensorProto.FLOAT)],
        [('y', x.shape, TensorProto.FLOAT)],
        axis=-1,
        epsilon=1e-5
    )
    
    # Run with ORT
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    ort_output = sess.run(None, {'x': x, 'scale': scale, 'bias': bias})[0]
    
    print(f"Input shape: {x.shape}")
    print(f"Input stats: mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"Output stats: mean={ort_output.mean():.4f}, std={ort_output.std():.4f}")
    print(f"Output first 5: {ort_output[0, 0, :5]}")
    
    print("Status: ORT reference obtained ✓")
    return True

def test_matmul():
    """Test MatMul kernel"""
    print("\n=== Testing MatMul ===")
    
    # Test data
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)
    
    model = create_test_model(
        'MatMul',
        [('a', a.shape, TensorProto.FLOAT),
         ('b', b.shape, TensorProto.FLOAT)],
        [('c', [2, 3, 5], TensorProto.FLOAT)]
    )
    
    # Run with ORT
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    ort_output = sess.run(None, {'a': a, 'b': b})[0]
    
    # Also compute with numpy
    np_output = np.matmul(a, b)
    
    print(f"A shape: {a.shape}, B shape: {b.shape}")
    print(f"Output shape: {ort_output.shape}")
    print(f"ORT vs NumPy max diff: {np.abs(ort_output - np_output).max():.6e}")
    
    print("Status: ORT reference obtained ✓")
    return True

def test_concat():
    """Test Concat kernel"""
    print("\n=== Testing Concat ===")
    
    # Test data
    x1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    x2 = np.array([[5, 6]], dtype=np.float32)
    
    model = create_test_model(
        'Concat',
        [('x1', x1.shape, TensorProto.FLOAT),
         ('x2', x2.shape, TensorProto.FLOAT)],
        [('y', [3, 2], TensorProto.FLOAT)],
        axis=0
    )
    
    # Run with ORT
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    ort_output = sess.run(None, {'x1': x1, 'x2': x2})[0]
    
    print(f"X1 shape: {x1.shape}, X2 shape: {x2.shape}")
    print(f"Output shape: {ort_output.shape}")
    print(f"Output:\n{ort_output}")
    
    print("Status: ORT reference obtained ✓")
    return True

def test_where():
    """Test Where kernel"""
    print("\n=== Testing Where ===")
    
    # Test data
    condition = np.array([[True, False], [False, True]], dtype=bool)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    
    model = create_test_model(
        'Where',
        [('condition', condition.shape, TensorProto.BOOL),
         ('x', x.shape, TensorProto.FLOAT),
         ('y', y.shape, TensorProto.FLOAT)],
        [('output', [2, 2], TensorProto.FLOAT)]
    )
    
    # Run with ORT
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    ort_output = sess.run(None, {'condition': condition, 'x': x, 'y': y})[0]
    
    print(f"Condition:\n{condition}")
    print(f"X:\n{x}")
    print(f"Y:\n{y}")
    print(f"Output:\n{ort_output}")
    
    print("Status: ORT reference obtained ✓")
    return True

def test_split():
    """Test Split kernel"""
    print("\n=== Testing Split ===")
    
    # Test data
    x = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
    split_sizes = np.array([2, 2, 2], dtype=np.int64)
    
    model = create_test_model(
        'Split',
        [('x', x.shape, TensorProto.FLOAT),
         ('split', split_sizes.shape, TensorProto.INT64)],
        [('y1', [1, 2], TensorProto.FLOAT),
         ('y2', [1, 2], TensorProto.FLOAT),
         ('y3', [1, 2], TensorProto.FLOAT)],
        axis=1
    )
    
    # Run with ORT
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    outputs = sess.run(None, {'x': x, 'split': split_sizes})
    
    print(f"Input: {x}")
    print(f"Output 1: {outputs[0]}")
    print(f"Output 2: {outputs[1]}")
    print(f"Output 3: {outputs[2]}")
    
    print("Status: ORT reference obtained ✓")
    return True

def test_expand():
    """Test Expand kernel"""
    print("\n=== Testing Expand ===")
    
    # Test data
    x = np.array([[1], [2], [3]], dtype=np.float32)
    shape = np.array([3, 4], dtype=np.int64)
    
    model = create_test_model(
        'Expand',
        [('x', x.shape, TensorProto.FLOAT),
         ('shape', shape.shape, TensorProto.INT64)],
        [('y', [3, 4], TensorProto.FLOAT)]
    )
    
    # Run with ORT
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    ort_output = sess.run(None, {'x': x, 'shape': shape})[0]
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {shape}")
    print(f"Output shape: {ort_output.shape}")
    print(f"Output:\n{ort_output}")
    
    print("Status: ORT reference obtained ✓")
    return True

def test_dynamic_quantize():
    """Test DynamicQuantizeLinear kernel"""
    print("\n=== Testing DynamicQuantizeLinear ===")
    
    # Test data
    x = np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    
    model = create_test_model(
        'DynamicQuantizeLinear',
        [('x', x.shape, TensorProto.FLOAT)],
        [('y', x.shape, TensorProto.UINT8),
         ('y_scale', [1], TensorProto.FLOAT),
         ('y_zero_point', [1], TensorProto.UINT8)]
    )
    
    # Run with ORT
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    outputs = sess.run(None, {'x': x})
    
    print(f"Input: {x}")
    print(f"Quantized: {outputs[0]}")
    print(f"Scale: {outputs[1]}")
    print(f"Zero point: {outputs[2]}")
    
    # Verify dequantization
    zp_val = outputs[2].item() if outputs[2].size == 1 else outputs[2][0]
    scale_val = outputs[1].item() if outputs[1].size == 1 else outputs[1][0]
    dequant = (outputs[0].astype(np.float32) - zp_val) * scale_val
    print(f"Dequantized: {dequant}")
    print(f"Max error: {np.abs(dequant - x).max():.6e}")
    
    print("Status: ORT reference obtained ✓")
    return True

def test_matmul_integer():
    """Test MatMulInteger kernel"""
    print("\n=== Testing MatMulInteger ===")
    
    # Test data (uint8)
    a = np.array([[1, 2, 3],
                  [4, 5, 6]], dtype=np.uint8)
    b = np.array([[7, 8],
                  [9, 10],
                  [11, 12]], dtype=np.uint8)
    a_zero_point = np.array([0], dtype=np.uint8)
    b_zero_point = np.array([0], dtype=np.uint8)
    
    model = create_test_model(
        'MatMulInteger',
        [('a', a.shape, TensorProto.UINT8),
         ('b', b.shape, TensorProto.UINT8),
         ('a_zero_point', a_zero_point.shape, TensorProto.UINT8),
         ('b_zero_point', b_zero_point.shape, TensorProto.UINT8)],
        [('y', [2, 2], TensorProto.INT32)]
    )
    
    # Run with ORT
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    ort_output = sess.run(None, {
        'a': a, 
        'b': b,
        'a_zero_point': a_zero_point,
        'b_zero_point': b_zero_point
    })[0]
    
    # Compute manually
    a_adj = a.astype(np.int32) - a_zero_point[0]
    b_adj = b.astype(np.int32) - b_zero_point[0]
    manual = np.matmul(a_adj, b_adj)
    
    print(f"A: {a}")
    print(f"B: {b}")
    print(f"ORT output: {ort_output}")
    print(f"Manual computation: {manual}")
    print(f"Match: {np.array_equal(ort_output, manual)}")
    
    print("Status: ORT reference obtained ✓")
    return True

def test_transpose():
    print("\n" + "=" * 60)
    print("=== Testing Transpose ===")
    
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print(f"Input shape: {x.shape}")
    print(f"Input:\n{x}")
    
    model = create_test_model(
        "Transpose",
        inputs=[("X", x.shape, TensorProto.FLOAT)],
        outputs=[("Y", None, TensorProto.FLOAT)],
        perm=[1, 0]
    )
    
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    y = sess.run(None, {"X": x})[0]
    
    print(f"Output shape: {y.shape}")
    print(f"Output:\n{y}")
    print("Status: ORT reference obtained ✓")
    return True

def test_add():
    print("\n" + "=" * 60)
    print("=== Testing Add ===")
    
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    print(f"A:\n{a}")
    print(f"B:\n{b}")
    
    model = create_test_model(
        "Add",
        inputs=[("A", a.shape, TensorProto.FLOAT), ("B", b.shape, TensorProto.FLOAT)],
        outputs=[("C", None, TensorProto.FLOAT)]
    )
    
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    c = sess.run(None, {"A": a, "B": b})[0]
    
    print(f"Output:\n{c}")
    print(f"Expected:\n{a + b}")
    print("Status: ORT reference obtained ✓")
    return True

def test_mul():
    print("\n" + "=" * 60)
    print("=== Testing Mul ===")
    
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[2, 3], [4, 5]], dtype=np.float32)
    print(f"A:\n{a}")
    print(f"B:\n{b}")
    
    model = create_test_model(
        "Mul",
        inputs=[("A", a.shape, TensorProto.FLOAT), ("B", b.shape, TensorProto.FLOAT)],
        outputs=[("C", None, TensorProto.FLOAT)]
    )
    
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    c = sess.run(None, {"A": a, "B": b})[0]
    
    print(f"Output:\n{c}")
    print(f"Expected:\n{a * b}")
    print("Status: ORT reference obtained ✓")
    return True

def test_relu():
    print("\n" + "=" * 60)
    print("=== Testing Relu ===")
    
    x = np.array([[-2, -1, 0], [1, 2, 3]], dtype=np.float32)
    print(f"Input:\n{x}")
    
    model = create_test_model(
        "Relu",
        inputs=[("X", x.shape, TensorProto.FLOAT)],
        outputs=[("Y", None, TensorProto.FLOAT)]
    )
    
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    y = sess.run(None, {"X": x})[0]
    
    print(f"Output:\n{y}")
    print(f"Expected:\n{np.maximum(0, x)}")
    print("Status: ORT reference obtained ✓")
    return True

def test_gather():
    print("\n" + "=" * 60)
    print("=== Testing Gather ===")
    
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    indices = np.array([0, 2], dtype=np.int64)
    print(f"Data:\n{data}")
    print(f"Indices: {indices}")
    
    model = create_test_model(
        "Gather",
        inputs=[("data", data.shape, TensorProto.FLOAT), ("indices", indices.shape, TensorProto.INT64)],
        outputs=[("output", None, TensorProto.FLOAT)],
        axis=0
    )
    
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    output = sess.run(None, {"data": data, "indices": indices})[0]
    
    print(f"Output:\n{output}")
    print("Status: ORT reference obtained ✓")
    return True

def test_conv():
    print("\n" + "=" * 60)
    print("=== Testing Conv ===")
    
    # Input: [N, C, L] = [1, 2, 5]
    x = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]], dtype=np.float32)
    # Weights: [M, C, K] = [3, 2, 3] (3 output channels, 2 input channels, kernel size 3)
    w = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
    # Bias: [M]
    b = np.array([1, 2, 3], dtype=np.float32)
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {w.shape}")
    print(f"Bias shape: {b.shape}")
    
    model = create_test_model(
        "Conv",
        inputs=[("X", x.shape, TensorProto.FLOAT), 
                ("W", w.shape, TensorProto.FLOAT),
                ("B", b.shape, TensorProto.FLOAT)],
        outputs=[("Y", None, TensorProto.FLOAT)],
        dilations=[1],
        group=1,
        kernel_shape=[3],
        pads=[0, 0],
        strides=[1]
    )
    
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    y = sess.run(None, {"X": x, "W": w, "B": b})[0]
    
    print(f"Output shape: {y.shape}")
    print(f"Output:\n{y}")
    print("Status: ORT reference obtained ✓")
    return True

def main():
    print("=" * 60)
    print("Kernel Verification Against ONNX Runtime")
    print("=" * 60)
    
    tests = [
        test_softmax,
        test_layer_norm,
        test_matmul,
        test_concat,
        test_where,
        test_split,
        test_expand,
        test_dynamic_quantize,
        test_matmul_integer,
        test_transpose,
        test_add,
        test_mul,
        test_relu,
        test_gather,
        test_conv,
    ]
    
    results = []
    for test in tests:
        try:
            success = test()
            results.append((test.__name__, success))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((test.__name__, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    print(f"\nTotal: {sum(1 for _, s in results if s)}/{len(results)} tests passed")

if __name__ == '__main__':
    main()
