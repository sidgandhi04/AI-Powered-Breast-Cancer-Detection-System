import torch
import torch.nn as nn
import time
import numpy as np

def test_cuda_performance():
    """Test CUDA performance with different optimizations"""
    print("🚀 Testing CUDA Performance...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device = torch.device('cuda')
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Test 1: Basic CUDA operations
    print("\n📊 Test 1: Basic CUDA Operations")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    
    start_time = time.time()
    for _ in range(100):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    basic_time = time.time() - start_time
    print(f"   Basic matrix multiplication: {basic_time:.4f}s")
    
    # Test 2: With optimizations
    print("\n📊 Test 2: With CUDA Optimizations")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    start_time = time.time()
    for _ in range(100):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    optimized_time = time.time() - start_time
    print(f"   Optimized matrix multiplication: {optimized_time:.4f}s")
    
    speedup = basic_time / optimized_time
    print(f"   Speedup: {speedup:.2f}x")
    
    # Test 3: Memory operations
    print("\n📊 Test 3: Memory Operations")
    torch.cuda.empty_cache()
    
    # Test batch processing
    batch_sizes = [16, 32, 64, 128]
    for batch_size in batch_sizes:
        try:
            x_batch = torch.randn(batch_size, 3, 224, 224).to(device)
            y_batch = torch.randn(batch_size, 1000).to(device)
            
            start_time = time.time()
            for _ in range(10):
                z_batch = torch.mm(x_batch.view(batch_size, -1), y_batch)
            torch.cuda.synchronize()
            batch_time = time.time() - start_time
            
            print(f"   Batch size {batch_size}: {batch_time:.4f}s")
            
            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"     Memory used: {memory_allocated:.2f}GB")
            
        except RuntimeError as e:
            print(f"   Batch size {batch_size}: Failed - {e}")
            break
    
    # Test 4: Mixed Precision
    print("\n📊 Test 4: Mixed Precision (AMP)")
    try:
        scaler = torch.cuda.amp.GradScaler()
        
        start_time = time.time()
        for _ in range(100):
            with torch.cuda.amp.autocast():
                z = torch.mm(x, y)
        torch.cuda.synchronize()
        amp_time = time.time() - start_time
        print(f"   Mixed precision: {amp_time:.4f}s")
        
        amp_speedup = basic_time / amp_time
        print(f"   AMP Speedup: {amp_speedup:.2f}x")
        
    except Exception as e:
        print(f"   Mixed precision failed: {e}")
    
    # Final memory report
    print("\n📊 Final Memory Report:")
    memory_allocated = torch.cuda.memory_allocated(0) / 1e9
    memory_reserved = torch.cuda.memory_reserved(0) / 1e9
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"   Allocated: {memory_allocated:.2f}GB")
    print(f"   Reserved: {memory_reserved:.2f}GB")
    print(f"   Total: {memory_total:.1f}GB")
    print(f"   Utilization: {memory_allocated/memory_total*100:.1f}%")
    
    # Cleanup
    torch.cuda.empty_cache()
    print("\n✅ CUDA Performance Test Complete!")

if __name__ == "__main__":
    test_cuda_performance()
