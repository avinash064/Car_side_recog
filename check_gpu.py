"""Quick script to check GPU availability"""
import tensorflow as tf

print("=" * 80)
print("GPU AVAILABILITY CHECK")
print("=" * 80)

print(f"\nTensorFlow version: {tf.__version__}")

# Check GPU devices
gpus = tf.config.list_physical_devices('GPU')
print(f"\nNumber of GPUs available: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}: {gpu}")
        print(f"  Name: {gpu.name}")
        print(f"  Type: {gpu.device_type}")
    
    # Get GPU details
    print("\n" + "=" * 80)
    print("GPU is AVAILABLE and will be used for training!")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("WARNING: No GPU detected!")
    print("=" * 80)
    print("\nPossible reasons:")
    print("  1. CUDA is not installed")
    print("  2. cuDNN is not installed")
    print("  3. GPU drivers are not properly configured")
    print("  4. TensorFlow cannot detect CUDA installation")

# Check if CUDA is built with
print(f"\nTensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"TensorFlow built with GPU support: {tf.test.is_built_with_gpu_support()}")
