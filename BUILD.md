# Build Instructions

## Windows (nvcc)

### Best Kernel with cuBLAS Comparison
```cmd
nvcc -O3 -arch=sm_86 kernels\5_rectangular_tiling_with_cublas.cu -o bin\rectangular.exe -lcublas
```

### Individual Kernels
```cmd
nvcc -O3 -arch=sm_86 kernels\1_naive.cu -o bin\naive.exe -lcublas
nvcc -O3 -arch=sm_86 kernels\2_tiled.cu -o bin\tiled.exe -lcublas
nvcc -O3 -arch=sm_86 kernels\3_register_blocking.cu -o bin\register_blocking.exe -lcublas
nvcc -O3 -arch=sm_86 kernels\4_vectorized.cu -o bin\vectorized.exe -lcublas
nvcc -O3 -arch=sm_86 kernels\5_rectangular_tiling.cu -o bin\rectangular_basic.exe -lcublas
```

## Visual Studio

1. Open `evantest1_matrixmul.sln` in Visual Studio 2022
2. Select "Release" configuration
3. Build → Build Solution (F7)
4. Executable will be in `x64\Release\evantest1_matrixmul.exe`

## Run

```cmd
cd bin
rectangular.exe
```

## Profile with Nsight Compute

```cmd
# Quick profile
ncu bin\rectangular.exe

# Detailed profile
ncu --set full -o profile bin\rectangular.exe

# Open in GUI
ncu-ui profile.ncu-rep
```

## Architecture Note

The `-arch=sm_86` flag is for RTX 30-series (Ampere). Adjust for your GPU:
- RTX 40-series (Ada): `-arch=sm_89`
- RTX 20-series (Turing): `-arch=sm_75`
- GTX 10-series (Pascal): `-arch=sm_61`

Check your GPU compute capability: https://developer.nvidia.com/cuda-gpus

