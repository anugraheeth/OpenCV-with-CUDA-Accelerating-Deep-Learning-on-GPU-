# OpenCV-with-CUDA-Accelerating-Deep-Learning-on-GPU-
This repository provides step-by-step instructions to set up OpenCV with CUDA for faster performance on NVIDIA GPUs, including building from source, configuring CUDA/cuDNN, and modifying code for GPU-based inference.


Using GPU acceleration with OpenCV for deep learning tasks involves installing a GPU-compatible build of OpenCV and ensuring that CUDA (NVIDIA's parallel computing platform) is properly configured. Here's a step-by-step guide on how to set this up:

### 1. Verify GPU Compatibility

Ensure that your system has a compatible NVIDIA GPU and that the appropriate CUDA drivers are installed.

- **NVIDIA GPU**: Check if your GPU is supported by CUDA [here](https://developer.nvidia.com/cuda-gpus).
- **CUDA Toolkit**: Download and install the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-toolkit).

### 2. Install CUDA and cuDNN

1. **Download CUDA Toolkit**:
   - Go to [NVIDIA's CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and download the version compatible with your GPU and operating system.
   - Follow the installation instructions provided.

2. **Download cuDNN**:
   - Go to [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) and download the version compatible with your CUDA Toolkit.
   - Extract the contents and copy them to your CUDA installation directory. For example:
     - Copy `bin/cudnn*.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin`
     - Copy `include/cudnn*.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\include`
     - Copy `lib/x64/cudnn*.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\lib\x64`

### 3. Install GPU-Enabled OpenCV

**Option 1: Use Pre-built Binaries**

1. **Download Pre-built OpenCV Binaries with CUDA Support**:
   - Pre-built binaries are available for various platforms. One source is [opencv.org](https://opencv.org/releases/), but ensure you download a version that includes CUDA support.

**Option 2: Build OpenCV from Source**

1. **Clone OpenCV Repositories**:
   ```bash
   git clone https://github.com/opencv/opencv.git
   git clone https://github.com/opencv/opencv_contrib.git
   ```

2. **Install Dependencies** (for Ubuntu):
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential cmake git libgtk-3-dev libcanberra-gtk* libtbb2 libtbb-dev libdcmtk-dev libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libatlas-base-dev gfortran python3-dev python3-pip
   ```

3. **Configure and Build OpenCV with CUDA Support**:
   ```bash
   cd opencv
   mkdir build
   cd build
   cmake -D CMAKE_BUILD_TYPE=RELEASE \
         -D CMAKE_INSTALL_PREFIX=/usr/local \
         -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
         -D WITH_CUDA=ON \
         -D CUDA_ARCH_BIN="6.1" \  # Adjust based on your GPU architecture
         -D WITH_CUDNN=ON \
         -D OPENCV_DNN_CUDA=ON \
         -D OPENCV_GENERATE_PKGCONFIG=ON ..
   make -j"$(nproc)"
   sudo make install
   ```

### 4. Verify GPU Acceleration in OpenCV

After installation, verify that OpenCV is using CUDA:

1. **Python Verification**:
   ```python
   import cv2
   print(cv2.getBuildInformation())
   ```

   Look for lines indicating CUDA support, such as `--   CUDA: YES` and `--   cuDNN: YES`.

2. **Test CUDA Functionality**:
   ```python
   import cv2
   if cv2.cuda.getCudaEnabledDeviceCount() > 0:
       print("CUDA is enabled")
   else:
       print("CUDA is not enabled")
   ```

### 5. Modify Your Code for GPU Usage

Replace the CPU-based operations in your code with their GPU equivalents. For example:

- **Loading the Model**:
  ```python
  yolo = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
  yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
  yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
  ```

- **Performing Inference**:
  ```python
  blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
  yolo.setInput(blob)
  outputs = yolo.forward(output_layers)
  ```

The key lines to use GPU acceleration are setting the preferable backend and target to CUDA, which you can do using the `cv2.dnn` module.

By following these steps, you can leverage GPU acceleration to improve the performance of your object detection tasks with OpenCV.
