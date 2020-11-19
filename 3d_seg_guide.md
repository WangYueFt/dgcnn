
# installation

https://medium.com/repro-repo/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e (SEE BELOW)

- Ubunut 16.04
- drivers: NVIDIA binary driver - version 384.130 from nvidia-384 (proprietary, tested)
- cuda: cat /usr/local/cuda/version.txt -> CUDA Version 9.0.176
- cudnn: cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2 -> #define CUDNN_MAJOR 7 #define CUDNN_MINOR 0 #define CUDNN_PATCHLEVEL 5
- pip install tensorflow-gpu==1.14 -> esto hace que se tenga que ahcer fuera de cualquier anaconda environment

# get data

- rosrun pcl_ros pointcloud_to_pcd input:=/stereo_narrow/points2 (ejecutar desde el folder donde se van a guardar las pc)

- pcd_to_ply.py

- marcar gt en ply  -> terminar con ply general + annotation folder con cada uno de los elementos
		    -> para conseguir floor, pasar general tambien por meshlab para que se formatee igual
		    -> plyfile lo he tenido que instalar dentro de un anaconda envirnment

- ply_to_txt.py	    -> plyfile lo he tenido que instalar dentro de un anaconda envirnment

- txt_to_npy.py

- npy_to_h5.py	    -> he cambiado ln116 de indoor3d_util.py añadiendo list() al range para poder concatenar

- meter npy y h5 en set/test y set/train


train: pcd -> ply -> txt -> npy -> h5
val: pcd -> ply -> txt -> npy -> h5
test: pcd -> ply -> txt -> npy -> h5

# data management

data
  classes.txt
  train_val
    train
      h5
    val
      h5
  test
    test1
      npy
    ...
  

# train and infer
- python3 train.py --path_data Desktop/data/train_val/ --cls 5 --log_dir RUNS/run_x --batch_size X  # con 32 no va

- python3 batch_inference.py --path_data Desktop/data/test/test1/ --path_cls Desktop/data/classes.txt --model_path RUNS/run_x/ --test_name "test1" --batch_size 8 --visu 

- python3 batch_inference_online.py --path_data Desktop/data/test/test_online/ --path_cls Desktop/data/valve/classes.txt --model_path RUNS/run_x/ --test_name "test_online" --batch_size 8 --out

# troubleshouting

mirar donde estan instaladas las cosas, quiza hay que hacer python3


# ----------------------------------------------------------------------------------------------
# https://medium.com/repro-repo/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e
# ----------------------------------------------------------------------------------------------

Top highlight
Install CUDA 9.0 and cuDNN 7.0 for TensorFlow/PyTorch (GPU) on Ubuntu 16.04
Zhanwen Chen
Zhanwen Chen
Follow
Apr 12, 2018 · 5 min read

Note: I just wrote a post on installing CUDA 9.2 and cuDNN 7.1 here.
You do not have to spend weeks going through official docs while figuring out how to “temporarily add the number ‘3’ and the word ‘nomodeset’ to the end of the system’s kernel boot parameters.” Others suggest turning off lightdm and nouveau which you haven’t even heard of. But what if you don’t actually have to do any of these?
It turns out that you can disregard them. I followed someone’s gist, clarified and updated the instructions, and recorded my steps.
Here’s what I did:


1. Install NVIDIA Graphics Driver via apt-get

CUDA 9.0 requires NVIDIA driver version 384 or above. To install the driver, use apt-get instead of the CUDA runfile:

sudo apt-get install nvidia-384 nvidia-modprobe

, and then you will be prompted to disable Secure Boot. Select Disable.

Reboot your machine but enter BIOS to disable Secure Boot. Typically you can enter BIOS by hitting F12 rapidly as soon as the system restarts.

Afterwards, you can check the installation with the nvidia-smi command, which will report all your CUDA-capable devices in the system, like this:

Wed Apr 11 23:34:18 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.111                Driver Version: 384.111                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1060    Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   60C    P5     8W /  N/A |    242MiB /  6072MiB |      3%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1003      G   /usr/lib/xorg/Xorg                           177MiB |
|    0      1646      G   compiz                                        60MiB |
|    0      2230      G   /usr/lib/firefox/firefox                       1MiB |
+-----------------------------------------------------------------------------+


2. Install CUDA 9.0 via Runfile

Installing CUDA from runfile is actually a breeze, compared to apt-get which involves adding NVIDIA repos and messing up your configuration.

The CUDA runfile installer can be downloaded from NVIDIA’s website, or using wget in case you can’t find it easily on NVIDIA:

(Note: the version I downloaded at first was 384.81 but NVidia constantly releases new minor versions. I recommend the latest which is 384.130 as of 8/29/2018)

$ cd

$ wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run

What you download is a package the following three components:

an NVIDIA driver installer, but usually of stale version;
the actual CUDA installer;
the CUDA samples installer;

I suggest extracting the above three components and executing 2 and 3 separately (remember we installed the driver ourselves already). To extract them, execute the runfile installer with --extract option:

$ chmod +x cuda_9.0.176_384.81_linux-run
$ ./cuda_9.0.176_384.81_linux-run --extract=$HOME

You should have unpacked three components: NVIDIA-Linux-x86_64-384.81.run (1. NVIDIA driver that we ignore), cuda-linux.9.0.176-22781540.run (2. CUDA 9.0 installer), and cuda-samples.9.0.176-22781540-linux.run (3. CUDA 9.0 Samples).

Execute the second one to install the CUDA Toolkit 9.0:

$ sudo ./cuda-linux.9.0.176-22781540.run

You now have to accept the license by scrolling down to the bottom (hit the “d” key on your keyboard) and enter “accept”. Next accept the defaults.

To verify our CUDA installation, install the sample tests by

$ sudo ./cuda-samples.9.0.176-22781540-linux.run

After the installation finishes, configure the runtime library.

$ sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
$ sudo ldconfig

It is also recommended for Ubuntu users to append string /usr/local/cuda/bin to system file /etc/environment so that nvcc will be included in $PATH. This will take effect after reboot. To do that, you just have to

$ sudo vim /etc/environment

and then add :/usr/local/cuda/bin (including the ":") at the end of the PATH="/blah:/blah/blah" string (inside the quotes).
After a reboot, let's test our installation by making and invoking our tests:

$ cd /usr/local/cuda-9.0/samples
$ sudo make

It’s a long process with many irrelevant warnings about deprecated architectures (sm_20 and such ancient GPUs). After it completes, run deviceQuery and p2pBandwidthLatencyTest:

$ cd /usr/local/cuda/samples/bin/x86_64/linux/release
$ ./deviceQuery

The result of running deviceQuery should look something like this:

./deviceQuery Starting...
 CUDA Device Query (Runtime API) version (CUDART static linking)
Detected 1 CUDA Capable device(s)
Device 0: "GeForce GTX 1060"
  CUDA Driver Version / Runtime Version          9.0 / 9.0
  CUDA Capability Major/Minor version number:    6.1
  Total amount of global memory:                 6073 MBytes (6367739904 bytes)
  (10) Multiprocessors, (128) CUDA Cores/MP:     1280 CUDA Cores
  GPU Max Clock rate:                            1671 MHz (1.67 GHz)
  Memory Clock rate:                             4004 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 1572864 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 9.0, CUDA Runtime Version = 9.0, NumDevs = 1
Result = PASS

Cleanup: if ./deviceQuery works, remember to rm the 4 files (1 downloaded and 3 extracted).


3. Install cuDNN 7.0

The recommended way to install cuDNN 7.0 is to download all 3 .deb files. I had previously recommended using the .tgz installation approach, but found out that it didn’t allow verification by running code samples (no way to install the code samples .deb after .tgz installation).

The following steps are pretty much the same as the installation guide using .deb files (strange that the cuDNN guide is better than the CUDA one).

Go to the cuDNN download page (need registration) and select the latest cuDNN 7.0.* version made for CUDA 9.0.

Download all 3 .deb files: the runtime library, the developer library, and the code samples library for Ubuntu 16.04.

In your download folder, install them in the same order:

$ sudo dpkg -i libcudnn7_7.0.5.15–1+cuda9.0_amd64.deb (the runtime library),
$ sudo dpkg -i libcudnn7-dev_7.0.5.15–1+cuda9.0_amd64.deb (the developer library), and
$ sudo dpkg -i libcudnn7-doc_7.0.5.15–1+cuda9.0_amd64.deb (the code samples).

Now we can verify the cuDNN installation (below is just the official guide, which surprisingly works out of the box):

Copy the code samples somewhere you have write access: cp -r /usr/src/cudnn_samples_v7/ ~.

Go to the MNIST example code: cd ~/cudnn_samples_v7/mnistCUDNN.

Compile the MNIST example: make clean && make.

Run the MNIST example: ./mnistCUDNN. If your installation is successful, you should see Test passed! at the end of the output.

Do NOT Install cuda-command-line-tools

Contrary to the official TensorFlow installation docs, you don’t need to install cuda-command-line-tools because it’s already installed in this version of CUDA and cuDNN. If you apt-get it, you won’t find it.

Configure the CUDA and cuDNN library paths

What you do need to do, however, is exporting environment variables LD_LIBRARY_PATH in your .bashrc file:

# put the following line in the end or your .bashrc file

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"

And source it by source ~/.bashrc.



