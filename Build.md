#### Building problems and solutions

1. fatal error: cuda_runtime.h: No such file or directory
    ```markdown
    CPATH=/path/to/your/cuda/include ./make.sh 
    ```
    [reference](https://github.com/roytseng-tw/Detectron.pytorch/issues/17)
2. error: ‘for’ loop initial declarations are only allowed in C99 mode;  
   CompileError: command 'gcc' failed with exit status 1
   
   gcc version should >= 5.4
   
    ```markdown
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install gcc-5 g++-5

    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 --slave /usr/bin/g++ g++ /usr/bin/g++-5
    ```
   [reference](https://github.com/jwyang/faster-rcnn.pytorch/issues/127)