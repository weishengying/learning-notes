## gtest 安装
参考https://github.com/google/googletest/blob/main/googletest/README.md

    git clone https://github.com/google/googletest.git -b release-1.11.0

    cd googletest        # Main directory of the cloned repository.

    mkdir build          # Create a directory to hold the build output.

    cd build

    cmake .. -DBUILD_GMOCK=OFF          # Generate native build scripts for GoogleTest.

    make

    sudo make install    # Install in /usr/local/ by default



