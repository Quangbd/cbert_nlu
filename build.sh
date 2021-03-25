rm -rf build
mkdir "build"
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_ENV=osx_x86_64 ..
#cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_ENV=ubuntu_x86_64 ..
make -j4
./cbert_nlu