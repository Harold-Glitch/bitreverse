nvcc program.cu rng.cpp secp256k1.cpp -std=c++11 -I/usr/local/include -L/usr/local/lib -lboost_system -lboost_filesystem -lnvidia-ml --compiler-options -fpermissive
