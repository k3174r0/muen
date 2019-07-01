# muen / 夢魘
## usage
before use this program make sure that you already built darknet and opencv and installed CUDA toolkits/CUDNN.
```
git clone https://github.com/k3174r0/muen.git
cd muen
```
you have to change CmakeLists.txt:12 to your darknet path.

like set(DARKNET_PATH /path/to/darknet) -> set(DARKNET_PATH /home/k3174r0/darknet)
```
mkdir data && cd data
wget https://github.com/pjreddie/darknet/blob/master/cfg/jnet-conv.cfg
wget http://pjreddie.com/media/files/jnet-conv.weights
cd ..
mkdir build && cd build
cmake ..
make
```
and
```
./muen 10 10
```
