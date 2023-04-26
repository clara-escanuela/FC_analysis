# FC_analysis
Characterisation of the FlashCam response

This analysis needs ctapipe: https://github.com/cta-observatory/ctapipe
You also need fcio to read FlashCam data:

```
cd <pick your local software repo directory>
git clone https://www.mpi-hd.mpg.de/software/FlashCam/fcio.git
cd fcio
make # required to fetch the needed submodules
cd python
# in case you don't have it installed
pip install build
make install
```
