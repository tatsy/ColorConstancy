Color Constancy Algorithms 
---

> Implementations of several color constancy algorithms.

## Infomation

Some of the algorithms are imported to an image editing library [lime](https://github.com/tatsy/lime).
As this library includes only headers, you can test the algorithms more easily :-)

## Overview

This contains following algorithms:
* Horn's algorithm [Horn 1974, 1986]
* Blake's algorithm [Blake 1985]
* Moore's algorithm [Moore 1991]
* Rahman's algorithm [Rahman et al. 1991]
* Homomorphic filtering [Stockham Jr. 1972]
  
All of the codes are dependent on OpenCV 2.4 (maybe it
will work with older varsions as well). Each folder named
with aforementioned algorithm containes "main.cpp."
These programes include header "clcnst.h" that is 
in the folder named "clcnst".

## Installation

Please execute following command.

```shell
make all
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd`/bin/
export LD_LIBRARY_PATH
```

## Copyright

MIT License, Copyright 2013-2015, tatsy.
