# GMRec
This is a Tensorflow implementation for GMRec and a part of baselines:

>Bin Wu, Xun Su, Jing Liang, Zhongchuan Sun, Lihong Zhong & Yangdong Ye. Graph Gating-Mixer for Sequential Recommendation, Expert Systems With Applications.

## Environment Requirement
The code has been tested running under Python 3.8.10. The required packages are as follows:
* tensorflow == 1.14.0
* numpy == 1.19.5
* scipy == 1.4.1
* pandas == 1.3.1

## C++ evaluator
We have implemented C++ code to output metrics during and after training, which is much more efficient than python evaluator. It needs to be compiled first using the following command. 
```
python setup.py build_ext --inplace
```
If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.

