# 行星的形成
基于期中的行星运动，加入合并检测，输出可视化视频。在速度上做了些许优化。

### 快速开始
从github拉取：
~~~
git clone https://github.com/Kouer-www/Birth-of-Planets.git
~~~

首先安装依赖包：
~~~
conda create -n bop python==3.12
conda activate bop
pip install -r requirements.txt
~~~

一键启动：
~~~
python model.py
~~~

可以更改的是`config.json`以设置参数，`init_utils`文件来自定义初始化状态。

### 实现思路
主要文件在`model.py`下，在期中项目的基础上实现`merge`检测，可以调整`config.json`来设置参数。`init_utils`文件下是状态初始化，包括初始的“位置-速度”状态和初始行星半径，可以自定义新的初始化状态。

尝试过的方向：
1. 对绘图做并行（失败，由于python的多线程有GIL限制很难用，但是多进程IPC消耗太大，几乎没有提升）
2. 使用matplotlib.animation或者imageio（失败，有内存限制，超过会被kill）

细节：
1. 现在使用opencv来制作.mp4可视化视频，并且只对部分步骤采样，否则太慢。由于步骤太多会导致内存不够，遂实现了chunk化操作，可以一个chunk一个chunk的进行“预测-绘图-重置”循环，然后最后把所有chunk的.mp4文件合并成final.mp4文件。
2. `merge`检测会先使用向量化操作“预检测”一次是否有合并，只有在有合并的时候才会搜索合并发生的位置，大大增加了吞吐量。

实现了自动调参器，

