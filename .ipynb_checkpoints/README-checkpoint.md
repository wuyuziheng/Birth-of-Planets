# 行星的形成
基于期中的行星运动，加入合并检测，输出可视化视频。在速度上做了许多优化，并支持自动调参，方便使用。

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

### 参数说明
在`config`文件中，我们有如下参数，假设长度量纲为x，质量量纲为y，时间量纲为z。
|参数|描述|单位|
| --- | --- | --- |
|G | 万有引力常数 | $x^3y^{-1}z^{-2}$|
|rho | 行星初始密度 | $x^{-3}y$|
|sun_mass | 中心恒心质量|$y$|
|merging_threshold | 合并检测阈值|无|
|per_step_time |每步预测间隔|$z$|
|num_steps |单个chunk的总步长|无|
|basic_radii |行星初始半径|$x$|
|num_chunks |chunk个数|无|
|init_config |关于初始化的参数||
|    num_planets |行星个数|无|
|    pos_norm |初始位置|$x$|
|    pos_bias |初始位置扰动|$x$|
|    v_norm |初始速度|$xz^{-1}$|
|    v_bias |初始速度扰动|$xz^{-1}$|
|figure_config |关于绘图的参数||
|    no_figure |不需要视频|无|
|    margin_bias |图片边缘大小|$x$|
|    plot_scale |图中行星的大小|无|
|    range_quantile |主要视图的分位数|无|
|    record_steps |视频记录的步长|无|
|auto_config_method |auto_config的方法|无|
|device |运行设备|无|

在`auto_config`文件中，我们有如下方法：

### 建模思路
直接使用`all-pairs`的方式暴力计算所有引力，并使用CUDA并行计算加速。假设所有行星密度相同，初始的半径也相同。初始位置为行星位于恒星的近日点，距离为`pos_norm`，初始速度`v_norm`在满足近日点的范围内。初始行星分布以初始位置为中心，以`pos_bias`往外呈正态分布，速度同理，在初始速度的基础上以`v_bias`往外呈正态分布。使用`Runge-Kutta`法打点计算行星位置，并在每步计算完后检测是否有合并，有则将两个行星合并到其质心，保持密度，质量，动量不变。

### 实现思路
主要文件在`model.py`下，在期中项目的基础上实现`merge`检测，可以调整`./config/config.json`来设置参数。`init_utils`文件下是状态初始化，包括初始的“位置-速度”状态和初始行星半径，可以自定义新的初始化状态。`auto_config`实现了自动调参功能，其配置文件可以在`./config/auto_config.json`中修改。

代码实现的目标是找到一种快速的方式模拟N-体运动，然后将结果绘制为动图或者视频呈现出来。主要瓶颈在加速度计算和绘图，我们经历了如下尝试：
1. 使用`Barnes-Hut`方法加速N-体问题计算，但是不方便并行计算，遂放弃，希望找到直接简单有效的并行计算方式。
2. 觉得代码中部需要多次使用`pytorch`计算，希望能直接使用`CUDA`编程将每个行星单元的加速度计算过程打包，分配到单个GPU core上计算，设置`CUDA`并行为：$$(GridSize, BlockSize) = (number_of_planets, 1024)$$将每个block上计算一个行星的加速度，其中分1024个thread并行计算其他行星对该行星的力，再除以质量总和为加速度。但是最后发现速度不如直接使用torch原生的cuda支持。
3. 对绘图做并行（失败，由于python的多线程有GIL限制很难用，但是多进程IPC消耗太大，几乎没有提升）
4. 使用`matplotlib.animation`原生支持或者`imageio`（失败，有内存限制，超过会被kill）

经过一些失败的尝试，认为应该高效化`torch`代码，并加速`merge`验证过程。于是设计思路为：
1. 全程使用向量化操作，并尽可能减少加速度的计算次数。
2. 使用`Runge Kutta 4`代替`Runge Kutta 5`，在精度几乎不变的情况下提升计算速度。
3. `merge`检测会先使用向量化操作“预检测”一次是否有合并，减少`merge`的判定次数。
4. 使用`opencv`来制作.mp4可视化视频，并且只对部分步骤采样，否则太慢。
5. 由于步骤太多会导致内存不够，遂实现了chunk化操作，可以一个chunk一个chunk的进行“预测-绘图-重置”循环，然后最后把所有chunk的.mp4文件合并成final.mp4文件。

在此基础上，我们加入了初始化代码，可以自行在`init_utils`下调整。此外，由于参数众多，我们实现了自动调参器`auto_config`。由于我们是忽略量纲的参数体系，所以我们有三个基础的单位恒等变换`id-length`,`id-mass`,`id-time`。在此基础上，我们根据加速度的计算式可以观察到一个新的恒等变换`id-scope`。在此基础上，我们自定义了6个有物理含义的基础参数`merger_checking`,`orbit_constant`,`force_ratio`,`angular_velocity`,`init_pos_ratio`,`init_vel_ratio`，并使用这些物理量加上四个重要的参数`G`,`sun_mass`,`merging_threshold`,`pos_norm`来重构了参数的依赖关系，从而只需要调节这些参数即可。



