# MoRobtrol D 组质量测试报告

[English](REPORT.md) | 中文

生成日期：2026-07-15  
数据范围：`mo_hopper_m3`、`mo_swimmer`；CUDA-MOEA 与 EvoX；NSGA-III 与 RVEA；每组 10 个 seed；名义种群 16384；第 0–100 代（每 5 代检查点）。

## 结论摘要

- **Hopper / NSGA-III：CUDA-MOEA 综合占优。** 第 100 代中位 HV 为 4.846e9（EvoX 4.538e9），EU 为 0.4906（EvoX 0.4856），100 代中位算法时间为 9.99 min（EvoX 11.65 min）。在相同的 2.93、5.85、8.75 min 预算下，CUDA-MOEA 的中位 HV 分别高 9.5%、7.4%、13.3%，中位 EU 分别高 3.4%、2.0%、3.8%。达到 EvoX 最终中位目标时，CUDA-MOEA 的 HV/EU 达到率分别为 90%/80%，EvoX 为 70%/60%；已达目标 seed 的时间中位数加速分别为 1.59×/1.52×。
- **Hopper / RVEA：后程 EvoX 质量更高。** EvoX 第 100 代中位 HV/EU 为 1.318e10/0.7051，CUDA-MOEA 为 1.229e10/0.6995。相同时间下 CUDA-MOEA 仅在 2.87 min 的早期预算领先（HV +5.7%、EU +2.0%）；到 5.73 和 8.59 min，EvoX 的 HV 分别高 13.5%、14.7%，EU 分别高 7.0%、5.4%。CUDA-MOEA 100 代时间略短（11.17 min 对 11.46 min），但未达到 EvoX 最终中位 HV，不能将小幅时间优势解释为同质量加速。
- **Swimmer / NSGA-III：最终质量基本持平，CUDA-MOEA 时间效率更高。** 第 100 代中位 HV 为 33.87 对 34.06，EU 为 0.74095 对 0.74092；CUDA-MOEA 100 代中位时间 11.59 min，EvoX 为 19.19 min（约 1.66× 总时间差）。在相同的 4.84 和 9.64 min 预算下，CUDA-MOEA 的中位 HV 分别高 10.1%、7.7%，中位 EU 分别高 4.3%、1.7%。
- **Swimmer / RVEA：CUDA-MOEA 在最终质量和等时质量上均显著占优。** 第 100 代中位 HV 为 11.43（EvoX 0.7765），EU 为 0.5163（EvoX 0.05566）；两者总时间接近。在相同的 2.96、5.92、8.88 min 预算下，CUDA-MOEA 的中位 HV 是 EvoX 的 26.6×、17.2×、15.1×，中位 EU 是 9.4×、8.3×、9.1×。由于以 EvoX 最终值定义的目标在初始检查点即可达到，相关 Time-to-target 的 0 min 结果只表示目标区分度不足，不代表无限加速。

## 判读口径

- 曲线实线/虚线为 10 个 seed 的中位数，阴影为四分位距（IQR）；HV 与 EU 均为越大越好。
- 时间仅统计优化算法的检查点累计时间，不含独立评估、指标计算、写盘和绘图。
- MoRobtrol 优化阶段包含仿真环境中的策略评估。仿真评估在每代耗时中占比较大且两框架都必须承担，因此会稀释 CUDA-MOEA 在进化算子上的速度优势；种群越大，选择、排序和种群操作的并行计算占比越高，速度优势才越容易体现。对于单步仿真更耗时的环境，共同的评估成本占比更高，观察到的整体速度优势通常会进一步减小。
- Quality-at-time 只使用两套实现共同可观测时间范围内的预算；`missing` 表示该预算下无可用检查点的 seed 数。
- Time-to-target 的目标取 EvoX 最终中位数及其 90%，仅在已观测的 100 代内判断；时间统计只覆盖实际达到目标的 seed。
- 最终前沿使用各“环境–实现–算法”中第 100 代 HV 最接近 10-seed 中位数的代表运行。

## mo_hopper_m3

### 按代数的收敛质量

![mo_hopper_m3 Generation–HV/EU](images/mo_hopper_m3_generation.png)

### 按算法时间的收敛质量

![mo_hopper_m3 Time–HV/EU](images/mo_hopper_m3_time.png)

### 最终非支配前沿

![mo_hopper_m3 最终非支配前沿](images/mo_hopper_m3_final_front.png)
## mo_swimmer

### 按代数的收敛质量

![mo_swimmer Generation–HV/EU](images/mo_swimmer_generation.png)

### 按算法时间的收敛质量

![mo_swimmer Time–HV/EU](images/mo_swimmer_time.png)

### 最终非支配前沿

![mo_swimmer 最终非支配前沿](images/mo_swimmer_final_front.png)


## Quality-at-time

### HV

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>environment</th>
      <th>algorithm</th>
      <th>budget_source</th>
      <th>budget_time_min</th>
      <th>common_horizon_min</th>
      <th>CUDA-MOEA median [q1, q3]</th>
      <th>CUDA-MOEA missing</th>
      <th>EvoX median [q1, q3]</th>
      <th>EvoX missing</th>
      <th>CUDA-MOEA−EvoX median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mo_hopper_m3</td>
      <td>nsga3</td>
      <td>EvoX generation 25 median</td>
      <td>2.9305</td>
      <td>9.9926</td>
      <td>3.833e+09 [3.538e+09, 3.981e+09]</td>
      <td>0</td>
      <td>3.502e+09 [3.375e+09, 3.559e+09]</td>
      <td>0</td>
      <td>3.31e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mo_hopper_m3</td>
      <td>nsga3</td>
      <td>EvoX generation 50 median</td>
      <td>5.8543</td>
      <td>9.9926</td>
      <td>4.235e+09 [4.005e+09, 4.498e+09]</td>
      <td>0</td>
      <td>3.945e+09 [3.783e+09, 4.101e+09]</td>
      <td>0</td>
      <td>2.901e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mo_hopper_m3</td>
      <td>nsga3</td>
      <td>EvoX generation 75 median</td>
      <td>8.7459</td>
      <td>9.9926</td>
      <td>4.791e+09 [4.486e+09, 4.971e+09]</td>
      <td>0</td>
      <td>4.228e+09 [4.148e+09, 4.499e+09]</td>
      <td>0</td>
      <td>5.632e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mo_hopper_m3</td>
      <td>rvea</td>
      <td>EvoX generation 25 median</td>
      <td>2.8655</td>
      <td>11.17</td>
      <td>5.486e+09 [5.275e+09, 5.651e+09]</td>
      <td>0</td>
      <td>5.192e+09 [5.056e+09, 5.375e+09]</td>
      <td>0</td>
      <td>2.941e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mo_hopper_m3</td>
      <td>rvea</td>
      <td>EvoX generation 50 median</td>
      <td>5.7305</td>
      <td>11.17</td>
      <td>7.096e+09 [6.525e+09, 7.622e+09]</td>
      <td>0</td>
      <td>8.053e+09 [6.803e+09, 1.109e+10]</td>
      <td>0</td>
      <td>-9.57e+08</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mo_hopper_m3</td>
      <td>rvea</td>
      <td>EvoX generation 75 median</td>
      <td>8.5949</td>
      <td>11.17</td>
      <td>1.038e+10 [7.752e+09, 1.144e+10]</td>
      <td>0</td>
      <td>1.191e+10 [9.251e+09, 1.278e+10]</td>
      <td>0</td>
      <td>-1.53e+09</td>
    </tr>
    <tr>
      <th>6</th>
      <td>mo_swimmer</td>
      <td>nsga3</td>
      <td>EvoX generation 25 median</td>
      <td>4.841</td>
      <td>11.59</td>
      <td>30.68 [29.6, 31.68]</td>
      <td>0</td>
      <td>27.86 [26.84, 28.72]</td>
      <td>0</td>
      <td>2.825</td>
    </tr>
    <tr>
      <th>7</th>
      <td>mo_swimmer</td>
      <td>nsga3</td>
      <td>EvoX generation 50 median</td>
      <td>9.6404</td>
      <td>11.59</td>
      <td>33.22 [32.8, 33.74]</td>
      <td>0</td>
      <td>30.84 [30.14, 31.53]</td>
      <td>0</td>
      <td>2.376</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mo_swimmer</td>
      <td>rvea</td>
      <td>EvoX generation 25 median</td>
      <td>2.9565</td>
      <td>11.571</td>
      <td>14.08 [8.554, 19.57]</td>
      <td>0</td>
      <td>0.5286 [0.08181, 1.681]</td>
      <td>0</td>
      <td>13.55</td>
    </tr>
    <tr>
      <th>9</th>
      <td>mo_swimmer</td>
      <td>rvea</td>
      <td>EvoX generation 50 median</td>
      <td>5.9237</td>
      <td>11.571</td>
      <td>13.82 [7.707, 16.71]</td>
      <td>0</td>
      <td>0.8055 [0.3276, 1.731]</td>
      <td>0</td>
      <td>13.01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>mo_swimmer</td>
      <td>rvea</td>
      <td>EvoX generation 75 median</td>
      <td>8.8758</td>
      <td>11.571</td>
      <td>9.814 [8.86, 13.96]</td>
      <td>0</td>
      <td>0.6484 [0.3994, 1.543]</td>
      <td>0</td>
      <td>9.166</td>
    </tr>
  </tbody>
</table>
</div>

### EU

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>environment</th>
      <th>algorithm</th>
      <th>budget_source</th>
      <th>budget_time_min</th>
      <th>common_horizon_min</th>
      <th>CUDA-MOEA median [q1, q3]</th>
      <th>CUDA-MOEA missing</th>
      <th>EvoX median [q1, q3]</th>
      <th>EvoX missing</th>
      <th>CUDA-MOEA−EvoX median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mo_hopper_m3</td>
      <td>nsga3</td>
      <td>EvoX generation 25 median</td>
      <td>2.9305</td>
      <td>9.9926</td>
      <td>0.4647 [0.4537, 0.4737]</td>
      <td>0</td>
      <td>0.4495 [0.448, 0.4523]</td>
      <td>0</td>
      <td>0.01526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mo_hopper_m3</td>
      <td>nsga3</td>
      <td>EvoX generation 50 median</td>
      <td>5.8543</td>
      <td>9.9926</td>
      <td>0.4736 [0.4635, 0.4845]</td>
      <td>0</td>
      <td>0.4645 [0.4588, 0.4792]</td>
      <td>0</td>
      <td>0.009148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mo_hopper_m3</td>
      <td>nsga3</td>
      <td>EvoX generation 75 median</td>
      <td>8.7459</td>
      <td>9.9926</td>
      <td>0.4931 [0.4842, 0.4966]</td>
      <td>0</td>
      <td>0.4752 [0.4717, 0.4842]</td>
      <td>0</td>
      <td>0.01795</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mo_hopper_m3</td>
      <td>rvea</td>
      <td>EvoX generation 25 median</td>
      <td>2.8655</td>
      <td>11.17</td>
      <td>0.511 [0.5006, 0.5177]</td>
      <td>0</td>
      <td>0.5011 [0.4988, 0.5099]</td>
      <td>0</td>
      <td>0.009927</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mo_hopper_m3</td>
      <td>rvea</td>
      <td>EvoX generation 50 median</td>
      <td>5.7305</td>
      <td>11.17</td>
      <td>0.5506 [0.5402, 0.5755]</td>
      <td>0</td>
      <td>0.5893 [0.5511, 0.6567]</td>
      <td>0</td>
      <td>-0.03868</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mo_hopper_m3</td>
      <td>rvea</td>
      <td>EvoX generation 75 median</td>
      <td>8.5949</td>
      <td>11.17</td>
      <td>0.6441 [0.5716, 0.6801]</td>
      <td>0</td>
      <td>0.6787 [0.6161, 0.704]</td>
      <td>0</td>
      <td>-0.03462</td>
    </tr>
    <tr>
      <th>6</th>
      <td>mo_swimmer</td>
      <td>nsga3</td>
      <td>EvoX generation 25 median</td>
      <td>4.841</td>
      <td>11.59</td>
      <td>0.7337 [0.7276, 0.7359]</td>
      <td>0</td>
      <td>0.7038 [0.6963, 0.7148]</td>
      <td>0</td>
      <td>0.02991</td>
    </tr>
    <tr>
      <th>7</th>
      <td>mo_swimmer</td>
      <td>nsga3</td>
      <td>EvoX generation 50 median</td>
      <td>9.6404</td>
      <td>11.59</td>
      <td>0.7409 [0.7377, 0.7453]</td>
      <td>0</td>
      <td>0.7283 [0.7157, 0.7353]</td>
      <td>0</td>
      <td>0.01263</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mo_swimmer</td>
      <td>rvea</td>
      <td>EvoX generation 25 median</td>
      <td>2.9565</td>
      <td>11.571</td>
      <td>0.5228 [0.4989, 0.6418]</td>
      <td>0</td>
      <td>0.05561 [0.03557, 0.118]</td>
      <td>0</td>
      <td>0.4672</td>
    </tr>
    <tr>
      <th>9</th>
      <td>mo_swimmer</td>
      <td>rvea</td>
      <td>EvoX generation 50 median</td>
      <td>5.9237</td>
      <td>11.571</td>
      <td>0.5266 [0.4994, 0.5402]</td>
      <td>0</td>
      <td>0.06314 [0.03408, 0.08756]</td>
      <td>0</td>
      <td>0.4635</td>
    </tr>
    <tr>
      <th>10</th>
      <td>mo_swimmer</td>
      <td>rvea</td>
      <td>EvoX generation 75 median</td>
      <td>8.8758</td>
      <td>11.571</td>
      <td>0.5087 [0.5043, 0.5289]</td>
      <td>0</td>
      <td>0.05617 [0.04086, 0.09104]</td>
      <td>0</td>
      <td>0.4525</td>
    </tr>
  </tbody>
</table>
</div>

## Time-to-target

### HV

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>environment</th>
      <th>algorithm</th>
      <th>target_label</th>
      <th>target_scope</th>
      <th>target_value</th>
      <th>CUDA-MOEA reached_rate [reached/total]</th>
      <th>CUDA-MOEA time min median [q1, q3]</th>
      <th>EvoX reached_rate [reached/total]</th>
      <th>EvoX time min median [q1, q3]</th>
      <th>Speedup（EvoX / CUDA-MOEA）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mo_hopper_m3</td>
      <td>nsga3</td>
      <td>90% EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>4.084e+09</td>
      <td>1 [10/10]</td>
      <td>3.997 [2.624, 4.999]</td>
      <td>1 [10/10]</td>
      <td>6.983 [5.182, 8.167]</td>
      <td>1.747×</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mo_hopper_m3</td>
      <td>nsga3</td>
      <td>EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>4.538e+09</td>
      <td>0.9 [9/10]</td>
      <td>5.501 [4, 6.492]</td>
      <td>0.7 [7/10]</td>
      <td>8.752 [7.072, 9.885]</td>
      <td>1.591×</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mo_hopper_m3</td>
      <td>rvea</td>
      <td>90% EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>1.186e+10</td>
      <td>0.6 [6/10]</td>
      <td>9.496 [9.074, 9.913]</td>
      <td>0.7 [7/10]</td>
      <td>7.441 [6.294, 8.038]</td>
      <td>0.7836×</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mo_hopper_m3</td>
      <td>rvea</td>
      <td>EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>1.318e+10</td>
      <td>0 [0/10]</td>
      <td>— [—, —]</td>
      <td>0.5 [5/10]</td>
      <td>9.151 [8.583, 9.746]</td>
      <td>—</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mo_swimmer</td>
      <td>nsga3</td>
      <td>90% EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>30.65</td>
      <td>1 [10/10]</td>
      <td>5 [4.206, 5.769]</td>
      <td>1 [10/10]</td>
      <td>9.217 [6.061, 10.33]</td>
      <td>1.844×</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mo_swimmer</td>
      <td>nsga3</td>
      <td>EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>34.06</td>
      <td>0.3 [3/10]</td>
      <td>8.797 [8.448, 9.336]</td>
      <td>0.5 [5/10]</td>
      <td>17.25 [15.39, 19.18]</td>
      <td>1.96×</td>
    </tr>
    <tr>
      <th>6</th>
      <td>mo_swimmer</td>
      <td>rvea</td>
      <td>90% EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>0.6988</td>
      <td>1 [10/10]</td>
      <td>0 [0, 0]</td>
      <td>1 [10/10]</td>
      <td>0 [0, 0]</td>
      <td>1×</td>
    </tr>
    <tr>
      <th>7</th>
      <td>mo_swimmer</td>
      <td>rvea</td>
      <td>EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>0.7765</td>
      <td>1 [10/10]</td>
      <td>0 [0, 0]</td>
      <td>1 [10/10]</td>
      <td>0 [0, 0]</td>
      <td>1×</td>
    </tr>
  </tbody>
</table>
</div>

### EU

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>environment</th>
      <th>algorithm</th>
      <th>target_label</th>
      <th>target_scope</th>
      <th>target_value</th>
      <th>CUDA-MOEA reached_rate [reached/total]</th>
      <th>CUDA-MOEA time min median [q1, q3]</th>
      <th>EvoX reached_rate [reached/total]</th>
      <th>EvoX time min median [q1, q3]</th>
      <th>Speedup（EvoX / CUDA-MOEA）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mo_hopper_m3</td>
      <td>nsga3</td>
      <td>90% EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>0.4371</td>
      <td>1 [10/10]</td>
      <td>0.4998 [0.4989, 0.9991]</td>
      <td>1 [10/10]</td>
      <td>0.8823 [0.5881, 1.178]</td>
      <td>1.765×</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mo_hopper_m3</td>
      <td>nsga3</td>
      <td>EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>0.4856</td>
      <td>0.8 [8/10]</td>
      <td>5.001 [3.997, 7.122]</td>
      <td>0.6 [6/10]</td>
      <td>7.618 [6.219, 8.614]</td>
      <td>1.524×</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mo_hopper_m3</td>
      <td>rvea</td>
      <td>90% EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>0.6346</td>
      <td>0.8 [8/10]</td>
      <td>7.54 [7.12, 8.692]</td>
      <td>1 [10/10]</td>
      <td>6.878 [5.722, 9.038]</td>
      <td>0.9122×</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mo_hopper_m3</td>
      <td>rvea</td>
      <td>EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>0.7051</td>
      <td>0.3 [3/10]</td>
      <td>11.17 [10.89, 11.17]</td>
      <td>0.6 [6/10]</td>
      <td>9.165 [8.583, 10.59]</td>
      <td>0.8207×</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mo_swimmer</td>
      <td>nsga3</td>
      <td>90% EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>0.6668</td>
      <td>1 [10/10]</td>
      <td>1.161 [0.5812, 1.167]</td>
      <td>1 [10/10]</td>
      <td>0.9967 [0.9893, 1.958]</td>
      <td>0.8588×</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mo_swimmer</td>
      <td>nsga3</td>
      <td>EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>0.7409</td>
      <td>0.6 [6/10]</td>
      <td>6.085 [5.788, 6.819]</td>
      <td>0.5 [5/10]</td>
      <td>14.44 [13.45, 17.41]</td>
      <td>2.374×</td>
    </tr>
    <tr>
      <th>6</th>
      <td>mo_swimmer</td>
      <td>rvea</td>
      <td>90% EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>0.05009</td>
      <td>1 [10/10]</td>
      <td>0 [0, 0]</td>
      <td>1 [10/10]</td>
      <td>0 [0, 0]</td>
      <td>1×</td>
    </tr>
    <tr>
      <th>7</th>
      <td>mo_swimmer</td>
      <td>rvea</td>
      <td>EvoX final median</td>
      <td>within observed 100-generation run</td>
      <td>0.05566</td>
      <td>1 [10/10]</td>
      <td>0 [0, 0]</td>
      <td>1 [10/10]</td>
      <td>0 [0, 0]</td>
      <td>1×</td>
    </tr>
  </tbody>
</table>
</div>

## 数据与复现

- 原始派生表位于 [`data/`](data/)；完整数值应以 CSV 为准。
- 图表和表格由 [`plot_quality_curves.ipynb`](../plot_quality_curves.ipynb) 的已保存输出提取，本报告生成器只负责整理，不重新优化或独立评估。
- 重新执行 notebook 后，运行 `python tests/benchmark/MoRobtrol/generate_quality_report.py` 可刷新本报告。
