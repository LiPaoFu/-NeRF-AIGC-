NeRF / AIGC 图像检测（FFT 频域分析 + ResNet18）

本仓库用于二分类检测：NeRF 渲染图像（nerf） vs 真实人脸图像（real），并包含频域（2D FFT）可解释性分析脚本。

包含内容
- scripts/train_resnet18.py：ResNet18 迁移学习训练与评测（两阶段：head -> finetune），输出 Acc/Precision/Recall/F1/AUC、混淆矩阵，并保存 result.json
- scripts/high_freq_power_ratio.py：高频功率占比统计（FFT）
- scripts/radial_spectrum_norm.py：归一化径向功率谱统计
- scripts/split_dataset.py：生成 train/val/test 数据划分目录
- results/：挑选的结果图与统计表（PNG/CSV）
- models/：权重输出目录（.pt 不随仓库提交）

重要：路径配置（必须）
本项目来自本地实验工程，部分脚本使用 Windows 绝对路径（如 E:\learn_pytorch\...）。
运行前请打开 scripts/train_resnet18.py，在 CFG 中修改：
- data_dir  -> 你的数据目录（data_split）
- out_dir   -> 你的输出目录（runs）
- model_dir -> 你的权重目录（models）
其他脚本也有绝对路径，同样需要改为你本机路径（不改变算法逻辑）。

数据目录结构（torchvision ImageFolder）
<data_dir>/
  train/nerf, train/real
  val/nerf,   val/real
  test/nerf,  test/real
可选跨域真实测试（CelebA）：
  test_celeba/nerf, test_celeba/real

运行方式
1）安装依赖
pip install -r requirements.txt

2）训练
python scripts/train_resnet18.py --seed 42

3）仅评测（不训练）
python scripts/train_resnet18.py --eval_only --ckpt <权重路径> --seed 42
结果会保存到 out_dir 下的 result_eval_only.json

频域分析（FFT）
python scripts/high_freq_power_ratio.py
python scripts/radial_spectrum_norm.py

仓库中已包含的代表性结果（精选）
- results/figures/box_high_freq_power_ratio.png
- results/tables/high_freq_power_ratio_stats.csv
- results/figures/box_high_freq_power_ratio_with_tiny.png
- results/tables/high_freq_power_ratio_stats_with_tiny.csv
- results/figures/radial_spectrum_comparison_norm_zoom_bins1_10.png
- results/tables/radial_spectrum_avg_norm.csv

权重说明
请不要提交 .pt/.pth 到 git。训练得到的权重放在：
models/best_head.pt
models/best_finetune.pt
