# 强化学习数学原理（赵世钰）

## 算法清单


## 环境配置

```bash
# 创建虚拟环境
conda create -n rl-learn python=3.7
conda activate rl-learn

# 安装依赖
pip install numpy matplotlib torch gymnasium tensorboard
```

## 使用示例
```bash
# 运行值迭代算法
python scripts/chapter4/value_iteration.py

# 运行DQN算法
python scripts/chapter8/dqn.py
```