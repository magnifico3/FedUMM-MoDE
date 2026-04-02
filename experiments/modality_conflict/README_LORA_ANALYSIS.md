# LoRA 权重变化相似性分析

分析四个任务（CC3M、InstructPix2Pix、WikiText、VQAv2）在训练过程中 LoRA 权重变化方向的相似性。

## 概述

LoRA 参数由两个矩阵 A 和 B 组成，实际的权重变化（deltaW）为 `B @ A`。本分析通过以下步骤进行：

1. **计算权重变化**：从 `round_params.pt` 中提取 LoRA 参数，计算 `B @ A` 得到每层的权重变化
2. **向量化**：将所有层的权重变化展平为单个向量
3. **相似性分析**：计算四个任务间两两的点积、余弦相似度、Pearson 相关系数

## 运行步骤

### 第一步：计算真实权重变化（预处理）

生成 `round_real_flat.pt` 文件，包含每轮的完整权重变化向量：

```bash
cd /root/ad

conda run -n nvflare_januspro python experiments/modality_conflict/compute_real_params.py \
  outputs/modality_conflict_sched/cc3m_s1000_e250_lr3e4 \
  outputs/modality_conflict_sched/instruct_s1000_e250_lr3e4 \
  outputs/modality_conflict_sched/text_s4000_e1000_lr2e4 \
  outputs/modality_conflict_sched/vqav2_s2000_e500_lr3e5
```

**输出**：每个任务的每一轮生成 `round_real_flat.pt` 文件
- 大小：约 3.1 GB/轮（8.3 亿参数）
- 处理时间：~5-10分钟（取决于硬件）

### 第二步：分析相似性

计算四个任务间的权重变化方向相似性：

```bash
cd /root/ad

conda run -n nvflare_januspro python experiments/modality_conflict/analyze_lora_similarity.py \
  outputs/modality_conflict_sched/cc3m_s1000_e250_lr3e4 \
  outputs/modality_conflict_sched/instruct_s1000_e250_lr3e4 \
  outputs/modality_conflict_sched/text_s4000_e1000_lr2e4 \
  outputs/modality_conflict_sched/vqav2_s2000_e500_lr3e5
```

**输出**：
- CSV 文件：`outputs/modality_conflict_sched/similarity_report_real.csv`
- 包含 60 个两两比较（4 个任务 × 10 轮 × C(4,2)=6 个任务对）

## 输出解释

### CSV 列说明

| 列名 | 描述 |
|------|------|
| `round` | 训练轮数（1-10） |
| `task_a` / `task_b` | 两个任务名称 |
| `dot_product` | 点积 = ΣA_i × B_i（绝对相似度） |
| `cosine_similarity` | 余弦相似度 = dot/(‖A‖×‖B‖)，范围[-1,1] |
| `pearson_correlation` | Pearson相关系数，消除尺度影响 |
| `l2_norm_a` / `l2_norm_b` | 权重变化的 L2 范数 |
| `conflict` | 是否冲突（dot < 0） |

### 统计数据示例

从上次运行的结果：

**总体统计**
- 总比较数：60
- 冲突比例：11.7%（7/60）
- 余弦相似度均值：0.0056

**任务对级别**
```
cc3m ↔ instruct:  Mean cosine = 0.0012  (最小相似度)
cc3m ↔ text:      Mean cosine = 0.0108
cc3m ↔ vqav2:     Mean cosine = 0.0150  (最大相似度)
instruct ↔ text:   Mean cosine = 0.0009
instruct ↔ vqav2:  Mean cosine = 0.0000  (7处冲突)
text ↔ vqav2:      Mean cosine = 0.0056
```

**解读**：
- 总体相似度低（<0.02），说明不同任务的权重变化方向差异明显
- `instruct-vqav2` 冲突最多，说明这两个任务对权重的更新方向最不一致
- `cc3m-vqav2` 相似度最高，但仍然很低

## 自定义选项

### 检查单个任务的配置

查看元数据：
```bash
cat outputs/modality_conflict_sched/cc3m_s1000_e250_lr3e4/metadata.json
```

### 指定输出文件名

```bash
conda run -n nvflare_januspro python experiments/modality_conflict/analyze_lora_similarity.py \
  outputs/modality_conflict_sched/cc3m_s1000_e250_lr3e4 \
  outputs/modality_conflict_sched/instruct_s1000_e250_lr3e4 \
  outputs/modality_conflict_sched/text_s4000_e1000_lr2e4 \
  outputs/modality_conflict_sched/vqav2_s2000_e500_lr3e5 \
  --output_csv custom_report.csv
```

## 文件说明

- [compute_real_params.py](compute_real_params.py) - 预处理脚本，生成 `round_real_flat.pt`
- [analyze_lora_similarity.py](analyze_lora_similarity.py) - 分析脚本，计算相似性指标

## 数据结构

```
outputs/modality_conflict_sched/
├── cc3m_s1000_e250_lr3e4/
│   ├── metadata.json
│   └── round_001/
│       ├── round_params.pt         (原始 LoRA 参数)
│       ├── round_grad.pt / round_grad_flat.pt
│       ├── round_delta.pt / round_delta_flat.pt
│       └── round_real_flat.pt      (新生成：B@A 结果)
│       └── summary.json
│   ├── round_002/
│   ├── ...
│   └── round_010/
├── instruct_s1000_e250_lr3e4/
├── text_s4000_e1000_lr2e4/
├── vqav2_s2000_e500_lr3e5/
└── similarity_report_real.csv      (最终结果)
```

## 内存要求

- **第一步**：~4-6 GB（处理单个round时的峰值）
- **第二步**：~2-3 GB（流式加载）
- 总空间：~3.1 GB × 40 rounds ≈ **124 GB**（所有任务所有轮）

## 性能提示

- 处理 40 轮数据（4任务×10轮）需要约 **15-20 分钟**
- 两个脚本都支持在内存受限的环境下运行（设计中已优化）
- 可以按任务批处理以节省内存
