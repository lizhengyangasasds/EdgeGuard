# EdgeGuard: Edge-Cloud Collaborative Multimodal Security System

<div align="center">

**[English](#english) | [中文](#chinese)**

<a name="english"></a>

## English

**EdgeGuard** is an edge-cloud collaborative multimodal anomaly behavior detection system. It performs real-time inference on video streams (visual + text) at the edge, while completing model training and updates in the cloud via PEFT, with TensorRT deployment on edge devices.

### Key Features

- **Multimodal Fusion**: DeiT (visual) + DistilBERT (text) + LSTM (temporal) + Cross-Attention
- **PEFT Fine-tuning**: LoRA + Adapter LoRA with <1% trainable parameters
- **TensorRT Deployment**: INT8 quantization, dynamic batching, <85ms latency
- **Edge-Cloud Sync**: MQTT-based model push and inference reporting
- **7 Behavior Classes**: Fighting, Falling, Climbing, Loitering, Retrograde, Gathering, Normal
- **5 Alert Classes**: Intrusion, Fault, Violation, Anomaly, Normal

### Architecture

```
Cloud (Training)                 Edge (Inference)
┌──────────────────────┐  ┌──────────────────────┐
│ Pre-train            │  │ TensorRT Engine       │
│ (COCO2017/UCF101)    │──push──>│ (INT8 Quantized)   │
│                      │  │                      │
│ PEFT Fine-tune       │  │ ┌──────┴──────┐      │
│ (LoRA + Adapter)     │  │ │DeiT(visual) │      │
│                      │  │ │DistilBERT   │      │
│ Model Registry       │  │ │LSTM(temp)   │      │
│ (versioned)          │  │ └──────┬──────┘      │
│                      │  │ Cross-Modal Attn     │
│ Alert Pipeline       │  │ Behavior Classifier  │
└──────────────────────┘  └──────────────────────┘
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourrepo/edgeguard.git
cd edgeguard

# Install dependencies
pip install -r requirements.txt

# Phase 1: Run model demo
python demo/demo_edge.py

# Phase 2: Fine-tune with PEFT
python training/train.py --config configs/train_config.yaml

# Phase 3: Export to TensorRT
python deployment/onnx_to_trt.py --config configs/deploy_config.yaml

# Phase 4: Start edge service
python edge_cloud/edge_service.py
```

### Project Structure

```
EdgeGuard/
├── configs/              # YAML configuration files
├── model/                # Model architecture (DeiT+DistilBERT+LSTM+CrossAttn)
├── training/             # PEFT fine-tuning pipeline
├── deployment/           # TensorRT export & inference
├── edge_cloud/           # MQTT communication & model registry
├── data/                 # Video/text processing pipeline
├── evaluation/           # Metrics & visualization
├── demo/                 # Demo scripts
└── tests/                # Unit tests
```

### Performance Targets

| Metric              | Target         |
|---------------------|----------------|
| Single Frame Latency| < 85ms         |
| Throughput          | >= 15 FPS      |
| GPU Memory          | < 2GB          |
| Trainable Params    | < 1%           |
| Supported Platform  | Jetson Orin Nano |

### Development Phases

| Phase | Description           | Duration |
|-------|-----------------------|----------|
| 1     | Model Architecture    | 1 week   |
| 2     | PEFT Fine-tuning      | 1 week   |
| 3     | TensorRT Deployment   | 1 week   |
| 4     | Edge-Cloud Comm       | 3 days   |
| 5     | Demo & Documentation  | 3 days   |

---

<a name="chinese"></a>

## 中文

**EdgeGuard** 是一个边云协同的多模态异常行为检测系统。边缘端实时推理视频流（视觉+文本），云端通过 PEFT 完成模型训练与更新，通过 TensorRT 实现边缘端高效部署。

### 核心特性

- **多模态融合**：DeiT（视觉）+ DistilBERT（文本）+ LSTM（时序）+ 跨模态注意力
- **PEFT 微调**：LoRA + Adapter LoRA，可训练参数 < 1%
- **TensorRT 部署**：INT8 量化、动态批处理、延迟 < 85ms
- **边云同步**：基于 MQTT 的模型推送与推理结果上报
- **7 类行为**：打架、摔倒、攀爬、滞留、逆行、聚集、正常
- **5 类告警**：入侵、故障、违规、异常、正常

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/yourrepo/edgeguard.git
cd edgeguard

# 安装依赖
pip install -r requirements.txt

# 阶段1：运行模型演示
python demo/demo_edge.py

# 阶段2：PEFT 微调训练
python training/train.py --config configs/train_config.yaml

# 阶段3：导出 TensorRT
python deployment/onnx_to_trt.py --config configs/deploy_config.yaml

# 阶段4：启动边缘服务
python edge_cloud/edge_service.py
```

### 技术栈

PyTorch 2.x · HuggingFace Transformers · PEFT · TensorRT 8.x · MQTT · FFmpeg · OpenCV · Docker

### License

MIT License

</div>
