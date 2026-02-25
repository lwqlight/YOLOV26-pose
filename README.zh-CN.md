# YOLOv26

这是一个基于 Ultralytics YOLO 的仓库，包含检测、姿态估计与示例推理脚本。仓库内含模型权重、导出工具、文档和示例代码，目标是提供易用且可复现的推理与部署流程。

**主要内容**
- 模型与权重：位于 `ultralytics/weights/`，包含预训练的 pose/detection 权重。
- 推理脚本：`infer.py`、`infer_video.py`、`infer_3_pose.py` 等，用于快速运行图像/视频/多人体姿态推理。
- 文档：位于 `docs/`，包含快速开始与更多使用示例。

## 特性

- 支持图像与视频的快速推理
- 支持姿态估计（pose）模型
- 多种导出与部署选项（见 `docker/` 与 `docs/`）

## 依赖与安装

推荐使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

如果你只想安装运行推理所需的依赖，可以参考 `pyproject.toml` 或 `docs/` 下的说明。

## 快速开始

1) 下载或准备好要使用的权重，例如仓库自带的：

```
ultralytics/weights/yolo26s-pose.pt
```

2) 运行单张图像推理（示例）：

```bash
python infer.py --weights ultralytics/weights/yolo26s-pose.pt --source path/to/image.jpg
```

3) 运行视频推理：

```bash
python infer_video.py --weights ultralytics/weights/yolo26s-pose.pt --source path/to/video.mp4
```

4) 运行多人体/三维姿态相关脚本（视具体脚本参数而定）：

```bash
python infer_3_pose.py --weights ultralytics/weights/yolo26m-pose.pt --source path/to/image_or_video
```

更多参数与用法请查看各脚本的 `--help`：

```bash
python infer.py --help
```

## 文档与示例

- 快速入门与更多用法见 `docs/en/quickstart.md` 和 `docs/` 下的其它页面。
- 仓库内 `examples/` 含若干 Jupyter Notebook 演示，可用于功能验证与可视化调试。

## 贡献

欢迎 Issue、讨论与 PR。贡献流程建议：

1. 新建 issue 讨论需求或 bug。
2. 在本地创建分支实现改动并包含简要说明与测试（若适用）。
3. 提交 PR，CI 会在有配置时运行测试，维护者会进行 review。

更多贡献细节请参阅仓库根目录的 `CONTRIBUTING.md`。

## 许可证

本项目的许可证见仓库根目录的 `LICENSE` 文件。

## 联系与支持

如需帮助或报告问题，请使用仓库 issue 页面或阅读 `CONTRIBUTING.md` 中的联系方式。

---


