# NextE Models

NextE 的可下载本地图像模型仓库。这里负责锁定模型来源、转换为 HarmonyOS 可用格式、
记录设备验证证据，并通过 GitHub Release 分发；模型不会打进 NextE 安装包。

当前首要目标是 `Real-ESRGAN x2plus` 的 MindSpore Lite / NNRT 版本。NextE 当前运行契约为：

- 输入：Float32、NCHW、`1 x 3 x 180 x 180`
- 输出：Float32、NCHW、`1 x 3 x 360 x 360`
- 阅读器切片：有效区域 `160 x 160`，四周预填充 `10 px`
- 像素：RGB，范围 `0...1`

## 安全边界

实际阅读页可以用于本地校准和画质评估，但不得提交、上传为 Actions Artifact，或附加到
Release。`.gitignore` 已隔离 `calibration-data/`、`evaluation-data/`、`private-data/`、
`downloads/`、`work/` 和生成的模型文件。

## 基本流程

安装转换依赖（建议 Python 3.11）：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-convert.txt
```

下载并校验官方 checkpoint：

```bash
python scripts/fetch_locked_source.py \
  models/realesrgan-x2plus/source.lock.json \
  downloads/RealESRGAN_x2plus.pth
```

导出固定输入的 ONNX：

```bash
python scripts/export_realesrgan_x2plus_onnx.py \
  --checkpoint downloads/RealESRGAN_x2plus.pth \
  --output work/realesrgan-x2plus-180.onnx
```

准备 100 到 500 个与阅读器真实预处理一致的校准切片；默认抽取 300 个：

```bash
python scripts/prepare_calibration.py \
  --input private-data/reader-pages \
  --output calibration-data/realesrgan-x2plus-180 \
  --samples 300 \
  --seed 20260719
```

MindSpore Lite 转换器必须使用来源锁中的 Linux x86_64 版本。macOS ARM 主机可在
Linux/amd64 容器中运行它。FP16 基线：

```bash
python scripts/convert_mindspore.py \
  --converter /path/to/converter_lite \
  --onnx work/realesrgan-x2plus-180.onnx \
  --output artifacts/realesrgan-x2plus-180-fp16.ms \
  --mode fp16
```

全量 INT8 候选在首个方法能够通过目标设备 NNRT 编译后，再比较 `MAX_MIN` 和 `KL`：

```bash
python scripts/convert_mindspore.py \
  --converter /path/to/converter_lite \
  --onnx work/realesrgan-x2plus-180.onnx \
  --output artifacts/realesrgan-x2plus-180-full-int8-max-min.ms \
  --mode full-int8-max-min \
  --calibration calibration-data/realesrgan-x2plus-180
```

`weight-int8` 只保留为对照实验。设备 103 的既有测试中，它的 45 次预测总耗时与 FP16
几乎相同，因此不能仅凭体积缩小就替换 FP16。

使用 100 个真实阅读页校准缓冲区生成的 `MAX_MIN` 全量 INT8 候选虽然转换成功、体积约为
FP16 的一半，但在 103、197、237 三台设备上均无法通过 NNRT 图编译，因此已经淘汰。
该失败发生在推理前；`KL` 只改变校准阈值，不能修复同一混合量化算子图的兼容性，所以
本轮不继续消耗算力生成 KL 版本。

FP16 版本已经在四台设备的 NPU 上完成 `800 x 1149` 到 `1600 x 2298` 的完整阅读器流程。
端到端耗时处于约 6 到 7 秒量级，前台事件循环 P95 延迟为 1 到 2 毫秒。发布前仍需使用本地
保留的实际页面进行肉眼画质对照；这些页面不会进入仓库或 Release。2026-07-19 的本地对照
已确认输出没有切片接缝或填充伪影，相比 2x Lanczos 能恢复更清晰的线条和眼部细节，因此
`realesrgan-x2plus-nnrt-v1` 已通过发布门禁。

`Real-ESRGAN animevideov3 2x` 的首个 FP16 图仍保留为未发布候选。同一文件在 103 的 CPU
后端可以加载和推理，但在 103、197、237 的 NNRT 后端均无法获得输入张量，说明当前图无法
通过这些 NPU 的编译或算子映射；修复图结构前不会进入应用下载清单。

`ESPCN x2` 的亮度通道 FP16 图已经固定为 `1 x 1 x 180 x 180`，并在 103、197、237 的
NNRT/NPU 上完成加载和三次推理。它暂不发布：NextE 还需要实现独立的亮度切片执行器，并将
2x 亮度结果与缩放后的色度通道重组，再完成整页性能和画质验证。

## 发布门禁

1. 来源锁与许可证通过仓库校验。
2. 生成的文件尺寸和 SHA-256 写入候选记录。
3. 在目标设备上验证模型 I/O、可加载性、稳定性和真实页面端到端耗时。
4. 用保留的实际页面比较 FP16 与 INT8；校准页和评估页不离开本地。
5. 只有通过门禁的条目才能在 `manifests/models-v1.json` 中改为 `published` 并填写真实
   Release URL。
6. Release 资产不可覆盖；任何改变使用新 tag 和新 SHA-256。

应用只应读取 `published` 条目。当前清单中的 `candidate` 不构成下载地址，也不会让
NextE 下载一个尚未发布或未经验证的模型。
