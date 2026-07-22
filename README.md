# NextE Models

NextE 的可下载本地图像模型仓库。这里负责锁定模型来源、转换为 HarmonyOS 可用格式、
记录设备验证证据，并通过 GitHub Release 分发；模型不会打进 NextE 安装包。

当前仓库包含按用途分开的两个模型清单：

- `manifests/models-v1.json`：阅读器超分模型；
- `manifests/comic-translation-models-v1.json`：漫画翻译的检测、OCR 与修复模型。

模型共用 NextE-Models 的下载与校验基础设施，但在应用中按角色独立安装、选择和展示许可，
不会把超分模型与漫画翻译模型混为同一设置项。

超分首要目标是 `Real-ESRGAN x2plus` 的 MindSpore Lite / NNRT 版本。NextE 当前运行契约为：

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
对应资产已通过发布门禁并纳入统一的 `model-pack-v1.0.0`。

`Real-ESRGAN animevideov3 2x` 的首个 FP16 图仍保留为未发布候选。同一文件在 103 的 CPU
后端可以加载和推理，但在 103、197、237 的 NNRT 后端均无法获得输入张量，说明当前图无法
通过这些 NPU 的编译或算子映射；修复图结构前不会进入应用下载清单。

`ESPCN x2` 的亮度通道 FP16 图固定为 `1 x 1 x 180 x 180`。NextE 已实现 172 px 有效亮度
切片、8 px 输出上下文裁剪，以及在同一双线性 2x 底图上的 RGB 重组。103、197、237 三台
设备的完整 Reader 管线均通过 NNRT/NPU：一张 800 x 1100 实际阅读图的输出相对固定 ncnn
参考最大 RGB 字节误差为 2；应用 AUTO 路径处理约 1.1 到 1.3 秒，事件循环采样 P95 不超过
2 毫秒，因此对应资产已通过发布门禁并纳入统一模型包。

`waifu2x photo noise0 2x` 已从锁定的 ncnn upconv7 参数和 FP16 权重重建为固定
`1 x 3 x 156 x 156` 输入、`1 x 3 x 284 x 284` 输出的 MindSpore Lite FP16 模型。修正
Deconvolution 权重布局后，103、197、237 三台设备的完整 `800 x 1149` Reader 管线均通过
NNRT/NPU 执行。与同一 ncnn 模型相比，拼接后 RGB 平均绝对误差约为 0.26 到 0.28、最大误差
3 到 4，端到端耗时由约 9 秒降至约 3 秒。因此对应资产已通过作为同模型加速后端的发布
门禁并纳入统一模型包；这里不把等价性测试表述为新模型的主观画质提升。

## 发布门禁

1. 来源锁与许可证通过仓库校验。
2. 生成的文件尺寸和 SHA-256 写入候选记录。
3. 在目标设备上验证模型 I/O、可加载性、稳定性和真实页面端到端耗时。
4. 用保留的实际页面比较 FP16 与 INT8；校准页和评估页不离开本地。
5. 只有通过门禁的条目才能在 `manifests/models-v1.json` 中改为 `published`。
6. 所有 `published` NNRT 和 ncnn 资产必须进入同一个版本化 `model-pack` Release；模型仍按
   文件独立下载，不要求应用下载整个模型包。
7. Release 资产不可覆盖；任何改变都发布新的模型包版本和 SHA-256。

应用只应读取 `published` 条目。当前清单中的 `candidate` 不构成下载地址，也不会让
NextE 下载一个尚未发布或未经验证的模型。

## 统一模型包

`manifests/models-v1.json` 管理已验证的 NNRT 资产，`manifests/ncnn-runtime-assets-v1.json`
管理无需重新转换的 ncnn 运行资产，`manifests/comic-translation-models-v1.json` 管理漫画翻译
模型及逐资产许可。唯一的发布工作流会收集三份清单中的全部有效文件，逐一
验证尺寸与 SHA-256，并发布到同一个 `model-pack-vX.Y.Z` Release。当前模型包还覆盖
Real-ESRGAN animevideov3 与 x2plus 的 ncnn 来源文件；应用优先使用本仓库 Release，并可保留
清单中的固定上游地址作为网络回退。

模型转换和候选验证工作流仍然按模型独立运行，但不再各自创建 Release。统一发布避免 tag、
下载地址和资产版本互相漂移，同时不改变应用按需下载单个模型文件的行为。

这份运行资产清单不表示对应模型已经支持 NPU。NPU 模型仍必须经过上面的转换、候选、多设备
NNRT 和整页画质门禁，只有 `manifests/models-v1.json` 中的 `published` 条目可以作为 NPU
下载资产。

## 漫画翻译模型

端侧漫画模型包中的 `YSGYolo 1.2 OS1.0` 仅检测漫画气泡/文字区域；它不进行 OCR、
翻译、文字像素分割或背景修复。固定 ONNX 通过 pnnx 20260526 转为 ncnn FP32：

```bash
python scripts/convert_ysgyolo_onnx_to_ncnn.py \
  --pnnx /path/to/pnnx \
  --onnx downloads/ysgyolo_1.2_OS1.0.onnx \
  --output-dir artifacts
```

该模型的作者页面声明 MIT，但固定 checkpoint 内嵌 Ultralytics/YOLO11 OBB 元数据以及
`AGPL-3.0` 许可字段。仓库不尝试把冲突解释成 MIT，而是对 YSGYolo checkpoint、ONNX 和
ncnn 衍生产物采用更严格的 `AGPL-3.0-only` 分发标记。NextE 自身适配代码仍使用项目原许可；
每个可选模型资产保留自己的许可。

YSGYolo 的 Release 同时携带 checkpoint、源 ONNX、转换来源锁、第三方说明和完整 AGPL 文本；
应用只下载运行所需的 param/bin。

`PP-OCRv5_mobile_rec` 是独立的文本行识别组件，来源锁固定到 PaddlePaddle 官方模型仓库的
不可变提交，采用 Apache-2.0。它不替代系统 OCR：NextE 只在 YSGYolo 检出、但系统 OCR 未覆盖
的区域调用该模型，并对低置信度输出保持不处理。固定 Paddle 推理文件通过
Paddle2ONNX 2.1.0 和 pnnx 20260526 转为 ncnn FP32：

```bash
python scripts/convert_ppocrv5_mobile_rec_to_ncnn.py \
  --paddle2onnx /path/to/paddle2onnx \
  --pnnx /path/to/pnnx \
  --model-dir downloads/ppocrv5-mobile-rec \
  --output-dir artifacts/comic-generated
```

Release 同时包含三份 Paddle 原始推理文件、来源锁、Apache-2.0 文本、ncnn param/bin 和字符
字典。应用设置中仍是一个“端侧漫画模型”能力包，但许可按 YSGYolo 与 PP-OCRv5 两个组件分别
展示和保留。
