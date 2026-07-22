# Third-party notices

## Real-ESRGAN / BasicSR model architecture

The Real-ESRGAN x2plus checkpoint and the RRDBNet architecture used by the export script originate
from the Real-ESRGAN and BasicSR projects.

- Project: https://github.com/xinntao/Real-ESRGAN
- License: BSD 3-Clause
- Pinned checkpoint: see `models/realesrgan-x2plus/source.lock.json`

Copyright (c) 2021, Xintao Wang. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
   and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of
   conditions and the following disclaimer in the documentation and/or other materials provided
   with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific prior written
   permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## waifu2x-ncnn-vulkan and realcugan-ncnn-vulkan runtime assets

The pinned ncnn parameter and model files distributed by the runtime-assets release originate from:

- https://github.com/nihui/waifu2x-ncnn-vulkan
- https://github.com/nihui/realcugan-ncnn-vulkan
- License: MIT

Copyright (c) 2019 nihui

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Real-CUGAN architecture and checkpoint

The Real-CUGAN SE 2x conservative checkpoint and UpCunet architecture used by the candidate export
originate from the official bilibili AI Lab repository and Real-CUGAN release.

- Project: https://github.com/bilibili/ailab/tree/main/Real-CUGAN
- License: MIT
- License copy: `licenses/Real-CUGAN-MIT.txt`
- Pinned sources: see `models/realcugan-se-2x-conservative/source.lock.json`

Copyright (c) 2022 bilibili

## Hailo Model Zoo ESPCN source package and ESPCN-PyTorch model

The pinned ESPCN source ZIP originates from the Hailo Model Zoo:

- https://github.com/hailo-ai/hailo_model_zoo
- Hailo Model Zoo distribution license: MIT
- Model source: https://github.com/Lornatang/ESPCN-PyTorch
- Model license: Apache-2.0
- Apache-2.0 license copy: `licenses/ESPCN-PyTorch-Apache-2.0.txt`

Copyright (c) 2021 Hailo Technologies Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## YSGYolo 1.2 OS1.0

YSGYolo is an optional, separately downloaded manga region detector. It is not linked into the
NextE application package.

- Model repository: https://huggingface.co/YSGforMTL/YSGYoloDetector
- Pinned revision: `563aba14fe3ba9e8c5ea1c8074598365190b3705`
- Runtime architecture: Ultralytics YOLO11 OBB
- Effective artifact distribution license: `AGPL-3.0-only`
- Release license copy: `YSGYolo-AGPL-3.0.txt`
- Canonical license text: https://www.gnu.org/licenses/agpl-3.0.txt
- Source and hashes: `models/ysgyolo-1.2-os1/source.lock.json`

The model card declares MIT. The pinned checkpoint itself contains Ultralytics module metadata,
`yolo11n-obb.yaml`, and an embedded `AGPL-3.0` license value. Because those two signals conflict,
this repository distributes the checkpoint and derived ONNX/ncnn artifacts under the stricter
AGPL-3.0-only label. This is an engineering compliance boundary, not a representation that one
party can relicense another party's work.

## PP-OCRv5 Mobile Recognition

The optional text-line recognizer is converted from the official PaddleOCR
`PP-OCRv5_mobile_rec` model and remains separate from the NextE application package.

- Project: https://github.com/PaddlePaddle/PaddleOCR
- Model repository: https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec
- Pinned revision: `682f20538d8c086cb2128e5cfac775e6c4904e85`
- License: Apache-2.0
- License copy: `licenses/ESPCN-PyTorch-Apache-2.0.txt`
- Source and hashes: `models/ppocrv5-mobile-rec/source.lock.json`

The generated ONNX and ncnn files, character dictionary, source inference files, source lock,
and Apache-2.0 license text are published together in the immutable model-pack release.
