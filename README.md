# EssayLens

[打开 EssayLens 网页版](https://mrsomebodyyy.github.io/essaylens/)

由 OpenAI Codex 制作的英文作文检测工具。粘贴英文即可查看全文的「AI率」，以及模型认为更像 AI 写的原句。检测模型由 Desklib 训练。[Desklib 模型说明](https://huggingface.co/desklib/ai-text-detector-v1.01)

## 使用

1. 使用新版桌面 Microsoft Edge 或 Chrome 打开网站。首次分析会从 Hugging Face 下载约 1.74 GB 的模型，请预留至少 2 GB 浏览器存储空间并保持网络连接。权重缓存在本机，后续分析会重复使用。手机或低内存电脑可能无法完成分析。
2. 粘贴英文作文或单句，也可以点击「填入示例」。每次最多 60,000 个字符。
3. 点击「分析作文」。全文分数先显示，达到 98% 的句子会陆续出现在下方，按原文顺序排列。只想看全文分数时，可以在分析前取消勾选逐句检测。全文分数出来后停止逐句检测，也可以保留已有结果。

直接打开上面的公开网址即可使用。作文留在浏览器内，由 Web Worker 运行模型。同一个标签页会在内存里暂存最近的评分，修改少量文字再检测时可以复用未改句子的结果。网页加载时会请求网站资源，模型推理所需的程序和权重也需要联网下载。

## 阅读结果

- **AI率**是检测模型给整篇作文的分数。分数越高，表示模型认为全文越像 AI 写作。这个分数不能换算成 AI 实际写了多少字。超出模型一次能处理的长度时，工具会分块评分，再按各块的词元数加权平均。
- **更像 AI 写的句子**按原文顺序列出单独评分达到 98% 的原句。如果没有句子达到判定线，页面会直接说明。模型评估全文时能看到上下文，因此整篇得分很高时，也可能没有单句入选。单句结果只能作为复核线索。

## 本地预览

在仓库目录运行以下命令，然后在桌面浏览器打开显示的本机地址（端口 8000）。

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

GitHub Pages 可从这个仓库的根目录发布静态文件。程序使用 Transformers.js 分词，并用 ONNX Runtime Web 的 WebAssembly 后端分批执行 Desklib v1.01 的四组模型层。[ONNX 权重和转换说明](https://huggingface.co/GabeuxDev/ai-text-detector-v1.01-onnx) · [Transformers.js](https://github.com/huggingface/transformers.js)
