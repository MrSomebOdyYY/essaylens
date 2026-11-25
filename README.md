# EssayLens

[打开 EssayLens 网页版](https://mrsomebodyyy.github.io/essaylens/)

由 OpenAI Codex 制作的英文作文 AI 写作倾向分析工具。把英文粘贴到页面，可查看全文分数、各段分数、短段标注和高分段落占全文的词数比例。模型由 Desklib 训练，网页和分析流程由 Codex 开发。[Desklib 模型说明](https://huggingface.co/desklib/ai-text-detector-v1.01)

## 使用

1. 在支持 WebGPU 的桌面浏览器中打开网站，并启用图形加速。首次分析会从 Hugging Face 下载约 1.74 GB 的模型，需要稳定的网络和数 GB 可用内存。手机或低内存电脑可能无法完成分析。
2. 粘贴英文并保留自然段换行，或点击「填入示例」。每次最多 60,000 个字符。
3. 点击「分析作文」。页面会显示加载和分析进度，也可以点击「停止」中止任务。全文评分至少需要 50 个英文单词和 3 个句子，短段可以单独提交。

作文留在浏览器内，由 Web Worker 运行模型。网页加载时会请求网站资源，模型推理所需的程序和权重也需要联网下载。作文不会作为请求发送到模型托管服务。

## 阅读结果

- **全文分数**显示模型对英文文本的 AI 写作倾向评分。达到 98% 时页面给出高分提示。超出单次模型输入长度的文章会分块推理，再按词元数量加权。
- **逐段分数**按输入中的换行划分。每个非空段落独立评分，少于 50 词的段落会注明「短段」。短文本可用于判断的上下文较少。
- **高分段落的词数占比**等于达到 98% 的段落词数之和除以全文词数。整段被标亮时会计入该段全部词数。同段中某些词经 AI 修改的情况无法据此定位。
- **逐句文风提示**列出固定衔接语或相同的句子开头，供修改表达时参考。

桌面 PyTorch 参考版曾用 32 篇已知人机词数占比的英文混合稿检验占比估算。8 篇真实学生与 ChatGPT 合写稿的平均绝对误差为 **29.2 个百分点**，24 篇同题人机文本拼接稿为 **5.4 个百分点**。实验中的 AI 贡献接近一半，样本数量有限。分数本身不能确定作者身份。[NIRVANA 真实合写研究](https://arxiv.org/abs/2604.07344) · [GEDE 同题大学作文语料](https://github.com/lukasgehring/Assessing-LLM-Text-Detection-in-Educational-Contexts)

## 本地预览

在仓库目录运行以下命令，然后在桌面浏览器打开显示的本机地址（端口 8000）。

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

GitHub Pages 可从这个仓库的根目录发布静态文件。浏览器推理使用 Desklib v1.01 的 ONNX 转换及 Transformers.js。[ONNX 权重和转换说明](https://huggingface.co/GabeuxDev/ai-text-detector-v1.01-onnx) · [Transformers.js](https://github.com/huggingface/transformers.js)
