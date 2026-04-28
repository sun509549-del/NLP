# NLP 环境包测试文档

本文档用于测试 `nlp` conda 环境中已安装的常用 NLP/深度学习包是否能正常导入和执行基础功能。

## 测试范围

- `jieba`：中文分词
- `gensim`：训练一个极小的 Word2Vec 模型
- `transformers`：用本地配置创建一个极小的 BERT 模型并前向推理
- `datasets`：创建内存数据集并执行 `map`
- `tensorboard`：写入一个标量日志文件
- `tqdm`：显示进度条并完成循环
- `jupyter`：确认模块可导入
- `torch`：确认 CUDA 可用，并在 GPU 上做一次矩阵乘法

## 运行方法

在 PowerShell 中进入当前目录：

```powershell
cd E:\sh\damoxing\NLP
conda activate nlp
python package_smoke_test.py
```

也可以不激活环境，直接运行：

```powershell
conda run -n nlp python package_smoke_test.py
```

注意：不要用 `C:\msys64\ucrt64\bin\python.exe` 运行这个脚本。这个 Python 不属于 `nlp` conda 环境，因此会出现类似下面的错误：

```text
ModuleNotFoundError: No module named 'datasets'
```

如果在 VS Code 中运行，请把 Python 解释器切换到：

```text
D:\Conda_Envs\nlp\python.exe
```

## 预期结果

终端应看到多行 `[OK]`，最后出现：

```text
All package checks passed.
```

其中 GPU 检查应包含类似内容：

```text
[OK] torch cuda - 2.11.0+cu128, device=NVIDIA GeForce RTX 5060 Laptop GPU
```

如果 `tensorboard` 测试通过，当前目录下会生成：

```text
runs/package_smoke_test/
```

可以用下面命令打开 TensorBoard：

```powershell
tensorboard --logdir runs
```

## Jupyter 额外验证

启动 JupyterLab：

```powershell
jupyter lab
```

新建 Notebook 后运行：

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

若输出为 `True` 并显示 `NVIDIA GeForce RTX 5060 Laptop GPU`，说明 Notebook 中也能正常使用 GPU。
