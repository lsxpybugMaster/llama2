## 🦙 LLaMA2
* llama2 code from scratch
* 构建的是llama-2-7B 原始模型

## 📦️ 权重文件下载
* https://www.modelscope.cn/models/angelala00/Llama-2-7b/files
* 根目录下创建文件夹`llama-2-7b`
* 下载以下文件至其中,得到文件结构：

```
llama-2-7b/
├── consolidated.00.pth
├── params.json
└── tokenizer.model
```

## 🤗 运行
* `z_Llama2.ipynb`仅用作笔记,不建议直接运行
* 直接运行`inference.py`即可
* `inference.py`中修改prompt的位置:

```python
if __name__ == '__main__':  
    torch.manual_seed(0)
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    ### 修改prompts内容即可
    prompts = [
        "Simply put, the theory of relativity states that ",
        
        "If Google was an Italian company founded in Milan, it would",
    ]
```
