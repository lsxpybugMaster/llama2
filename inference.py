from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

# model.py
from model import ModelArgs, Transformer


#🦙
class LLaMA:
    def __init__(
        self,
        model: Transformer,
        tokenizer: SentencePieceProcessor,
        model_args: ModelArgs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    # 读取权重文件llama-2-7b
    @staticmethod
    def build(
        checkpoints_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        device: str
    ):
        # 计时
        prev_time = time.time()

        # 加载权重文件
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"{checkpoints_dir} 路径下未发现权重文件"
            ckpt_path = checkpoints[0]
            print(f"加载权重文件 {ckpt_path}")
            # 得到权重字典
            checkpoint = torch.load(ckpt_path,map_location='cpu')

            print(f"加载权重花费时间: {time.time() - prev_time:.2f}s")
            prev_time = time.time()

        # 加载配置文件
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
            
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params  #传入其他配置自动匹配对应项
        )

        # 加载分词器
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # 半精度计算
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        # ⚠️Pytorch高版本上面的代码会报出警告
        # if device == "cuda":
        #     torch.set_default_dtype(torch.float16)
        #     torch.set_default_device("cuda")
        # else:
        #     torch.set_default_dtype(torch.bfloat16)
        #     torch.set_default_device("cpu")

        print(f"将模型部署至设备: {device}")
        model = Transformer(model_args).to(device)
        print(f"完成将模型部署至设备")

        # 真正地逐Key匹配加载权重
        if load_model:
            # 删除旋转位置编码数据,我们直接计算
            # 权重是字典
            
            del checkpoint['rope.freqs']
            
            print(f"加载权重字典中")
            model.load_state_dict(checkpoint,strict=True)
            print(f"加载了权重字典,耗时{time.time() - prev_time:.2f}s")

        return LLaMA(model,tokenizer,model_args)
    


if __name__ == '__main__':
    torch.manual_seed(0)

    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path=str(Path('llama-2-7b') / 'tokenizer.model'),
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device,
    )

    print("PASS")