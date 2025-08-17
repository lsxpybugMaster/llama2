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
    
    
    ## top_p 预测
    ## 找到第一个满足 “累积概率 - 当前概率 > P” 的位置
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        # 为了方便找前x个总和接近p的token进行随机，我们先排序
        # 由于排序后位置变化,所以使用probs_idx存储排序前位置
        prob_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # 前缀和
        prob_sum = torch.cumsum(prob_sort, dim = -1)
        # prob_sum - prob_sort 相当于实现右移
        '''
        prob_sort [5, 4,  3,  2,  1]
        prob_sum  [5, 9, 12, 14, 15]
        sum - sort[0, 5,  9, 12, 14]
        P = 10
        mask =    [F, F,  F,  T,  T]
        prob[mask][5, 4,  3,  0,  0]
        '''
        mask = prob_sum - prob_sort > p
        # 对于为T的,我们不再需要,将其置为0
        prob_sort[mask] = 0.0
        # 一些元素被抹除为0后,需要重新归一化
        prob_sort.div_(prob_sort.sum(dim = -1, keepdim=True))
        # 随机选取,输入向量的值代表选取概率(因此0永远不选)
        next_token = torch.multinomial(prob_sort,num_samples=1)
        # 将位置映射回原位置(排序前)
        next_token = torch.gather(probs_idx,-1,next_token)
        return next_token


    # Token预测
    def text_completion(
        self,
        prompts: list[str], # prompts可以成batch
        temperature: float = 0.6, # 温度调节logit的信心度
        top_p: float = 0.9, # top_p预测方式
        max_gen_len: Optional[int] = None
    ):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        ## prompt -> tokens
        prompt_tokens = [
            self.tokenizer.encode(prompt,out_type=int,add_bos=True,add_eos=False) 
            for prompt in prompts  # 逐batch转换
        ]

        ## 处理长度信息
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"输入的prompt句子数需要 <= {self.args.max_batch_size}"

        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, f"prompt的token长度需要 <= {self.args.max_seq_len}"

        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        ## 初始化token序列
        # 全部填充为[PAD]
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size,total_len),pad_id,dtype=torch.long,device=device)

        for k,t in enumerate(prompt_tokens):
            # 将token序列中添加初始prompt token
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long,device=device)

        # 每个token序列是否到达eos(生成完毕)
        eos_reached = torch.tensor([False] * batch_size, device=device)
        # PAD Mask, True代表当前位置为有效token
        prompt_tokens_mask = tokens != pad_id

        ## 准备预测token
        cur_iterator = tqdm(range(1, total_len),desc="Generating tokens")
        for cur_pos in cur_iterator:
            ## 前向计算
            with torch.no_grad():
                # 使用KV-cache : 仅需传入1个token
                # RoPE : 需传入当前token位置
                # logits : (B, seq_len = 1?, vocab_size)
                logits = self.model.forward(tokens[:, cur_pos - 1 : cur_pos], cur_pos)
            if temperature > 0:
                # 给logits加入温度
                probs = torch.softmax(logits[:, -1] / temperature, dim = -1)
                next_token = self._sample_top_p(probs,top_p)
            else:
                # 不加温度就做贪婪搜索
                next_token = torch.argmax(logits[:, -1],dim = -1)

            # (B, 1) -> (B,)
            next_token = next_token.reshape(-1)
            # 是[pad]才做token替换,否则保留原token
            # 生成应从每个 prompt 的实际结束位置开始
            # where逻辑类似三元表达式
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:,cur_pos], next_token)
            # 🤗终于生成该位置token
            tokens[:, cur_pos] = next_token
            # |= 表示累积逻辑或,用于反复判断是否到达[eos],且到达后不再变化
            # 后续逻辑:  不为初始prompt & 当前预测词为[eos]
            eos_reached |= (~prompt_tokens_mask[:,cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break
        
        ## 🤗 到达该位置说明所有batch生成完毕
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # 意思是找到第一个[eos] 就立刻处理 (有可能生成多个eos)
            if self.tokenizer.eos_id in current_prompt_tokens:
                # 找eos对应索引
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                # 截取有效部分 -> [eos]
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            # 加入一句完整的token答案
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)   
    


if __name__ == '__main__':
    
    torch.manual_seed(0)

    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    # batch = 4 
    prompts = [
        "Simply put, the theory of relativity states that ",
        
        "If Google was an Italian company founded in Milan, it would",
        
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        
        # Zero shot prompt
        """Tell me if the following formula is correct:
        Formula: a - b = b - a
        Decision: incorrect
        Formula: a + b = b + a
        Decision: 
        """
    ]

    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path=str(Path('llama-2-7b') / 'tokenizer.model'),
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    out_tokens, out_texts = (model.text_completion(prompts,max_gen_len=64))
    assert len(out_texts) == len(prompts) , "输出batch与输入batch不一致"
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)