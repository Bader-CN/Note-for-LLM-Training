import os
import json
import string
from pathlib import Path

from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers, Tokenizer
from transformers import PreTrainedTokenizerFast

# 获取项目根目录
# Path(__file__).resolve() 总是指向当前模块的绝对路径
project_root = Path(__file__).resolve().parent.parent

# 训练数据文件路径
data_path = os.path.join(project_root, "./dataset/pretrain_hq.jsonl")
# 训练好的 tokenizer 保存目录
tokenizer_dir = os.path.join(project_root, "./model/train_tokenizer")
# jinja 模板
jinja_chat_template_path = os.path.join(project_root, "./trainer/chat_template.jinja")

# 词表大小
vocab_size = 25600
# 最长 Tokens
max_tokens = 32768

def get_text_from_jsonl(data_path):
    """
    从 jsonl 文件中读取数据, 并返回一个迭代器对象

    Args:
    - data_path:    JSONL 格式的数据文件路径

    Yields:
    - str:          每行数据中 'text' 字段的内容
    """
    with open(data_path, mode='r', encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            yield data['text']

def set_tokenizer():
    """
    创建并返回一个分词器对象
    """
    # 创建一个 BPE 算法的 tokenizer
    tokenizer = Tokenizer(model=models.BPE())

    # tokenizer 标准化
    tokenizer.normalizer = normalizers.Sequence([
        # 兼容等价
        normalizers.NFKC(),
        # 去掉所有重音
        normalizers.StripAccents(),
    ])

    # tokenizer 预分词规则
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        # 将数字拆分为单个字符
        pre_tokenizers.Digits(individual_digits=True),
        # 根据不同语系的字符将文本分隔成单词
        pre_tokenizers.UnicodeScripts(),
        # 将标点符号作为单独的字符进行拆分
        pre_tokenizers.Punctuation(behavior="isolated"),

        # 将指定字符独立出来, 例如空格字符
        # 不要使用 BertPreTokenizer() / Whitespace(), 因为会干扰特定字符的分割
        pre_tokenizers.Split(pattern=" ",  behavior="isolated"),
        pre_tokenizers.Split(pattern="\t", behavior="isolated"),     # \u0009 Horizontal Tab (HT)
        pre_tokenizers.Split(pattern="\n", behavior="isolated"),     # \u000A Line Feed (LF)
        pre_tokenizers.Split(pattern="\f", behavior="isolated"),     # \u000C Form Feed (FF)
        pre_tokenizers.Split(pattern="\r", behavior="isolated"),     # \u000D Carriage Return (CR)

        # 忽略掉控制字符 - ASCII 控制字符 (\u0000 ~ \u001F and \u007F)
        pre_tokenizers.Split(pattern="\u0000", behavior="removed"),  # Null (NUL)
        pre_tokenizers.Split(pattern="\u0001", behavior="removed"),  # Start of Header (SOH)
        pre_tokenizers.Split(pattern="\u0002", behavior="removed"),  # Start of Text (STX)
        pre_tokenizers.Split(pattern="\u0003", behavior="removed"),  # End of Text (ETX)
        pre_tokenizers.Split(pattern="\u0004", behavior="removed"),  # End of Transmission (EOT)
        pre_tokenizers.Split(pattern="\u0005", behavior="removed"),  # Enquiry (ENQ)
        pre_tokenizers.Split(pattern="\u0006", behavior="removed"),  # Acknowledge (ACK)
        pre_tokenizers.Split(pattern="\u0007", behavior="removed"),  # Bell (BEL)
        pre_tokenizers.Split(pattern="\u0008", behavior="removed"),  # Backspace (BS)
        pre_tokenizers.Split(pattern="\u000B", behavior="removed"),  # Vertical Tab (VT)
        pre_tokenizers.Split(pattern="\u000E", behavior="removed"),  # Shift Out (SO)
        pre_tokenizers.Split(pattern="\u000F", behavior="removed"),  # Shift In (SI)
        pre_tokenizers.Split(pattern="\u0010", behavior="removed"),  # Data Link Escape (DLE)
        pre_tokenizers.Split(pattern="\u0011", behavior="removed"),  # Device Control 1 (DC1)
        pre_tokenizers.Split(pattern="\u0012", behavior="removed"),  # Device Control 2 (DC2)
        pre_tokenizers.Split(pattern="\u0013", behavior="removed"),  # Device Control 3 (DC3)
        pre_tokenizers.Split(pattern="\u0014", behavior="removed"),  # Device Control 4 (DC4)
        pre_tokenizers.Split(pattern="\u0015", behavior="removed"),  # Negative Acknowledge (NAK)
        pre_tokenizers.Split(pattern="\u0016", behavior="removed"),  # Synchronous Idle (SYN)
        pre_tokenizers.Split(pattern="\u0017", behavior="removed"),  # End of Transmission Block (ETB)
        pre_tokenizers.Split(pattern="\u0018", behavior="removed"),  # Cancel (CAN)
        pre_tokenizers.Split(pattern="\u0019", behavior="removed"),  # End of Medium (EM)
        pre_tokenizers.Split(pattern="\u001A", behavior="removed"),  # Substitute (SUB)
        pre_tokenizers.Split(pattern="\u001B", behavior="removed"),  # Escape (ESC)
        pre_tokenizers.Split(pattern="\u001C", behavior="removed"),  # File Separator (FS)
        pre_tokenizers.Split(pattern="\u001D", behavior="removed"),  # Group Separator (GS)
        pre_tokenizers.Split(pattern="\u001E", behavior="removed"),  # Record Separator (RS)
        pre_tokenizers.Split(pattern="\u001F", behavior="removed"),  # Unit Separator (US)
        pre_tokenizers.Split(pattern="\u007F", behavior="removed"),  # Delete (DEL)
        # 忽略掉控制字符 - 扩展控制字符 (C1 控制字符, Unicode 范围 \u0080 ~ \u009F)
        pre_tokenizers.Split(pattern="\u0080", behavior="removed"),  # Padding Character (PAD)
        pre_tokenizers.Split(pattern="\u0081", behavior="removed"),  # High Octet Preset (HOP)
        pre_tokenizers.Split(pattern="\u0082", behavior="removed"),  # Break Permitted Here (BPH)
        pre_tokenizers.Split(pattern="\u0083", behavior="removed"),  # No Break Here (NBH)
        pre_tokenizers.Split(pattern="\u0084", behavior="removed"),  # Index (IND)
        pre_tokenizers.Split(pattern="\u0085", behavior="removed"),  # Next Line (NEL)
        pre_tokenizers.Split(pattern="\u0086", behavior="removed"),  # Start of Selected Area (SSA)
        pre_tokenizers.Split(pattern="\u0087", behavior="removed"),  # End of Selected Area (ESA)
        pre_tokenizers.Split(pattern="\u0088", behavior="removed"),  # Character Tabulation Set (HTS)
        pre_tokenizers.Split(pattern="\u0089", behavior="removed"),  # Character Tabulation with Justification (HTJ)
        pre_tokenizers.Split(pattern="\u008A", behavior="removed"),  # Line Tabulation Set (VTS)
        pre_tokenizers.Split(pattern="\u008B", behavior="removed"),  # Partial Line Down (PLD)
        pre_tokenizers.Split(pattern="\u008C", behavior="removed"),  # Partial Line Up (PLU)
        pre_tokenizers.Split(pattern="\u008D", behavior="removed"),  # Reverse Line Feed (RI)
        pre_tokenizers.Split(pattern="\u008E", behavior="removed"),  # Single Shift Two (SS2)
        pre_tokenizers.Split(pattern="\u008F", behavior="removed"),  # Single Shift Three (SS3)
        pre_tokenizers.Split(pattern="\u0090", behavior="removed"),  # Device Control String (DCS)
        pre_tokenizers.Split(pattern="\u0091", behavior="removed"),  # Private Use One (PU1)
        pre_tokenizers.Split(pattern="\u0092", behavior="removed"),  # Private Use Two (PU2)
        pre_tokenizers.Split(pattern="\u0093", behavior="removed"),  # Set Transmit State (STS)
        pre_tokenizers.Split(pattern="\u0094", behavior="removed"),  # Cancel Character (CCH)
        pre_tokenizers.Split(pattern="\u0095", behavior="removed"),  # Message Waiting (MW)
        pre_tokenizers.Split(pattern="\u0096", behavior="removed"),  # Start of Protected Area (SPA)
        pre_tokenizers.Split(pattern="\u0097", behavior="removed"),  # End of Protected Area (EPA)
        pre_tokenizers.Split(pattern="\u0098", behavior="removed"),  # Start of String (SOS)
        pre_tokenizers.Split(pattern="\u0099", behavior="removed"),  # Single Graphic Character Introducer (SGCI)
        pre_tokenizers.Split(pattern="\u009A", behavior="removed"),  # Single Character Introducer (SCI)
        pre_tokenizers.Split(pattern="\u009B", behavior="removed"),  # Control Sequence Introducer (CSI)
        pre_tokenizers.Split(pattern="\u009C", behavior="removed"),  # String Terminator (ST)
        pre_tokenizers.Split(pattern="\u009D", behavior="removed"),  # Operating System Command (OSC)
        pre_tokenizers.Split(pattern="\u009E", behavior="removed"),  # Privacy Message (PM)
        pre_tokenizers.Split(pattern="\u009F", behavior="removed"),  # Application Program Command (APC)
        # 忽略掉控制字符 - 其余控制字符
        pre_tokenizers.Split(pattern="\u00AD", behavior="removed"),  # Soft Hyphen (SHY)
        pre_tokenizers.Split(pattern="\u200B", behavior="removed"),  # Zero Width Space (ZWSP)
        pre_tokenizers.Split(pattern="\u200C", behavior="removed"),  # Zero Width Non-Joiner (ZWNJ)
        pre_tokenizers.Split(pattern="\u200D", behavior="removed"),  # Zero Width Joiner (ZWJ)
        pre_tokenizers.Split(pattern="\u200E", behavior="removed"),  # Left-to-Right Mark (LRM)
        pre_tokenizers.Split(pattern="\u200F", behavior="removed"),  # Right-to-Left Mark (RLM)
        pre_tokenizers.Split(pattern="\u2028", behavior="removed"),  # Line Separator (LS)
        pre_tokenizers.Split(pattern="\u2029", behavior="removed"),  # Paragraph Separator (PS)
        pre_tokenizers.Split(pattern="\u202A", behavior="removed"),  # Left-to-Right Embedding (LRE)
        pre_tokenizers.Split(pattern="\u202B", behavior="removed"),  # Right-to-Left Embedding (RLE)
        pre_tokenizers.Split(pattern="\u202C", behavior="removed"),  # Pop Directional Formatting (PDF)
        pre_tokenizers.Split(pattern="\u2060", behavior="removed"),  # Word Joiner (WJ)
        pre_tokenizers.Split(pattern="\u2061", behavior="removed"),  # Function Application (FA)
        pre_tokenizers.Split(pattern="\uFEFF", behavior="removed"),  # Zero Width No-Break Space (ZWNBSP)
    ])

    # tokenizer 解码
    tokenizer.decoder = decoders.BPEDecoder()

    return tokenizer

def set_tokenizer_trainer(
    vocab_size=vocab_size, 
    min_frequency=4,
    special_tokens=["<|im_unk|>", "<|im_start|>", "<|im_end|>", "<|im_pad|>"],  # 定义特殊 token: 文本结束符、指令开始符、指令结束符等
    ):
    # 常用汉字
    # chinese_str = [chr(i) for i in range(0x4E00, 0x9FFF + 1)]
    # 日语平假名
    # jp_hiragana = [chr(i) for i in range(0x3041, 0x3096 + 1)]
    # 日语片假名
    # jp_katakana = [chr(i) for i in range(0x30A0, 0x30FF + 1)]
    # 韩语
    # kr_hangul = [chr(i) for i in range(0xAC00, 0xD7A3 + 1)]

    # 基于上述语种构造初始词表
    # string.printable: 数字 + 字母 + 标点符号 + 空白符
    initial_alphabet = list(string.printable[:-2])
    # initial_alphabet.extend(chinese_str)
    # initial_alphabet.extend(jp_hiragana)
    # initial_alphabet.extend(jp_katakana)
    # initial_alphabet.extend(kr_hangul)
    initial_alphabet = list(sorted(set(initial_alphabet)))

    # 创建一个训练器
    trainer = trainers.BpeTrainer(
        # 词表最大的数量
        vocab_size=vocab_size,
        # 最小合并频率
        min_frequency=min_frequency,
        # 特殊标记, 注意顺序很重要, 因为 ID 号会从 0 开始标记, 依次增加
        special_tokens=special_tokens,
        # 初始字母表中, 即使在训练数据集中未见
        initial_alphabet=initial_alphabet,
        # 显示进度
        show_progress=True,
    )

    return trainer

if __name__ == "__main__":
    # 实例化 tokenizer 和 trainer
    tokenizer = set_tokenizer()
    trainer = set_tokenizer_trainer()

    # 加载数据 & 开始训练
    texts = get_text_from_jsonl(data_path=data_path)
    tokenizer.train_from_iterator(texts, trainer)

    # 使用 PreTrainedTokenizerFast 来实例化定制好的分词器
    custom_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,

        # PreTrainedTokenizerFast 默认会返回 "input_ids", "attention_mask", "token_type_ids"
        # - 因果模型只需要 "input_ids", "attention_mask", 不需要 "token_type_ids"
        # - 不添加这个会导致模型解码时出现错误
        model_input_names=["input_ids", "attention_mask"],

        # 指定特殊字符
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        unk_token="<|im_unk|>",
        pad_token="<|im_pad|>",
    )

    # 设置最大序列长度和截断策略
    custom_tokenizer.model_max_length = max_tokens
    custom_tokenizer.truncation_side = "right"

    # 指定聊天模板
    custom_tokenizer.chat_template = Path(jinja_chat_template_path).read_text(encoding='utf-8')

    # 保存分词器
    custom_tokenizer.save_pretrained(tokenizer_dir)