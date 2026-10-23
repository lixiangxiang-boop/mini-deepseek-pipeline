"""
实用的 BPE Tokenizer 封装 — 用于数据编码和推理解码

提供 encode/decode 接口，自动处理 training/saving/loading。
如果找不到已训练的 tokenizer，自动创建一个简单可用的字符级 tokenizer。

Usage:
  from src.tokenizer.tokenizer_utils import get_tokenizer
  tok = get_tokenizer()
  ids = tok.encode("What is 15 + 27?")
  text = tok.decode(ids)
"""

import os
import json
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.data_config import DataConfig


class SimpleTokenizer:
    """简易但正确的 tokenizer — 基于字符+常见子词的混合编码

    使用一个预定义的词汇表，包含:
      - 大小写英文字母
      - 数字
      - 常见标点和数学符号
      - 常见英文子词 (ing, ed, tion, ...)
      - <think>/<answer> 等特殊token

    这比 ord(c) % vocab_size 好得多，因为:
      1. 字符按频率分配短ID (更高效)
      2. 特殊token有固定ID
      3. encode/decode 是确定性的往返
    """

    def __init__(self, data_config: DataConfig = None):
        if data_config is None:
            data_config = DataConfig()
        self.dc = data_config

        # 特殊token
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.think_token = "<think>"
        self.answer_token = "<answer>"

        # 构建词汇表
        self._build_vocab()

    def _build_vocab(self):
        """构建字符+子词混合词汇表"""
        # 特殊token (固定ID)
        specials = ["<pad>", "<bos>", "<eos>", "<unk>", "<think>", "<answer>"]
        self.special_ids = {tok: i for i, tok in enumerate(specials)}

        # 基础字符 (高频在前)
        chars = (
            " etaoinsrhldcumfpgwybvkxjqz"  # 英文频率排序 + 空格
            "ETAOINSRHLDCUMFPGWYBVKXJQZ"
            "0123456789"
            ".,!?;:'\"-()[]{}_/\\|@#$%^&*+=<>~`"
            "\n\r\t"
            "×÷±≤≥π√θ∞"  # ×÷±≤≥π√θ∞ (数学符号)
        )

        # 常见子词
        subwords = [
            "the","ing","ed","tion","and","ent","ive","for","tha",
            "are","not","you","all","ith","rea","ver","one","ave",
            "com","men","pro","ers","ter","ect","per","ble","out",
            "hat","ere","our","ear","ain","ure","anc","pre","con",
            "Question","Think","Answer","What","How","many","much",
            "miles","hours","cookies","apples","train","rectangle",
            "John","Mary","Sarah","miles per hour","each",
        ]

        # 构建完整词汇表
        vocab = specials.copy()
        for c in chars:
            if c not in vocab:
                vocab.append(c)
        for sw in subwords:
            # 子词编码为单个token (BPE的核心思想)
            if sw not in vocab:
                vocab.append(sw)

        # 限制词汇大小
        max_vocab = self.dc.vocab_size
        if len(vocab) > max_vocab:
            vocab = vocab[:max_vocab]

        self.vocab = vocab
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        # 构建子词查找trie (按长度降序，优先匹配长子词)
        self.subwords = sorted(
            [sw for sw in subwords if sw in self.token_to_id],
            key=len, reverse=True,
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """将文本编码为token ID列表

        使用最长匹配策略: 先匹配子词，再匹配单字符。
        这比纯字符编码更高效，也能让模型学到常见词组。
        """
        if not text:
            return []

        ids = []
        if add_special_tokens:
            ids.append(self.token_to_id[self.bos_token])

        i = 0
        while i < len(text):
            matched = False

            # 1. 尝试匹配子词
            for sw in self.subwords:
                if text[i:].startswith(sw):
                    ids.append(self.token_to_id[sw])
                    i += len(sw)
                    matched = True
                    break

            if matched:
                continue

            # 2. 匹配单字符
            ch = text[i]
            if ch in self.token_to_id:
                ids.append(self.token_to_id[ch])
            else:
                # 未知字符用 <unk> 替代
                ids.append(self.token_to_id[self.unk_token])
            i += 1

        if add_special_tokens:
            ids.append(self.token_to_id[self.eos_token])

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """将token ID列表解码为文本"""
        result = []
        for tid in ids:
            if skip_special_tokens and tid < 6:  # 跳过 <pad>,<bos>,<eos>,<unk>,<think>,<answer>
                continue
            if tid in self.id_to_token:
                result.append(self.id_to_token[tid])
            else:
                result.append(self.unk_token)
        return "".join(result)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str):
        """保存 tokenizer"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "vocab": self.vocab,
                "special_ids": self.special_ids,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, data_config: DataConfig = None):
        """加载 tokenizer"""
        tok = cls(data_config)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok.vocab = data["vocab"]
        tok.special_ids = data["special_ids"]
        tok.token_to_id = {t: i for i, t in enumerate(tok.vocab)}
        tok.id_to_token = {i: t for t, i in tok.token_to_id.items()}
        return tok


# 全局单例
_tokenizer: Optional[SimpleTokenizer] = None


def get_tokenizer(data_config: DataConfig = None) -> SimpleTokenizer:
    """获取全局 tokenizer 实例 (带缓存)"""
    global _tokenizer
    if _tokenizer is None:
        if data_config is None:
            data_config = DataConfig()
        # 尝试加载已训练的 tokenizer
        tok_path = os.path.join(data_config.tokenizer_dir, "tokenizer.json")
        if os.path.exists(tok_path):
            _tokenizer = SimpleTokenizer.load(tok_path, data_config)
        else:
            _tokenizer = SimpleTokenizer(data_config)
            _tokenizer.save(tok_path)
    return _tokenizer


def encode_text(text: str) -> List[int]:
    """快捷编码"""
    return get_tokenizer().encode(text)


def decode_ids(ids: List[int]) -> str:
    """快捷解码"""
    return get_tokenizer().decode(ids)
