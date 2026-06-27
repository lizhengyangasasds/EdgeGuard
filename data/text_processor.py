"""
EdgeGuard: Text Processor Module

Text tokenization, encoding, augmentation, and synthetic alert text generation
for the multimodal inference pipeline.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np


class TextProcessor:
    """
    Text tokenization and encoding for EdgeGuard.

    Wraps a HuggingFace tokenizer with augmentation utilities and
    consistent interface for the inference pipeline.

    Args:
        tokenizer_name: HuggingFace tokenizer identifier (default distilbert-base-uncased).
        max_length: Maximum sequence length (default 128).
        padding: Padding strategy ("max_length" or "longest").

    Example:
        >>> processor = TextProcessor(tokenizer_name="distilbert-base-uncased")
        >>> tokens = processor.tokenize("Unauthorized entry detected", return_tensors="pt")
        >>> decoded = processor.decode(tokens["input_ids"])
    """

    def __init__(
        self,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        padding: str = "max_length",
    ) -> None:
        self.max_length = max_length
        self.padding = padding
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except ImportError:
            self._tokenizer = None

    def tokenize(
        self,
        texts: str | list[str],
        return_tensors: str | None = None,
    ) -> dict[str, np.ndarray | list]:
        """
        Tokenize text(s) into input IDs and attention masks.

        Args:
            texts: Single text string or list of texts.
            return_tensors: If "pt", return PyTorch tensors; else return numpy lists.

        Returns:
            Dictionary with "input_ids" and "attention_mask" keys.
        """
        if self._tokenizer is None:
            raise ImportError("transformers is required for TextProcessor. Install with: pip install transformers")

        is_single = isinstance(texts, str)
        texts = [texts] if is_single else texts

        outputs = self._tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors=return_tensors or "np",
        )

        result = {k: v for k, v in outputs.items()}
        return result

    def decode(self, input_ids: np.ndarray | list) -> str | list[str]:
        """
        Decode token IDs back to text strings.

        Args:
            input_ids: Token ID array of shape (seq_len,) or (B, seq_len).
            Can be numpy array, Python list, or PyTorch tensor.

        Returns:
            Decoded string or list of strings.
        """
        if self._tokenizer is None:
            raise ImportError("transformers is required for TextProcessor. Install with: pip install transformers")

        if hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()

        is_batch = isinstance(input_ids, np.ndarray) and input_ids.ndim > 1

        if not is_batch:
            return self._tokenizer.decode(input_ids.tolist() if isinstance(input_ids, np.ndarray) else input_ids, skip_special_tokens=True)
        return self._tokenizer.batch_decode(input_ids.tolist() if isinstance(input_ids, np.ndarray) else input_ids, skip_special_tokens=True)

    def augment_synonym_replacement(
        self,
        text: str,
        augmentation_ratio: float = 0.3,
        seed: int = 42,
    ) -> str:
        """
        Perform synonym replacement augmentation on text.

        Args:
            text: Input text string.
            augmentation_ratio: Fraction of words to replace (0.0 - 1.0).
            seed: Random seed for reproducibility.

        Returns:
            Augmented text string.
        """
        synonym_map = {
            "unauthorized": "illegal",
            "detected": "identified",
            "person": "individual",
            "entry": "access",
            "zone": "area",
            "perimeter": "boundary",
            "crowd": "group",
            "gathering": "assembly",
            "reverse": "backward",
            "abnormal": "unusual",
            "motion": "movement",
            "restricted": "restricted",
            "alert": "warning",
            "intrusion": "breach",
            "climbing": "scaling",
            "vehicle": "car",
            "unattended": "abandoned",
            "deviation": "departure",
            "patrol": "guard",
        }

        words = text.lower().split()
        random.seed(seed)
        num_to_replace = max(1, int(len(words) * augmentation_ratio))
        indices = random.sample(range(len(words)), min(num_to_replace, len(words)))

        for idx in indices:
            word = words[idx]
            if word in synonym_map:
                words[idx] = synonym_map[word]

        result = " ".join(words)
        if text[0].isupper():
            result = result[0].upper() + result[1:]
        return result


class AlertTextGenerator:
    """
    Generate synthetic alert text samples with behavior and alert labels.

    Provides templates for realistic security-alert scenarios across
    the 7 behavior classes and 5 alert types.

    Args:
        language: Language for generated texts ("en" or "zh").

    Example:
        >>> generator = AlertTextGenerator(language="en")
        >>> samples = generator.generate(num_samples=50)
        >>> print(samples[0])
        # {"text": "Fighting detected in corridor", "behavior": 0, "alert": 3}
    """

    BEHAVIOR_TEMPLATES = {
        0: [  # fighting
            "Fighting detected in corridor",
            "Physical altercation identified between individuals",
            "Violent confrontation recorded at main entrance",
            "Two persons engaged in physical fight",
            "Assault detected in zone B2",
        ],
        1: [  # falling
            "Person falling detected near staircase",
            "Individual collapse detected in hallway",
            "Sudden fall detected in loading area",
            "Fall event detected - medical attention may be required",
            "Unusual falling motion detected in parking lot",
        ],
        2: [  # climbing
            "Person climbing fence at perimeter",
            "Unauthorized climbing detected on boundary wall",
            "Individual scaling restricted fence",
            "Climbing behavior detected at west wall",
            "Trespasser climbing barrier fence",
        ],
        3: [  # loitering
            "Person loitering near restricted area",
            "Prolonged presence detected in security zone",
            "Unauthorized individual lingering at entrance",
            "Loitering detected near vault room",
            "Suspicious lingering behavior in monitored area",
        ],
        4: [  # retrograde
            "Vehicle moving in reverse direction",
            "Retrograde movement detected on access road",
            "Vehicle traveling wrong way on driveway",
            "Backward movement detected in one-way zone",
            "Reverse trajectory detected at checkpoint",
        ],
        5: [  # gathering
            "Abnormal crowd gathering detected",
            "Large group assembly in unauthorized area",
            "Unusual gathering of persons near facility",
            "Crowd accumulation detected at main gate",
            "Group formation detected in restricted zone",
        ],
        6: [  # normal
            "Normal activity in monitored area",
            "Regular patrol movement detected",
            "Authorized personnel activity confirmed",
            "Routine operation in progress",
            "No suspicious activity in coverage area",
        ],
    }

    ALERT_TEMPLATES = {
        0: "intrusion",
        1: "fault",
        2: "violation",
        3: "anomaly",
        4: "normal",
    }

    def __init__(self, language: str = "en") -> None:
        self.language = language
        self._zh_templates = {
            0: ["走廊检测到打架", "出入口检测到肢体冲突", "主入口记录到暴力对抗"],
            1: ["楼梯口检测到人员跌倒", "大厅检测到人员摔倒", "装卸区检测到突然跌倒"],
            2: ["周界检测到翻越围栏", "边界墙检测到违规攀爬", "西侧围栏检测到攀爬行为"],
            3: ["限制区域检测到徘徊", "安保区域检测到长时间滞留", "入口检测到未经授权人员"],
            4: ["车辆反向行驶", "入口通道检测到逆向行驶", "单向区域检测到倒车行为"],
            5: ["检测到异常人群聚集", "未授权区域检测到大规模人员聚集", "主大门检测到人群聚集"],
            6: ["监控区域正常活动", "常规巡逻行为确认", "授权人员正常作业"],
        }

    def generate(self, num_samples: int = 100) -> list[dict]:
        """
        Generate synthetic alert text samples.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            List of dicts with "text", "behavior" (class index), and "alert" (class index).
        """
        samples = []
        for _ in range(num_samples):
            behavior_class = random.randint(0, 6)
            alert_class = random.randint(0, 4)

            if self.language == "zh":
                texts = self._zh_templates.get(behavior_class, ["正常活动"])
                text = random.choice(texts)
            else:
                texts = self.BEHAVIOR_TEMPLATES.get(behavior_class, ["Normal activity"])
                text = random.choice(texts)

            samples.append({
                "text": text,
                "behavior": behavior_class,
                "alert": alert_class,
                "label": behavior_class,
            })

        return samples
