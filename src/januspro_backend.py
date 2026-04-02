# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JanusPro backend: LoRA on the DeepSeek language_model only.

JanusPro architecture (MultiModalityCausalLM):
  - vision_model       (SigLIP-L, frozen)
  - aligner            (MLP projector, frozen)
  - language_model      (DeepSeek-LLM, LoRA target)
  - gen_vision_model   (generation vision encoder, frozen)
  - gen_head           (generation head, frozen)

We inject LoRA into ``language_model`` targeting the MLP projections
(``gate_proj`` / ``up_proj`` / ``down_proj``), keeping everything else frozen.
Only LoRA weights are exchanged in FL.

Note: JanusPro requires ``trust_remote_code=True`` and the ``janus``
      package (``pip install -e .`` from the Janus repo).
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset

from .common import clean_generation_text, prefer_local_model_path, vqa_soft_score
from .model_registry import register_backend

_DEFAULT_MODEL = "deepseek-ai/Janus-Pro-1B"
_LOCAL_MODEL_CANDIDATES = [
    "/home/Models/Janus-Pro-1B",
]


def _torch_version_tuple() -> Tuple[int, ...]:
    version = torch.__version__.split("+", 1)[0]
    parts = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def _has_safetensors(model_id: str) -> bool:
    model_path = Path(model_id)
    return model_path.is_dir() and any(model_path.glob("*.safetensors"))


def _has_pickle_weights(model_id: str) -> bool:
    model_path = Path(model_id)
    return model_path.is_dir() and any(model_path.glob("*.bin"))


def _check_safe_weight_loading(model_id: str) -> None:
    # Transformers now blocks loading pickle-based checkpoints with torch < 2.6.
    if _has_pickle_weights(model_id) and not _has_safetensors(model_id):
        if _torch_version_tuple() < (2, 6):
            raise RuntimeError(
                f"JanusPro checkpoint at '{model_id}' only provides .bin weights, "
                f"but the current torch version is {torch.__version__}. "
                "Transformers requires torch>=2.6 to safely load pickle-based "
                "checkpoints. Please upgrade the nvflare_januspro environment "
                "to torch 2.6+ or replace the checkpoint with safetensors files."
            )


def _ensure_transformers_compatibility_for_janus() -> None:
    """Fail fast when the env drifts to unsupported Transformers releases."""

    import transformers

    major_version = int(transformers.__version__.split(".", 1)[0])
    if major_version < 5:
        return

    raise RuntimeError(
        "The editable Janus checkout in this environment is not compatible with "
        f"transformers {transformers.__version__}. "
        "Please install the versions pinned by envs/env_januspro.yml, for example:\n"
        '  conda run -n nvflare_januspro pip install "transformers==4.44.2" '
        '"tokenizers<0.20" "huggingface-hub<1.0"'
    )


def _format_vqa_question(question: str) -> str:
    return (
        "<image_placeholder>\n"
        f"{question}\n"
        "Answer the visual question with the most likely short answer only. "
        "Use one word or a short phrase when possible. "
        "Do not explain or add extra text."
    )


class JanusProVQADataset(Dataset):
    """Wraps VQAv2 for JanusPro's conversation-style input.

    JanusPro uses ``VLChatProcessor`` which expects a list-of-dict
    conversation format.  We pre-tokenize here and store the tensors.
    """

    def __init__(self, hf_ds, processor, max_q_len=128, max_a_len=32):
        self.ds = hf_ds
        self.proc = processor
        self.tokenizer = processor.tokenizer
        self.max_q = max_q_len
        self.max_a = max_a_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        image = ex["image"].convert("RGB")
        question = ex["question"]
        answer = ex["multiple_choice_answer"]
        gt_answers = [a["answer"] for a in ex["answers"]]

        # Prompt-only conversation for evaluation/generation start point.
        prompt_conversation = [
            {
                "role": "<|User|>",
                "content": _format_vqa_question(question),
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Full conversation for teacher-forced training labels.
        train_conversation = [
            {
                "role": "<|User|>",
                "content": _format_vqa_question(question),
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": answer},
        ]
  
        pil_images = [image]
        prompt_prepare = self.proc(
            conversations=prompt_conversation,
            images=pil_images,
            force_batchify=True,
        )
        train_prepare = self.proc(
            conversations=train_conversation,
            images=pil_images,
            force_batchify=True,
        )

        input_ids = train_prepare.input_ids.squeeze(0)
        attention_mask = train_prepare.attention_mask.squeeze(0)
        labels = input_ids.clone()
        prompt_len = prompt_prepare.input_ids.shape[-1]

        # Only compute loss on the assistant answer span.
        labels[:prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        image_token_id = getattr(self.proc, "image_id", None)
        if image_token_id is not None:
            labels[input_ids == image_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": train_prepare.pixel_values.squeeze(0)
            if hasattr(train_prepare, "pixel_values") and train_prepare.pixel_values is not None
            else train_prepare.images.squeeze(0),
            "images_seq_mask": train_prepare.images_seq_mask.squeeze(0)
            if hasattr(train_prepare, "images_seq_mask")
            else torch.zeros(train_prepare.input_ids.shape[-1], dtype=torch.bool),
            "images_emb_mask": train_prepare.images_emb_mask.squeeze(0)
            if hasattr(train_prepare, "images_emb_mask")
            else torch.zeros(1, dtype=torch.bool),
            "labels": labels,
            "gt_answers": gt_answers,
            "question": question,
            "multiple_choice_answer": answer,
            "question_id": ex.get("question_id"),
            "image_id": ex.get("image_id"),
            "question_type": ex.get("question_type", ""),
            "answer_type": ex.get("answer_type", ""),
            "prompt_input_ids": prompt_prepare.input_ids.squeeze(0),
            "prompt_attention_mask": prompt_prepare.attention_mask.squeeze(0),
            "prompt_images_seq_mask": prompt_prepare.images_seq_mask.squeeze(0)
            if hasattr(prompt_prepare, "images_seq_mask")
            else torch.zeros(prompt_prepare.input_ids.shape[-1], dtype=torch.bool),
        }


class JanusProBackend:
    name = "januspro"

    def build_model_and_processor(self, model_name_or_path, lora_r, lora_alpha,
                                  lora_dropout, device):
        from transformers import AutoModelForCausalLM

        model_id = prefer_local_model_path(
            model_name_or_path, _DEFAULT_MODEL, _LOCAL_MODEL_CANDIDATES)

        # JanusPro uses custom model code
        _ensure_transformers_compatibility_for_janus()
        from janus.models import MultiModalityCausalLM, VLChatProcessor

        processor = VLChatProcessor.from_pretrained(model_id)
        _check_safe_weight_loading(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True
        )

        # Apply LoRA to the language_model MLP blocks only
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["gate_proj", "up_proj", "down_proj"],
        )
        model.language_model = get_peft_model(model.language_model, cfg)
        model.language_model.config.use_cache = False
        model.language_model.gradient_checkpointing_enable()

        # Freeze everything except LoRA
        for n, p in model.named_parameters():
            p.requires_grad = "lora_" in n

        model.to(torch.bfloat16).to(device)
        return model, processor

    def build_dataset(self, hf_ds, processor, max_q_len, max_a_len):
        return JanusProVQADataset(hf_ds, processor, max_q_len, max_a_len)

    def collate_fn(self, batch):
        def _pad_1d(tensors, pad_value):
            max_len = max(t.shape[0] for t in tensors)
            out = tensors[0].new_full((len(tensors), max_len), pad_value)
            for i, t in enumerate(tensors):
                out[i, -t.shape[0]:] = t
            return out

        out = {}
        out["input_ids"] = _pad_1d(
            [b["input_ids"] for b in batch], self._pad_token_id(batch)
        )
        out["attention_mask"] = _pad_1d(
            [b["attention_mask"] for b in batch], 0
        )
        out["images_seq_mask"] = _pad_1d(
            [b["images_seq_mask"] for b in batch], False
        )
        out["labels"] = _pad_1d(
            [b["labels"] for b in batch], -100
        )
        out["prompt_input_ids"] = _pad_1d(
            [b["prompt_input_ids"] for b in batch], self._pad_token_id(batch)
        )
        out["prompt_attention_mask"] = _pad_1d(
            [b["prompt_attention_mask"] for b in batch], 0
        )
        out["prompt_images_seq_mask"] = _pad_1d(
            [b["prompt_images_seq_mask"] for b in batch], False
        )
        for k in ["pixel_values", "images_emb_mask"]:
            out[k] = torch.stack([b[k] for b in batch])
        out["gt_answers"] = [b["gt_answers"] for b in batch]
        out["questions"] = [b["question"] for b in batch]
        out["multiple_choice_answers"] = [b["multiple_choice_answer"] for b in batch]
        out["question_ids"] = [b["question_id"] for b in batch]
        out["image_ids"] = [b["image_id"] for b in batch]
        out["question_types"] = [b["question_type"] for b in batch]
        out["answer_types"] = [b["answer_type"] for b in batch]
        return out

    @staticmethod
    def _pad_token_id(batch):
        return 100002

    def train_step(self, model, batch, device):
        vision_dtype = next(model.vision_model.parameters()).dtype
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device=device, dtype=vision_dtype)
        images_seq_mask = batch["images_seq_mask"].to(device)
        images_emb_mask = batch["images_emb_mask"].to(device)
        labels = batch["labels"].to(device)

        # Prepare multimodal embeddings
        inputs_embeds = model.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )
        # Janus feeds precomputed multimodal embeddings directly into the language
        # model. When gradient checkpointing is enabled, these inputs must require
        # grad or the loss can be detached even though LoRA weights are trainable.
        inputs_embeds = inputs_embeds.requires_grad_()

        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def _evaluate_impl(self, model, dataloader, processor, device, collect_details=False):
        model.eval()
        tokenizer = processor.tokenizer
        total_score, total = 0.0, 0
        details = []

        for batch in dataloader:
            vision_dtype = next(model.vision_model.parameters()).dtype
            input_ids = batch["prompt_input_ids"].to(device)
            attention_mask = batch["prompt_attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device=device, dtype=vision_dtype)
            images_seq_mask = batch["prompt_images_seq_mask"].to(device)
            images_emb_mask = batch["images_emb_mask"].to(device)

            inputs_embeds = model.prepare_inputs_embeds(
                input_ids=input_ids,
                pixel_values=pixel_values,
                images_seq_mask=images_seq_mask,
                images_emb_mask=images_emb_mask,
            )

            gen_ids = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for idx, (p, gt) in enumerate(zip(preds, batch["gt_answers"])):
                answer = clean_generation_text(p)
                score = vqa_soft_score(answer, gt)
                total_score += score
                total += 1
                if collect_details:
                    normalized_gt = [a.strip().lower() for a in gt]
                    normalized_pred = answer.strip().lower()
                    match_count = sum(1 for a in normalized_gt if normalized_pred == a)
                    details.append({
                        "question": batch["questions"][idx],
                        "prediction_raw": p,
                        "prediction_clean": answer,
                        "multiple_choice_answer": batch["multiple_choice_answers"][idx],
                        "gt_answers": gt,
                        "soft_score": score,
                        "match_count": match_count,
                        "num_gt_answers": len(gt),
                        "question_id": batch["question_ids"][idx],
                        "image_id": batch["image_ids"][idx],
                        "question_type": batch["question_types"][idx],
                        "answer_type": batch["answer_types"][idx],
                    })

        metric = total_score / max(total, 1)
        if collect_details:
            return metric, details
        return metric

    @torch.no_grad()
    def evaluate(self, model, dataloader, processor, device):
        return self._evaluate_impl(model, dataloader, processor, device, collect_details=False)

    @torch.no_grad()
    def evaluate_with_details(self, model, dataloader, processor, device):
        return self._evaluate_impl(model, dataloader, processor, device, collect_details=True)

    def hf_dataset_name(self):
        return "HuggingFaceM4/VQAv2"

    def hf_train_split(self):
        return "train"

    def hf_eval_split(self):
        return "validation[:50%]"

    def keep_columns(self):
        return [
            "image",
            "question",
            "multiple_choice_answer",
            "answers",
            "question_id",
            "image_id",
            "question_type",
            "answer_type",
        ]


register_backend("januspro", JanusProBackend())
