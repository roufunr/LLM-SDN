#!/usr/bin/env python3
"""
Packet Info Predictor using Transformer Model with QLoRA + LoRA + Optional 8-bit Adam

- Uses BitsAndBytesConfig for 4-bit quantization when bitsandbytes is available.
- Falls back to fp16 + LoRA if bitsandbytes isn't present.
- Avoids device_map="auto" in 4-bit path to prevent internal .to(...) calls.
- Derives a safe device handle from model parameters (works for bnb & non-bnb).
- Short context/target lengths + grad checkpointing to fit ~12 GB VRAM.
"""

import os
import json
import logging
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.generation.logits_process import LogitsProcessor

from peft import LoraConfig, get_peft_model

# ---- Optional bitsandbytes / 8-bit Adam --------------------------------------
try:
    import bitsandbytes as bnb  # noqa: F401
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

try:
    from bitsandbytes.optim import Adam8bit
    _HAS_8BIT_OPT = True
except Exception:
    Adam8bit = None
    _HAS_8BIT_OPT = False

# ---- Optional BERTScore -------------------------------------------------------
try:
    from bert_score import score as bertscore_score  # type: ignore
    _HAS_BERTSCORE = True
except Exception:
    bertscore_score = None  # type: ignore
    _HAS_BERTSCORE = False

# ---- BERTScore config ---------------------------------------------------------
BERTSCORE_DEVICE = "cpu"
BERTSCORE_MODEL_TYPE = "roberta-large"


@dataclass
class PredictionResult:
    predicted_info: List[str]
    actual_info: List[str]
    accuracy: float
    bit_accuracy: float      # stores BERTScore F1
    word_accuracy: float
    position_accuracy: float
    reward: float
    iteration: int


class PacketInfoPredictor:
    def __init__(
        self,
        model_name: str = "distilgpt2",
        sequence_length: int = 100,
        prediction_length: int = 1,
        learning_rate: float = 1e-4,
        # short lengths to avoid OOM
        ctx_max_len: int = 128,
        tgt_max_len: int = 128,
        train_max_tokens: int = 128,
        # compute BERTScore every N steps (0 to disable)
        compute_bertscore_every_n: int = 200,
        # LoRA config
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        # optimizer
        use_8bit_adam: bool = True,
        # quantization toggle (disable to avoid accelerate dispatch and .to issues)
        use_qlora: bool = False,
    ):
        self.sequence_length = int(sequence_length)
        self.prediction_length = int(prediction_length)
        self.learning_rate = float(learning_rate)
        self.ctx_max_len = int(ctx_max_len)
        self.tgt_max_len = int(tgt_max_len)
        self.train_max_tokens = int(train_max_tokens)
        self.compute_bertscore_every_n = int(compute_bertscore_every_n)
        self.use_8bit_adam = use_8bit_adam and _HAS_8BIT_OPT
        self.use_qlora = bool(use_qlora)

        self.logger = self._setup_logging()
        self.model, self.tokenizer, self._device = self._initialize_model_and_lora(
            model_name, lora_r, lora_alpha, lora_dropout
        )

        if self.use_8bit_adam:
            self.logger.info("Using 8-bit Adam optimizer (bitsandbytes).")
            self.optimizer = Adam8bit(self.model.parameters(), lr=self.learning_rate)
        else:
            if use_8bit_adam and not _HAS_8BIT_OPT:
                self.logger.warning("bitsandbytes 8-bit Adam not available -> falling back to AdamW.")
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.training_history = {
            "iterations": [],
            "accuracies": [],
            "bit_accuracies": [],
            "word_accuracies": [],
            "position_accuracies": [],
            "rewards": [],
        }
        self.csv_results: List[Dict[str, Any]] = []

    # ----------------------------- Setup / Init --------------------------------

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("packet_predictor")
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            logger.addHandler(ch)

            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            fh = logging.FileHandler(log_dir / "packet_predictor.log")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        return logger

    def _initialize_model_and_lora(self, model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
        using_bnb = _HAS_BNB and self.use_qlora
        self.logger.info(
            f"Loading base model: {model_name} "
            f"({'4-bit QLoRA' if using_bnb else 'fp16 (no QLoRA)' })."
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if using_bnb:
            # Quantized 4-bit config (no torch_dtype here to avoid .to() calls)
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                low_cpu_mem_usage=True
            )
            # pick a primary device from model params
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Fallback fp16 (safe to move manually)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=None,
                low_cpu_mem_usage=True
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Memory-saving settings
        if hasattr(model, "config"):
            try:
                model.config.use_cache = False
            except Exception:
                pass
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        # LoRA adapters (include GPT-2 style proj names for broader coverage)
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            "c_attn", "c_proj"
        ]
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        self.logger.info("Base loaded, LoRA attached.")

        return model, tokenizer, device

    # ----------------------------- Metrics -------------------------------------
    def _calculate_bertscore(self, predicted: str, actual: str) -> float:
        if not _HAS_BERTSCORE:
            return 0.0
        P, R, F1 = bertscore_score(
            [predicted], [actual],
            lang="en",
            rescale_with_baseline=True,
            device=BERTSCORE_DEVICE,
            model_type=BERTSCORE_MODEL_TYPE,
        )
        return float(F1.mean().item())

    def _calculate_word_accuracy(self, predicted: str, actual: str) -> float:
        pw = predicted.split()
        aw = actual.split()
        if not aw:
            return 1.0 if not pw else 0.0
        m = sum(1 for i in range(min(len(pw), len(aw))) if pw[i] == aw[i])
        return m / len(aw)

    def _calculate_position_accuracy(self, predicted: str, actual: str) -> float:
        pw = predicted.split()
        aw = actual.split()
        if not aw:
            return 1.0 if not pw else 0.0
        ok = 0
        for i, w in enumerate(aw):
            s = max(0, i - 2)
            e = min(len(pw), i + 3)
            if w in pw[s:e]:
                ok += 1
        return ok / len(aw)

    def _position_counts(self, predicted: str, actual: str) -> Tuple[int, int]:
        pw = predicted.split()
        aw = actual.split()
        total = len(aw)
        if total == 0:
            return (0, 0)
        ok = 0
        for i, w in enumerate(aw):
            s = max(0, i - 2)
            e = min(len(pw), i + 3)
            if w in pw[s:e]:
                ok += 1
        return ok, total

    def _calculate_reward(self, bert_f1: float) -> float:
        return 2.0 / (1.0 + np.exp(-10 * (bert_f1 - 0.5))) - 1.0

    # ----------------------------- Inference -----------------------------------
    def predict_next_packets(self, context_packets: List[str]) -> List[str]:
        self.model.eval()
        ctx = " ".join(context_packets)
        toks = self.tokenizer(
            ctx, return_tensors="pt", truncation=True, max_length=self.ctx_max_len, padding=False
        )
        input_ids = toks["input_ids"].to(self._device)
        attention_mask = toks.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        logits_processor = [NanInfClampLogitsProcessor()]
        amp_ctx = torch.amp.autocast("cuda", enabled=False) if getattr(self._device, "type", "cpu") == "cuda" else nullcontext()
        with torch.inference_mode(), amp_ctx:
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max(10, self.prediction_length * 10),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return self._extract_packet_info(gen, self.prediction_length)

    def _extract_packet_info(self, generated_text: str, num_packets: int) -> List[str]:
        lines = generated_text.split("\n")
        out: List[str] = []
        for line in lines:
            if len(out) >= num_packets:
                break
            if any(k in line.lower() for k in ["tcp", "udp", "http", "dns", "icmp", "arp"]):
                out.append(line.strip())
        while len(out) < num_packets:
            out.append("Generic packet info")
        return out[:num_packets]

    # ----------------------------- Training ------------------------------------
    def train_iteration(self, context_packets: List[str], actual_next_packets: List[str], iteration: int) -> PredictionResult:
        self.model.train()
        predicted_info = self.predict_next_packets(context_packets)
        bert_scores, word_accs, pos_accs = [], [], []
        total_pos_ok, total_words = 0, 0
        compute_bs = (self.compute_bertscore_every_n > 0 and iteration % self.compute_bertscore_every_n == 0)
        for pred, actual in zip(predicted_info, actual_next_packets):
            bert_f1 = self._calculate_bertscore(pred, actual) if compute_bs else 0.0
            wacc = self._calculate_word_accuracy(pred, actual)
            pacc = self._calculate_position_accuracy(pred, actual)
            ok, t = self._position_counts(pred, actual)
            bert_scores.append(bert_f1)
            word_accs.append(wacc)
            pos_accs.append(pacc)
            total_pos_ok += ok
            total_words += t
        avg_bert_f1 = float(np.mean(bert_scores)) if bert_scores else 0.0
        avg_word_accuracy = float(np.mean(word_accs)) if word_accs else 0.0
        avg_position_accuracy = float(np.mean(pos_accs)) if pos_accs else 0.0
        reward = self._calculate_reward(avg_bert_f1)
        exact = sum(1 for p, a in zip(predicted_info, actual_next_packets) if p.strip() == a.strip())
        accuracy = exact / max(1, len(actual_next_packets))
        self._update_model(reward, context_packets, actual_next_packets)
        result = PredictionResult(
            predicted_info=predicted_info,
            actual_info=actual_next_packets,
            accuracy=accuracy,
            bit_accuracy=avg_bert_f1,
            word_accuracy=avg_word_accuracy,
            position_accuracy=avg_position_accuracy,
            reward=reward,
            iteration=iteration,
        )
        self._log_iteration_result(result, iteration)
        self.csv_results.append({
            "iteration": iteration,
            "accuracy": accuracy,
            "bit_accuracy": avg_bert_f1,
            "word_accuracy": avg_word_accuracy,
            "position_accuracy": avg_position_accuracy,
            "reward": reward,
            "bit_correct": "",
            "bit_total": "",
            "word_pos_correct": int(total_pos_ok),
            "word_total": int(total_words),
            "predicted_samples": " | ".join(predicted_info[:3]),
            "actual_samples": " | ".join(actual_next_packets[:3]),
        })
        return result

    def _update_model(self, reward: float, context: List[str], actual: List[str]):
        ctx_text = " ".join(context)
        tgt_text = " ".join(actual)
        ctx = self.tokenizer(ctx_text, return_tensors="pt", truncation=True, max_length=self.ctx_max_len, padding=False)
        tgt = self.tokenizer(tgt_text, return_tensors="pt", truncation=True, max_length=self.tgt_max_len, padding=False)
        ctx_ids = ctx["input_ids"].to(self._device)
        tgt_ids = tgt["input_ids"].to(self._device)
        ctx_mask = ctx.get("attention_mask")
        tgt_mask = tgt.get("attention_mask")
        if ctx_mask is not None:
            ctx_mask = ctx_mask.to(self._device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(self._device)
        ctx_len = ctx_ids.shape[1]
        allow_tgt = max(1, self.train_max_tokens - ctx_len)
        if tgt_ids.shape[1] > allow_tgt:
            tgt_ids = tgt_ids[:, :allow_tgt]
            if tgt_mask is not None:
                tgt_mask = tgt_mask[:, :allow_tgt]
        input_ids = torch.cat([ctx_ids, tgt_ids], dim=1)
        if ctx_mask is not None and tgt_mask is not None:
            attention_mask = torch.cat([ctx_mask, tgt_mask], dim=1)
        else:
            attention_mask = torch.ones_like(input_ids, device=self._device)
        labels = input_ids.clone()
        labels[:, :ctx_len] = -100
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        adjusted_loss = loss * (1.0 - reward)
        adjusted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

    # ----------------------------- Logging / I/O --------------------------------
    def _log_iteration_result(self, result: PredictionResult, iteration: int):
        self.logger.info(
            f"Iter {iteration}: Acc={result.accuracy:.2%} | "
            f"BERTF1={result.bit_accuracy:.2%} | WordAcc={result.word_accuracy:.2%} | "
            f"PosAcc={result.position_accuracy:.2%} | Reward={result.reward:.4f}"
        )
        self.training_history["iterations"].append(iteration)
        self.training_history["accuracies"].append(result.accuracy)
        self.training_history["bit_accuracies"].append(result.bit_accuracy)
        self.training_history["word_accuracies"].append(result.word_accuracy)
        self.training_history["position_accuracies"].append(result.position_accuracy)
        self.training_history["rewards"].append(result.reward)

    def run_training(self, packet_data: List[str], num_iterations: int = 200):
        self.logger.info(
            f"Starting training on {len(packet_data)} packets | "
            f"seq_len={self.sequence_length}, pred_len={self.prediction_length}, "
            f"ctx_max={self.ctx_max_len}, tgt_max={self.tgt_max_len}, "
            f"train_max_tokens={self.train_max_tokens}"
        )
        for iteration in tqdm(range(num_iterations), desc="Training Progress"):
            start = iteration * self.prediction_length
            end = start + self.sequence_length
            pstart = end
            pend = pstart + self.prediction_length
            if pend > len(packet_data):
                self.logger.info(f"Reached end of data at iteration {iteration}")
                break
            ctx_packets = packet_data[start:end]
            actual_next = packet_data[pstart:pend]
            result = self.train_iteration(ctx_packets, actual_next, iteration)
            if iteration % 10 == 0:
                print(f"Iter {iteration}: Acc={result.accuracy:.2%} "
                      f"BERTF1={result.bit_accuracy:.2%} Reward={result.reward:.4f}")
        self.logger.info("Training completed!")
        self._plot_training_history()

    def _plot_training_history(self):
        it = self.training_history["iterations"]
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].plot(it, self.training_history["accuracies"]); axs[0, 0].set_title("Accuracy"); axs[0, 0].grid(True)
        axs[0, 1].plot(it, self.training_history["bit_accuracies"]); axs[0, 1].set_title("BERTScore F1"); axs[0, 1].grid(True)
        axs[1, 0].plot(it, self.training_history["word_accuracies"]); axs[1, 0].set_title("Word Accuracy"); axs[1, 0].grid(True)
        axs[1, 1].plot(it, self.training_history["position_accuracies"]); axs[1, 1].set_title("Position Accuracy"); axs[1, 1].grid(True)
        plt.tight_layout()
        plt.savefig("training_history.png")
        # plt.show()

    def save_csv_results(self, filename: str = "training_results.csv"):
        if not self.csv_results:
            return
        df = pd.DataFrame(self.csv_results)
        df.to_csv(filename, index=False)
        self.logger.info(f"Saved results to {filename}")
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total iterations: {len(self.csv_results)}")
        print(f"Final Accuracy: {df['accuracy'].iloc[-1]:.2%}")
        print(f"Final Bit Accuracy: {df['bit_accuracy'].iloc[-1]:.2%}")
        print(f"Final Word Accuracy: {df['word_accuracy'].iloc[-1]:.2%}")
        print(f"Final Position Accuracy: {df['position_accuracy'].iloc[-1]:.2%}")
        print(f"Best Accuracy: {df['accuracy'].max():.2%}")
        print(f"Best Bit Accuracy: {df['bit_accuracy'].max():.2%}")
        print(f"Best Word Accuracy: {df['word_accuracy'].max():.2%}")
        print(f"Best Position Accuracy: {df['position_accuracy'].max():.2%}")
        print("=" * 60)

    def save_model(self, path: str = "packet_predictor_model"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(f"{path}/training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str = "packet_predictor_model"):
        self.model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if hasattr(self.model, "config"):
            try:
                self.model.config.use_cache = False
            except Exception:
                pass
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        try:
            with open(f"{path}/training_history.json", "r") as f:
                self.training_history = json.load(f)
        except FileNotFoundError:
            self.logger.warning("No training history found")
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Model loaded from {path}")

    # ----------------------------- Metrics -------------------------------------

class NanInfClampLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, -1e4))
        return torch.clamp(scores, min=-1e4, max=1e4)

    def _calculate_bertscore(self, predicted: str, actual: str) -> float:
        if not _HAS_BERTSCORE:
            return 0.0
        P, R, F1 = bertscore_score(
            [predicted], [actual],
            lang="en",
            rescale_with_baseline=True,
            device=BERTSCORE_DEVICE,
            model_type=BERTSCORE_MODEL_TYPE,
        )
        return float(F1.mean().item())

    def _calculate_word_accuracy(self, predicted: str, actual: str) -> float:
        pw = predicted.split()
        aw = actual.split()
        if not aw:
            return 1.0 if not pw else 0.0
        m = sum(1 for i in range(min(len(pw), len(aw))) if pw[i] == aw[i])
        return m / len(aw)

    def _calculate_position_accuracy(self, predicted: str, actual: str) -> float:
        pw = predicted.split()
        aw = actual.split()
        if not aw:
            return 1.0 if not pw else 0.0
        ok = 0
        for i, w in enumerate(aw):
            s = max(0, i - 2)
            e = min(len(pw), i + 3)
            if w in pw[s:e]:
                ok += 1
        return ok / len(aw)

    def _position_counts(self, predicted: str, actual: str) -> Tuple[int, int]:
        pw = predicted.split()
        aw = actual.split()
        total = len(aw)
        if total == 0:
            return (0, 0)
        ok = 0
        for i, w in enumerate(aw):
            s = max(0, i - 2)
            e = min(len(pw), i + 3)
            if w in pw[s:e]:
                ok += 1
        return ok, total

    def _calculate_reward(self, bert_f1: float) -> float:
        # smooth mapping of [0,1] -> (-1,1), centered around 0.5
        return 2.0 / (1.0 + np.exp(-10 * (bert_f1 - 0.5))) - 1.0

    # ----------------------------- Inference -----------------------------------

    def predict_next_packets(self, context_packets: List[str]) -> List[str]:
        self.model.eval()
        ctx = " ".join(context_packets)
        toks = self.tokenizer(
            ctx, return_tensors="pt", truncation=True, max_length=self.ctx_max_len, padding=False
        )
        # use safe device handle
        input_ids = toks["input_ids"].to(self._device)
        attention_mask = toks.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        # Clamp unsafe logits to avoid NaN/inf during sampling
        logits_processor = [NanInfClampLogitsProcessor()]
        amp_ctx = torch.amp.autocast("cuda", enabled=False) if getattr(self._device, "type", "cpu") == "cuda" else nullcontext()

        with torch.inference_mode(), amp_ctx:
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max(10, self.prediction_length * 10),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return self._extract_packet_info(gen, self.prediction_length)

    def _extract_packet_info(self, generated_text: str, num_packets: int) -> List[str]:
        lines = generated_text.split("\n")
        out: List[str] = []
        for line in lines:
            if len(out) >= num_packets:
                break
            if any(k in line.lower() for k in ["tcp", "udp", "http", "dns", "icmp", "arp"]):
                out.append(line.strip())
        while len(out) < num_packets:
            out.append("Generic packet info")
        return out[:num_packets]

    # ----------------------------- Training ------------------------------------

    def train_iteration(
        self, context_packets: List[str], actual_next_packets: List[str], iteration: int
    ) -> PredictionResult:
        self.model.train()

        predicted_info = self.predict_next_packets(context_packets)

        # metrics
        bert_scores, word_accs, pos_accs = [], [], []
        total_pos_ok, total_words = 0, 0
        compute_bs = (self.compute_bertscore_every_n > 0 and
                      iteration % self.compute_bertscore_every_n == 0)

        for pred, actual in zip(predicted_info, actual_next_packets):
            bert_f1 = self._calculate_bertscore(pred, actual) if compute_bs else 0.0
            wacc = self._calculate_word_accuracy(pred, actual)
            pacc = self._calculate_position_accuracy(pred, actual)
            ok, t = self._position_counts(pred, actual)
            bert_scores.append(bert_f1)
            word_accs.append(wacc)
            pos_accs.append(pacc)
            total_pos_ok += ok
            total_words += t

        avg_bert_f1 = float(np.mean(bert_scores)) if bert_scores else 0.0
        avg_word_accuracy = float(np.mean(word_accs)) if word_accs else 0.0
        avg_position_accuracy = float(np.mean(pos_accs)) if pos_accs else 0.0
        reward = self._calculate_reward(avg_bert_f1)

        exact = sum(1 for p, a in zip(predicted_info, actual_next_packets) if p.strip() == a.strip())
        accuracy = exact / max(1, len(actual_next_packets))

        # teacher-forced update (labels ignore context)
        self._update_model(reward, context_packets, actual_next_packets)

        result = PredictionResult(
            predicted_info=predicted_info,
            actual_info=actual_next_packets,
            accuracy=accuracy,
            bit_accuracy=avg_bert_f1,
            word_accuracy=avg_word_accuracy,
            position_accuracy=avg_position_accuracy,
            reward=reward,
            iteration=iteration,
        )
        self._log_iteration_result(result, iteration)
        self.csv_results.append({
            "iteration": iteration,
            "accuracy": accuracy,
            "bit_accuracy": avg_bert_f1,
            "word_accuracy": avg_word_accuracy,
            "position_accuracy": avg_position_accuracy,
            "reward": reward,
            "bit_correct": "",
            "bit_total": "",
            "word_pos_correct": int(total_pos_ok),
            "word_total": int(total_words),
            "predicted_samples": " | ".join(predicted_info[:3]),
            "actual_samples": " | ".join(actual_next_packets[:3]),
        })
        return result

    def _update_model(self, reward: float, context: List[str], actual: List[str]):
        ctx_text = " ".join(context)
        tgt_text = " ".join(actual)

        ctx = self.tokenizer(
            ctx_text, return_tensors="pt", truncation=True, max_length=self.ctx_max_len, padding=False
        )
        tgt = self.tokenizer(
            tgt_text, return_tensors="pt", truncation=True, max_length=self.tgt_max_len, padding=False
        )

        ctx_ids = ctx["input_ids"].to(self._device)
        tgt_ids = tgt["input_ids"].to(self._device)
        ctx_mask = ctx.get("attention_mask")
        tgt_mask = tgt.get("attention_mask")
        if ctx_mask is not None:
            ctx_mask = ctx_mask.to(self._device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(self._device)

        # truncate to keep <= train_max_tokens
        ctx_len = ctx_ids.shape[1]
        allow_tgt = max(1, self.train_max_tokens - ctx_len)
        if tgt_ids.shape[1] > allow_tgt:
            tgt_ids = tgt_ids[:, :allow_tgt]
            if tgt_mask is not None:
                tgt_mask = tgt_mask[:, :allow_tgt]

        input_ids = torch.cat([ctx_ids, tgt_ids], dim=1)
        if ctx_mask is not None and tgt_mask is not None:
            attention_mask = torch.cat([ctx_mask, tgt_mask], dim=1)
        else:
            attention_mask = torch.ones_like(input_ids, device=self._device)

        labels = input_ids.clone()
        labels[:, :ctx_len] = -100  # ignore context

        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        adjusted_loss = loss * (1.0 - reward)
        adjusted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

    # ----------------------------- Logging / I/O --------------------------------

    def _log_iteration_result(self, result: PredictionResult, iteration: int):
        self.logger.info(
            f"Iter {iteration}: Acc={result.accuracy:.2%} | "
            f"BERTF1={result.bit_accuracy:.2%} | WordAcc={result.word_accuracy:.2%} | "
            f"PosAcc={result.position_accuracy:.2%} | Reward={result.reward:.4f}"
        )
        self.training_history["iterations"].append(iteration)
        self.training_history["accuracies"].append(result.accuracy)
        self.training_history["bit_accuracies"].append(result.bit_accuracy)
        self.training_history["word_accuracies"].append(result.word_accuracy)
        self.training_history["position_accuracies"].append(result.position_accuracy)
        self.training_history["rewards"].append(result.reward)

    # Re-affirm method definition (ensure available on the instance)
    def run_training(self, packet_data: List[str], num_iterations: int = 200):
        self.logger.info(
            f"Starting training on {len(packet_data)} packets | "
            f"seq_len={self.sequence_length}, pred_len={self.prediction_length}, "
            f"ctx_max={self.ctx_max_len}, tgt_max={self.tgt_max_len}, "
            f"train_max_tokens={self.train_max_tokens}"
        )

        for iteration in tqdm(range(num_iterations), desc="Training Progress"):
            start = iteration * self.prediction_length
            end = start + self.sequence_length
            pstart = end
            pend = pstart + self.prediction_length
            if pend > len(packet_data):
                self.logger.info(f"Reached end of data at iteration {iteration}")
                break

            ctx_packets = packet_data[start:end]
            actual_next = packet_data[pstart:pend]
            result = self.train_iteration(ctx_packets, actual_next, iteration)

            if iteration % 10 == 0:
                print(
                    f"Iter {iteration}: Acc={result.accuracy:.2%} "
                    f"BERTF1={result.bit_accuracy:.2%} Reward={result.reward:.4f}"
                )

        self.logger.info("Training completed!")
        self._plot_training_history()

    def run_training(self, packet_data: List[str], num_iterations: int = 200):
        self.logger.info(
            f"Starting training on {len(packet_data)} packets | "
            f"seq_len={self.sequence_length}, pred_len={self.prediction_length}, "
            f"ctx_max={self.ctx_max_len}, tgt_max={self.tgt_max_len}, "
            f"train_max_tokens={self.train_max_tokens}"
        )

        for iteration in tqdm(range(num_iterations), desc="Training Progress"):
            start = iteration * self.prediction_length
            end = start + self.sequence_length
            pstart = end
            pend = pstart + self.prediction_length
            if pend > len(packet_data):
                self.logger.info(f"Reached end of data at iteration {iteration}")
                break

            ctx_packets = packet_data[start:end]
            actual_next = packet_data[pstart:pend]
            result = self.train_iteration(ctx_packets, actual_next, iteration)

            if iteration % 10 == 0:
                print(f"Iter {iteration}: Acc={result.accuracy:.2%} "
                      f"BERTF1={result.bit_accuracy:.2%} Reward={result.reward:.4f}")

        self.logger.info("Training completed!")
        self._plot_training_history()

    def _plot_training_history(self):
        it = self.training_history["iterations"]
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].plot(it, self.training_history["accuracies"]); axs[0, 0].set_title("Accuracy"); axs[0, 0].grid(True)
        axs[0, 1].plot(it, self.training_history["bit_accuracies"]); axs[0, 1].set_title("BERTScore F1"); axs[0, 1].grid(True)
        axs[1, 0].plot(it, self.training_history["word_accuracies"]); axs[1, 0].set_title("Word Accuracy"); axs[1, 0].grid(True)
        axs[1, 1].plot(it, self.training_history["position_accuracies"]); axs[1, 1].set_title("Position Accuracy"); axs[1, 1].grid(True)
        plt.tight_layout()
        plt.savefig("training_history.png")
        # plt.show()

    def save_csv_results(self, filename: str = "training_results.csv"):
        if not self.csv_results:
            return
        df = pd.DataFrame(self.csv_results)
        df.to_csv(filename, index=False)
        self.logger.info(f"Saved results to {filename}")
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total iterations: {len(self.csv_results)}")
        print(f"Final Accuracy: {df['accuracy'].iloc[-1]:.2%}")
        print(f"Final Bit Accuracy: {df['bit_accuracy'].iloc[-1]:.2%}")
        print(f"Final Word Accuracy: {df['word_accuracy'].iloc[-1]:.2%}")
        print(f"Final Position Accuracy: {df['position_accuracy'].iloc[-1]:.2%}")
        print(f"Best Accuracy: {df['accuracy'].max():.2%}")
        print(f"Best Bit Accuracy: {df['bit_accuracy'].max():.2%}")
        print(f"Best Word Accuracy: {df['word_accuracy'].max():.2%}")
        print(f"Best Position Accuracy: {df['position_accuracy'].max():.2%}")
        print("=" * 60)

    def save_model(self, path: str = "packet_predictor_model"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(f"{path}/training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str = "packet_predictor_model"):
        # Note: loading a saved LoRA model in 4-bit should use the same quantization logic
        self.model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if hasattr(self.model, "config"):
            try:
                self.model.config.use_cache = False
            except Exception:
                pass
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        try:
            with open(f"{path}/training_history.json", "r") as f:
                self.training_history = json.load(f)
        except FileNotFoundError:
            self.logger.warning("No training history found")
        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Model loaded from {path}")


# ----------------------------- Data Loader -------------------------------------

def load_pcap_data(file_path: str) -> List[str]:
    print(f"Loading PCAP data from {file_path}...")
    df = pd.read_csv(file_path)
    out = []
    for info in df["Info"]:
        out.append("No info available" if pd.isna(info) else str(info))
    print(f"Loaded {len(out)} packets")
    return out


# ----------------------------- Main -------------------------------------------

if __name__ == "__main__":
    print("Packet Info Predictor with QLoRA + LoRA (+ optional 8-bit Adam)")
    print("=" * 50)

    # Optional: helps with CUDA fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    packet_data = load_pcap_data("Pcap/pcap.csv")

    predictor = PacketInfoPredictor(
        model_name="distilgpt2",
        sequence_length=1000,
        prediction_length=1,
        learning_rate=1e-4,
        ctx_max_len=512,
        tgt_max_len=512,
        train_max_tokens=512,
        compute_bertscore_every_n=0,  # disable to save compute
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        use_8bit_adam=False,
        use_qlora=False,
    )

    predictor.run_training(packet_data, num_iterations=250000)
    predictor.save_model()
    predictor.save_csv_results()

    print("Done. See logs/, training_history.png, training_results.csv.")
