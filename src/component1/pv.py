"""
pv.py - Premise Validation Scorer using NLI models

Implements PVScorer class for scoring evidence items using transformer-based
NLI models (e.g., DeBERTa-v3-base-mnli). Supports batching, fp16, and robust
label mapping.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .evidence import PVResult

logger = logging.getLogger(__name__)


@dataclass
class PVConfig:
    """Configuration for PV Scorer."""
    model_name: str = "microsoft/deberta-v3-base-mnli"
    device: str = "auto"  # "auto", "cuda", "cpu"
    batch_size: int = 8  # Conservative for 8GB GPU with concurrent processes
    max_length: int = 256
    use_fp16: bool = True  # Auto-enabled if CUDA available


class PVScorer:
    """
    Premise Validation Scorer using NLI models.
    
    Scores evidence items by treating evidence text as premise and claim text
    as hypothesis in an NLI setup.
    
    Attributes:
        config: PVConfig with model settings
        model: Loaded HuggingFace NLI model
        tokenizer: Corresponding tokenizer
        device: torch device (cuda/cpu)
        label_map: Mapping from model output indices to entail/neutral/contra
    """
    
    def __init__(self, config: PVConfig):
        """
        Initialize PV scorer with model and configuration.
        
        Args:
            config: PVConfig instance
            
        Raises:
            ValueError: If model fails to load or label mapping cannot be determined
        """
        self.config = config
        
        # Determine device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        logger.info(f"Loading PV model: {config.model_name}")
        logger.info(f"Device: {self.device}")
        
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Determine entail/neutral/contra label indices
            self.label_map = self._determine_label_mapping()
            
            logger.info(f"Label mapping: {self.label_map}")
            logger.info(f"PV scorer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PV scorer: {e}")
            raise ValueError(f"Could not load model {config.model_name}: {e}")
    
    def _determine_label_mapping(self) -> dict:
        """
        Robustly determine entail/neutral/contra label indices from model config.
        
        Strategy:
        1. Try id2label from model config
        2. Fall back to MNLI standard order: [contradiction, neutral, entailment]
        
        Returns:
            Dict with keys "entail", "neutral", "contra" mapping to int indices
            
        Raises:
            ValueError: If label mapping cannot be determined
        """
        # Try to get id2label from model config
        if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
            id2label = self.model.config.id2label
            logger.info(f"Found id2label in model config: {id2label}")
            
            # Search for entail/neutral/contra labels (case-insensitive)
            label_map = {}
            for idx, label in id2label.items():
                label_lower = label.lower()
                if 'entail' in label_lower and 'non' not in label_lower:
                    label_map['entail'] = int(idx)
                elif 'neutral' in label_lower:
                    label_map['neutral'] = int(idx)
                elif 'contra' in label_lower:
                    label_map['contra'] = int(idx)
            
            # Verify all labels found
            if len(label_map) == 3:
                return label_map
            else:
                logger.warning(f"Incomplete label mapping from id2label: {label_map}")
        
        # Fallback: MNLI standard order [contradiction, neutral, entailment]
        logger.warning("Using MNLI standard label order fallback: [contradiction, neutral, entailment]")
        return {
            'entail': 2,
            'neutral': 1,
            'contra': 0
        }
    
    def pv_score_many(
        self,
        claim_text: str,
        evidence_texts: List[str]
    ) -> List[PVResult]:
        """
        Score multiple evidence texts against a single claim using batched inference.
        
        Args:
            claim_text: The claim (hypothesis in NLI)
            evidence_texts: List of evidence texts (premises in NLI)
            
        Returns:
            List of PVResult objects, one per evidence text
            
        Example:
            >>> scorer = PVScorer(PVConfig())
            >>> claim = "The sky is blue"
            >>> evidence = ["The sky has a blue color", "Grass is green"]
            >>> results = scorer.pv_score_many(claim, evidence)
            >>> len(results)
            2
        """
        if not evidence_texts:
            return []
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(evidence_texts), self.config.batch_size):
            batch_evidence = evidence_texts[i:i + self.config.batch_size]
            batch_results = self._score_batch(claim_text, batch_evidence)
            all_results.extend(batch_results)
        
        return all_results
    
    def _score_batch(
        self,
        claim_text: str,
        evidence_texts: List[str]
    ) -> List[PVResult]:
        """
        Score a single batch of evidence texts.
        
        CRITICAL: NLI models expect (premise, hypothesis) order.
        For PV scoring: premise=evidence, hypothesis=claim
        
        Args:
            claim_text: The claim (hypothesis in NLI)
            evidence_texts: Batch of evidence texts (premises in NLI)
            
        Returns:
            List of PVResult objects
        """
        # CORRECT ORDER: premise (evidence) first, hypothesis (claim) second
        inputs = self.tokenizer(
            evidence_texts,  # premise = evidence
            [claim_text] * len(evidence_texts),  # hypothesis = claim
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference with fp16 if CUDA and enabled
        with torch.no_grad():
            if self.config.use_fp16 and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        
        # Get probabilities
        logits = outputs.logits.cpu()
        probs = torch.softmax(logits, dim=-1).numpy()
        
        # Extract entail/neutral/contra probabilities and compute rel/pol
        results = []
        for prob_vec in probs:
            p_entail = float(prob_vec[self.label_map['entail']])
            p_neutral = float(prob_vec[self.label_map['neutral']])
            p_contra = float(prob_vec[self.label_map['contra']])
            
            rel = p_entail + p_contra  # Relevance
            pol = p_entail - p_contra  # Polarity
            
            results.append(PVResult(
                p_entail=p_entail,
                p_contra=p_contra,
                p_neutral=p_neutral,
                rel=rel,
                pol=pol
            ))
        
        return results
