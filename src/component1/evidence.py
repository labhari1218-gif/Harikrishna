"""
evidence.py - Evidence abstraction and verbalization for Component 1

Defines EvidenceItem and PVResult dataclasses, plus helper functions for:
- Converting KG triples to natural language (evidence_to_text)
- Normalizing relations (URI tail extraction, underscore replacement)
- Stable hashing for cache keys
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple


# Verbalization version for cache key versioning
# v3: Improved triple verbalization (camelCase split + natural templates)
# v2: Fixed PV premise/hypothesis order (was reversed in v1)
VERBALIZER_VERSION = "v3"


@dataclass
class PVResult:
    """Premise validation scoring result for a single evidence item."""
    p_entail: float
    p_contra: float
    p_neutral: float
    rel: float  # p_entail + p_contra (relevance)
    pol: float  # p_entail - p_contra (polarity)


@dataclass
class EvidenceItem:
    """
    Evidence abstraction supporting both KG triples and sentences.
    
    Attributes:
        evidence_id: Unique identifier (e.g., "train_0_triple_5")
        kind: "kg_triple" or "sentence"
        content: Evidence data (structure depends on kind)
            - For kg_triple: {"triple": [subj, rel, obj], "surface_text": str (optional)}
            - For sentence: {"text": str, "doc_id": str (optional), "sent_id": int (optional)}
        retrieval_score: Retrieval score (default 0.0, for future use)
        pv: Premise validation result (populated after scoring)
        meta: Extensible metadata dict
    """
    evidence_id: str
    kind: str  # "kg_triple" or "sentence"
    content: Dict[str, Any]
    retrieval_score: float = 0.0
    pv: Optional[PVResult] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def normalize_relation(rel: str) -> str:
    """
    Normalize relation string for human readability.
    
    - Strips URI prefixes (e.g., http://dbpedia.org/property/successor -> successor)
    - Replaces underscores with spaces
    Normalize relation string for natural language verbalization.
    
    Improvements:
    - Split camelCase (e.g., birthPlace → birth place)
    - Handle underscores and hyphens
    - Add spacing for readability
    
    Args:
        rel: Raw relation string
        
    Returns:
        Normalized relation string suitable for templates
    """
    import re
    
    # Remove common prefixes
    rel = rel.replace('http://', '').replace('https://', '')
    
    # Extract tail if URI-like
    if '/' in rel:
        rel = rel.split('/')[-1]
    if '#' in rel:
        rel = rel.split('#')[-1]
    
    # Split camelCase: insert space before uppercase letters
    # birthPlace → birth Place → birth place
    rel = re.sub(r'([a-z])([A-Z])', r'\1 \2', rel)
    
    # Replace underscores and hyphens with spaces
    rel = rel.replace('_', ' ').replace('-', ' ')
    
    # Lowercase and strip
    rel = rel.lower().strip()
    
    # Collapse multiple spaces
    rel = re.sub(r'\s+', ' ', rel)
    
    return rel.strip()


def extract_entity_name(entity: str) -> str:
    """
    Extract human-readable name from entity URI or string.
    
    - Extracts URI tail (last component after last /)
    - Replaces underscores with spaces
    - Strips quotes from literals
    
    Args:
        entity: Raw entity string (URI or literal)
        
    Returns:
        Human-readable entity name
        
    Example:
        >>> extract_entity_name("http://dbpedia.org/resource/John_E._Beck")
        'John E. Beck'
        >>> extract_entity_name('"2006"')
        '2006'
    """
    # Strip quotes from literals
    entity = entity.strip('"\'')
    
    # Extract URI tail
    if '/' in entity:
        entity = entity.split('/')[-1]
    
    # Replace underscores with spaces
    entity = entity.replace('_', ' ')
    
    return entity.strip()


def stable_hash(text: str) -> str:
    """
    Generate stable SHA256 hash for cache keys.
    
    Args:
        text: Input text
        
    Returns:
        Full 64-character hexadecimal hash (not truncated to avoid collisions)
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def evidence_to_text(item: EvidenceItem, verbalizer_id: str = VERBALIZER_VERSION) -> Tuple[str, str]:
    """
    Convert evidence to natural language text for NLI.
    
    v3: Uses natural language templates instead of raw concatenation
        to reduce neutral dominance in MNLI models.
    
    Args:
        item: EvidenceItem to verbalize
        verbalizer_id: Versioning for cache invalidation
        
    Returns:
        Tuple of (verbalized_text, original_representation)
    """
    if item.kind == "kg_triple":
        triple = item.content["triple"]
        if len(triple) != 3:
            raise ValueError(f"Invalid triple structure: {triple}. Expected [subj, rel, obj]")
        
        subj, rel, obj = triple
        
        # Handle inverse relations (~relation means reverse direction)
        if rel.startswith('~'):
            # Swap subject and object, remove tilde
            rel_clean = rel[1:]
            subj_name = extract_entity_name(obj)  # Swapped
            obj_name = extract_entity_name(subj)  # Swapped
        else:
            rel_clean = rel
            subj_name = extract_entity_name(subj)
            obj_name = extract_entity_name(obj)
        
        # Normalize relation for natural language
        rel_norm = normalize_relation(rel_clean)
        
        # v3: Use natural language template
        # Default: "{subj} has {relation} {obj}."
        # This is more natural than raw concatenation and reduces MNLI neutral rates
        premise_text = f"{subj_name} has {rel_norm} {obj_name}."
        
        # Store raw triple for audit (debugging ~relation handling)
        raw_triple = f"{subj} | {rel} | {obj}"
        
        return premise_text, raw_triple
    
    elif item.kind == "sentence":
        # Sentence evidence: use text directly
        text = item.content.get('text', '')
        doc_id = item.content.get('doc_id', 'unknown')
        sent_id = item.content.get('sent_id', -1)
        
        raw_repr = f"doc={doc_id}, sent={sent_id}"
        return text, raw_repr
    
    else:
        raise ValueError(f"Unknown evidence kind: {item.kind}")
