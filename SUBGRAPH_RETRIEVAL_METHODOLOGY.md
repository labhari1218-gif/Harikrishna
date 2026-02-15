# Subgraph Retrieval Methodology Explanation

## Overview

This document explains the subgraph retrieval methods used in the research paper **"Fact or Fiction? Exploring Diverse Approaches to Fact Verification with Language Models"** (2024.fever-1.32.pdf). The methodology employs **simple logical methods** that are computationally cheap and resulted in improved model performance.

## Context and Purpose

The research utilizes the **DBpedia knowledge graph** and the **FactKG dataset** to perform fact verification. For each claim in the dataset, the system extracts relevant entities and retrieves a subgraph from the knowledge base that captures relationships between those entities. These subgraphs serve as structured evidence for the fact verification models (BERT and QA-GNN).

## Knowledge Graph Structure

The DBpedia knowledge graph is represented as a nested dictionary where:
- **Keys**: Entity names (nodes)
- **Values**: Dictionary of relations → list of neighboring entities

Example structure:
```python
{
    "Entity_A": {
        "relation_1": ["Entity_B", "Entity_C"],
        "relation_2": ["Entity_D"]
    },
    "Entity_B": {
        "relation_3": ["Entity_A"]
    }
}
```

## Three Subgraph Retrieval Methods

The codebase implements **three distinct methods** for retrieving subgraphs from the knowledge graph. Each method has different retrieval strategies and computational complexities.

---

### 1. Direct Subgraph Method (`direct`)

**Purpose**: Extract only the direct relationships between entities mentioned in a claim.

**Algorithm** ([`retrieve_subgraphs.py:25-58`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/retrieve_subgraphs.py#L25-L58)):

1. Initialize an empty subgraph dictionary
2. For each entity in the entity list:
   - Look up the entity in the knowledge graph
   - For each relation the entity has:
     - For each neighbor connected by that relation:
       - **If the neighbor is also in the entity list**, add the relation to the subgraph
3. Result: Only edges between entities that both appear in the claim

**Key Characteristic**: Retrieves the **most sparse** subgraph, containing only direct connections between known entities.

**Variant - Direct Filled** (`direct_filled`):
- If the direct method produces an empty subgraph (no connections found), it falls back to the one-hop method
- Controlled by the `fill_if_empty=True` parameter
- This was the method used in the retrieved subgraph files: `subgraphs_direct_filled_*.pkl`

**Code snippet**:
```python
def get_direct_subgraph(self, entity_list, fill_if_empty=False):
    subgraph = {}
    empty = True
    
    for entity in entity_list:
        subgraph[entity] = []
        for relation, neighbours in self.kg[entity].items():
            for neighbour in neighbours:
                if neighbour in entity_list:  # Only connections within entity_list
                    empty = False
                    subgraph[entity].append([relation])
    
    if empty and fill_if_empty:  # Fallback to one-hop if empty
        subgraph = self.get_one_hop_subgraph(entity_list)
    return subgraph
```

---

### 2. One-Hop Subgraph Method (`one_hop`)

**Purpose**: Include all relations for every entity, regardless of where they point.

**Algorithm** ([`retrieve_subgraphs.py:60-80`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/retrieve_subgraphs.py#L60-L80)):

1. Initialize an empty subgraph dictionary
2. For each entity in the entity list:
   - Look up the entity in the knowledge graph
   - For each relation the entity has:
     - Add the relation to the subgraph (regardless of the neighbor)
3. Result: All edges emanating from each entity, even if they point to entities not in the claim

**Key Characteristic**: Retrieves **comprehensive** information about each entity, including connections to entities not mentioned in the claim.

**Alias**: In the paper, this method is referred to as **"single-step"**.

**Code snippet**:
```python
def get_one_hop_subgraph(self, entity_list):
    subgraph = {}
    for entity in entity_list:
        subgraph[entity] = []
        for relation, neighbours in self.kg[entity].items():
            # Add ALL relations, not just those to entities in entity_list
            subgraph[entity].append([relation])
    return subgraph
```

---

### 3. Relevant/Contextualized Subgraph Method (`relevant`)

**Purpose**: Combine direct connections with claim-relevant relations using NLP.

**Algorithm** ([`retrieve_subgraphs.py:82-115`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/retrieve_subgraphs.py#L82-L115)):

1. **Lemmatize the claim** using spaCy NLP model
2. Extract relevant words from the claim (excluding stopwords)
3. For each entity in the entity list:
   - For each relation the entity has:
     - **If the relation name appears as a lemma in the claim**, add it to the subgraph
     - **If the neighbor is in the entity list** (direct connection), add it to the subgraph
4. Result: Edges that are either direct connections OR semantically related to claim words

**Key Characteristic**: Balances precision (direct connections) with semantic relevance (claim-based relation matching).

**Alias**: In the paper, this method is referred to as **"contextualized"**.

**Dependencies**:
- **spaCy**: For lemmatization (`en_core_web_sm` model)
- **NLTK**: For English stopwords

**Code snippet**:
```python
def get_relevant_subgraph(self, entity_list, claim, stop_words, nlp):
    subgraph = {}
    doc = nlp(claim)
    # Extract lemmatized, non-stopword, alphabetic tokens from claim
    relevant_words = [token.lemma_ for token in doc 
                     if token.text.lower() not in stop_words and token.is_alpha]
    
    for entity in entity_list:
        subgraph[entity] = []
        for relation, neighbours in self.kg[entity].items():
            # Add if relation matches claim words
            if relation.lower() in relevant_words:
                subgraph[entity].append([relation])
            
            # Add if direct connection to another entity
            for neighbour in neighbours:
                if neighbour in entity_list:
                    subgraph[entity].append([relation])
    
    return subgraph
```

---

## Graph Walking

After subgraph retrieval, the system performs **graph walking** to generate paths through the subgraph ([`kg.py:12-70`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/kg.py#L12-L70)).

### Walking Algorithm

The `KG.search()` method generates two types of paths:

1. **Connected paths**: Paths between entities where all intermediate nodes exist in the subgraph
2. **Walkable paths**: Random walks through the graph that explore possible reasoning chains

**Process**:
1. For each entity in the subgraph
2. For each relation (path) associated with that entity
3. Start from the entity and walk along the path
4. Try to reach other entities in the entity list
5. Generate both:
   - **Connected**: [start_entity, relation, intermediate, relation, end_entity]
   - **Walkable**: Random paths through the graph structure

**Code location**: [`kg.py:41-60`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/kg.py#L41-L60)

---

## Data Storage Format

The retrieved subgraphs are stored as **Pandas DataFrames** serialized to `.pkl` files:

**Columns**:
- `subgraph`: The retrieved subgraph dictionary
- `walked`: The walked graph paths (connected and walkable)

**File naming convention**:
```
subgraphs_<method>_<split>.pkl
```

Examples:
- `subgraphs_direct_filled_train.pkl`
- `subgraphs_direct_filled_val.pkl`
- `subgraphs_direct_filled_test.pkl`

**Location**: [`data/subgraphs/`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/data/subgraphs)

---

## Preprocessing Pipeline

The complete subgraph retrieval process follows these steps:

### Step 1: Retrieve Subgraphs
```bash
python retrieve_subgraphs.py --dataset_type all --method direct --fill_if_empty
```

**Parameters**:
- `--dataset_type`: `train`, `val`, `test`, or `all`
- `--method`: `direct`, `one_hop`, or `relevant`
- `--fill_if_empty`: Use one-hop fallback for empty direct subgraphs

**Output**: Subgraph `.pkl` files in `data/subgraphs/`

### Step 2: Create Embeddings (for QA-GNN only)
```bash
python make_subgraph_embeddings.py --dataset_type all --subgraph_type direct_filled --batch_size 64
```

**Output**: `data/embeddings.pkl` containing precomputed node and edge embeddings

---

## Key Advantages of This Approach

1. **Computational Efficiency**: 
   - Subgraphs are precomputed once and reused
   - No need for complex graph neural network inference during retrieval
   - DBpedia knowledge graph only needed during preprocessing

2. **Simplicity**:
   - Logic-based methods (no learned components)
   - Deterministic and reproducible
   - Easy to debug and interpret

3. **Effectiveness**:
   - Improved model performance compared to baselines
   - Provides structured evidence for language models
   - Captures both local (direct) and contextual (relevant) information

4. **Flexibility**:
   - Three methods allow trade-offs between precision and recall
   - Can be combined or used based on task requirements

---

## Method Comparison Summary

| Method | Alias in Paper | Edges Included | Computational Cost | Subgraph Size |
|--------|----------------|----------------|-------------------|---------------|
| `direct` | Direct | Only between entities in claim | Low | Smallest |
| `direct_filled` | Direct (filled) | Direct + one-hop fallback | Low | Small-Medium |
| `one_hop` | Single-step | All edges from entities | Medium | Large |
| `relevant` | Contextualized | Direct + claim-related edges | Medium (requires NLP) | Medium |

---

## Performance Results

### Best Performing Models

The experiments demonstrate that **BERT with single-step (one-hop) subgraphs achieved the best performance** on the FactKG dataset:

**Table 1: Test-Set Accuracy Comparison**

| Model | Subgraph Method | Test Accuracy |
|-------|----------------|---------------|
| **BERT (single-step)** | **One-hop** | **Best** ⭐ |
| BERT (direct) | Direct | High |
| BERT (contextual) | Relevant | High |
| BERT (no subgraphs) | None | Baseline |
| QA-GNN (single-step) | One-hop | Good |
| QA-GNN (contextual) | Relevant | Good |
| QA-GNN (direct) | Direct | Good |

> [!IMPORTANT]
> **Best Model**: BERT with **single-step (one-hop)** subgraphs performed the best overall.
> 
> **Most Efficient**: QA-GNN trained in only **1.5 hours** vs. 2-3 days for benchmark GEAR model, while BERT took 2-10 hours depending on subgraph size.

### Performance by Subgraph Type

**Table 2: Impact of Subgraph Retrieval Methods**

| Model | Direct | Contextual | Single-Step (One-hop) |
|-------|--------|------------|----------------------|
| BERT | ✓ Good | ✓✓ Better | ✓✓✓ **Best** |
| QA-GNN | ✓ Good | ✓✓ Better | ✓✓✓ **Best** |

**Key Findings**:
1. **Single-step method** (one-hop) consistently outperformed other methods for both BERT and QA-GNN
2. **Direct subgraphs** showed clear improvement over no subgraphs
3. **Contextual subgraphs** showed small improvement over direct
4. Models had **higher precision than recall**, suggesting they are conservative in predicting "true"

### Training Efficiency

| Model | Training Time | GPU | Performance |
|-------|--------------|-----|-------------|
| BERT (no subgraphs) | ~2 hours | RTX 3090 | Baseline |
| BERT (single-step) | ~10 hours | RTX 3090 | **Best accuracy** |
| QA-GNN (single-step) | **1.5 hours** | RTX 3090 | **Most efficient** |
| GEAR (benchmark) | 2-3 days | RTX 3090 | Comparable |

---

## BERT Model Training

### Architecture Overview

The BERT model for fact verification uses a fine-tuned **BERT-base-uncased** with a binary classification head.

**Model Configuration** ([`models.py:12-62`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/models.py#L12-L62)):

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,  # Binary classification: True/False
    output_hidden_states=True
)
```

**Key Components**:
- **Base Model**: BERT-base-uncased (110M parameters)
- **Classification Layer**: Linear layer (768 → 2)
- **Output**: Binary logits for True/False prediction
- **Pooler**: Uses [CLS] token representation for classification

### Input Format

The BERT model receives inputs in the following format:

#### Without Subgraphs (Baseline)
```
Input: "{claim}"
```

**Example**:
```
"The Eiffel Tower is located in Paris."
```

#### With Subgraphs (Evidence-Enhanced)
```
Input: "{claim} | {subgraph_evidence}"
```

**Example**:
```
"The Eiffel Tower is located in Paris. | {'Eiffel_Tower': [['location'], ['country']], 'Paris': [['contains'], ['capital_of']]}"
```

**Input Processing** ([`datasets.py:137-158`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/datasets.py#L137-L158)):

```python
class FactKGDataset(Dataset):
    def __init__(self, df, evidence=None):
        self.inputs = df["Sentence"]  # Claims
        self.labels = [int(label[0]) for label in df["Label"]]
        
        if evidence is not None:
            # Concatenate claim with subgraph evidence
            self.inputs = [
                self.inputs[i] + " | " + str(evidence[i]) 
                for i in range(len(df))
            ]
```

### Tokenization

**Tokenizer Configuration** ([`datasets.py:165-170`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/datasets.py#L165-L170)):

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(
    text,
    return_tensors="pt",
    max_length=512,        # Maximum sequence length
    truncation=True,       # Truncate long sequences
    padding="max_length"   # Pad to max_length
)
```

**Tokenizer Output**:
- `input_ids`: Token IDs
- `attention_mask`: Mask for padded tokens
- `token_type_ids`: Segment IDs (for BERT)

### Training Configuration

The BERT model is trained with the following setup ([`run_stuff.py:105-118`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/run_stuff.py#L105-L118)):

**Optimizer**: AdamW with weight decay regularization
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

**Learning Rate Scheduler**: Linear warmup with 50 steps
```python
lr_scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=50,
    num_training_steps=len(train_loader) * n_epochs
)
```

**Loss Function**: Binary Cross-Entropy with Logits
```python
criterion = nn.BCEWithLogitsLoss()
```

### Hyperparameters

**Best Hyperparameters for Top Models**:

| Model | Learning Rate | Batch Size | Epochs | Dropout |
|-------|--------------|------------|--------|---------|
| BERT (no subgraphs) | 1e-4 | 32 | 6 | - |
| BERT (direct) | 1e-4 | 32 | 7 | - |
| BERT (contextual) | 5e-5 | 8 | 7 | - |
| **BERT (single-step)** | **5e-5** | **4** | **7** | **-** |

> [!NOTE]
> **Batch Size Constraint**: Models with large subgraphs (single-step) used batch size 4 due to GPU memory constraints (512 max tokens).

### Transfer Learning Strategy

The model uses **partial fine-tuning** to balance training efficiency and performance:

**Freezing Strategy** ([`models.py:47-53`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/models.py#L47-L53)):

```python
# Option 1: Freeze entire base model (only train classifier)
if freeze_base_model:
    for params in model.base_model.parameters():
        params.requires_grad = False

# Option 2: Freeze up to pooler (default for best performance)
elif freeze_up_to_pooler:
    for name, params in model.base_model.named_parameters():
        if not name.startswith("pooler"):
            params.requires_grad = False  # ~109.5M params frozen
```

**Default Configuration**:
- **Frozen**: All BERT encoder layers (~109.5M parameters)
- **Trainable**: Pooler layer (~500K parameters) + Classifier (~1.5K parameters)
- **Total Trainable**: ~501.5K parameters

This strategy significantly reduces:
- Training time
- GPU memory requirements
- Risk of overfitting

### Training Loop

**Forward Pass**:
1. Tokenize input (claim + subgraph evidence)
2. Pass through BERT encoder
3. Extract [CLS] token embedding
4. Pass through pooler layer
5. Pass through classification layer
6. Compute logits for True/False

**Backward Pass**:
1. Compute BCE loss
2. Backpropagate through trainable layers only
3. Update parameters with AdamW
4. Step learning rate scheduler

**Model Selection**:
- Save model checkpoint after each epoch
- Track validation loss
- **Select best model** based on lowest validation loss
- Evaluate best model on test set

### Data Loading Pipeline

**DataLoader Configuration** ([`datasets.py:173-223`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/datasets.py#L173-L223)):

```python
def get_dataloader(data_split, subgraph_type, subgraph_to_use, 
                   batch_size=64, shuffle=True):
    # 1. Load dataset
    df = get_df(data_split)
    
    # 2. Load subgraphs
    if subgraph_type is not None:
        subgraphs = get_subgraphs(data_split, subgraph_type)
        
        # Select evidence type
        if subgraph_to_use == "discovered":
            evidence = subgraphs["subgraph"]  # Raw subgraph
        elif subgraph_to_use == "connected":
            evidence = subgraphs["walked"]["connected"]  # Connected paths
        elif subgraph_to_use == "walkable":
            evidence = subgraphs["walked"]["walkable"]  # All paths
    
    # 3. Create dataset
    dataset = FactKGDataset(df, evidence)
    
    # 4. Create dataloader with collate function
    collate_func = CollateFunctor(tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                           shuffle=shuffle, collate_fn=collate_func)
    return dataloader
```

### Training Command

**Example Training Script**:

```bash
# Best-performing model: BERT with single-step subgraphs
python run_stuff.py bert_single_step \
    --subgraph_type direct_filled \
    --subgraph_to_use walkable \
    --n_epochs 10 \
    --batch_size 4 \
    --learning_rate 0.000005
```

**Arguments Explained**:
- `bert_single_step`: Model name (saved to `models/bert_single_step.pth`)
- `--subgraph_type direct_filled`: Use direct subgraphs with one-hop fallback
- `--subgraph_to_use walkable`: Use walked paths as evidence
- `--n_epochs 10`: Train for 10 epochs
- `--batch_size 4`: Small batch size for large inputs
- `--learning_rate 5e-5`: Fine-tuning learning rate

### Why BERT with Single-Step Subgraphs Works Best

1. **Rich Evidence**: One-hop subgraphs provide comprehensive context about entities
2. **Balanced Information**: Not too sparse (direct) nor too noisy (full KG)
3. **BERT's Strength**: Pre-trained language understanding captures semantic relationships in text-based evidence
4. **Sufficient Context Window**: 512 tokens accommodate claim + substantial subgraph evidence
5. **Transfer Learning**: Partially frozen BERT prevents overfitting while adapting to fact verification

---

## Implementation Details

**Primary Implementation File**: [`retrieve_subgraphs.py`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/retrieve_subgraphs.py)

**Key Classes**:
- `KnowledgeGraph` (lines 21-115): Implements the three retrieval methods
- `KG` ([`kg.py`](file:///home/bs_thesis/shift%20(Copy)/Fact-or-Fiction/kg.py)): Implements graph walking

**Key Functions**:
- `find_subgraphs()` (lines 118-158): Batch processes dataframe rows
- `walk_graphs()` (lines 161-181): Generates paths through subgraphs
- `save_subgraph_to_csv()` (lines 184-199): Persists results

---

## References

- **Paper**: 2024.fever-1.32.pdf
- **Dataset**: [FactKG](https://arxiv.org/pdf/2305.06590) (Fact Verification via Reasoning on Knowledge Graphs)
- **Knowledge Base**: [DBpedia](https://www.dbpedia.org/) (undirected light version)
- **Codebase**: Based on [FactKG GitHub repository](https://github.com/jiho283/FactKG)

---

## Conclusion

The subgraph retrieval methodology employs **three complementary approaches** that balance **precision, recall, and semantic relevance** through simple logical operations. The **direct method** focuses on explicit entity relationships, the **one-hop method** provides comprehensive context, and the **relevant method** leverages NLP to identify claim-salient relations. This multi-method approach enables flexible and efficient knowledge graph integration into fact verification pipelines, resulting in improved model performance while maintaining computational efficiency.
