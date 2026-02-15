import torch
from sklearn.metrics import precision_recall_fscore_support

from datasets import get_df

FACTKG_CLAIM_TYPES = ["existence", "substitution", "multi hop", "multi claim", "negation"]


def _decode_predictions(outputs, labels):
    if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.size(-1) == 1):
        logits = outputs.view(-1)
        flat_labels = labels.view(-1).long()
        if flat_labels.numel() > 0:
            min_label = int(flat_labels.min().item())
            max_label = int(flat_labels.max().item())
            if min_label < 0 or max_label > 1:
                raise ValueError(
                    "Binary-logit model received non-binary labels during evaluation. "
                    f"Observed label range: [{min_label}, {max_label}]."
                )
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        return preds, flat_labels

    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)
    return preds.long(), labels.view(-1).long()


def _safe_prf(y_true, y_pred, average):
    return precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )


def _flatten_factkg_label(label_value):
    if isinstance(label_value, (list, tuple)):
        if len(label_value) == 0:
            return 0
        return int(label_value[0])
    return int(label_value)


def _extract_factkg_types_from_dataset(dataset):
    if dataset is None:
        return None

    claim_types = getattr(dataset, "claim_types", None)
    if claim_types is not None:
        normalized = []
        for meta in claim_types:
            if isinstance(meta, (list, tuple, set)):
                normalized.append([str(v) for v in meta])
            elif meta is None:
                normalized.append([])
            else:
                normalized.append([str(meta)])
        return normalized

    base_dataset = getattr(dataset, "dataset", None)
    indices = getattr(dataset, "indices", None)
    if base_dataset is not None and indices is not None:
        base_types = _extract_factkg_types_from_dataset(base_dataset)
        if base_types is None:
            return None
        return [base_types[int(i)] for i in indices]

    return None


def _evaluate_factkg_per_type(test_types, all_preds, all_labels):
    metrics_dict = {
        ct: {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        for ct in FACTKG_CLAIM_TYPES + ["single hop"]
    }
    count_dict = {ct: 0 for ct in FACTKG_CLAIM_TYPES + ["single hop"]}

    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
        metadata = test_types[i]
        if not isinstance(metadata, (list, tuple, set)):
            metadata = []
        correct = int(pred == label)
        for claim_type in FACTKG_CLAIM_TYPES:
            if claim_type in metadata:
                metrics_dict[claim_type]["accuracy"] += correct
                count_dict[claim_type] += 1
        if "multi hop" not in metadata:
            metrics_dict["single hop"]["accuracy"] += correct
            count_dict["single hop"] += 1

    for claim_type in metrics_dict:
        if count_dict[claim_type] == 0:
            continue
        metrics_dict[claim_type]["accuracy"] /= count_dict[claim_type]

        if claim_type == "single hop":
            type_pairs = [
                (pred, label)
                for pred, label, meta in zip(all_preds, all_labels, test_types)
                if "multi hop" not in meta
            ]
        else:
            type_pairs = [
                (pred, label)
                for pred, label, meta in zip(all_preds, all_labels, test_types)
                if claim_type in meta
            ]

        if type_pairs:
            type_preds = [p for p, _ in type_pairs]
            type_labels = [l for _, l in type_pairs]
            p, r, f, _ = _safe_prf(type_labels, type_preds, average="binary")
            metrics_dict[claim_type]["precision"] = p
            metrics_dict[claim_type]["recall"] = r
            metrics_dict[claim_type]["f1"] = f

    return metrics_dict


def _evaluate_fever_breakdown(all_preds, all_labels):
    labels_sorted = sorted(set(int(v) for v in all_labels))
    p_macro, r_macro, f_macro, _ = _safe_prf(all_labels, all_preds, average="macro")
    p_micro, r_micro, f_micro, _ = _safe_prf(all_labels, all_preds, average="micro")

    p_cls, r_cls, f_cls, support_cls = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=labels_sorted,
        average=None,
        zero_division=0,
    )

    per_class = {}
    for idx, label_id in enumerate(labels_sorted):
        per_class[str(label_id)] = {
            "precision": float(p_cls[idx]),
            "recall": float(r_cls[idx]),
            "f1": float(f_cls[idx]),
            "support": int(support_cls[idx]),
        }

    return {
        "macro": {
            "precision": float(p_macro),
            "recall": float(r_macro),
            "f1": float(f_macro),
        },
        "micro": {
            "precision": float(p_micro),
            "recall": float(r_micro),
            "f1": float(f_micro),
        },
        "per_class": per_class,
    }


def evaluate_on_test_set(
    qa_gnn,
    model,
    test_loader,
    criterion=None,
    device=None,
    dataset_name="factkg",
):
    """
    Evaluate a model on a test loader.

    Args:
        qa_gnn (bool): True if model is a QA-GNN style model.
        model: PyTorch model.
        test_loader: DataLoader for evaluation.
        criterion: Optional loss function. Required for QA-GNN loss reporting.
        device: Optional torch device.
        dataset_name (str): Dataset key for dataset-specific breakdowns.

    Returns:
        dict: Metrics dictionary.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_name = str(dataset_name).strip().lower()

    factkg_test_types = None
    if dataset_name == "factkg":
        factkg_test_types = _extract_factkg_types_from_dataset(getattr(test_loader, "dataset", None))
        if factkg_test_types is None:
            test_df = get_df("test")
            canonical_labels = [_flatten_factkg_label(v) for v in test_df["Label"]]
            if len(canonical_labels) != len(test_loader.dataset):
                raise ValueError(
                    "Cannot compute FactKG per-type metrics: test loader size differs from canonical FactKG test split."
                )
            factkg_test_types = []
            for meta in test_df["types"]:
                if isinstance(meta, (list, tuple, set)):
                    factkg_test_types.append([str(v) for v in meta])
                elif meta is None:
                    factkg_test_types.append([])
                else:
                    factkg_test_types.append([str(meta)])
        else:
            canonical_labels = None

    total_loss = 0.0
    total_correct = 0

    all_preds = []
    all_labels = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for inputs in test_loader:
            if qa_gnn:
                inputs, data_graph, labels = inputs
                batch = inputs.to(device)
                data_graph = data_graph.to(device)
                labels = labels.to(device)

                outputs = model(batch, data_graph)
                if criterion is None:
                    loss = torch.zeros((), device=device)
                else:
                    loss = criterion(outputs, labels)

                preds, eval_labels = _decode_predictions(outputs, labels)
            else:
                batch = inputs.to(device)

                outputs = model(**batch)
                loss = outputs.loss

                preds = torch.argmax(torch.softmax(outputs.logits, dim=1), dim=1)
                eval_labels = batch["labels"].view(-1).long()

            correct_list = preds.view(-1) == eval_labels.view(-1)
            total_loss += float(loss.item()) * batch["input_ids"].size(0)
            total_correct += int(correct_list.sum().item())

            all_preds.extend(preds.view(-1).cpu().tolist())
            all_labels.extend(eval_labels.view(-1).cpu().tolist())

    if not all_labels:
        raise ValueError("Evaluation received an empty test loader.")

    n_samples = len(test_loader.dataset)
    overall_accuracy = total_correct / max(n_samples, 1)
    overall_loss = total_loss / max(n_samples, 1)

    if dataset_name == "factkg":
        if factkg_test_types is None:
            raise ValueError("FactKG evaluation requires claim-type metadata.")
        if len(factkg_test_types) != len(all_labels):
            raise ValueError(
                "Cannot compute FactKG per-type metrics: metadata length does not match evaluated sample count."
            )
        if canonical_labels is not None:
            observed_labels = [int(v) for v in all_labels]
            if observed_labels != canonical_labels:
                raise ValueError(
                    "Cannot compute FactKG per-type metrics: dataloader order/content does not match canonical "
                    "FactKG test split."
                )

        precision, recall, f1, _ = _safe_prf(all_labels, all_preds, average="binary")
        metrics_dict = _evaluate_factkg_per_type(
            test_types=factkg_test_types,
            all_preds=all_preds,
            all_labels=all_labels,
        )
        metrics_dict["overall"] = {
            "accuracy": overall_accuracy,
            "loss": overall_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        return metrics_dict

    fever_breakdown = _evaluate_fever_breakdown(all_preds=all_preds, all_labels=all_labels)
    metrics = {
        "overall": {
            "accuracy": overall_accuracy,
            "loss": overall_loss,
            "precision": fever_breakdown["macro"]["precision"],
            "recall": fever_breakdown["macro"]["recall"],
            "f1": fever_breakdown["macro"]["f1"],
        },
        "macro": fever_breakdown["macro"],
        "micro": fever_breakdown["micro"],
        "per_class": fever_breakdown["per_class"],
    }
    return metrics
