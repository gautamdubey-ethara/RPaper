from __future__ import annotations

from typing import Optional

from datasets import load_dataset, Dataset

TASK_SEQUENCE = ["sst2", "mrpc", "cola", "mnli"]

TASK_METRICS = {
    "sst2": "accuracy",
    "mrpc": "f1",
    "cola": "matthews_correlation",
    "mnli": "accuracy",
}

LABEL_MAPS = {
    "sst2": {0: "negative", 1: "positive"},
    "mrpc": {0: "not paraphrase", 1: "paraphrase"},
    "cola": {0: "unacceptable", 1: "acceptable"},
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
}

PROMPT_TEMPLATES = {
    "sst2": (
        "Classify the sentiment of the following sentence as positive or negative.\n"
        "Sentence: {sentence}\n"
        "Answer:"
    ),
    "mrpc": (
        "Determine whether the two sentences below are paraphrases of each other.\n"
        "Answer 'paraphrase' or 'not paraphrase'.\n"
        "Sentence 1: {sentence1}\n"
        "Sentence 2: {sentence2}\n"
        "Answer:"
    ),
    "cola": (
        "Determine whether the following English sentence is grammatically acceptable.\n"
        "Answer 'acceptable' or 'unacceptable'.\n"
        "Sentence: {sentence}\n"
        "Answer:"
    ),
    "mnli": (
        "Given the premise and hypothesis, determine their logical relationship.\n"
        "Answer 'entailment', 'neutral', or 'contradiction'.\n"
        "Premise: {premise}\n"
        "Hypothesis: {hypothesis}\n"
        "Answer:"
    ),
}

def _format_example(task_name: str, example: dict) -> dict:
    template = PROMPT_TEMPLATES[task_name]
    label_map = LABEL_MAPS[task_name]

    if task_name == "sst2":
        prompt = template.format(sentence=example["sentence"])
    elif task_name == "mrpc":
        prompt = template.format(
            sentence1=example["sentence1"],
            sentence2=example["sentence2"],
        )
    elif task_name == "cola":
        prompt = template.format(sentence=example["sentence"])
    elif task_name == "mnli":
        prompt = template.format(
            premise=example["premise"],
            hypothesis=example["hypothesis"],
        )
    else:
        raise ValueError(f"Unsupported task: {task_name!r}")

    label_int = int(example["label"])
    return {
        "prompt": prompt,
        "target": label_map[label_int],
        "label": label_int,
    }

def load_task(
    task_name: str,
    split: str,
    n_examples: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    if task_name not in TASK_SEQUENCE:
        raise ValueError(
            f"Unknown task {task_name!r}. Choose from {TASK_SEQUENCE}."
        )

    actual_split = split
    if task_name == "mnli" and split == "validation":
        actual_split = "validation_matched"  # GLUE MNLI has no plain 'validation' split

    raw = load_dataset("glue", task_name, split=actual_split)

    if n_examples is not None and n_examples < len(raw):
        raw = raw.shuffle(seed=seed).select(range(n_examples))

    formatted = raw.map(
        lambda ex: _format_example(task_name, ex),
        remove_columns=raw.column_names,
        desc=f"Formatting {task_name}/{actual_split}",
    )
    return formatted


def build_replay_buffer(
    task_name: str,
    n_per_task: int = 200,
    seed: int = 42,
) -> Dataset:
    return load_task(task_name, split="train", n_examples=n_per_task, seed=seed)
