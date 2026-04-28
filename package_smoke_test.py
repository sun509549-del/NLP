from pathlib import Path
import sys
from importlib.metadata import version

try:
    import datasets
    import gensim
    import jieba
    import jupyter
    import jupyterlab
    import tensorboard
    import torch
    import transformers
    import tqdm
    from datasets import Dataset
    from gensim.models import Word2Vec
    from torch.utils.tensorboard import SummaryWriter
    from transformers import BertConfig, BertModel
except ModuleNotFoundError as exc:
    print(f"Missing package: {exc.name}")
    print(f"Current Python: {sys.executable}")
    print()
    print("Please run this script with the nlp conda environment:")
    print(r"  conda activate nlp")
    print(r"  python package_smoke_test.py")
    print()
    print("Or run it without activating:")
    print(r"  conda run -n nlp python package_smoke_test.py")
    raise SystemExit(1) from exc


def ok(name: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[OK] {name}{suffix}")


def test_jieba() -> None:
    words = list(jieba.cut("我正在测试自然语言处理环境"))
    assert "自然语言" in words or "自然语言处理" in words
    ok("jieba", "/".join(words))


def test_gensim() -> None:
    sentences = [
        ["nlp", "is", "fun"],
        ["torch", "supports", "gpu"],
        ["gensim", "trains", "word2vec"],
    ]
    model = Word2Vec(sentences=sentences, vector_size=8, min_count=1, epochs=5)
    vector = model.wv["nlp"]
    assert vector.shape == (8,)
    ok("gensim", f"Word2Vec vector shape={vector.shape}")


def test_transformers() -> None:
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = BertModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    output = model(input_ids=input_ids)
    assert output.last_hidden_state.shape == (2, 6, 16)
    ok("transformers", f"BertModel output shape={tuple(output.last_hidden_state.shape)}")


def test_datasets() -> None:
    dataset = Dataset.from_dict({"text": ["hello", "world"], "label": [0, 1]})
    mapped = dataset.map(lambda row: {"length": len(row["text"])})
    assert mapped[0]["length"] == 5
    ok("datasets", f"rows={len(mapped)}, columns={mapped.column_names}")


def test_tensorboard() -> None:
    log_dir = Path("runs") / "package_smoke_test"
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_scalar("loss", 0.123, 1)
    writer.close()
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert event_files
    ok("tensorboard", f"event file={event_files[-1].name}")


def test_tqdm() -> None:
    total = 0
    for value in tqdm.tqdm(range(5), desc="tqdm smoke test"):
        total += value
    assert total == 10
    ok("tqdm", f"sum={total}")


def test_jupyter() -> None:
    assert version("jupyter")
    assert jupyter is not None
    assert jupyterlab is not None
    ok("jupyter", f"jupyter={version('jupyter')}, jupyterlab={version('jupyterlab')}")


def test_torch_cuda() -> None:
    assert torch.cuda.is_available(), "CUDA is not available in this environment"
    x = torch.randn(128, 128, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    ok("torch cuda", f"{torch.__version__}, device={torch.cuda.get_device_name(0)}, result={tuple(y.shape)}")


def main() -> None:
    print("NLP package smoke test")
    print(f"torch={torch.__version__}, transformers={transformers.__version__}")
    print(f"datasets={datasets.__version__}, gensim={gensim.__version__}")
    print(f"tensorboard={tensorboard.__version__}, tqdm={tqdm.__version__}")
    print()

    test_jieba()
    test_gensim()
    test_transformers()
    test_datasets()
    test_tensorboard()
    test_tqdm()
    test_jupyter()
    test_torch_cuda()

    print()
    print("All package checks passed.")


if __name__ == "__main__":
    main()
