from mini_rag.chunking import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("", doc_id="d") == []
    assert chunk_text("   \n ", doc_id="d") == []


def test_short_text_is_single_chunk():
    chunks = chunk_text("hello world", doc_id="d", chunk_size=200)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].doc_id == "d"
    assert chunks[0].chunk_id == 0


def test_overlap_shares_words():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = chunk_text(text, doc_id="d", chunk_size=4, overlap=2)
    # step = 2 -> chunks: [0:4], [2:6], [4:8], [6:10]
    assert len(chunks) == 4
    assert chunks[0].text == "w0 w1 w2 w3"
    assert chunks[1].text == "w2 w3 w4 w5"
    assert chunks[2].text == "w4 w5 w6 w7"
    assert chunks[3].text == "w6 w7 w8 w9"


def test_no_words_lost():
    text = " ".join(f"w{i}" for i in range(37))
    chunks = chunk_text(text, doc_id="d", chunk_size=10, overlap=3)
    seen = set()
    for c in chunks:
        seen.update(c.text.split())
    assert seen == {f"w{i}" for i in range(37)}


def test_chunk_ids_sequential():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_text(text, doc_id="d", chunk_size=10, overlap=2)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
