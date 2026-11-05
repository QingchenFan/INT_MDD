from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calc_doc_cosine_tfidf(doc1: str, doc2: str) -> float:
    """
    计算两个文档TF‑IDF余弦相似度（相关性，0~1）
    """
    corpus = [doc1, doc2]
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(corpus)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return float(sim[0][0])


if __name__ == "__main__":
    codex = "/Users/qingchen/Desktop/1000003_codex.txt"
    res = "/Volumes/T7 Shield/test22/sub-01000003V01_22.txt"
    #res = "/Volumes/T7 Shield/T7 Shield/T7 Shield.txt"
    #res = "/Users/qingchen/Desktop/1000003_codex.txt"
    # 读取两个文档
    with open(codex, "r", encoding="utf-8") as f:
        text1 = f.read()
    with open(res, "r", encoding="utf-8") as f:
        text2 = f.read()

    score = calc_doc_cosine_tfidf(text1, text2)
    print(f"TF‑IDF余弦相似度（相关性）: {score:.4f}")
