from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# --------------------------------------------------------------
# 1) 서버 실행 시: 기존 학습 데이터 불러와서 TF-IDF 학습
# --------------------------------------------------------------
corpus_df = pd.read_csv(
    "acdttest.csv",
    header=None,
    names=["group", "name", "label", "content"]
)

corpus_df = corpus_df.dropna(subset=["content"])
corpus = corpus_df["content"].astype(str).tolist()

# norm=None : 나중에 우리가 직접 norm(벡터 길이)을 쓰기 위해
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", norm=None)
vectorizer.fit(corpus)


# --------------------------------------------------------------
# 2) 카카오톡 파싱 함수
# --------------------------------------------------------------
def parse_kakao(text):
    lines = text.split("\n")

    names = []
    contents = []

    # 패턴: [이름] [오전 1:23] 내용
    pattern = re.compile(
        r"^\[(.*?)\]\s\[(오전|오후)\s\d{1,2}:\d{2}\]\s?(.*)$"
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if not match:
            continue  # 불필요한 줄은 무시

        name, daytime, message = match.groups()

        # 시스템 메시지/사진/파일 제거
        if message.startswith("사진") or message.startswith("파일") or message.startswith("메시지가 삭제되었습니다"):
            continue

        # 정제된 데이터 저장
        names.append(name.strip())
        contents.append(message.strip())

    df = pd.DataFrame({"name": names, "content": contents})
    df = df[df["content"] != ""]  # 공백 메시지 제거

    return df


# --------------------------------------------------------------
# 3) 문장 중요도 + 토큰 계산 (학습된 vectorizer 사용)
# --------------------------------------------------------------
def compute_importance(df):
    # TF-IDF 벡터
    X = vectorizer.transform(df["content"])
    df["importance"] = np.linalg.norm(X.toarray(), axis=1)

    # TF-IDF에서 사용하는 토크나이저 그대로 사용
    analyzer = vectorizer.build_analyzer()
    df["tokens"] = df["content"].apply(analyzer)

    return df


# --------------------------------------------------------------
# 4) Flask 웹페이지
# --------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        text = uploaded_file.read().decode("utf-8")

        df = parse_kakao(text)
        df = compute_importance(df)

        # --------------------------------------------------
        # 사람별 단어 다양성 계산
        #   - total : 전체 단어 수
        #   - unique : 서로 다른 단어의 set
        # --------------------------------------------------
        token_stats = {}

        for _, row in df.iterrows():
            name = row["name"]
            toks = row["tokens"]

            if name not in token_stats:
                token_stats[name] = {"total": 0, "unique": set()}

            token_stats[name]["total"] += len(toks)
            token_stats[name]["unique"].update(toks)

        # diversity = (서로 다른 단어 수) / (전체 단어 수)
        diversity = {
            name: (len(stat["unique"]) / stat["total"]) if stat["total"] > 0 else 0
            for name, stat in token_stats.items()
        }

        diversity_series = pd.Series(diversity)

        # 원래 TF-IDF 기반 기여도
        raw_contrib = df.groupby("name")["importance"].sum()

        # 단어 다양성 보정 적용
        contrib_adjusted = (raw_contrib * diversity_series).sort_values(ascending=False)

        return render_template(
            "result.html",
            tables=df.to_html(classes="table table-striped"),
            contrib=contrib_adjusted.to_dict()
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
