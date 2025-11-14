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
# 3) 문장 중요도 계산 (학습된 vectorizer 사용)
# --------------------------------------------------------------
def compute_importance(df):
    X = vectorizer.transform(df["content"])
    df["importance"] = np.linalg.norm(X.toarray(), axis=1)
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

        contrib = df.groupby("name")["importance"].sum().sort_values(ascending=False)

        return render_template(
            "result.html",
            tables=df.to_html(classes="table table-striped"),
            contrib=contrib.to_dict()
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)