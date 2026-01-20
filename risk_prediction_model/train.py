# train.py

import pandas as pd
from pathlib import Path

from models import MODEL_FACTORY
from sampling.sampler import apply_sampling, get_class_weight
from utils import load_data
from evaluate import evaluate_binary


def run_experiment(df_param, model_name):
    results = []

    for _, row in df_param.iterrows():

        disease = row["disease"]
        sampling_method = row["sampling_method"]
        sampling_ratio = row["sampling_ratio"]

        print(f"[{model_name}] {disease} | {sampling_method} | {sampling_ratio}")

        # 데이터
        X, y = load_data(disease)

        # sampling
        X_s, y_s = apply_sampling(X, y, sampling_method, sampling_ratio)

        # class_weight
        class_weight = get_class_weight(
            y_s,
            model_name=model_name,
            sampling_method=sampling_method
        )

        # 모델 생성
        model = MODEL_FACTORY[model_name].build(
            row,
            class_weight=class_weight
        )

        model.fit(X_s, y_s)

        # 평가 (원본 기준)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        metrics = evaluate_binary(y, y_pred, y_prob)

        results.append({
            "disease": disease,
            "model": model_name,
            "sampling_method": sampling_method,
            "sampling_ratio": sampling_ratio,
            **metrics
        })

    return pd.DataFrame(results)

