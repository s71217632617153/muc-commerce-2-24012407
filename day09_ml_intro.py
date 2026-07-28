"""
Day 9 学生实验手册：第一次接触机器学习
======================================
本脚本完成全部 6 个任务，生成 4 个 CSV 成果文件。

运行方式:
    pip install -r requirements.txt
    python generate_data.py     # 先生成数据
    python day09_ml_intro.py    # 运行全部任务

任务清单:
    任务1: 数据验收 (5630行, 22列, 无缺失, 流失率~16.84%)
    任务2: 填写建模口径 (TARGET=Churn, ID_COL=CustomerID)
    任务3: 查看特征方案 (feature_schema.csv)
    任务4: 完成分层划分 (STRATIFY_TARGET=y, split_summary.csv)
    任务5: 运行预处理流水线 (36列, model_matrix_preview.csv)
    任务6: 运行最低参照线 (baseline_metrics.csv)
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

# ============================================================
#  配置区
# ============================================================
DATA_PATH = Path("data/ecommerce_churn_data.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- 任务2: 填写建模口径 ----
TARGET = "Churn"          # 预测目标：是否流失
ID_COL = "CustomerID"     # 用户唯一标识，不作为特征

# ---- 任务4: 分层划分 ----
# 原始代码为 STRATIFY_TARGET = None，修改为 y 实现分层划分
STRATIFY_TARGET = None    # <-- 上午先不完成，搜索 TODO 8-4


def banner(title):
    """打印分隔横幅"""
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


# ============================================================
#  任务1: 数据验收
# ============================================================
def task1_data_validation(df):
    """验证数据集行数、列数、缺失值、流失率"""
    banner("任务1: 数据验收")

    n_rows, n_cols = df.shape
    n_missing = df.isnull().sum().sum()
    n_churn = int(df[TARGET].sum())
    churn_rate = df[TARGET].mean()

    print(f"  行数:     {n_rows}")
    print(f"  列数:     {n_cols}")
    print(f"  缺失值:   {n_missing}")
    print(f"  流失人数: {n_churn}")
    print(f"  流失率:   {churn_rate:.4%}")

    # 断言检查
    assert n_rows == 5630, f"行数不是5630，实际{n_rows}"
    assert n_cols == 22, f"列数不是22，实际{n_cols}"
    assert n_missing == 0, f"存在{ n_missing }个缺失值"
    assert abs(churn_rate - 0.1684) < 0.01, f"流失率偏差过大: {churn_rate:.4%}"

    print("\n  ✓ 数据验收全部通过!")
    print(f"    - 5630 行 × 22 列")
    print(f"    - 无缺失值")
    print(f"    - 总体流失率 {churn_rate:.2%} (约16.84%)")

    return True


# ============================================================
#  任务2: 填写建模口径
# ============================================================
def task2_modeling_spec(df):
    """设置 TARGET 和 ID_COL，确认 X 中不含 ID 和答案"""
    banner("任务2: 填写建模口径")

    y = df[TARGET].copy()
    # 排除 ID 和目标列
    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET]]
    X = df[feature_cols].copy()

    print(f"  TARGET  = {TARGET}")
    print(f"  ID_COL  = {ID_COL}")
    print(f"  特征数:  {len(feature_cols)}")
    print(f"  特征列:  {feature_cols}")

    # 确认 X 中不含 ID 和答案
    assert ID_COL not in X.columns, f"{ID_COL} 不应出现在特征中!"
    assert TARGET not in X.columns, f"{TARGET} 不应出现在特征中!"

    print(f"\n  ✓ 确认 X 中不含 CustomerID 和 Churn")
    print(f"    - CustomerID 是唯一标识，不是特征")
    print(f"    - Churn 是预测目标（标签），不能作为特征")
    print(f"    - 实际特征矩阵: {X.shape[0]} 行 × {X.shape[1]} 列")

    return X, y


# ============================================================
#  任务3: 查看特征方案
# ============================================================
def task3_feature_schema(X):
    """自动识别数值列和类别列，生成 feature_schema.csv"""
    banner("任务3: 查看特征方案")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"  数值列 ({len(numeric_cols)}): {numeric_cols}")
    print(f"  类别列 ({len(categorical_cols)}): {categorical_cols}")

    # 构建特征方案表
    schema_rows = []
    for col in numeric_cols:
        schema_rows.append({
            "feature_name": col,
            "dtype": "numeric",
            "role": "特征",
            "processing": "StandardScaler (标准化)",
            "example_value": str(X[col].iloc[0]),
        })
    for col in categorical_cols:
        n_unique = X[col].nunique()
        schema_rows.append({
            "feature_name": col,
            "dtype": "categorical",
            "role": "特征",
            "processing": f"OneHotEncoder ({n_unique} 个类别)",
            "example_value": str(X[col].iloc[0]),
        })

    schema_df = pd.DataFrame(schema_rows)
    schema_path = OUTPUT_DIR / "feature_schema.csv"
    schema_df.to_csv(schema_path, index=False, encoding="utf-8-sig")
    print(f"\n  ✓ feature_schema.csv 已生成 ({len(schema_df)} 个特征)")
    print(f"    保存路径: {schema_path}")

    # 解释为什么文字类别不能直接交给模型
    print(f"\n  【解释】为什么文字类别不能直接交给模型计算?")
    print(f"    1. 数学运算只认数字: 模型用矩阵乘法计算权重之和,")
    print(f"       'Mobile Phone' 这样的文字无法参与加法和乘法运算。")
    print(f"    2. 序号编码有误导: 如果把 Fashion=1, Grocery=2 赋数字,")
    print(f"       模型会认为 Grocery 比 Fashion '大一倍', 这是无意义的。")
    print(f"    3. OneHot 解决方案: 把每个类别变成独立的 0/1 列,")
    print(f"       让模型理解 '是/否属于该类别', 不引入虚假的大小关系。")

    return numeric_cols, categorical_cols


# ============================================================
#  任务4: 完成分层划分
# ============================================================
def task4_stratified_split(X, y):
    """分层划分：STRATIFY_TARGET = y"""
    banner("任务4: 完成分层划分")

    # ---- 核心修改：将 STRATIFY_TARGET = None 改为 y ----
    STRATIFY_TARGET = y  # ← 修改此处！原来是 None
    print(f"  STRATIFY_TARGET = y (原为 None)")
    print(f"  分层划分让训练集和测试集的流失比例保持接近\n")

    # 不分层（错误做法）对比
    X_train_no, X_test_no, y_train_no, y_test_no = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    # 分层划分（正确做法）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=STRATIFY_TARGET
    )

    # 对比表
    summary_rows = [
        {
            "dataset": "train (分层)",
            "n_samples": len(X_train),
            "n_churn": int(y_train.sum()),
            "churn_rate": f"{y_train.mean():.4%}",
            "stratify": "y",
        },
        {
            "dataset": "test (分层)",
            "n_samples": len(X_test),
            "n_churn": int(y_test.sum()),
            "churn_rate": f"{y_test.mean():.4%}",
            "stratify": "y",
        },
        {
            "dataset": "train (未分层)",
            "n_samples": len(X_train_no),
            "n_churn": int(y_train_no.sum()),
            "churn_rate": f"{y_train_no.mean():.4%}",
            "stratify": "None",
        },
        {
            "dataset": "test (未分层)",
            "n_samples": len(X_test_no),
            "n_churn": int(y_test_no.sum()),
            "churn_rate": f"{y_test_no.mean():.4%}",
            "stratify": "None",
        },
        {
            "dataset": "full",
            "n_samples": len(X),
            "n_churn": int(y.sum()),
            "churn_rate": f"{y.mean():.4%}",
            "stratify": "-",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  分层划分后:")
    print(f"    训练集: {len(X_train)} 行, 流失率 {y_train.mean():.4%}")
    print(f"    测试集: {len(X_test)} 行, 流失率 {y_test.mean():.4%}")
    print(f"    全量:   {len(X)} 行, 流失率 {y.mean():.4%}")

    print(f"\n  未分层（对比）:")
    print(f"    训练集: {len(X_train_no)} 行, 流失率 {y_train_no.mean():.4%}")
    print(f"    测试集: {len(X_test_no)} 行, 流失率 {y_test_no.mean():.4%}")

    print(f"\n  ✓ split_summary.csv 已生成")
    print(f"    保存路径: {summary_path}")
    print(f"    分层后训练集/测试集流失率接近全量比例 (~16.84%)")

    return X_train, X_test, y_train, y_test


# ============================================================
#  任务5: 运行预处理流水线
# ============================================================
def task5_preprocessing(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols):
    """运行教师提供的预处理流水线"""
    banner("任务5: 运行预处理流水线")

    # ---- 教师提供的预处理流水线 ----
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
    ])

    # 拟合并转换
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # 获取转换后的列名
    cat_feature_names = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_cols)
        .tolist()
    )
    all_feature_names = numeric_cols + cat_feature_names

    # 转为 DataFrame
    X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)

    # ---- 检查项 ----
    print(f"  预处理前:")
    print(f"    训练集: {X_train.shape[0]} 行 × {X_train.shape[1]} 列 (含文字类别)")
    print(f"    测试集: {X_test.shape[0]} 行 × {X_test.shape[1]} 列")

    print(f"\n  预处理后:")
    print(f"    训练集: {X_train_df.shape[0]} 行 × {X_train_df.shape[1]} 列")
    print(f"    测试集: {X_test_df.shape[0]} 行 × {X_test_df.shape[1]} 列")

    # 检查1: 都是数值
    all_numeric_train = X_train_df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
    all_numeric_test = X_test_df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
    print(f"\n  检查1 - 全部为数值:")
    print(f"    训练集: {'✓' if all_numeric_train else '✗'}")
    print(f"    测试集: {'✓' if all_numeric_test else '✗'}")

    # 检查2: 列数相同
    same_cols = X_train_df.shape[1] == X_test_df.shape[1]
    print(f"  检查2 - 列数相同: {'✓' if same_cols else '✗'}")

    # 检查3: 无缺失值
    no_missing_train = X_train_df.isnull().sum().sum() == 0
    no_missing_test = X_test_df.isnull().sum().sum() == 0
    print(f"  检查3 - 无缺失值:")
    print(f"    训练集: {'✓' if no_missing_train else '✗'} ({X_train_df.isnull().sum().sum()} 个)")
    print(f"    测试集: {'✓' if no_missing_test else '✗'} ({X_test_df.isnull().sum().sum()} 个)")

    # 检查4: 无无穷值
    no_inf_train = not np.isinf(X_train_df.select_dtypes(include=[np.number]).values).any()
    no_inf_test = not np.isinf(X_test_df.select_dtypes(include=[np.number]).values).any()
    print(f"  检查4 - 无无穷值:")
    print(f"    训练集: {'✓' if no_inf_train else '✗'}")
    print(f"    测试集: {'✓' if no_inf_test else '✗'}")

    # 检查5: 转换后列数
    n_cols = X_train_df.shape[1]
    print(f"\n  检查5 - 转换后列数: {n_cols}")
    print(f"    数值列: {len(numeric_cols)} 列 (StandardScaler)")
    print(f"    类别OneHot: {len(cat_feature_names)} 列")
    print(f"    合计: {len(numeric_cols)} + {len(cat_feature_names)} = {n_cols}")

    # 保存前20行预览
    preview_df = X_train_df.head(20).round(4)
    preview_path = OUTPUT_DIR / "model_matrix_preview.csv"
    preview_df.to_csv(preview_path, index=False, encoding="utf-8-sig")
    print(f"\n  ✓ model_matrix_preview.csv 已生成 (前20行)")
    print(f"    保存路径: {preview_path}")

    # 打印前5行前5列的值作为示例
    print(f"\n  预览 (前5行前5列):")
    print(preview_df.iloc[:5, :5].to_string(index=False))

    return X_train_df, X_test_df, all_feature_names


# ============================================================
#  任务6: 运行最低参照线
# ============================================================
def task6_baseline(X_train, X_test, y_train, y_test):
    """最低参照线：永远预测人数最多的类别（未流失）"""
    banner("任务6: 运行最低参照线")

    # 使用 DummyClassifier，strategy='most_frequent' 永远预测最多的类别
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    # recall_score for churn=1
    recall_churn = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    # 预测的流失人数
    n_pred_churn = int(y_pred.sum())
    n_pred_nochurn = int((y_pred == 0).sum())
    n_actual_churn = int(y_test.sum())
    n_actual_nochurn = int((y_test == 0).sum())

    print(f"  最低参照线策略: strategy='most_frequent'")
    print(f"  训练集中最多的类别: 未流失 (Churn=0, {int(y_train.sum())}/{len(y_train)} = {1-y_train.mean():.2%})")
    print(f"  预测结果: 全部预测为 '未流失'")
    print(f"\n  测试集情况:")
    print(f"    总人数:     {len(y_test)}")
    print(f"    实际流失:   {n_actual_churn}")
    print(f"    实际未流失: {n_actual_nochurn}")
    print(f"    预测流失:   {n_pred_churn}")
    print(f"    预测未流失: {n_pred_nochurn}")

    print(f"\n  指标:")
    print(f"    准确率:       {accuracy:.4f} ({accuracy:.2%})")
    print(f"    流失召回率:   {recall_churn:.4f}")
    print(f"    流失预测数:   {n_pred_churn}")

    # 保存 baseline_metrics.csv
    metrics_rows = [
        {
            "metric": "accuracy",
            "value": round(accuracy, 4),
            "description": "准确率 = 预测正确数 / 总数",
        },
        {
            "metric": "churn_recall",
            "value": round(recall_churn, 4),
            "description": "流失召回率 = 正确预测的流失 / 实际流失",
        },
        {
            "metric": "predicted_churn_count",
            "value": n_pred_churn,
            "description": "预测为流失的人数",
        },
        {
            "metric": "actual_churn_count",
            "value": n_actual_churn,
            "description": "实际流失人数",
        },
        {
            "metric": "total_test_samples",
            "value": len(y_test),
            "description": "测试集总人数",
        },
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = OUTPUT_DIR / "baseline_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n  ✓ baseline_metrics.csv 已生成")
    print(f"    保存路径: {metrics_path}")

    # 解释
    print(f"\n  【解释】为什么 {accuracy:.2%} 的准确率不能用于寻找流失用户?")
    print(f"    1. 假阳性为零: 模型预测的流失人数为 {n_pred_churn},")
    print(f"       没有找出任何一个流失用户。")
    print(f"    2. 召回率为零: 流失召回率 = {recall_churn:.4f},")
    print(f"       说明所有 {n_actual_churn} 个真正流失的用户全部被漏掉了。")
    print(f"    3. 准确率虚高: 83% 的准确率来自 '大部分用户本来就没流失' 这个事实,")
    print(f"       模型只是猜 '不流失', 没有任何预测能力。")
    print(f"    4. 结论: 当类别不平衡时, 准确率是误导性指标,")
    print(f"       需要关注召回率、精确率等针对少数类的指标。")

    return accuracy, recall_churn


# ============================================================
#  上午：6人小实验
# ============================================================
def morning_experiment(df):
    """人工规则判断6名用户是否流失"""
    banner("上午: 6人小实验 — 人工规则判断")

    # 随机选6个用户，包含3个流失3个未流失
    churn_sample = df[df["Churn"] == 1].sample(3, random_state=42)
    nochurn_sample = df[df["Churn"] == 0].sample(3, random_state=42)
    sample = pd.concat([churn_sample, nochurn_sample]).sample(frac=1, random_state=7)

    # 只展示判断时能看的信息（不含 Churn 和 CustomerID）
    view_cols = ["Tenure", "SatisfactionScore", "Complain", "DaySinceLastOrder",
                 "OrderCount", "PreferedOrderCat"]

    print("  6名用户的部分信息（不含 Churn 和 CustomerID）:\n")
    display_df = sample[view_cols + ["Churn"]].copy()
    display_df.index = [f"用户{i+1}" for i in range(6)]
    print(display_df.to_string())

    # 人工规则: 满意度<=2 且 投诉=1 且 Tenure<6 → 判断为流失
    print("\n  人工规则: 满意度<=2 且 投诉=1 且 Tenure<6 → 预测流失")
    print()

    correct = 0
    for idx, row in sample.iterrows():
        user_label = f"用户{list(sample.index).index(idx)+1}"
        actual = int(row["Churn"])
        pred = 1 if (row["SatisfactionScore"] <= 2 and row["Complain"] == 1 and row["Tenure"] < 6) else 0
        match = "✓ 正确" if pred == actual else "✗ 错误"
        if pred == actual:
            correct += 1
        print(f"    {user_label}: 满意度={row['SatisfactionScore']}, 投诉={row['Complain']}, "
              f"Tenure={row['Tenure']} → 预测={'流失' if pred else '未流失'} | "
              f"实际={'流失' if actual else '未流失'} {match}")

    print(f"\n  人工规则判断正确: {correct}/6")
    print(f"\n  【回答】人工规则为什么不能保证对所有用户都有效?")
    print(f"    1. 规则过于简单: 只用3个特征判断, 忽略了订单数、品类、支付方式等信息。")
    print(f"    2. 阈值靠经验: 'Tenure<6' 和 '满意度<=2' 是人为设定的,")
    print(f"       换一批用户可能就不适用了。")
    print(f"    3. 无法覆盖组合: 用户行为是多特征组合的结果,")
    print(f"       人工规则难以枚举所有有效组合。")
    print(f"    4. 这正是机器学习的价值: 从数据中自动学习特征与标签的关系,")
    print(f"       不依赖人工设定的固定阈值。")


# ============================================================
#  主函数
# ============================================================
def main():
    print("=" * 60)
    print("  Day 9: 第一次接触机器学习")
    print("  电商用户流失预测 — 数据准备与基线")
    print("=" * 60)

    # 加载数据
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"\n  数据加载完成: {df.shape[0]} 行 × {df.shape[1]} 列")

    # ---- 上午：6人小实验 ----
    morning_experiment(df)

    # ---- 任务1: 数据验收 ----
    task1_data_validation(df)

    # ---- 任务2: 填写建模口径 ----
    X, y = task2_modeling_spec(df)

    # ---- 任务3: 查看特征方案 ----
    numeric_cols, categorical_cols = task3_feature_schema(X)

    # ---- 任务4: 完成分层划分 ----
    X_train, X_test, y_train, y_test = task4_stratified_split(X, y)

    # ---- 任务5: 运行预处理流水线 ----
    X_train_proc, X_test_proc, feature_names = task5_preprocessing(
        X_train, X_test, y_train, y_test, numeric_cols, categorical_cols
    )

    # ---- 任务6: 运行最低参照线 ----
    accuracy, recall = task6_baseline(X_train_proc, X_test_proc, y_train, y_test)

    # ---- 总结 ----
    banner("Day 9 全部任务完成!")
    print(f"  生成的成果文件:")
    print(f"    1. outputs/feature_schema.csv      — 字段角色与处理方式")
    print(f"    2. outputs/split_summary.csv        — 训练集/测试集规模与流失比例")
    print(f"    3. outputs/model_matrix_preview.csv — 模型输入矩阵前20行")
    print(f"    4. outputs/baseline_metrics.csv     — 最低参照线三项结果")
    print(f"\n  关键指标:")
    print(f"    数据集: 5630 行 × 22 列, 无缺失, 流失率 16.84%")
    print(f"    预处理后: {X_train_proc.shape[1]} 列 (16 数值 + 20 OneHot)")
    print(f"    最低参照线: 准确率 {accuracy:.2%}, 流失召回率 {recall:.4f}")


if __name__ == "__main__":
    main()
