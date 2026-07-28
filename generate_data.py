"""
Day 9 数据生成脚本
生成符合第7天统计指标的电商用户流失数据集 (5630行 x 22列)
- 总体流失率约 16.84% (948/5630)
- 品类分布: Mobile Phone 2080, Laptop & Accessory 2050, Fashion 826, Grocery 410, Others 264
- 各品类流失率: Mobile 27.4%, Fashion 15.5%, Laptop 10.2%, Others 7.6%, Grocery 4.9%
- 平均订单数 2.96, 平均优惠券 1.72, 平均返现 177.22, 平均App时长 2.93
- 平均满意度 3.07, 平均距上次下单天数 4.46
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 5630

# ---- 品类分布 ----
categories = (
    ["Mobile Phone"] * 2080
    + ["Laptop & Accessory"] * 2050
    + ["Fashion"] * 826
    + ["Grocery"] * 410
    + ["Others"] * 264
)
np.random.shuffle(categories)

# 各品类流失率
cat_churn_rate = {
    "Mobile Phone": 0.2740,
    "Laptop & Accessory": 0.1024,
    "Others": 0.0758,
    "Fashion": 0.1550,
    "Grocery": 0.0488,
}

# ---- 生成 Churn 标签 ----
churn = np.zeros(N, dtype=int)
for i in range(N):
    if np.random.rand() < cat_churn_rate[categories[i]]:
        churn[i] = 1

# 调整使总流失人数接近 948
current_churn = churn.sum()
target_churn = 948
if current_churn < target_churn:
    zero_idx = np.where(churn == 0)[0]
    flip = np.random.choice(zero_idx, target_churn - current_churn, replace=False)
    churn[flip] = 1
elif current_churn > target_churn:
    one_idx = np.where(churn == 1)[0]
    flip = np.random.choice(one_idx, current_churn - target_churn, replace=False)
    churn[flip] = 0

print(f"生成流失用户: {churn.sum()} / {N} = {churn.sum()/N:.4%}")

# ---- 生成特征 ----
# Tenure: 流失用户通常 tenure 较短
tenure = np.where(
    churn == 1,
    np.random.exponential(8, N) + 1,
    np.random.exponential(18, N) + 2,
)
tenure = np.clip(tenure, 0, 72).round(1)

# PreferredLoginDevice
login_device = np.random.choice(
    ["Mobile Phone", "Computer", "Phone"], N, p=[0.71, 0.28, 0.01]
)

# CityTier
city_tier = np.random.choice([1, 2, 3], N, p=[0.55, 0.30, 0.15])

# WarehouseToHome
warehouse_to_home = np.where(
    churn == 1,
    np.random.exponential(16, N) + 5,
    np.random.exponential(12, N) + 5,
)
warehouse_to_home = np.clip(warehouse_to_home, 5, 128).round(1)

# PreferredPaymentMode
payment_mode = np.random.choice(
    ["Debit Card", "Credit Card", "E wallet", "Cash on Delivery", "UPI", "COD"],
    N,
    p=[0.41, 0.30, 0.12, 0.09, 0.05, 0.03],
)

# Gender
gender = np.random.choice(["Male", "Female"], N, p=[0.60, 0.40])

# HourSpendOnApp
hour_spend = np.where(
    churn == 1,
    np.random.normal(2.5, 1.2, N),
    np.random.normal(3.0, 1.0, N),
)
hour_spend = np.clip(hour_spend, 0, 24).round(1)

# NumberOfDeviceRegistered
n_devices = np.random.randint(1, 7, N)

# SatisfactionScore: 流失用户满意度偏低
satisfaction = np.where(
    churn == 1,
    np.random.choice([1, 2, 3, 4, 5], N, p=[0.25, 0.25, 0.25, 0.15, 0.10]),
    np.random.choice([1, 2, 3, 4, 5], N, p=[0.05, 0.10, 0.25, 0.30, 0.30]),
)

# MaritalStatus
marital = np.random.choice(["Single", "Married", "Divorced"], N, p=[0.50, 0.40, 0.10])

# NumberOfAddress
n_address = np.random.randint(1, 22, N)

# Complain: 流失用户投诉率更高
complain = np.where(
    churn == 1,
    np.random.choice([0, 1], N, p=[0.45, 0.55]),
    np.random.choice([0, 1], N, p=[0.88, 0.12]),
)

# OrderAmountHikeFromlastYear (percentage)
order_hike = np.where(
    churn == 1,
    np.random.normal(12, 5, N),
    np.random.normal(15, 4, N),
)
order_hike = np.clip(order_hike, 0, 50).round(1)

# CouponUsed: 按品类调整
coupon_base = {"Mobile Phone": 1.37, "Laptop & Accessory": 1.65, "Fashion": 2.32, "Grocery": 2.19, "Others": 2.33}
coupon_used = np.array([
    max(0, np.random.normal(coupon_base[c], 0.8)) for c in categories
]).round(1)

# OrderCount: 按品类调整
order_base = {"Mobile Phone": 2.18, "Laptop & Accessory": 2.77, "Fashion": 3.87, "Grocery": 4.60, "Others": 5.25}
order_count = np.array([
    max(0, np.random.normal(order_base[c], 1.0)) for c in categories
]).round(0).astype(int)

# DaySinceLastOrder
day_since = np.where(
    churn == 1,
    np.random.exponential(6, N) + 1,
    np.random.exponential(3.5, N) + 1,
)
day_since = np.clip(day_since, 0, 46).round(0).astype(int)

# CashbackAmount: 按品类调整
cashback_base = {"Mobile Phone": 140.2, "Laptop & Accessory": 167.2, "Fashion": 210.4, "Grocery": 266.2, "Others": 304.6}
cashback = np.array([
    max(0, np.random.normal(cashback_base[c], 40)) for c in categories
]).round(2)

# AppEngagementScore (1-10)
engagement = np.where(
    churn == 1,
    np.random.normal(4, 2, N),
    np.random.normal(6, 2, N),
)
engagement = np.clip(engagement, 1, 10).round(0).astype(int)

# CustomerSegment (Bronze/Silver/Gold)
segment = np.where(
    (tenure > 24) & (order_count > 3),
    "Gold",
    np.where(
        (tenure > 12) & (order_count > 1),
        "Silver",
        "Bronze",
    ),
)

# ---- 构建 DataFrame ----
df = pd.DataFrame({
    "CustomerID": np.arange(50001, 50001 + N),
    "Churn": churn,
    "Tenure": tenure,
    "PreferredLoginDevice": login_device,
    "CityTier": city_tier,
    "WarehouseToHome": warehouse_to_home,
    "PreferredPaymentMode": payment_mode,
    "Gender": gender,
    "HourSpendOnApp": hour_spend,
    "NumberOfDeviceRegistered": n_devices,
    "PreferedOrderCat": categories,
    "SatisfactionScore": satisfaction,
    "MaritalStatus": marital,
    "NumberOfAddress": n_address,
    "Complain": complain,
    "OrderAmountHikeFromlastYear": order_hike,
    "CouponUsed": coupon_used,
    "OrderCount": order_count,
    "DaySinceLastOrder": day_since,
    "CashbackAmount": cashback,
    "AppEngagementScore": engagement,
    "CustomerSegment": segment,
})

# 确保无缺失值
assert df.isnull().sum().sum() == 0, "存在缺失值!"

# 保存
output_path = "data/ecommerce_churn_data.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"数据集已保存: {output_path}")
print(f"行数: {len(df)}, 列数: {len(df.columns)}")
print(f"流失人数: {df['Churn'].sum()}, 流失率: {df['Churn'].mean():.4%}")
print(f"\n品类分布:")
print(df.groupby("PreferedOrderCat")["Churn"].agg(["count", "mean"]))
print(f"\n平均订单数: {df['OrderCount'].mean():.4f}")
print(f"平均优惠券: {df['CouponUsed'].mean():.4f}")
print(f"平均返现: {df['CashbackAmount'].mean():.2f}")
print(f"平均App时长: {df['HourSpendOnApp'].mean():.4f}")
print(f"平均满意度: {df['SatisfactionScore'].mean():.4f}")
print(f"平均距上次下单天数: {df['DaySinceLastOrder'].mean():.4f}")
print(f"\n列名: {list(df.columns)}")
print(f"\n数值列: {list(df.select_dtypes(include=[np.number]).columns)}")
print(f"类别列: {list(df.select_dtypes(exclude=[np.number]).columns)}")
