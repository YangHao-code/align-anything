import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 配置可视化风格
sns.set(style="whitegrid")

# CSV 文件路径（替换为你的真实路径）
csv_path = "/root/autodl-tmp/align-anything/eval_vs_outputs/reward_scores.csv"

# 读取数据
df = pd.read_csv(csv_path)
print("列名为：", df.columns.tolist())

# 计算差值（如果你还没在 CSV 中算这个列的话）
if "original_reward - dpo_reward" not in df.columns:
    df["original_reward - dpo_reward"] = df["original_reward"] - df["dpo_reward"]

# 计算 DPO 胜率（即 dpo_reward > original_reward 的比例）
dpo_win_rate = (df["dpo_reward"] > df["original_reward"]).mean()
print(f"（dpo_reward > original_reward）：{dpo_win_rate:.2%}")

# 差值的描述统计
diff = df["original_reward - dpo_reward"]
print("\nReward 差值的统计信息：")
print(diff.describe())

# 1. Reward 差值直方图
plt.figure(figsize=(8, 4))
sns.histplot(diff, kde=True, bins=50, color="skyblue")
plt.axvline(0, color='red', linestyle='--')
plt.title("Original Reward - DPO Reward Distribution")
plt.xlabel("Reward diff(Original - DPO)")
plt.ylabel("number of samples")
plt.tight_layout()
plt.savefig("reward_difference_distribution.png")

# 2. DPO vs 原始模型 reward 散点图
plt.figure(figsize=(6, 6))
sns.scatterplot(x="original_reward", y="dpo_reward", data=df, alpha=0.5)
plt.plot([df["original_reward"].min(), df["original_reward"].max()],
         [df["original_reward"].min(), df["original_reward"].max()],
         'r--', label="y = x")
plt.title("DPO Reward vs Original Reward")
plt.xlabel("Original Reward")
plt.ylabel("DPO Reward")
plt.legend()
plt.tight_layout()
plt.savefig("dpo_vs_original_reward_scatter.png")

# 3. Reward 差值箱型图
plt.figure(figsize=(6, 4))
sns.boxplot(data=diff, color="lightgreen")
plt.title("Reward Difference Box Plot (Original - DPO)")
plt.xlabel("Reward diff")
plt.tight_layout()
plt.savefig("reward_difference_boxplot.png")
