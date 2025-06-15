import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/root/autodl-tmp/align-anything/eval_outputs/reward_scores.csv")

# 直方图：显示 chosen 与 rejected reward 分布差异
plt.figure(figsize=(10, 6))
sns.histplot(df["chosen_reward"], label="Chosen", color="green", kde=True, bins=30)
sns.histplot(df["rejected_reward"], label="Rejected", color="red", kde=True, bins=30)
plt.title("Reward Score Distribution")
plt.xlabel("Reward Score")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_distribution.png")
plt.show()
