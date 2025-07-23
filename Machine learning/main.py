import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Excel 文件路径
file_path = 'F:\ML program\data.xlsx'

# 读取数据
sheet1_data = pd.read_excel(file_path, sheet_name='Sheet2')
sheet2_data = pd.read_excel(file_path, sheet_name='Sheet1')

# 提取HSV特征与标签
# X_train = sheet1_data[['HSV平均值(H)', 'HSV平均值(S)', 'HSV平均值(V)']].values
# y_train = np.array([1] * 50 + [0] * 50)
# X_test = sheet2_data[['HSV平均值(H)', 'HSV平均值(S)', 'HSV平均值(V)']].values
# y_test = np.array([1] * 50 + [0] * 50)

# 处理 sheet1 数据
sheet1_data['编号'] = sheet1_data['文件名'].str.extract(r'(\d+)', expand=False).astype(int)
sheet1_data = sheet1_data[sheet1_data['编号'] != 50]
sheet1_data['标签'] = np.where(sheet1_data['编号'] <= 49, 1, 0)

# 处理 sheet2 数据
sheet2_data['编号'] = sheet2_data['文件名'].str.extract(r'(\d+)', expand=False).astype(int)
sheet2_data = sheet2_data[sheet2_data['编号'] != 50]
sheet2_data['标签'] = np.where(sheet2_data['编号'] <= 49, 1, 0)

# 构造特征和标签
X_train = sheet1_data[['HSV平均值(H)', 'HSV平均值(S)', 'HSV平均值(V)']].values
y_train = sheet1_data['标签'].values

X_test = sheet2_data[['HSV平均值(H)', 'HSV平均值(S)', 'HSV平均值(V)']].values
y_test = sheet2_data['标签'].values



# 初始化并训练SVM模型
svm_model = SVC(kernel='linear', probability=False)
svm_model.fit(X_train, y_train)

# 模型预测
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, digits=5)

# 可视化：3D散点图 + 决策超平面
def visualize_data(X_train, y_train, X_test, y_test, y_pred, svm_model=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2], label='Positive (Train)', marker='o')
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2], label='Negative (Train)', marker='x')
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], X_test[y_test == 1, 2], label='Positive (Test)', marker='s')
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], X_test[y_test == 0, 2], label='Negative (Test)', marker='d')

    if svm_model is not None:
        coef = svm_model.coef_[0]
        intercept = svm_model.intercept_[0]
        H_range = np.linspace(min(X_train[:, 0]), max(X_train[:, 0]), 50)
        S_range = np.linspace(min(X_train[:, 1]), max(X_train[:, 1]), 50)
        H, S = np.meshgrid(H_range, S_range)
        V = (-coef[0] * H - coef[1] * S - intercept) / coef[2]
        ax.plot_surface(H, S, V, alpha=0.45, rstride=100, cstride=100, edgecolor='none')

    ax.set_xlabel('HSV (H)')
    ax.set_ylabel('HSV (S)')
    ax.set_zlabel('HSV (V)')
    ax.set_title('3D Feature Distribution with SVM Decision Boundary')
    ax.legend()
    plt.show()

# 显示3D图
visualize_data(X_train, y_train, X_test, y_test, y_pred, svm_model=svm_model)

# 打印性能指标
print(f"\n测试集准确率: {accuracy:.5f}")
print("\n分类报告:\n", classification_rep)

# ROC曲线
y_scores = svm_model.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.5f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM Model')
plt.legend()
plt.show()

# 找出距离超平面最近的40个点
decision_scores = svm_model.decision_function(X_test)
closest_indices = np.argsort(np.abs(decision_scores))[:40]
closest_points = X_test[closest_indices]
closest_distances = decision_scores[closest_indices]
closest_labels = y_test[closest_indices]

# 计算法向量单位方向分量
w = svm_model.coef_[0]
unit_w = w / np.linalg.norm(w)
component_distances = np.outer(closest_distances, unit_w)

# 构建结果 DataFrame
result_df = pd.DataFrame(closest_points, columns=['H', 'S', 'V'])
result_df['Dist'] = closest_distances
result_df['Label'] = closest_labels
component_df = pd.DataFrame(component_distances, columns=['H_Dis', 'S_Dis', 'V_Dis'])
result_df = pd.concat([result_df, component_df], axis=1)

# 添加预测结果列
closest_predictions = y_pred[closest_indices]
result_df['Pred'] = closest_predictions

# 添加原始索引列
closest_original_indices = sheet2_data.iloc[closest_indices].index
result_df.insert(0, 'Index', closest_original_indices.values)

# 保留三位小数
pd.set_option('display.max_columns', None)
float_cols = result_df.select_dtypes(include=[np.float64]).columns
result_df[float_cols] = result_df[float_cols].round(3)

# 打印结果
print("\n距离SVM分界面最近的40个点及其HSV方向垂直距离（保留三位小数）：\n")
print(result_df)

# ==========================================
# 新增：每个点仅修改 H / S / V 单一分量实现翻类所需值和距离
# ==========================================

def get_critical_value_to_flip_class(x, w, b, target_index):
    """
    计算将样本x仅在一个分量方向上移动，使其落在SVM决策边界上的临界值
    """
    rest = np.dot(w, x) - w[target_index] * x[target_index]
    return -(rest + b) / w[target_index]

# 遍历最近的点，计算每个维度的翻类所需修改
delta_records = []
for idx, x in enumerate(closest_points):
    record = {'Index': closest_original_indices[idx]}
    for i, dim in enumerate(['H', 'S', 'V']):
        critical_val = get_critical_value_to_flip_class(x, w, svm_model.intercept_[0], i)
        delta = critical_val - x[i]
        record[f'{dim}_Delta'] = round(delta, 3)
        record[f'{dim}_CriticalVal'] = round(critical_val, 3)
    delta_records.append(record)

# 合并为 DataFrame
delta_df = pd.DataFrame(delta_records)
final_df = pd.merge(result_df, delta_df, on='Index')

print("\n最终结果（距离超平面最近点及每个分量实现翻类的修改量和目标值）：\n")
pd.set_option('display.max_columns', None)      # 显示所有列
pd.set_option('display.width', 150)             # 控制台宽度
pd.set_option('display.float_format', '{:.3f}'.format)  # 小数格式

print(final_df)
