import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from torch.utils.data import DataLoader, TensorDataset

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# -------------------- 第一部分：数据准备与预处理 --------------------
# 加载数据
df = pd.read_csv(r'C:\Users\34923\Desktop\RL_SuperMario-main\郭洪刚 张珈硕 糖尿病预测\preprocessed_diabetes_data.csv')

# 平衡采样并添加随机性
negative_data = df[df['Diabetes_012'] == 0].sample(n=10000, random_state=42)
positive_data = df[df['Diabetes_012'] == 1].sample(n=10000, random_state=42)
selected_data = pd.concat([negative_data, positive_data])

# 特征工程
X = selected_data.drop('Diabetes_012', axis=1)

y = selected_data['Diabetes_012'].astype(np.int64)  # 改为整型
# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- 修正后的数据预处理部分 --------------------
# 特征选择（选择Top 10特征）
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y).astype(np.float32)  # 添加类型转换

# -------------------- 特征选择后的可视化 --------------------
# 获取选中的特征名称
selected_features = X.columns[selector.get_support()]

if 'BMI' in selected_features:
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=selected_data,
        x='BMI',
        hue='Diabetes_012',
        kde=True,
        bins=30,
        palette={0: 'blue', 1: 'red'},  # 显式定义颜色映射
        element='step',
        hue_order=[0, 1]
    )
    plt.title('BMI在糖尿病与非糖尿病中的分布', fontsize=14)
    plt.xlabel('BMI', fontsize=12)
    plt.ylabel('数量', fontsize=12)

    # 手动创建图例（解决自动图例颜色顺序问题）
    handles = [plt.Rectangle((0, 0), 1, 1, color='blue', ec="k"),
               plt.Rectangle((0, 0), 1, 1, color='red', ec="k")]
    plt.legend(handles=handles,
               title='糖尿病状态',
               labels=['非糖尿病患者', '糖尿病患者'],  # 注意标签顺序与hue_order一致
               title_fontsize=12,
               fontsize=11)

    plt.show()

# 特征相关性热图
plt.figure(figsize=(12, 8))
corr_matrix = pd.DataFrame(X_selected, columns=selected_features).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Top 10 特征间的相关系数矩阵', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)


# -------------------- 第二部分：PyTorch模型封装 --------------------
class EnhancedDiabetesNet(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        # 更深的网络结构
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1)
        )
        # 残差连接
        self.shortcut = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64)
        )
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Linear(64, 1)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # 残差连接
        x += identity
        # 注意力加权
        attn_weights = self.attention(x)
        x = x * attn_weights
        return self.classifier(x).squeeze(-1)

# 自定义NeuralNetClassifier以处理标签形状
class CustomNet(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = y_true.float()  # 确保标签是浮点型
        return super().get_loss(y_pred, y_true, X=X, training=training)

# 使用skorch封装PyTorch模型
torch_clf = CustomNet(
    module=EnhancedDiabetesNet,
    criterion=nn.BCEWithLogitsLoss(),
    optimizer=optim.Adam,
    optimizer__lr=0.001,
    max_epochs=50,
    batch_size=64,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    iterator_train__shuffle=True,
    train_split=None,
# 添加以下参数
    iterator_train__drop_last=True
)

# -------------------- 集成学习部分 --------------------
# 调整子模型参数控制复杂度
estimators = [
    ('svm', SVC(probability=True, C=0.5, kernel='linear')),
    ('dt', DecisionTreeClassifier(max_depth=5)),
    ('nb', GaussianNB()),
    ('lr', LogisticRegression(C=0.5, max_iter=1000)),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(100,),  # 增加隐藏层神经元
        alpha=0.1,
        max_iter=500,  # 增加迭代次数
        early_stopping=True,  # 启用早停
        random_state=42
    )),
    ('pytorch', torch_clf)  # 添加PyTorch模型
]

# 使用分层交叉验证
ensemble = VotingClassifier(estimators=estimators, voting='soft')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy')
print(f'交叉验证准确率: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})')

# 训练模型
ensemble.fit(X_train, y_train)



# -------------------- 评估与可视化 --------------------
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'fpr': roc_curve(y, y_proba)[0] if y_proba is not None else None,
        'tpr': roc_curve(y, y_proba)[1] if y_proba is not None else None
    }


# 评估集成模型
train_metrics = evaluate_model(ensemble, X_train, y_train)
test_metrics = evaluate_model(ensemble, X_test, y_test)

# 评估各子模型并收集ROC数据
submodel_roc_data = []
name_mapping = {
    'svm': '支持向量机',
    'dt': '决策树',
    'nb': '朴素贝叶斯',
    'lr': '逻辑回归',
    'mlp': '神经网络',
    'pytorch': 'PyTorch模型'  # 新增模型名称映射
}

for model_name in ['svm', 'dt', 'nb', 'lr', 'mlp', 'pytorch']:
    model = ensemble.named_estimators_[model_name]
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        # 处理PyTorch模型的特殊情况
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(torch_clf.device)
            y_proba = model.forward(X_test_tensor).cpu().numpy().flatten()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    submodel_roc_data.append((
        name_mapping[model_name],
        fpr,
        tpr,
        roc_auc
    ))

# 打印评估结果
print("\n训练集评估:")
print(f"准确率: {train_metrics['accuracy']:.4f}")
print(f"精准率: {train_metrics['precision']:.4f}")
print(f"召回率: {train_metrics['recall']:.4f}")
print(f"F1分数: {train_metrics['f1']:.4f}")

print("\n测试集评估:")
print(f"准确率: {test_metrics['accuracy']:.4f}")
print(f"精准率: {test_metrics['precision']:.4f}")
print(f"召回率: {test_metrics['recall']:.4f}")
print(f"F1分数: {test_metrics['f1']:.4f}")

# -------------------- 可视化部分 --------------------
plt.figure(figsize=(15, 6))

# ROC曲线（左子图）
plt.subplot(1, 2, 1)
# 绘制各子模型曲线
for name, fpr, tpr, auc_val in submodel_roc_data:
    plt.plot(fpr, tpr, lw=2,
             label=f'{name} (AUC={auc_val:.2f})')

# 绘制集成模型曲线
plt.plot(test_metrics['fpr'], test_metrics['tpr'],
         label=f'集成模型 (AUC={auc(test_metrics["fpr"], test_metrics["tpr"]):.2f})',
         lw=2, linestyle='--', color='black')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('假阳性率', fontsize=12)
plt.ylabel('真阳性率', fontsize=12)
plt.title('测试集ROC曲线对比', fontsize=14)
plt.legend(loc='lower right', fontsize=8)

# 指标对比（右子图）
plt.subplot(1, 2, 2)
metrics = ['accuracy', 'precision', 'recall', 'f1']
labels = ['训练集', '测试集']
colors = ['#1f77b4', '#ff7f0e']

x = np.arange(len(metrics))
width = 0.35

# 准备数据
train_scores = [train_metrics[m] for m in metrics]
test_scores = [test_metrics[m] for m in metrics]

plt.bar(x - width/2, train_scores, width, label='训练集', color=colors[0], alpha=0.6)
plt.bar(x + width/2, test_scores, width, label='测试集', color=colors[1], alpha=0.6)

plt.xticks(x, [m.capitalize() for m in metrics])
plt.ylabel('分数', fontsize=12)
plt.title('性能指标对比', fontsize=14)
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()
plt.show()


# -------------------- 新增评估可视化 --------------------
# 混淆矩阵
from sklearn.metrics import confusion_matrix

y_pred_test = ensemble.predict(X_test)
cm = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['非糖尿病', '糖尿病'],
            yticklabels=['非糖尿病', '糖尿病'])
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.title('混淆矩阵', fontsize=14)
plt.show()

# 子模型性能对比
submodel_performance = []
for model_name in ['svm', 'dt', 'nb', 'lr', 'mlp', 'pytorch']:
    model = ensemble.named_estimators_[model_name]
    # 处理PyTorch模型的预测
    if model_name == 'pytorch':
        X_test_tensor = torch.FloatTensor(X_test).to(torch_clf.device)
        with torch.no_grad():
            logits = model.forward(X_test_tensor).cpu().numpy().flatten()
            y_pred = (logits > 0).astype(int)  # 使用0作为阈值（因为使用BCEWithLogitsLoss）
    else:
        y_pred = model.predict(X_test)
    # 计算指标
    submodel_performance.append({
        '模型': name_mapping[model_name],
        '准确率': accuracy_score(y_test, y_pred),
        '精准率': precision_score(y_test, y_pred),
        '召回率': recall_score(y_test, y_pred),
        'F1分数': f1_score(y_test, y_pred)
    })

# 转换为DataFrame并绘图
sub_df = pd.DataFrame(submodel_performance)
melted_df = sub_df.melt(id_vars='模型', var_name='指标', value_name='值')

plt.figure(figsize=(12, 6))
sns.barplot(x='模型', y='值', hue='指标', data=melted_df, palette='tab10')
plt.title('各子模型在测试集上的性能对比', fontsize=14)
plt.xticks(rotation=30)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 校准曲线
from sklearn.calibration import calibration_curve

y_proba = ensemble.predict_proba(X_test)[:, 1]
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='集成模型')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='理想校准')
plt.xlabel('平均预测概率', fontsize=12)
plt.ylabel('实际阳性比例', fontsize=12)
plt.title('校准曲线', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 特征分布示例（以BMI为例）
if 'BMI' in selected_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=selected_data, x='BMI', hue='Diabetes_012',
                 kde=True, bins=30, palette='Set1', element='step')
    plt.title('BMI在糖尿病与非糖尿病中的分布', fontsize=14)
    plt.xlabel('BMI', fontsize=12)
    plt.ylabel('数量', fontsize=12)
    plt.legend(title='糖尿病', labels=['否', '是'])
    plt.show()