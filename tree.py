import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from scipy import stats
import warnings

# ============================
# TẠO THƯ MỤC LƯU TRỮ
# ============================
os.makedirs('img', exist_ok=True)
os.makedirs('result', exist_ok=True)
print("✅ Đã tạo thư mục 'img' và 'result'")

# ============================
# BƯỚC 1: ĐỌC VÀ KIỂM TRA DỮ LIỆU 
# ============================
print("PHÂN TÍCH DỮ LIỆU VỚI CÂY QUYẾT ĐỊNH - CẢI TIẾN (10 LẦN)")

try:
    xls = pd.ExcelFile('Folds5x2_pp.xlsx')
    df_list = []
    # Lặp qua tên của từng sheet
    for sheet_name in xls.sheet_names:
        print(f"Đang đọc sheet: {sheet_name}...")
        df_list.append(pd.read_excel(xls, sheet_name=sheet_name))
    
    # Gộp tất cả các DataFrame từ các sheet lại
    df = pd.concat(df_list, ignore_index=True)
    
    print(f" Đã đọc và gộp {len(xls.sheet_names)} sheets thành công!")
except FileNotFoundError:
    print(" LỖI: Không tìm thấy file 'Folds5x2_pp.xlsx'.")
    print("Vui lòng đảm bảo file dữ liệu nằm cùng thư mục với script.")
    exit()

print(f"\n THÔNG TIN DATASET (SAU KHI GỘP):")
print(f"    . Kích thước: {df.shape} ({df.shape[0]:,} mẫu × {df.shape[1]} đặc trưng)")
print(f"     Cột: {list(df.columns)}")
print(f"     Kiểm tra NaN (dữ liệu thiếu): {df.isna().sum().sum()} (Nếu > 0 là có lỗi)")

# ============================
# BƯỚC 2: PHÂN TÍCH VÀ TIỀN XỬ LÝ
# ============================
X = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

# BƯỚC 2.5: FEATURE ENGINEERING NÂNG CAO
print("\n🔧 FEATURE ENGINEERING NÂNG CAO")

# Tạo các feature tương tác và đa thức
X_enhanced = X.copy()
X_enhanced['AT_V'] = X['AT'] * X['V']           # Tương tác nhiệt độ và áp suất hơi
X_enhanced['AT_RH'] = X['AT'] * X['RH']         # Tương tác nhiệt độ và độ ẩm
X_enhanced['V_AP'] = X['V'] * X['AP']           # Tương tác áp suất hơi và áp suất khí
X_enhanced['AT_squared'] = X['AT'] ** 2         # Đa thức bậc 2 cho nhiệt độ
X_enhanced['V_squared'] = X['V'] ** 2           # Đa thức bậc 2 cho áp suất hơi

print(f"    Số feature ban đầu: {X.shape[1]}")
print(f"    Số feature sau engineering: {X_enhanced.shape[1]}")
print(f"    Feature mới: {list(X_enhanced.columns)[X.shape[1]:]}")

# Thêm lựa chọn sử dụng feature engineering (tùy chọn)
use_enhanced_features = False  # Đổi thành True để dùng feature mới

if use_enhanced_features:
    X = X_enhanced
    print("    ✅ Đã sử dụng feature engineering")
else:
    print("    ℹ️  Sử dụng feature gốc (để so sánh công bằng)")

# ============================
# BƯỚC 3: LẶP 10 LẦN HUẤN LUYỆN
# ============================
print("\nCẢI TIẾN: HUẤN LUYỆN 10 LẦN VÀ TÍNH TRUNG BÌNH")

# Lists để lưu kết quả của 10 lần chạy
all_train_metrics = []
all_test_metrics = []
all_feature_importances = []
best_models = []

# Thử nhiều bộ siêu tham số khác nhau
param_sets = [
    {'max_depth': 5, 'min_samples_split': 20, 'min_samples_leaf': 10},
    {'max_depth': 7, 'min_samples_split': 15, 'min_samples_leaf': 5},
    {'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 3},
    {'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2},
    {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'max_depth': 8, 'min_samples_split': 20, 'min_samples_leaf': 8},
    {'max_depth': 12, 'min_samples_split': 8, 'min_samples_leaf': 4},
    {'max_depth': 6, 'min_samples_split': 25, 'min_samples_leaf': 12},
    {'max_depth': 9, 'min_samples_split': 12, 'min_samples_leaf': 6},
    {'max_depth': 4, 'min_samples_split': 30, 'min_samples_leaf': 15}
]

# Hàm tính metrics 
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Xử lý trường hợp y_true = 0 để tránh lỗi chia cho 0
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-10))) * 100
    
    return {
        'mse': mse, 'rmse': rmse, 'mae': mae, 
        'r2': r2, 'mape': mape
    }

for i in range(10):
    print(f"\n🔄 LẦN CHẠY THỨ {i+1}/10")
    
    # Chuẩn hóa dữ liệu cho mỗi lần chạy
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Phân chia train-test với random_state khác nhau
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=40 + i, shuffle=True
    )
    
    # Lấy bộ tham số cho lần chạy này
    params = param_sets[i]
    print(f"     Tham số: {params}")
    
    # Tạo và huấn luyện mô hình
    dt_model = DecisionTreeRegressor(
        random_state=40 + i,
        **params
    )
    
    dt_model.fit(X_train, y_train)
    
    # Dự đoán
    y_pred_train = dt_model.predict(X_train)
    y_pred_test = dt_model.predict(X_test)
    
    # Tính metrics
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Lưu kết quả
    all_train_metrics.append(train_metrics)
    all_test_metrics.append(test_metrics)
    all_feature_importances.append(dt_model.feature_importances_)
    
    best_models.append({
        'model': dt_model,
        'params': params,
        'test_r2': test_metrics['r2'],
        'run_id': i,
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'y_pred_test': y_pred_test,
        'scaler': scaler
    })
    
    print(f"    ✓ Train R²: {train_metrics['r2']:.4f}")
    print(f"    ✓ Test R²:  {test_metrics['r2']:.4f}")
    print(f"    ✓ Test RMSE: {test_metrics['rmse']:.4f}")

# ============================
# BƯỚC 4: PHÂN TÍCH KẾT QUẢ 10 LẦN CHẠY
# ============================
print("\n" + "="*50)
print("PHÂN TÍCH TỔNG HỢP 10 LẦN CHẠY")
print("="*50)

train_df = pd.DataFrame(all_train_metrics)
test_df = pd.DataFrame(all_test_metrics)

# Tính trung bình và độ lệch chuẩn
print("\n THỐNG KÊ TẬP TRAIN (10 lần):")
print(f"     R²:     {train_df['r2'].mean():.4f} (±{train_df['r2'].std():.4f})")
print(f"     RMSE:   {train_df['rmse'].mean():.4f} (±{train_df['rmse'].std():.4f})")
print(f"     MSE:    {train_df['mse'].mean():.4f} (±{train_df['mse'].std():.4f})")
print(f"     MAE:    {train_df['mae'].mean():.4f} (±{train_df['mae'].std():.4f})")
print(f"     MAPE:   {train_df['mape'].mean():.2f}% (±{train_df['mape'].std():.2f}%)")

print("\n THỐNG KÊ TẬP TEST (10 lần):")
print(f"     R²:     {test_df['r2'].mean():.4f} (±{test_df['r2'].std():.4f})")
print(f"     RMSE:   {test_df['rmse'].mean():.4f} (±{test_df['rmse'].std():.4f})")
print(f"     MSE:    {test_df['mse'].mean():.4f} (±{test_df['mse'].std():.4f})")
print(f"     MAE:    {test_df['mae'].mean():.4f} (±{test_df['mae'].std():.4f})")
print(f"     MAPE:   {test_df['mape'].mean():.2f}% (±{test_df['mape'].std():.2f}%)")

# Tính độ quan trọng đặc trưng trung bình
avg_feature_importance = np.mean(all_feature_importances, axis=0)
feature_importance_df = pd.DataFrame({
    'Đặc trưng': ['AT', 'V', 'AP', 'RH'],
    'Độ quan trọng trung bình': avg_feature_importance,
    'Độ lệch chuẩn': np.std(all_feature_importances, axis=0)
}).sort_values('Độ quan trọng trung bình', ascending=False)

print("\n ĐỘ QUAN TRỌNG ĐẶC TRƯNG TRUNG BÌNH:")
for idx, row in feature_importance_df.iterrows():
    print(f"    ✓ {row['Đặc trưng']}: {row['Độ quan trọng trung bình']:.4f} (±{row['Độ lệch chuẩn']:.4f})")

# ============================
# BƯỚC 5: CHỌN MÔ HÌNH TỐT NHẤT
# ============================
best_models.sort(key=lambda x: x['test_r2'], reverse=True)
best_model_info = best_models[0] 
best_model = best_model_info['model']

# Lưu mô hình và scaler
model_path = os.path.join('result', 'best_decision_tree_model.pkl')
scaler_path = os.path.join('result', 'scaler.pkl')
joblib.dump(best_model, model_path)
joblib.dump(best_model_info['scaler'], scaler_path)

print("\n✅ Đã lưu mô hình và scaler thành công vào thư mục 'result':")
print(f"   • {model_path}")
print(f"   • {scaler_path}")

print(f"\n🏆 MÔ HÌNH TỐT NHẤT (Lần chạy {best_model_info['run_id'] + 1}):")
print(f"     Tham số: {best_model_info['params']}")
print(f"     Test R²: {best_model_info['test_r2']:.4f}")

# Lấy dữ liệu từ mô hình tốt nhất để so sánh
X_train_best = best_model_info['X_train']
X_test_best = best_model_info['X_test'] 
y_train_best = best_model_info['y_train']
y_test_best = best_model_info['y_test']
y_pred_test_best = best_model_info['y_pred_test']

# ============================
# BƯỚC 5.5: CROSS-VALIDATION
# ============================
print("\n🔄 ĐÁNH GIÁ ĐỘ ỔN ĐỊNH VỚI CROSS-VALIDATION (5-fold)")

cv_results = cross_validate(
    best_model, X_scaled, y, 
    cv=5, 
    scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
    return_train_score=True,
    n_jobs=-1
)

cv_train_r2 = cv_results['train_r2']
cv_test_r2 = cv_results['test_r2']
cv_test_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'])
cv_test_mae = -cv_results['test_neg_mean_absolute_error']

print(f"\n📊 KẾT QUẢ CROSS-VALIDATION (5-fold):")
print(f"    Train R²:     {cv_train_r2.mean():.4f} (±{cv_train_r2.std():.4f})")
print(f"    Test R²:      {cv_test_r2.mean():.4f} (±{cv_test_r2.std():.4f})")
print(f"    Test RMSE:    {cv_test_rmse.mean():.4f} (±{cv_test_rmse.std():.4f})")
print(f"    Test MAE:     {cv_test_mae.mean():.4f} (±{cv_test_mae.std():.4f})")

cv_stability = "RẤT ỔN ĐỊNH" if cv_test_r2.std() < 0.02 else "KHÁ ỔN ĐỊNH" if cv_test_r2.std() < 0.05 else "CÓ BIẾN ĐỘNG"
print(f"    Độ ổn định:    {cv_stability} (độ lệch chuẩn: {cv_test_r2.std():.4f})")

print(f"\n📈 SO SÁNH PHƯƠNG PHÁP:")
print(f"    10 lần split:  R² = {test_df['r2'].mean():.4f} (±{test_df['r2'].std():.4f})")
print(f"    5-fold CV:     R² = {cv_test_r2.mean():.4f} (±{cv_test_r2.std():.4f})")

# ============================
# BƯỚC 6: SO SÁNH VỚI MÔ HÌNH KHÁC
# ============================
print("\nSO SÁNH VỚI RANDOM FOREST")

# Huấn luyện Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100, random_state=42, max_depth=10,
    min_samples_split=10, n_jobs=-1
)
rf_model.fit(X_train_best, y_train_best)
y_pred_rf = rf_model.predict(X_test_best)
rf_metrics = calculate_metrics(y_test_best, y_pred_rf)

# Tính metrics cho Decision Tree tốt nhất
y_pred_dt_best = best_model.predict(X_test_best)
dt_metrics_best = calculate_metrics(y_test_best, y_pred_dt_best)

print("\n SO SÁNH HIỆU SUẤT TRÊN TẬP TEST TỐT NHẤT:")
print(f"    Decision Tree (tốt nhất):")
print(f"       R²:   {dt_metrics_best['r2']:.4f}")
print(f"       RMSE: {dt_metrics_best['rmse']:.4f}")
print(f"       MAE:  {dt_metrics_best['mae']:.4f}")
print(f"       MAPE: {dt_metrics_best['mape']:.2f}%")

print(f"    Random Forest:")
print(f"       R²:   {rf_metrics['r2']:.4f}")
print(f"       RMSE: {rf_metrics['rmse']:.4f}")
print(f"       MAE:  {rf_metrics['mae']:.4f}")
print(f"       MAPE: {rf_metrics['mape']:.2f}%")

# SO SÁNH VỚI KNN
print("\n🔍 SO SÁNH THÊM VỚI KNN REGRESSOR (TỐI ƯU HÓA THAM SỐ)")

knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn_grid = GridSearchCV(
    KNeighborsRegressor(), knn_param_grid, cv=5, 
    scoring='r2', n_jobs=-1, verbose=0
)

print("Đang tìm tham số tối ưu cho KNN...")
knn_grid.fit(X_train_best, y_train_best)

best_knn = knn_grid.best_estimator_
y_pred_knn = best_knn.predict(X_test_best)
knn_metrics = calculate_metrics(y_test_best, y_pred_knn)

print(f"\n✅ KNN Regressor (ĐÃ TỐI ƯU):")
print(f"    Tham số tốt nhất: {knn_grid.best_params_}")
print(f"    R²:   {knn_metrics['r2']:.4f}")
print(f"    RMSE: {knn_metrics['rmse']:.4f}")
print(f"    MAE:  {knn_metrics['mae']:.4f}")
print(f"    MAPE: {knn_metrics['mape']:.2f}%")

# ============================
# BƯỚC 7: TRỰC QUAN HÓA KẾT QUẢ
# ============================
print("\n🎨 BẮT ĐẦU TRỰC QUAN HÓA KẾT QUẢ")

# 7.1 Biểu đồ so sánh mô hình
print("\n📊 1. Biểu đồ so sánh mô hình")
comparison_path = os.path.join('img', 'model_comparison.png')
plt.figure(figsize=(10, 6))
models = ['Decision Tree', 'Random Forest', 'KNN']
r2_scores = [dt_metrics_best['r2'], rf_metrics['r2'], knn_metrics['r2']]
colors = ['#2ECC71', '#3498DB', '#9B59B6']

bars = plt.bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black')
plt.ylabel('R² Score', fontsize=12)
plt.title('SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH', fontweight='bold', fontsize=14)
plt.ylim(0.8, 1.0)
plt.grid(True, alpha=0.3)

for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Đã lưu: {comparison_path}")

# 7.2 Biểu đồ feature importance
print("📊 2. Biểu đồ feature importance")
feature_img_path = os.path.join('img', 'feature_importance.png')
plt.figure(figsize=(10, 6))
features = feature_importance_df['Đặc trưng']
importances = feature_importance_df['Độ quan trọng trung bình']
std_dev = feature_importance_df['Độ lệch chuẩn']

bars = plt.bar(features, importances, yerr=std_dev, capsize=8, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], 
               alpha=0.8, edgecolor='black')
plt.ylabel('Độ quan trọng trung bình', fontsize=12)
plt.title('ĐỘ QUAN TRỌNG ĐẶC TRƯNG (10 LẦN CHẠY)', fontweight='bold', fontsize=14)
plt.xticks(fontsize=11)
plt.grid(True, alpha=0.3)

for bar, importance, std in zip(bars, importances, std_dev):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{importance:.3f} (±{std:.3f})', ha='center', va='bottom', 
             fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(feature_img_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Đã lưu: {feature_img_path}")

# 7.3 Biểu đồ Actual vs Predicted
print("📊 3. Biểu đồ Actual vs Predicted")
actual_pred_path = os.path.join('img', 'actual_vs_predicted.png')
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test_best, y_pred_dt_best, alpha=0.6, s=30, color='blue')
plt.plot([y_test_best.min(), y_test_best.max()], [y_test_best.min(), y_test_best.max()], 'r--', lw=2)
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title(f'Decision Tree\nR² = {dt_metrics_best["r2"]:.4f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(y_test_best, y_pred_rf, alpha=0.6, s=30, color='green')
plt.plot([y_test_best.min(), y_test_best.max()], [y_test_best.min(), y_test_best.max()], 'r--', lw=2)
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title(f'Random Forest\nR² = {rf_metrics["r2"]:.4f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(y_test_best, y_pred_knn, alpha=0.6, s=30, color='purple')
plt.plot([y_test_best.min(), y_test_best.max()], [y_test_best.min(), y_test_best.max()], 'r--', lw=2)
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title(f'KNN\nR² = {knn_metrics["r2"]:.4f}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(actual_pred_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Đã lưu: {actual_pred_path}")

# 7.4 Biểu đồ tổng hợp 10 lần chạy
print("📊 4. Biểu đồ tổng hợp 10 lần chạy")
summary_plots_path = os.path.join('img', 'summary_plots.png')
plt.figure(figsize=(20, 12))
plt.suptitle("PHÂN TÍCH TỔNG HỢP 10 LẦN HUẤN LUYỆN DECISION TREE", fontsize=20, fontweight='bold', y=1.03)

# Biểu đồ 1: So sánh R² qua 10 lần chạy
plt.subplot(2, 3, 1)
runs = range(1, 11)
plt.plot(runs, train_df['r2'], marker='o', linewidth=2, markersize=8, label='Train R²', color='#2ECC71')
plt.plot(runs, test_df['r2'], marker='s', linewidth=2, markersize=8, label='Test R²', color='#E74C3C')
plt.axhline(y=test_df['r2'].mean(), color='red', linestyle='--', alpha=0.7, label=f"Test R² TB: {test_df['r2'].mean():.3f}")
plt.xlabel('Lần chạy')
plt.ylabel('R² Score')
plt.title('SO SÁNH R² QUA 10 LẦN CHẠY', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Biểu đồ 2: So sánh RMSE qua 10 lần chạy
plt.subplot(2, 3, 2)
plt.plot(runs, train_df['rmse'], marker='o', linewidth=2, markersize=8, label='Train RMSE', color='#3498DB')
plt.plot(runs, test_df['rmse'], marker='s', linewidth=2, markersize=8, label='Test RMSE', color='#F39C12')
plt.axhline(y=test_df['rmse'].mean(), color='orange', linestyle='--', alpha=0.7, label=f"Test RMSE TB: {test_df['rmse'].mean():.3f}")
plt.xlabel('Lần chạy')
plt.ylabel('RMSE')
plt.title('SO SÁNH RMSE QUA 10 LẦN CHẠY', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Biểu đồ 3: Phân bố R² trên tập test
plt.subplot(2, 3, 3)
sns.boxplot(data=[train_df['r2'], test_df['r2']], palette=['#AED6F1', '#FAD7A0'])
plt.xticks([0, 1], ['Train R²', 'Test R²'])
plt.ylabel('R² Score')
plt.title('PHÂN BỐ R² SCORE (10 LẦN)', fontweight='bold')
plt.grid(True, alpha=0.3)

# Biểu đồ 4: Hiệu suất theo bộ tham số
plt.subplot(2, 3, 4)
param_names = [f"Set {i+1}" for i in range(10)]
test_r2_values = test_df['r2']
plt.scatter(param_names, test_r2_values, s=100, alpha=0.7, c=test_r2_values, cmap='viridis')
plt.axhline(y=test_r2_values.mean(), color='red', linestyle='--', label='Trung bình')
plt.xlabel('Bộ tham số')
plt.ylabel('Test R²')
plt.title('HIỆU SUẤT THEO BỘ THAM SỐ', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.colorbar(label='R² Score')
plt.grid(True, alpha=0.3)

# Biểu đồ 5: So sánh 3 mô hình
plt.subplot(2, 3, 5)
models_compare = ['DT', 'RF', 'KNN']
r2_compare = [dt_metrics_best['r2'], rf_metrics['r2'], knn_metrics['r2']]
plt.bar(models_compare, r2_compare, color=['#2ECC71', '#3498DB', '#9B59B6'])
plt.ylabel('R² Score')
plt.title('SO SÁNH 3 MÔ HÌNH', fontweight='bold')
for i, v in enumerate(r2_compare):
    plt.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3)

# Biểu đồ 6: Cross-validation results
plt.subplot(2, 3, 6)
cv_folds = range(1, 6)
plt.plot(cv_folds, cv_test_r2, marker='o', linewidth=2, markersize=8, color='#E74C3C')
plt.axhline(y=cv_test_r2.mean(), color='red', linestyle='--', label=f'Trung bình: {cv_test_r2.mean():.3f}')
plt.xlabel('Fold')
plt.ylabel('Test R²')
plt.title('CROSS-VALIDATION (5-fold)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(summary_plots_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Đã lưu: {summary_plots_path}")

# 7.5 Phân tích sai số (Residuals Analysis)
print("📊 5. Phân tích sai số")
residuals_dt = y_test_best - y_pred_dt_best
residuals_rf = y_test_best - y_pred_rf
residuals_knn = y_test_best - y_pred_knn

residuals_path = os.path.join('img', 'residuals_analysis.png')
plt.figure(figsize=(18, 12))

# Biểu đồ 1: Residuals vs Predicted cho DT
plt.subplot(2, 3, 1)
plt.scatter(y_pred_dt_best, residuals_dt, alpha=0.6, s=30, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Giá trị dự đoán')
plt.ylabel('Sai số (Residuals)')
plt.title(f'Decision Tree\nStd: {residuals_dt.std():.3f}')
plt.grid(True, alpha=0.3)

# Biểu đồ 2: Residuals vs Predicted cho RF
plt.subplot(2, 3, 2)
plt.scatter(y_pred_rf, residuals_rf, alpha=0.6, s=30, color='green')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Giá trị dự đoán')
plt.ylabel('Sai số (Residuals)')
plt.title(f'Random Forest\nStd: {residuals_rf.std():.3f}')
plt.grid(True, alpha=0.3)

# Biểu đồ 3: Residuals vs Predicted cho KNN
plt.subplot(2, 3, 3)
plt.scatter(y_pred_knn, residuals_knn, alpha=0.6, s=30, color='purple')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Giá trị dự đoán')
plt.ylabel('Sai số (Residuals)')
plt.title(f'KNN\nStd: {residuals_knn.std():.3f}')
plt.grid(True, alpha=0.3)

# Biểu đồ 4: Phân phối residuals
plt.subplot(2, 3, 4)
plt.hist(residuals_dt, bins=30, alpha=0.7, label=f'DT (std: {residuals_dt.std():.3f})', color='blue')
plt.hist(residuals_rf, bins=30, alpha=0.7, label=f'RF (std: {residuals_rf.std():.3f})', color='green')
plt.hist(residuals_knn, bins=30, alpha=0.7, label=f'KNN (std: {residuals_knn.std():.3f})', color='purple')
plt.xlabel('Sai số (Residuals)')
plt.ylabel('Tần suất')
plt.title('PHÂN PHỐI SAI SỐ CỦA CÁC MÔ HÌNH')
plt.legend()
plt.grid(True, alpha=0.3)

# Biểu đồ 5: Q-Q plot cho Decision Tree
plt.subplot(2, 3, 5)
stats.probplot(residuals_dt, dist="norm", plot=plt)
plt.title('Q-Q Plot: Decision Tree Residuals')

# Biểu đồ 6: So sánh độ lớn sai số
plt.subplot(2, 3, 6)
residuals_abs = [np.abs(residuals_dt).mean(), np.abs(residuals_rf).mean(), np.abs(residuals_knn).mean()]
models_resid = ['Decision Tree', 'Random Forest', 'KNN']
bars = plt.bar(models_resid, residuals_abs, color=['blue', 'green', 'purple'], alpha=0.7)
plt.ylabel('Sai số tuyệt đối trung bình (MAE)')
plt.title('SO SÁNH ĐỘ LỚN SAI SỐ')
for bar, value in zip(bars, residuals_abs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.3f}', 
             ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Đã lưu: {residuals_path}")

# Phân tích thống kê residuals
print(f"\n📊 PHÂN TÍCH THỐNG KÊ SAI SỐ:")
print(f"    Decision Tree: Mean = {residuals_dt.mean():.4f}, Std = {residuals_dt.std():.4f}")
print(f"    Random Forest: Mean = {residuals_rf.mean():.4f}, Std = {residuals_rf.std():.4f}")
print(f"    KNN:           Mean = {residuals_knn.mean():.4f}, Std = {residuals_knn.std():.4f}")

# 7.6 Learning Curves
print("📊 6. Learning Curves")

def plot_and_save_learning_curve(estimator, title, filename, X, y, cv=5):
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2', n_jobs=-1, random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score", linewidth=2)
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score", linewidth=2)
    
    plt.xlabel("Số lượng mẫu training", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    plt.title(f"Learning Curve: {title}", fontweight='bold')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    filepath = os.path.join('img', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Đã lưu: {filepath}")
    return train_sizes, train_mean, test_mean

print("Đang vẽ và lưu learning curves...")
plot_and_save_learning_curve(best_model, "Decision Tree (Best Model)", "learning_curve_dt.png", X_scaled, y)
plot_and_save_learning_curve(rf_model, "Random Forest", "learning_curve_rf.png", X_scaled, y)
# 7.7 Biểu đồ so sánh chi tiết 10 lần lặp
print("📊 7. Biểu đồ so sánh chi tiết 10 lần lặp")

comparison_10_runs_path = os.path.join('img', 'comparison_10_runs.png')
plt.figure(figsize=(18, 12))

# Biểu đồ 1: So sánh R² train vs test qua 10 lần
plt.subplot(2, 3, 1)
runs = range(1, 11)
plt.plot(runs, train_df['r2'], marker='o', linewidth=3, markersize=8, 
         label=f'Train R² (TB: {train_df["r2"].mean():.4f})', color='#2ECC71')
plt.plot(runs, test_df['r2'], marker='s', linewidth=3, markersize=8, 
         label=f'Test R² (TB: {test_df["r2"].mean():.4f})', color='#E74C3C')
plt.axhline(y=train_df['r2'].mean(), color='#2ECC71', linestyle='--', alpha=0.5)
plt.axhline(y=test_df['r2'].mean(), color='#E74C3C', linestyle='--', alpha=0.5)
plt.xlabel('Lần chạy')
plt.ylabel('R² Score')
plt.title('SO SÁNH R² TRAIN vs TEST QUA 10 LẦN', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.8, 1.0)

# Thêm annotation cho lần chạy tốt nhất
best_run = best_model_info['run_id'] + 1
best_test_r2 = best_model_info['test_r2']
plt.annotate(f'Tốt nhất\nLần {best_run}\nR² = {best_test_r2:.4f}', 
             xy=(best_run, best_test_r2), 
             xytext=(best_run+0.5, best_test_r2-0.02),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontweight='bold', color='red')

# Biểu đồ 2: So sánh RMSE train vs test qua 10 lần
plt.subplot(2, 3, 2)
plt.plot(runs, train_df['rmse'], marker='o', linewidth=3, markersize=8, 
         label=f'Train RMSE (TB: {train_df["rmse"].mean():.4f})', color='#3498DB')
plt.plot(runs, test_df['rmse'], marker='s', linewidth=3, markersize=8, 
         label=f'Test RMSE (TB: {test_df["rmse"].mean():.4f})', color='#F39C12')
plt.axhline(y=train_df['rmse'].mean(), color='#3498DB', linestyle='--', alpha=0.5)
plt.axhline(y=test_df['rmse'].mean(), color='#F39C12', linestyle='--', alpha=0.5)
plt.xlabel('Lần chạy')
plt.ylabel('RMSE')
plt.title('SO SÁNH RMSE TRAIN vs TEST QUA 10 LẦN', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Biểu đồ 3: So sánh MAE train vs test qua 10 lần
plt.subplot(2, 3, 3)
plt.plot(runs, train_df['mae'], marker='o', linewidth=3, markersize=8, 
         label=f'Train MAE (TB: {train_df["mae"].mean():.4f})', color='#9B59B6')
plt.plot(runs, test_df['mae'], marker='s', linewidth=3, markersize=8, 
         label=f'Test MAE (TB: {test_df["mae"].mean():.4f})', color='#E67E22')
plt.axhline(y=train_df['mae'].mean(), color='#9B59B6', linestyle='--', alpha=0.5)
plt.axhline(y=test_df['mae'].mean(), color='#E67E22', linestyle='--', alpha=0.5)
plt.xlabel('Lần chạy')
plt.ylabel('MAE')
plt.title('SO SÁNH MAE TRAIN vs TEST QUA 10 LẦN', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Biểu đồ 4: Phân bố chênh lệch R² (Overfitting)
plt.subplot(2, 3, 4)
r2_diff = train_df['r2'] - test_df['r2']
plt.bar(runs, r2_diff, color=np.where(r2_diff > 0.1, '#E74C3C', '#2ECC71'), alpha=0.7)
plt.axhline(y=r2_diff.mean(), color='red', linestyle='--', 
           label=f'Trung bình: {r2_diff.mean():.4f}')
plt.xlabel('Lần chạy')
plt.ylabel('Chênh lệch R² (Train - Test)')
plt.title('ĐÁNH GIÁ OVERFITTING QUA 10 LẦN', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Thêm giá trị trên các cột
for i, v in enumerate(r2_diff):
    plt.text(i+1, v + 0.001, f'{v:.4f}', ha='center', va='bottom', 
             fontsize=9, fontweight='bold', 
             color='red' if v > 0.1 else 'green')

# Biểu đồ 5: Hiệu suất theo bộ tham số (Heatmap style)
plt.subplot(2, 3, 5)
# Tạo dữ liệu cho heatmap
param_names = [f"Lần {i+1}" for i in range(10)]
metrics = ['R²', 'RMSE', 'MAE']
performance_data = np.array([
    test_df['r2'].values,
    test_df['rmse'].values,
    test_df['mae'].values
])

im = plt.imshow(performance_data, cmap='RdYlGn', aspect='auto')
plt.xticks(range(10), param_names, rotation=45)
plt.yticks(range(3), metrics)
plt.title('MA TRẬN HIỆU SUẤT 10 LẦN CHẠY', fontweight='bold', fontsize=14)

# Thêm giá trị vào ô - SỬA ĐỔI: R² hiển thị 3 chữ số thập phân
for i in range(3):
    for j in range(10):
        if i == 0:  # R²
            text = f'{performance_data[i, j]:.3f}'  # ĐÃ SỬA: .4f thành .3f
            color = 'white' if performance_data[i, j] < 0.95 else 'black'
        else:  # RMSE, MAE
            text = f'{performance_data[i, j]:.2f}'
            color = 'white' if performance_data[i, j] > performance_data[i].mean() else 'black'
        plt.text(j, i, text, ha='center', va='center', 
                fontweight='bold', color=color, fontsize=9)

plt.colorbar(im, label='Hiệu suất (Xanh = Tốt, Đỏ = Kém)')
# Biểu đồ 6: Tổng quan độ ổn định
plt.subplot(2, 3, 6)
metrics_std = [test_df['r2'].std(), test_df['rmse'].std(), test_df['mae'].std()]
metrics_names = ['R²', 'RMSE', 'MAE']
colors_std = ['#2ECC71' if std < 0.02 else '#F39C12' if std < 0.05 else '#E74C3C' for std in metrics_std]

bars = plt.bar(metrics_names, metrics_std, color=colors_std, alpha=0.7, edgecolor='black')
plt.ylabel('Độ lệch chuẩn')
plt.title('ĐÁNH GIÁ ĐỘ ỔN ĐỊNH 10 LẦN CHẠY', fontweight='bold', fontsize=14)
plt.grid(True, alpha=0.3)

# Thêm giá trị và đánh giá
for bar, std, metric in zip(bars, metrics_std, metrics_names):
    if metric == 'R²':
        rating = "Rất ổn định" if std < 0.01 else "Ổn định" if std < 0.02 else "Biến động"
    else:
        rating = "Rất ổn định" if std < 0.5 else "Ổn định" if std < 1.0 else "Biến động"
    
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{std:.4f}\n{rating}', ha='center', va='bottom', 
             fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(comparison_10_runs_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Đã lưu: {comparison_10_runs_path}")
# 7.8 Biểu đồ chi tiết từng lần chạy với tham số
print("📊 8. Biểu đồ chi tiết từng lần chạy")

detailed_runs_path = os.path.join('img', 'detailed_runs_analysis.png')
plt.figure(figsize=(20, 15))

# Biểu đồ 1: Hiệu suất theo max_depth
plt.subplot(3, 3, 1)
max_depths = [params.get('max_depth', 'None') for params in param_sets]
test_r2_by_depth = test_df['r2'].values
colors_depth = ['#2ECC71' if r2 > test_df['r2'].mean() else '#E74C3C' for r2 in test_r2_by_depth]

bars = plt.bar(range(1, 11), test_r2_by_depth, color=colors_depth, alpha=0.7)
plt.xlabel('Lần chạy')
plt.ylabel('Test R²')
plt.title('HIỆU SUẤT THEO LẦN CHẠY', fontweight='bold')
plt.xticks(range(1, 11), [f'Lần {i}' for i in range(1, 11)], rotation=45)
plt.grid(True, alpha=0.3)

# Thêm giá trị R²
for i, (bar, r2, depth) in enumerate(zip(bars, test_r2_by_depth, max_depths)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
             f'{r2:.4f}\n(depth: {depth})', ha='center', va='bottom', 
             fontsize=8, fontweight='bold')

# Biểu đồ 2: Phân tích tham số max_depth
plt.subplot(3, 3, 2)
unique_depths = list(set(max_depths))
depth_performance = []
for depth in unique_depths:
    indices = [i for i, d in enumerate(max_depths) if d == depth]
    avg_r2 = test_df.iloc[indices]['r2'].mean()
    depth_performance.append(avg_r2)

plt.bar([str(d) for d in unique_depths], depth_performance, 
        color='#3498DB', alpha=0.7, edgecolor='black')
plt.xlabel('Max Depth')
plt.ylabel('R² Trung bình')
plt.title('HIỆU SUẤT THEO MAX DEPTH', fontweight='bold')
plt.grid(True, alpha=0.3)

for i, (depth, perf) in enumerate(zip(unique_depths, depth_performance)):
    plt.text(i, perf + 0.002, f'{perf:.4f}', ha='center', va='bottom', 
             fontweight='bold')

# Biểu đồ 3: Phân tích min_samples_split
plt.subplot(3, 3, 3)
min_splits = [params.get('min_samples_split', 'N/A') for params in param_sets]
split_groups = {}
for i, split in enumerate(min_splits):
    if split not in split_groups:
        split_groups[split] = []
    split_groups[split].append(test_df.iloc[i]['r2'])

split_means = {k: np.mean(v) for k, v in split_groups.items()}
plt.bar([str(k) for k in split_means.keys()], split_means.values(),
        color='#9B59B6', alpha=0.7, edgecolor='black')
plt.xlabel('Min Samples Split')
plt.ylabel('R² Trung bình')
plt.title('HIỆU SUẤT THEO MIN SAMPLES SPLIT', fontweight='bold')
plt.grid(True, alpha=0.3)

# Biểu đồ 4: Tương quan giữa các metrics
plt.subplot(3, 3, 4)
plt.scatter(test_df['rmse'], test_df['r2'], s=100, alpha=0.7, 
           c=test_df['r2'], cmap='RdYlGn')
plt.xlabel('RMSE')
plt.ylabel('R²')
plt.title('TƯƠNG QUAN RMSE vs R²', fontweight='bold')
plt.colorbar(label='R² Score')
plt.grid(True, alpha=0.3)

# Thêm annotation cho các điểm
for i, (rmse, r2) in enumerate(zip(test_df['rmse'], test_df['r2'])):
    plt.annotate(f'Lần {i+1}', (rmse, r2), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.8)

# Biểu đồ 5: Phân bố R² của 10 lần chạy
plt.subplot(3, 3, 5)
plt.hist(test_df['r2'], bins=8, color='#2ECC71', alpha=0.7, edgecolor='black')
plt.axvline(test_df['r2'].mean(), color='red', linestyle='--', 
           label=f'Trung bình: {test_df["r2"].mean():.4f}')
plt.xlabel('R² Score')
plt.ylabel('Tần suất')
plt.title('PHÂN BỐ R² 10 LẦN CHẠY', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Biểu đồ 6: Biểu đồ radar so sánh metrics 
plt.subplot(3, 3, 6, projection='polar')  # THÊM projection='polar'

# Chuẩn hóa dữ liệu cho radar chart
metrics_to_plot = ['r2', 'rmse', 'mae', 'mape']
normalized_data = []
for metric in metrics_to_plot:
    if metric == 'r2':  # R² càng cao càng tốt
        normalized = test_df[metric] / test_df[metric].max()
    else:  # RMSE, MAE, MAPE càng thấp càng tốt
        normalized = 1 - (test_df[metric] / test_df[metric].max())
    normalized_data.append(normalized.values)

normalized_data = np.array(normalized_data)
angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False).tolist()
angles += angles[:1]  # Đóng vòng

# Vẽ 3 lần chạy tốt nhất
best_runs_indices = test_df['r2'].nlargest(3).index
colors_best = ['#E74C3C', '#3498DB', '#2ECC71']

for idx, color in zip(best_runs_indices, colors_best):
    values = normalized_data[:, idx].tolist()
    values += values[:1]  # Đóng vòng
    plt.plot(angles, values, 'o-', linewidth=2, label=f'Lần {idx+1}', color=color)
    plt.fill(angles, values, alpha=0.1, color=color)

plt.thetagrids(np.degrees(angles[:-1]), metrics_to_plot)
plt.title('RADAR CHART: 3 LẦN CHẠY TỐT NHẤT', fontweight='bold', pad=20)
plt.legend(bbox_to_anchor=(1.3, 1.1))
# Biểu đồ 7: Trend hiệu suất theo thời gian
plt.subplot(3, 3, 7)
# Tính cumulative mean
cumulative_mean = [test_df['r2'].iloc[:i+1].mean() for i in range(len(test_df))]
cumulative_std = [test_df['r2'].iloc[:i+1].std() for i in range(len(test_df))]

plt.plot(range(1, 11), cumulative_mean, marker='o', linewidth=2, 
         label='R² trung bình tích lũy', color='#E74C3C')
plt.fill_between(range(1, 11), 
                 np.array(cumulative_mean) - np.array(cumulative_std),
                 np.array(cumulative_mean) + np.array(cumulative_std),
                 alpha=0.2, color='#E74C3C', label='±1 std')
plt.xlabel('Số lần chạy')
plt.ylabel('R² Trung bình tích lũy')
plt.title('XU HƯỚNG HIỆU SUẤT THEO SỐ LẦN CHẠY', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Biểu đồ 8: So sánh độ biến động
plt.subplot(3, 3, 8)
metrics_variability = {
    'R²': test_df['r2'].std(),
    'RMSE': test_df['rmse'].std(),
    'MAE': test_df['mae'].std(),
    'MAPE': test_df['mape'].std()
}

plt.bar(metrics_variability.keys(), metrics_variability.values(),
        color=['#2ECC71', '#3498DB', '#9B59B6', '#E67E22'], alpha=0.7)
plt.ylabel('Độ lệch chuẩn')
plt.title('ĐỘ BIẾN ĐỘNG CÁC CHỈ SỐ', fontweight='bold')
plt.grid(True, alpha=0.3)

for i, (metric, std) in enumerate(metrics_variability.items()):
    plt.text(i, std + 0.001, f'{std:.4f}', ha='center', va='bottom', 
             fontweight='bold')

# Biểu đồ 9: Tổng kết ranking
plt.subplot(3, 3, 9)
ranking = test_df['r2'].rank(ascending=False)
colors_rank = ['gold' if rank == 1 else 'silver' if rank == 2 else 'brown' if rank == 3 else '#3498DB' 
               for rank in ranking]

plt.bar(range(1, 11), test_df['r2'], color=colors_rank, alpha=0.7)
plt.xlabel('Lần chạy')
plt.ylabel('R² Score')
plt.title('RANKING 10 LẦN CHẠY', fontweight='bold')
plt.xticks(range(1, 11), [f'#{int(r)}' for r in ranking], rotation=45)

for i, (r2, rank) in enumerate(zip(test_df['r2'], ranking)):
    medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else ''
    plt.text(i+1, r2 + 0.002, f'{r2:.4f}\n{medal}', ha='center', va='bottom', 
             fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(detailed_runs_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Đã lưu: {detailed_runs_path}")
# ============================
# BƯỚC 8: VẼ VÀ LƯU CÂY QUYẾT ĐỊNH
# ============================
print(f"\n🌳 7. Vẽ và lưu cây quyết định")

tree_path = os.path.join('img', 'decision_tree.png')
plt.figure(figsize=(25, 12))
plot_tree(
    best_model,
    feature_names=['AT', 'V', 'AP', 'RH'],
    filled=True,
    rounded=True,
    impurity=True,
    fontsize=8,
    max_depth=3
)
plt.title(f"CÂY QUYẾT ĐỊNH - MÔ HÌNH TỐT NHẤT (Lần {best_model_info['run_id'] + 1})", 
          fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(tree_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Đã lưu: {tree_path}")

# ============================
# BƯỚC 9: LƯU KẾT QUẢ VÀO FILE EXCEL
# ============================
print("\n📊 8. Lưu kết quả vào file Excel")

excel_path = os.path.join('result', 'results_summary.xlsx')

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    
    # Sheet 1: Tổng quan kết quả
    summary_data = {
        'Metric': ['R² Trung bình', 'RMSE Trung bình', 'MAE Trung bình', 'MAPE Trung bình',
                  'R² Tốt nhất', 'Độ lệch chuẩn R²', 'Số lần chạy', 'Mô hình tốt nhất',
                  'Cross-Val R²', 'Cross-Val RMSE'],
        'Giá trị': [f"{test_df['r2'].mean():.4f}", f"{test_df['rmse'].mean():.4f}", 
                   f"{test_df['mae'].mean():.4f}", f"{test_df['mape'].mean():.2f}%",
                   f"{best_model_info['test_r2']:.4f}", f"{test_df['r2'].std():.4f}",
                   '10', f"Lần {best_model_info['run_id'] + 1}",
                   f"{cv_test_r2.mean():.4f}", f"{cv_test_rmse.mean():.4f}"],
        'Đánh giá': [f"{'✅ Tốt' if test_df['r2'].mean() > 0.9 else '⚠️ Khá'}", 
                    f"{'✅ Tốt' if test_df['rmse'].mean() < 5 else '⚠️ Trung bình'}",
                    f"{'✅ Tốt' if test_df['mae'].mean() < 4 else '⚠️ Trung bình'}",
                    f"{'✅ Tốt' if test_df['mape'].mean() < 5 else '⚠️ Khá'}",
                    '🏆 Tốt nhất', f"{'Ổn định' if test_df['r2'].std() < 0.02 else 'Biến động'}",
                    'Đủ', 'Đã chọn',
                    f"{'✅ Tốt' if cv_test_r2.mean() > 0.9 else '⚠️ Khá'}",
                    f"{'✅ Tốt' if cv_test_rmse.mean() < 5 else '⚠️ Trung bình'}"]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Tổng quan', index=False)
    
    # Sheet 2: So sánh mô hình
    model_comparison = {
        'Mô hình': ['Decision Tree', 'Random Forest', 'KNN'],
        'R²': [dt_metrics_best['r2'], rf_metrics['r2'], knn_metrics['r2']],
        'RMSE': [dt_metrics_best['rmse'], rf_metrics['rmse'], knn_metrics['rmse']],
        'MAE': [dt_metrics_best['mae'], rf_metrics['mae'], knn_metrics['mae']],
        'MAPE': [f"{dt_metrics_best['mape']:.2f}%", f"{rf_metrics['mape']:.2f}%", f"{knn_metrics['mape']:.2f}%"],
        'Đánh giá': [f"{'✅ Tốt' if dt_metrics_best['r2'] > 0.9 else '⚠️ Khá'}",
                    f"{'✅ Tốt' if rf_metrics['r2'] > 0.9 else '⚠️ Khá'}",
                    f"{'✅ Tốt' if knn_metrics['r2'] > 0.9 else '⚠️ Khá'}"]
    }
    pd.DataFrame(model_comparison).to_excel(writer, sheet_name='So sánh mô hình', index=False)
    
    # Sheet 3: Feature Importance
    feature_importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
    
    # Sheet 4: Kết quả 10 lần chạy
    detailed_results = test_df.copy()
    detailed_results['Lần chạy'] = range(1, 11)
    detailed_results.to_excel(writer, sheet_name='10 Lần chạy', index=False)
    
    # Sheet 5: Tham số mô hình tốt nhất
    best_params_df = pd.DataFrame([best_model_info['params']])
    best_params_df['Test_R2'] = best_model_info['test_r2']
    best_params_df['Lần_chạy'] = best_model_info['run_id'] + 1
    best_params_df.to_excel(writer, sheet_name='Mô hình tốt nhất', index=False)
    
    # Sheet 6: Cross-validation results
    cv_details = pd.DataFrame({
        'Fold': range(1, 6),
        'Train_R2': cv_train_r2,
        'Test_R2': cv_test_r2,
        'Test_RMSE': cv_test_rmse,
        'Test_MAE': cv_test_mae
    })
    cv_details.to_excel(writer, sheet_name='Cross-Validation', index=False)

print(f"✅ Đã lưu file Excel tổng hợp: {excel_path}")

# ============================
# BƯỚC 10: TỔNG KẾT VÀ KẾT LUẬN
# ============================
print("\n" + "="*60)
print("🎯 TỔNG KẾT KẾT QUẢ")
print("="*60)

# Đánh giá chất lượng tổng thể
avg_test_r2 = test_df['r2'].mean()
std_test_r2 = test_df['r2'].std()

if avg_test_r2 > 0.95 and std_test_r2 < 0.01:
    stability = "RẤT ỔN ĐỊNH VÀ XUẤT SẮC 🏆"
elif avg_test_r2 > 0.9 and std_test_r2 < 0.02:
    stability = "ỔN ĐỊNH VÀ TỐT ✅"
elif avg_test_r2 > 0.85:
    stability = "KHÁ ỔN ĐỊNH 📊"
else:
    stability = "CẦN CẢI THIỆN ⚠️"

print(f"\n📈 KẾT QUẢ TỔNG HỢP:")
print(f"   • Số lần huấn luyện: 10")
print(f"   • Số bộ tham số khác nhau: 10")
print(f"   • Mô hình tốt nhất đạt Test R²: {best_model_info['test_r2']:.4f}")

print(f"\n📊 CHẤT LƯỢNG TRUNG BÌNH (10 lần):")
print(f"   • R² trung bình: {avg_test_r2:.4f} (±{std_test_r2:.4f})")
print(f"   • RMSE trung bình: {test_df['rmse'].mean():.4f} (±{test_df['rmse'].std():.4f})")
print(f"   • MAE trung bình: {test_df['mae'].mean():.4f} (±{test_df['mae'].std():.4f})")
print(f"   • Độ ổn định: {stability}")

print(f"\n🔍 ĐẶC TRƯNG QUAN TRỌNG NHẤT:")
best_feature = feature_importance_df.iloc[0]
print(f"   • {best_feature['Đặc trưng']}: {best_feature['Độ quan trọng trung bình']:.4f} "
      f"(±{best_feature['Độ lệch chuẩn']:.4f})")

print(f"\n⚙️ BỘ THAM SỐ TỐT NHẤT (Lần {best_model_info['run_id'] + 1}):")
for key, value in best_model_info['params'].items():
    print(f"   • {key}: {value if value is not None else 'Không giới hạn'}")

print(f"\n📁 KẾT QUẢ ĐÃ ĐƯỢC LƯU:")
print(f"   • 📊 Ảnh biểu đồ: {len(os.listdir('img'))} file trong thư mục 'img/'")
print(f"   • 💾 Model & Data: {len(os.listdir('result'))} file trong thư mục 'result/'")
print(f"   • 📈 File Excel: result/results_summary.xlsx")
print(f"\n📈 BIỂU ĐỒ SO SÁNH 10 LẦN LẶP ĐÃ ĐƯỢC TẠO:")
print(f"   • {comparison_10_runs_path}")
print(f"   • {detailed_runs_path}")
print(f"   • Tổng cộng: {len(os.listdir('img'))} file ảnh trong thư mục 'img/'")
print(f"\n🎉 HOÀN THÀNH PHÂN TÍCH!")
print("="*60)