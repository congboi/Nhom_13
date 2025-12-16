import papermill as pm
import os
import json
from datetime import datetime

os.makedirs("notebooks/runs", exist_ok=True)

# Tạo thư mục lưu kết quả thí nghiệm
experiment_dir = f"experiments/exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(experiment_dir, exist_ok=True)

print(f"Thư mục thí nghiệm: {experiment_dir}")

# Lưu tham số thí nghiệm
experiment_params = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "parameters": {
        "MIN_SUPPORT": 0.02,
        "FILTER_MIN_SUPPORT": 0.02,
        "FILTER_MIN_CONF": 0.45,
        "FILTER_MIN_LIFT": 1.7,
        "FILTER_MAX_ANTECEDENTS": 2,
        "FILTER_MAX_CONSEQUENTS": 1
    },
    "description": "Thí nghiệm STRICT parameters"
}

with open(f"{experiment_dir}/experiment_config.json", "w") as f:
    json.dump(experiment_params, f, indent=2)

# run_preprocessing_and_eda.py
pm.execute_notebook(
    "notebooks/preprocessing_and_eda.ipynb",
    "notebooks/runs/preprocessing_and_eda_run.ipynb",
    parameters=dict(
        DATA_PATH="data/raw/online_retail.csv",
        COUNTRY="United Kingdom",
        OUTPUT_DIR="data/processed",
        PLOT_REVENUE=False,         # tắt bớt plot khi chạy batch
        PLOT_TIME_PATTERNS=False,
        PLOT_PRODUCTS=False,
        PLOT_CUSTOMERS=False,
        PLOT_RFM=False,
    ),
    kernel_name="python3",
)

# run_basket_preparation.py
pm.execute_notebook(
    "notebooks/basket_preparation.ipynb",
    "notebooks/runs/basket_preparation_run.ipynb",
    parameters=dict(
        CLEANED_DATA_PATH="data/processed/cleaned_uk_data.csv",
        BASKET_BOOL_PATH="data/processed/basket_bool.parquet",
        INVOICE_COL="InvoiceNo",
        ITEM_COL="Description",
        QUANTITY_COL="Quantity",
        THRESHOLD=1,
    ),
    kernel_name="python3",
)

# Chạy Notebook Apriori Modelling - THÊM CODE IN METRICS
pm.execute_notebook(
    "notebooks/apriori_modelling.ipynb",
    f"{experiment_dir}/apriori_strict_results.ipynb",  # Lưu riêng vào thư mục experiment
    parameters=dict(
        BASKET_BOOL_PATH="data/processed/basket_bool.parquet",
        RULES_OUTPUT_PATH=f"{experiment_dir}/rules_strict.csv",  # Lưu rules theo experiment

        # Tham số Apriori
        MIN_SUPPORT=0.02,
        MAX_LEN=3,

        # Generate rules
        METRIC="lift",
        MIN_THRESHOLD=1.0,

        # Lọc luật
        FILTER_MIN_SUPPORT=0.02,
        FILTER_MIN_CONF=0.45,
        FILTER_MIN_LIFT=1.7,
        FILTER_MAX_ANTECEDENTS=2,
        FILTER_MAX_CONSEQUENTS=1,

        # Số luật để vẽ
        TOP_N_RULES=20,

        # Tắt plot khi chạy batch
        PLOT_TOP_LIFT=False,
        PLOT_TOP_CONF=False,
        PLOT_SCATTER=False,
        PLOT_NETWORK=False,
        PLOT_PLOTLY_NETWORK=False,
        PLOT_PLOTLY_SCATTER=False,
        
        # THÊM: Flag để in metrics
        PRINT_METRICS=True,  # Thêm flag mới
        EXPERIMENT_NAME="STRICT"  # Thêm tên experiment
    ),
    kernel_name="python3",
)

# SAU KHI CHẠY XONG - THÊM PHẦN ĐỌC VÀ IN KẾT QUẢ
print("\n" + "="*70)
print("PHÂN TÍCH KẾT QUẢ THÍ NGHIỆM")
print("="*70)

# Đọc rules đã tạo
try:
    import pandas as pd
    rules_df = pd.read_csv(f"{experiment_dir}/rules_strict.csv")
    
    print(f"\n📊 KẾT QUẢ THÍ NGHIỆM: {experiment_params['description']}")
    print(f"📅 Thời gian: {experiment_params['timestamp']}")
    
    print(f"\n✅ Tổng số rules thu được: {len(rules_df):,}")
    
    if not rules_df.empty:
        print(f"\n📈 THỐNG KÊ CHI TIẾT:")
        print(f"   • Support trung bình: {rules_df['support'].mean():.4f}")
        print(f"   • Confidence trung bình: {rules_df['confidence'].mean():.4f}")
        print(f"   • Lift trung bình: {rules_df['lift'].mean():.4f}")
        
        print(f"\n📊 KHOẢNG GIÁ TRỊ:")
        print(f"   • Support: [{rules_df['support'].min():.4f} - {rules_df['support'].max():.4f}]")
        print(f"   • Confidence: [{rules_df['confidence'].min():.4f} - {rules_df['confidence'].max():.4f}]")
        print(f"   • Lift: [{rules_df['lift'].min():.4f} - {rules_df['lift'].max():.4f}]")
        
        print(f"\n🏆 TOP 3 RULES THEO LIFT:")
        top_lift = rules_df.nlargest(3, 'lift')
        for idx, row in top_lift.iterrows():
            print(f"   {idx+1}. {row['antecedents_str']} → {row['consequents_str']}")
            print(f"      Lift: {row['lift']:.4f}, Confidence: {row['confidence']:.4f}, Support: {row['support']:.4f}")
        
        print(f"\n🎯 TOP 3 RULES THEO CONFIDENCE:")
        top_conf = rules_df.nlargest(3, 'confidence')
        for idx, row in top_conf.iterrows():
            print(f"   {idx+1}. {row['antecedents_str']} → {row['consequents_str']}")
            print(f"      Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}, Support: {row['support']:.4f}")
        
        # Lưu summary
        summary = {
            "total_rules": int(len(rules_df)),
            "avg_support": float(rules_df['support'].mean()),
            "avg_confidence": float(rules_df['confidence'].mean()),
            "avg_lift": float(rules_df['lift'].mean()),
            "min_support": float(rules_df['support'].min()),
            "max_support": float(rules_df['support'].max()),
            "min_confidence": float(rules_df['confidence'].min()),
            "max_confidence": float(rules_df['confidence'].max()),
            "min_lift": float(rules_df['lift'].min()),
            "max_lift": float(rules_df['lift'].max()),
            "top_rules_lift": [
                {
                    "antecedents": row['antecedents_str'],
                    "consequents": row['consequents_str'],
                    "lift": float(row['lift']),
                    "confidence": float(row['confidence'])
                }
                for _, row in top_lift.iterrows()
            ]
        }
        
        with open(f"{experiment_dir}/experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n📁 Kết quả đã được lưu tại: {experiment_dir}/")
        print(f"   • rules_strict.csv - Toàn bộ rules")
        print(f"   • experiment_config.json - Cấu hình thí nghiệm")
        print(f"   • experiment_summary.json - Tóm tắt kết quả")
        print(f"   • apriori_strict_results.ipynb - Notebook kết quả")
        
    else:
        print("\n⚠️ Không có rules nào thỏa mãn điều kiện!")
        summary = {"total_rules": 0, "message": "No rules found with given parameters"}
        with open(f"{experiment_dir}/experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
except FileNotFoundError:
    print(f"\n❌ Không tìm thấy file rules tại: {experiment_dir}/rules_strict.csv")
except Exception as e:
    print(f"\n❌ Lỗi khi đọc kết quả: {str(e)}")

print("\n" + "="*70)
print("ĐÃ CHẠY XONG PIPELINE")
print("="*70)