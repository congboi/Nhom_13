# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
from apriori_library import BasketPreparer, AssociationRulesMiner, FPGrowthMiner, WeightedAssociationMiner
from mlxtend.frequent_patterns import fpgrowth, association_rules

warnings.filterwarnings('ignore')

def solve_topic_2(input_csv, save_dir="reports/topic_2"):
    """
    Hàm chính thực hiện chủ đề 2: So sánh Apriori vs FP-Growth và phân tích luật có trọng số.
    """
    print("🚀 Đang khởi động phân tích Chủ đề 2...")
    print("=" * 60)
    
    # ====================== 1. CHUẨN BỊ DỮ LIỆU ======================
    print("📂 1. Đang tải và chuẩn bị dữ liệu...")
    try:
        df_raw = pd.read_csv(input_csv, low_memory=False)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {input_csv}")
        print("⚠️  Vui lòng kiểm tra đường dẫn file.")
        return
    
    # Tạo basket matrix
    print("   • Đang tạo basket matrix...")
    bp = BasketPreparer(df_raw)
    basket_bool = bp.encode_basket(threshold=1)
    
    # Tính trọng số (Monetary) cho mỗi hóa đơn
    df_raw['TotalValue'] = df_raw['Quantity'] * df_raw['UnitPrice']
    invoice_weights = df_raw.groupby('InvoiceNo')['TotalValue'].sum()
    total_revenue = invoice_weights.sum()
    
    print(f"   • Kích thước basket: {basket_bool.shape}")
    print(f"   • Tổng số hóa đơn: {len(basket_bool)}")
    print(f"   • Tổng số sản phẩm: {len(basket_bool.columns)}")
    print(f"   • Tổng doanh thu: £{total_revenue:,.2f}")
    
    # ====================== 2. PHÂN TÍCH HUB SẢN PHẨM (QUAN TRỌNG) ======================
    print("\n📊 2. Đang phân tích Hub sản phẩm theo tần suất và giá trị...")
    
    # Tính weights vector
    weights_v = basket_bool.index.map(invoice_weights).fillna(0).values
    
    # Phân tích từng sản phẩm
    hub_list = []
    for prod in basket_bool.columns:
        mask = basket_bool[prod].values == 1
        freq = mask.mean()  # Tần suất (Support thường)
        if mask.any():
            val = weights_v[mask].sum() / total_revenue  # Weighted support
        else:
            val = 0.0
        
        hub_list.append({
            'Product': prod, 
            'Frequency': freq, 
            'Value': val
        })
    
    df_hub = pd.DataFrame(hub_list)
    
    # ====================== 3. TRỰC QUAN HÓA HUB SẢN PHẨM ======================
    print("\n🎨 3. Đang tạo biểu đồ Hub sản phẩm...")
    
    # Tạo thư mục lưu kết quả
    os.makedirs(save_dir, exist_ok=True)
    
    # Thiết lập style cho biểu đồ
    plt.style.use('seaborn-v0_8')
    sns.set_palette("viridis")
    
    # ---- BIỂU ĐỒ HUB SẢN PHẨM ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Top 10 Hub theo tần suất
    top_f = df_hub.sort_values('Frequency', ascending=False).head(10)
    if not top_f.empty:
        # Tạo barplot với palette 'viridis'
        bars1 = sns.barplot(data=top_f, x='Frequency', y='Product', ax=ax1, palette='viridis')
        ax1.set_title('TOP 10 HUB THEO TẦN SUẤT\n(Sản phẩm xuất hiện nhiều nhất)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Frequency (Support)', fontsize=12)
        ax1.set_ylabel('Sản phẩm', fontsize=12)
        
        # Điều chỉnh font size cho tên sản phẩm
        ax1.tick_params(axis='y', labelsize=10)
        
        # Thêm giá trị trên các cột
        for i, (freq, product) in enumerate(zip(top_f['Frequency'], top_f['Product'])):
            ax1.text(freq + 0.001, i, f'{freq:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # Top 10 Hub theo giá trị
    top_v = df_hub.sort_values('Value', ascending=False).head(10)
    if not top_v.empty:
        # Tạo barplot với palette 'magma'
        bars2 = sns.barplot(data=top_v, x='Value', y='Product', ax=ax2, palette='magma')
        ax2.set_title('TOP 10 HUB THEO GIÁ TRỊ\n(Sản phẩm đóng góp doanh thu lớn nhất)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Value (Weighted Support)', fontsize=12)
        ax2.set_ylabel('Sản phẩm', fontsize=12)
        
        # Điều chỉnh font size cho tên sản phẩm
        ax2.tick_params(axis='y', labelsize=10)
        
        # Thêm giá trị trên các cột
        for i, (val, product) in enumerate(zip(top_v['Value'], top_v['Product'])):
            ax2.text(val + 0.00001, i, f'{val:.5f}', va='center', fontsize=10, fontweight='bold')
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu biểu đồ
    plt.savefig(f"{save_dir}/hub_comparison_report.png", dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ Hub tại: {save_dir}/hub_comparison_report.png")
    
    # Hiển thị biểu đồ
    plt.show()
    
    # ====================== 4. THỬ NGHIỆM SO SÁNH THUẬT TOÁN (ĐƠN GIẢN) ======================
    print("\n🔬 4. Đang thực nghiệm so sánh Apriori vs FP-Growth...")
    
    # Chỉ chạy với 2 giá trị min_support để tránh lỗi
    support_values = [0.05, 0.03]
    min_confidence = 0.3
    
    experiment_results = []
    
    for min_sup in support_values:
        print(f"   • Đang chạy với min_support = {min_sup:.3f}...")
        
        # FP-Growth
        try:
            fp_start = time.time()
            fp_miner = FPGrowthMiner(basket_bool)
            freq_items_fp = fp_miner.run(min_support=min_sup, use_colnames=True)
            
            if len(freq_items_fp) > 0:
                rules_fp = association_rules(freq_items_fp, metric="confidence", 
                                            min_threshold=min_confidence)
                rules_fp = rules_fp[rules_fp['lift'] >= 1.0]
            else:
                rules_fp = pd.DataFrame()
            
            fp_time = time.time() - fp_start
            
        except Exception as e:
            print(f"     - FP-Growth lỗi: {str(e)[:50]}...")
            fp_time = 0
            rules_fp = pd.DataFrame()
        
        # Apriori (chỉ chạy với min_sup >= 0.03)
        if min_sup >= 0.03:
            try:
                ap_start = time.time()
                ap_miner = AssociationRulesMiner(basket_bool)
                freq_items_ap = ap_miner.mine_frequent_itemsets(min_support=min_sup, 
                                                              use_colnames=True)
                
                if len(freq_items_ap) > 0:
                    rules_ap = ap_miner.generate_rules(metric="confidence", 
                                                      min_threshold=min_confidence)
                    rules_ap = rules_ap[rules_ap['lift'] >= 1.0]
                else:
                    rules_ap = pd.DataFrame()
                
                ap_time = time.time() - ap_start
                
            except Exception as e:
                print(f"     - Apriori lỗi: {str(e)[:50]}...")
                ap_time = 0
                rules_ap = pd.DataFrame()
        else:
            ap_time = 0
            rules_ap = pd.DataFrame()
        
        experiment_results.append({
            'min_support': min_sup,
            'FP_Time': fp_time,
            'AP_Time': ap_time,
            'FP_Rules': len(rules_fp),
            'AP_Rules': len(rules_ap),
        })
    
    df_results = pd.DataFrame(experiment_results)
    
    # ====================== 5. TÍNH TOÁN LUẬT CÓ TRỌNG SỐ ======================
    print("\n⚖️ 5. Đang tính toán luật kết hợp có trọng số...")
    
    # Chọn min_support = 0.03 để đảm bảo có kết quả
    target_min_support = 0.03
    
    try:
        fp_miner = FPGrowthMiner(basket_bool)
        freq_items = fp_miner.run(min_support=target_min_support, use_colnames=True)
        
        if len(freq_items) > 0:
            rules = association_rules(freq_items, metric="confidence", 
                                     min_threshold=min_confidence)
            rules = rules[rules['lift'] >= 1.0]
            
            # Tính toán các metrics có trọng số
            w_miner = WeightedAssociationMiner()
            rules_weighted = w_miner.compute_weighted_metrics(rules.copy(), 
                                                             basket_bool, df_raw)
            
            # Thêm cột đọc được
            rules_weighted['antecedents_str'] = rules_weighted['antecedents'].apply(
                lambda x: ', '.join(sorted(list(x)))
            )
            rules_weighted['consequents_str'] = rules_weighted['consequents'].apply(
                lambda x: ', '.join(sorted(list(x)))
            )
            rules_weighted['rule_str'] = rules_weighted['antecedents_str'] + ' → ' + rules_weighted['consequents_str']
            
            print(f"   • Số luật có trọng số: {len(rules_weighted)}")
            
            # Lưu kết quả
            rules_weighted.to_csv(f"{save_dir}/weighted_association_rules.csv", index=False)
        else:
            rules_weighted = pd.DataFrame()
            print(f"   • Không tìm thấy luật nào với min_support = {target_min_support}")
            
    except Exception as e:
        print(f"   • Lỗi khi tính toán luật có trọng số: {str(e)[:100]}...")
        rules_weighted = pd.DataFrame()
    
    # ====================== 6. TẠO BIỂU ĐỒ SO SÁNH ======================
    print("\n📈 6. Đang tạo biểu đồ so sánh thuật toán...")
    
    if len(df_results) > 0 and df_results['FP_Time'].sum() > 0:
        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
        
        # Biểu đồ thời gian chạy
        valid_data = df_results[df_results['FP_Time'] > 0]
        
        if len(valid_data) > 0:
            axes2[0].plot(valid_data['min_support'], valid_data['FP_Time'], 
                         marker='o', markersize=8, linewidth=2.5, label='FP-Growth')
            axes2[0].plot(valid_data['min_support'], valid_data['AP_Time'], 
                         marker='s', markersize=8, linewidth=2.5, label='Apriori')
            axes2[0].set_xlabel('Min Support Threshold', fontsize=12)
            axes2[0].set_ylabel('Thời gian chạy (giây)', fontsize=12)
            axes2[0].set_title('SO SÁNH THỜI GIAN CHẠY\nFP-Growth vs Apriori', 
                             fontsize=14, fontweight='bold')
            axes2[0].legend(fontsize=11)
            axes2[0].grid(True, linestyle='--', alpha=0.7)
            axes2[0].invert_xaxis()
        
        # Biểu đồ số lượng luật
        axes2[1].plot(valid_data['min_support'], valid_data['FP_Rules'], 
                     marker='o', markersize=8, linewidth=2.5, label='FP-Growth')
        axes2[1].plot(valid_data['min_support'], valid_data['AP_Rules'], 
                     marker='s', markersize=8, linewidth=2.5, label='Apriori')
        axes2[1].set_xlabel('Min Support Threshold', fontsize=12)
        axes2[1].set_ylabel('Số lượng luật sinh ra', fontsize=12)
        axes2[1].set_title('SO SÁNH SỐ LƯỢNG LUẬT\nFP-Growth vs Apriori', 
                         fontsize=14, fontweight='bold')
        axes2[1].legend(fontsize=11)
        axes2[1].grid(True, linestyle='--', alpha=0.7)
        axes2[1].invert_xaxis()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/algorithm_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✅ Đã lưu biểu đồ so sánh thuật toán")
    
    # ====================== 7. LƯU KẾT QUẢ VÀ BÁO CÁO ======================
    print("\n💾 7. Đang lưu kết quả phân tích...")
    
    # Lưu kết quả thử nghiệm
    df_results.to_csv(f"{save_dir}/experiment_results.csv", index=False)
    
    # Lưu kết quả hub sản phẩm
    df_hub.to_csv(f"{save_dir}/product_hub_analysis.csv", index=False)
    
    # Tạo báo cáo
    print("\n" + "="*60)
    print("📋 BÁO CÁO KẾT QUẢ")
    print("="*60)
    
    # Hiển thị top sản phẩm
    print("\nTOP 5 SẢN PHẨM THEO TẦN SUẤT:")
    print("-" * 40)
    for i, (_, row) in enumerate(df_hub.nlargest(5, 'Frequency').iterrows(), 1):
        print(f"{i}. {row['Product'][:50]}... - Frequency: {row['Frequency']:.4f}")
    
    print("\nTOP 5 SẢN PHẨM THEO GIÁ TRỊ:")
    print("-" * 40)
    for i, (_, row) in enumerate(df_hub.nlargest(5, 'Value').iterrows(), 1):
        print(f"{i}. {row['Product'][:50]}... - Value: {row['Value']:.6f}")
    
    print(f"\n✅ HOÀN THÀNH PHÂN TÍCH!")
    print(f"📁 Tất cả kết quả đã được lưu tại: {save_dir}")
    print("="*60)

if __name__ == "__main__":
    # Đường dẫn đến dữ liệu đã làm sạch
    PATH = "data/processed/cleaned_uk_data.csv"
    
    # Chạy phân tích
    solve_topic_2(PATH, save_dir="reports/topic_2")