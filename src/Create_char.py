# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
import networkx as nx
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
    
    # Tạo basket matrix với sampling để giảm kích thước
    print("   • Đang tạo basket matrix...")
    bp = BasketPreparer(df_raw)
    basket_bool = bp.encode_basket(threshold=1)
    
    # Lấy mẫu dữ liệu nếu basket quá lớn
    max_transactions = 10000
    if len(basket_bool) > max_transactions:
        print(f"   • Basket quá lớn ({len(basket_bool)} giao dịch), đang lấy mẫu {max_transactions} giao dịch...")
        basket_bool = basket_bool.sample(n=max_transactions, random_state=42)
    
    # Tính trọng số (Monetary) cho mỗi hóa đơn
    df_raw['TotalValue'] = df_raw['Quantity'] * df_raw['UnitPrice']
    invoice_weights = df_raw.groupby('InvoiceNo')['TotalValue'].sum()
    total_revenue = invoice_weights.sum()
    
    print(f"   • Kích thước basket (sau sampling): {basket_bool.shape}")
    print(f"   • Tổng số hóa đơn: {len(basket_bool)}")
    print(f"   • Tổng số sản phẩm: {len(basket_bool.columns)}")
    print(f"   • Tổng doanh thu: £{total_revenue:,.2f}")
    
    # ====================== 2. THỬ NGHIỆM SO SÁNH THUẬT TOÁN ======================
    print("\n🔬 2. Đang thực nghiệm so sánh Apriori vs FP-Growth...")
    print("   (Với các mức min_support khác nhau)")
    
    # Điều chỉnh các tham số thử nghiệm để tránh lỗi bộ nhớ
    support_values = [0.05, 0.04, 0.03, 0.025, 0.02]  # Loại bỏ 0.01 để tránh lỗi
    min_confidence = 0.3
    min_lift = 1.0
    
    # Lưu trữ kết quả thử nghiệm
    experiment_results = []
    
    for min_sup in support_values:
        print(f"\n   • Đang chạy với min_support = {min_sup:.3f}...")
        
        # ---- FP-Growth ----
        try:
            print(f"     - FP-Growth: Đang chạy...")
            fp_start = time.time()
            fp_miner = FPGrowthMiner(basket_bool)
            freq_items_fp = fp_miner.run(min_support=min_sup, use_colnames=True)
            fp_itemset_time = time.time() - fp_start
            
            # Chỉ tạo rules nếu có frequent itemsets
            if len(freq_items_fp) > 0:
                print(f"     - FP-Growth: Tìm thấy {len(freq_items_fp)} itemsets, đang tạo rules...")
                rules_fp = association_rules(freq_items_fp, metric="confidence", 
                                            min_threshold=min_confidence)
                rules_fp = rules_fp[rules_fp['lift'] >= min_lift]
                fp_rule_time = time.time() - fp_start - fp_itemset_time
                fp_total_time = fp_itemset_time + fp_rule_time
            else:
                rules_fp = pd.DataFrame()
                fp_total_time = fp_itemset_time
                print(f"     - FP-Growth: Không tìm thấy frequent itemsets")
            
            fp_time = fp_total_time
            
        except Exception as e:
            print(f"     - FP-Growth: LỖI - {str(e)[:100]}...")
            fp_time = 0
            freq_items_fp = pd.DataFrame()
            rules_fp = pd.DataFrame()
        
        # ---- Apriori ----
        # Với min_support thấp, bỏ qua Apriori để tránh lỗi
        if min_sup >= 0.02:  # Chỉ chạy Apriori với min_support >= 0.02
            try:
                print(f"     - Apriori: Đang chạy...")
                ap_start = time.time()
                ap_miner = AssociationRulesMiner(basket_bool)
                freq_items_ap = ap_miner.mine_frequent_itemsets(min_support=min_sup, 
                                                              use_colnames=True)
                ap_itemset_time = time.time() - ap_start
                
                # Chỉ tạo rules nếu có frequent itemsets
                if len(freq_items_ap) > 0:
                    print(f"     - Apriori: Tìm thấy {len(freq_items_ap)} itemsets, đang tạo rules...")
                    rules_ap = ap_miner.generate_rules(metric="confidence", 
                                                      min_threshold=min_confidence)
                    rules_ap = rules_ap[rules_ap['lift'] >= min_lift]
                    ap_rule_time = time.time() - ap_start - ap_itemset_time
                    ap_total_time = ap_itemset_time + ap_rule_time
                else:
                    rules_ap = pd.DataFrame()
                    ap_total_time = ap_itemset_time
                    print(f"     - Apriori: Không tìm thấy frequent itemsets")
                
                ap_time = ap_total_time
                
            except Exception as e:
                print(f"     - Apriori: LỖI - {str(e)[:100]}...")
                ap_time = 0
                freq_items_ap = pd.DataFrame()
                rules_ap = pd.DataFrame()
        else:
            print(f"     - Apriori: Bỏ qua (min_support quá thấp, dễ gây lỗi bộ nhớ)")
            ap_time = 0
            freq_items_ap = pd.DataFrame()
            rules_ap = pd.DataFrame()
        
        # Tính các chỉ số về itemset
        avg_itemset_length_fp = freq_items_fp['itemsets'].apply(len).mean() if len(freq_items_fp) > 0 else 0
        avg_itemset_length_ap = freq_items_ap['itemsets'].apply(len).mean() if len(freq_items_ap) > 0 else 0
        
        # Tính chất lượng luật trung bình
        avg_confidence_fp = rules_fp['confidence'].mean() if len(rules_fp) > 0 else 0
        avg_lift_fp = rules_fp['lift'].mean() if len(rules_fp) > 0 else 0
        avg_confidence_ap = rules_ap['confidence'].mean() if len(rules_ap) > 0 else 0
        avg_lift_ap = rules_ap['lift'].mean() if len(rules_ap) > 0 else 0
        
        # Lưu kết quả
        experiment_results.append({
            'min_support': min_sup,
            'FP_Time': fp_time,
            'AP_Time': ap_time,
            'FP_FreqItems': len(freq_items_fp),
            'AP_FreqItems': len(freq_items_ap),
            'FP_Rules': len(rules_fp),
            'AP_Rules': len(rules_ap),
            'FP_AvgItemsetLength': avg_itemset_length_fp,
            'AP_AvgItemsetLength': avg_itemset_length_ap,
            'FP_AvgConfidence': avg_confidence_fp,
            'AP_AvgConfidence': avg_confidence_ap,
            'FP_AvgLift': avg_lift_fp,
            'AP_AvgLift': avg_lift_ap,
        })
        
        # Xóa biến để giải phóng bộ nhớ
        del freq_items_fp, rules_fp, freq_items_ap, rules_ap
    
    # Chuyển kết quả thành DataFrame
    df_results = pd.DataFrame(experiment_results)
    
    # ====================== 3. TÍNH TOÁN LUẬT CÓ TRỌNG SỐ ======================
    print("\n⚖️ 3. Đang tính toán luật kết hợp có trọng số...")
    
    # Chọn bộ luật từ FP-Growth với min_support = 0.03 để đảm bảo ổn định
    target_min_support = 0.03
    
    try:
        fp_miner = FPGrowthMiner(basket_bool)
        freq_items_for_weighted = fp_miner.run(min_support=target_min_support, use_colnames=True)
        
        if len(freq_items_for_weighted) > 0:
            rules_for_weighted = association_rules(freq_items_for_weighted, metric="confidence", 
                                                  min_threshold=min_confidence)
            rules_for_weighted = rules_for_weighted[rules_for_weighted['lift'] >= min_lift]
            
            # Tính toán các metrics có trọng số
            w_miner = WeightedAssociationMiner()
            rules_weighted = w_miner.compute_weighted_metrics(rules_for_weighted.copy(), 
                                                             basket_bool, df_raw)
            
            # Thêm cột đọc được cho antecedents và consequents
            rules_weighted['antecedents_str'] = rules_weighted['antecedents'].apply(
                lambda x: ', '.join(sorted(list(x)))
            )
            rules_weighted['consequents_str'] = rules_weighted['consequents'].apply(
                lambda x: ', '.join(sorted(list(x)))
            )
            rules_weighted['rule_str'] = rules_weighted['antecedents_str'] + ' → ' + rules_weighted['consequents_str']
            
            print(f"   • Số luật có trọng số: {len(rules_weighted)}")
            print(f"   • Weighted Support trung bình: {rules_weighted['weighted_support'].mean():.6f}")
            print(f"   • Weighted Confidence trung bình: {rules_weighted['weighted_confidence'].mean():.4f}")
            print(f"   • Weighted Lift trung bình: {rules_weighted['weighted_lift'].mean():.4f}")
            
            # Lưu luật có trọng số vào file
            rules_weighted.to_csv(f"{save_dir}/weighted_association_rules.csv", index=False)
        else:
            rules_weighted = pd.DataFrame()
            print(f"   • Không tìm thấy luật nào với min_support = {target_min_support}")
            
    except Exception as e:
        print(f"   • LỖI khi tính toán luật có trọng số: {str(e)[:100]}...")
        rules_weighted = pd.DataFrame()
    
    # ====================== 4. TẠO THƯ MỤC LƯU KẾT QUẢ ======================
    os.makedirs(save_dir, exist_ok=True)
    
    # ====================== 5. TẠO BIỂU ĐỒ CƠ BẢN ======================
    print("\n🎨 5. Đang tạo biểu đồ trực quan hóa kết quả...")
    
    plt.style.use('seaborn-v0_8')
    
    # ---- BIỂU ĐỒ 1: SO SÁNH THỜI GIAN CHẠY ----
    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
    
    # Lọc các điểm có dữ liệu hợp lệ
    valid_data = df_results[df_results['FP_Time'] > 0]
    
    if len(valid_data) > 0:
        # Biểu đồ thời gian chạy
        axes1[0].plot(valid_data['min_support'], valid_data['FP_Time'], 
                     marker='o', markersize=8, linewidth=2.5, label='FP-Growth')
        axes1[0].plot(valid_data['min_support'], valid_data['AP_Time'], 
                     marker='s', markersize=8, linewidth=2.5, label='Apriori')
        axes1[0].set_xlabel('Min Support Threshold', fontsize=12)
        axes1[0].set_ylabel('Thời gian chạy (giây)', fontsize=12)
        axes1[0].set_title('SO SÁNH THỜI GIAN CHẠY\nFP-Growth vs Apriori', fontsize=14, fontweight='bold')
        axes1[0].legend(fontsize=11)
        axes1[0].grid(True, linestyle='--', alpha=0.7)
        axes1[0].invert_xaxis()
        
        # Biểu đồ số lượng luật
        axes1[1].plot(valid_data['min_support'], valid_data['FP_Rules'], 
                     marker='o', markersize=8, linewidth=2.5, label='FP-Growth')
        axes1[1].plot(valid_data['min_support'], valid_data['AP_Rules'], 
                     marker='s', markersize=8, linewidth=2.5, label='Apriori')
        axes1[1].set_xlabel('Min Support Threshold', fontsize=12)
        axes1[1].set_ylabel('Số lượng luật sinh ra', fontsize=12)
        axes1[1].set_title('SO SÁNH SỐ LƯỢNG LUẬT\nFP-Growth vs Apriori', fontsize=14, fontweight='bold')
        axes1[1].legend(fontsize=11)
        axes1[1].grid(True, linestyle='--', alpha=0.7)
        axes1[1].invert_xaxis()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/1_algorithm_comparison.png", dpi=300, bbox_inches='tight')
        print(f"   • Đã tạo biểu đồ so sánh thuật toán")
    
    # ---- BIỂU ĐỒ 2: SCATTER PLOT ĐƠN GIẢN ----
    if len(rules_weighted) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        
        # Lấy top 50 luật để tránh quá tải
        top_rules = rules_weighted.nlargest(50, 'weighted_lift')
        
        scatter = ax2.scatter(top_rules['weighted_support'], 
                            top_rules['weighted_confidence'],
                            c=top_rules['weighted_lift'],
                            s=50, alpha=0.7, cmap='viridis')
        
        ax2.set_xlabel('Weighted Support', fontsize=12)
        ax2.set_ylabel('Weighted Confidence', fontsize=12)
        ax2.set_title('PHÂN BỐ LUẬT CÓ TRỌNG SỐ\n(Màu sắc thể hiện Lift)', fontsize=14, fontweight='bold')
        
        plt.colorbar(scatter, ax=ax2).set_label('Weighted Lift', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/2_weighted_rules_scatter.png", dpi=300, bbox_inches='tight')
        print(f"   • Đã tạo scatter plot cho luật có trọng số")
    
    # ====================== 6. LƯU KẾT QUẢ VÀ BÁO CÁO ======================
    print("\n💾 6. Đang lưu kết quả phân tích...")
    
    # Lưu kết quả thử nghiệm
    df_results.to_csv(f"{save_dir}/experiment_results.csv", index=False)
    
    # Tạo báo cáo đơn giản
    print("\n" + "="*60)
    print("📋 BÁO CÁO KẾT QUẢ")
    print("="*60)
    
    if len(df_results) > 0:
        print("\nKẾT QUẢ SO SÁNH THUẬT TOÁN:")
        print("-" * 50)
        
        for _, row in df_results.iterrows():
            print(f"\nmin_support = {row['min_support']:.3f}:")
            print(f"  FP-Growth: {row['FP_Time']:.2f}s, {row['FP_Rules']} luật")
            print(f"  Apriori:   {row['AP_Time']:.2f}s, {row['AP_Rules']} luật")
            
            if row['AP_Time'] > 0 and row['FP_Time'] > 0:
                speedup = row['AP_Time'] / row['FP_Time']
                print(f"  → FP-Growth nhanh hơn {speedup:.1f} lần")
    
    print(f"\n✅ HOÀN THÀNH! Kết quả đã được lưu tại: {save_dir}")
    print("="*60)
    
    # Hiển thị biểu đồ
    if 'fig1' in locals():
        plt.show()

if __name__ == "__main__":
    # Đường dẫn đến dữ liệu đã làm sạch
    PATH = "data/processed/cleaned_uk_data.csv"
    
    # Chạy phân tích
    solve_topic_2(PATH, save_dir="reports/topic_2")