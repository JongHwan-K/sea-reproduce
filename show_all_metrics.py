#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEA 모델 추론 결과의 모든 메트릭 출력
"""

import pickle
import numpy as np

def show_all_metrics():
    """모든 메트릭을 출력합니다."""
    
    # 결과 파일 로드
    try:
        with open('results_fci_sergio_reproduce.pkl', 'rb') as f:
            results = pickle.load(f)
        print(f"✅ 결과 파일 로드 성공")
        print(f"📊 총 데이터셋 수: {len(results)}")
        print("="*80)
    except FileNotFoundError:
        print("❌ results_fci_sergio_reproduce.pkl 파일을 찾을 수 없습니다.")
        return
    
    # 모든 데이터셋의 모든 메트릭 출력
    all_auc = []
    all_prc = []
    all_time = []
    
    print("📋 각 데이터셋별 상세 메트릭:")
    print("="*80)
    
    for i, (key, result) in enumerate(results.items()):
        print(f"\n🔍 데이터셋 {i+1}: {key}")
        print("-" * 60)
        
        # 각 데이터셋의 개별 메트릭
        for j in range(len(result['auc'])):
            auc_val = result['auc'][j]
            prc_val = result['prc'][j]
            time_val = result['time'][j].item() if hasattr(result['time'][j], 'item') else result['time'][j]
            
            print(f"  샘플 {j+1}: AUC={auc_val:.4f}, AUPRC={prc_val:.4f}, 시간={time_val:.4f}초")
            
            # 전체 통계용 리스트에 추가
            all_auc.append(auc_val)
            all_prc.append(prc_val)
            all_time.append(time_val)
        
        # 각 데이터셋의 요약 통계
        dataset_auc_mean = np.mean(result['auc'])
        dataset_auc_std = np.std(result['auc'])
        dataset_prc_mean = np.mean(result['prc'])
        dataset_prc_std = np.std(result['prc'])
        dataset_time_mean = np.mean([t.item() if hasattr(t, 'item') else t for t in result['time']])
        dataset_time_std = np.std([t.item() if hasattr(t, 'item') else t for t in result['time']])
        
        print(f"  📊 데이터셋 요약: AUC={dataset_auc_mean:.4f}±{dataset_auc_std:.4f}, "
              f"AUPRC={dataset_prc_mean:.4f}±{dataset_prc_std:.4f}, "
              f"시간={dataset_time_mean:.4f}±{dataset_time_std:.4f}초")
    
    # 전체 통계
    print("\n" + "="*80)
    print("🎯 전체 성능 요약")
    print("="*80)
    print(f"총 데이터셋 수: {len(results)}")
    print(f"총 샘플 수: {len(all_auc)}")
    print(f"평균 AUC: {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
    print(f"평균 AUPRC: {np.mean(all_prc):.4f} ± {np.std(all_prc):.4f}")
    print(f"평균 추론 시간: {np.mean(all_time):.4f}초 ± {np.std(all_time):.4f}초")
    print(f"총 추론 시간: {np.sum(all_time):.2f}초")
    print(f"최고 AUC: {np.max(all_auc):.4f}")
    print(f"최고 AUPRC: {np.max(all_prc):.4f}")
    print(f"최저 AUC: {np.min(all_auc):.4f}")
    print(f"최저 AUPRC: {np.min(all_prc):.4f}")
    print(f"가장 빠른 추론: {np.min(all_time):.4f}초")
    print(f"가장 느린 추론: {np.max(all_time):.4f}초")
    
    # 성능 등급 분류
    excellent_count = sum(1 for auc, prc in zip(all_auc, all_prc) if auc >= 0.9 and prc >= 0.7)
    good_count = sum(1 for auc, prc in zip(all_auc, all_prc) if auc >= 0.8 and prc >= 0.6 and not (auc >= 0.9 and prc >= 0.7))
    fair_count = sum(1 for auc, prc in zip(all_auc, all_prc) if auc >= 0.7 and prc >= 0.5 and not (auc >= 0.8 and prc >= 0.6))
    poor_count = len(all_auc) - excellent_count - good_count - fair_count

if __name__ == "__main__":
    show_all_metrics()
