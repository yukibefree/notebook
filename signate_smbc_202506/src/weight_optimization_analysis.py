#!/usr/bin/env python3
"""
重み最適化分析スクリプト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from .ensemble import optimize_weights_from_history, suggest_optimal_weights, analyze_weight_performance

def main():
    # ユーザーのデータ
    current_results = {
        '0.2:0.8': 8.0622387,
        '0.1:0.9': 8.2409318,
        '0.3:0.7': 8.0720854,
        '0.4:0.6': 8.2697982,
        '0.25:0.75': 8.0433271,
        '0.22:0.78': 8.0489563,
        '0.237:0.763': 8.0436570,
        #'0.227:0.763:0.01': 8.1440280,
        #'0.05:0.9:0.05': 8.7221483,
        #'0.23:0.75:0.02': 8.2714483,
    }
    
    print("=== 重み最適化分析 ===\n")
    
    # 1. 性能分析
    print("1. 性能分析")
    analysis = analyze_weight_performance(current_results)
    
    print(f"最良スコア: {analysis['best_score']:.7f} (重み: {analysis['best_weight']})")
    print(f"最悪スコア: {analysis['worst_score']:.7f} (重み: {analysis['worst_weight']})")
    print(f"スコア範囲: {analysis['score_range']:.7f}")
    print()
    
    print("傾向分析:")
    print(f"  model1重み高 (0.3以上): {analysis['trends']['model1_high_performance']:.7f}")
    print(f"  model2重み高 (0.7以上): {analysis['trends']['model2_high_performance']:.7f}")
    print(f"  バランス型 (0.2-0.3): {analysis['trends']['balanced_performance']:.7f}")
    print()
    
    print("推奨事項:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")
    print()
    
    # 2. 最適化実行
    print("2. 最適化結果")
    lb_scores = list(current_results.values())
    weights_history = []
    for weight_ratio in current_results.keys():
        w1, w2 = map(float, weight_ratio.split(':'))
        weights_history.append((w1, w2))
    
    optimization_result = optimize_weights_from_history(lb_scores, weights_history)
    
    print("履歴最良:")
    print(f"  スコア: {optimization_result['best_from_history']['score']:.7f}")
    print(f"  重み: {optimization_result['best_from_history']['weight_ratio']}")
    print()
    
    if 'grid_search' in optimization_result:
        print("グリッドサーチ最適化:")
        print(f"  予測スコア: {optimization_result['grid_search']['predicted_score']:.7f}")
        print(f"  推奨重み: {optimization_result['grid_search']['weight_ratio']}")
        print()
    
    # 3. 次に試すべき重みの提案
    print("3. 次に試すべき重みの組み合わせ")
    suggestions = suggest_optimal_weights(current_results, n_models=2)
    
    for i, (w1, w2) in enumerate(suggestions, 1):
        print(f"  提案{i}: {w1:.3f}:{w2:.3f}")
    
    print()
    
    # 4. 詳細な推奨
    print("4. 詳細推奨")
    print("現在のデータから以下の戦略を推奨します:")
    print()
    print("【即座に試すべき】")
    print("1. 0.15:0.85 - model2をさらに重視")
    print("2. 0.10:0.90 - model2をさらに重視") 
    print("3. 0.05:0.95 - model2をさらに重視")
    print()
    print("【微調整】")
    print("4. 0.20:0.80 - 現在の2番目に良い組み合わせの周辺")
    print("5. 0.30:0.70 - 現在の3番目に良い組み合わせの周辺")
    print()
    print("【注意点】")
    print("- model1の重みが0.4を超えるとスコアが悪化する傾向")
    print("- model2の重みを0.8以上にすると良い結果が期待できる")
    print("- 重みの影響範囲は約0.23ポイント（8.04 → 8.27）")

if __name__ == "__main__":
    main() 