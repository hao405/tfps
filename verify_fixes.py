#!/usr/bin/env python3
"""
自动验证所有NaN修复是否已正确应用
"""

import os
import sys
import re

def check_file_fix(filepath, checks):
    """检查文件中的修复"""
    if not os.path.exists(filepath):
        return False, f"文件不存在: {filepath}"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = []
    for check_name, pattern, should_exist in checks:
        found = bool(re.search(pattern, content, re.MULTILINE))
        passed = found == should_exist
        
        status = "✅" if passed else "❌"
        expected = "应该存在" if should_exist else "不应该存在"
        actual = "存在" if found else "不存在"
        
        results.append({
            'passed': passed,
            'name': check_name,
            'status': status,
            'message': f"{status} {check_name}: {expected}, 实际{actual}"
        })
    
    all_passed = all(r['passed'] for r in results)
    return all_passed, results

def main():
    print("="*70)
    print("🔍 NaN修复验证工具")
    print("="*70)
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    all_checks = {
        "Constraint.py": [
            ("设备匹配修复1", r"torch\.zeros\(dim,\s*dim,\s*device=", True),
            ("设备匹配修复2", r"torch\.ones\([^)]+,\s*device=", True),
            ("旧代码已移除", r"torch\.zeros\(dim,\s*dim\)\s*(?!.*device)", False),
        ],
        "Cluster.py": [
            ("L2归一化", r"F\.normalize\(z", True),
            ("Softmax归一化", r"F\.softmax\(s,\s*dim=1\)", True),
            ("温度缩放", r"temperature\s*=\s*0\.1", True),
            ("KL散度epsilon", r"eps\s*=\s*1e-8", True),
            ("KL散度上界", r"torch\.clamp\(kl_loss,\s*max=", True),
        ],
        "InitializeD.py": [
            ("数据中心化", r"data_centered\s*=\s*data\s*-\s*data_mean", True),
            ("数据归一化", r"data_normalized", True),
            ("奇异值筛选", r"threshold\s*=\s*ss\[0\]\s*\*\s*0\.01", True),
            ("QR补全", r"np\.linalg\.qr\(random_complement\)", True),
        ],
        "RevIN.py": [
            ("方差clamp", r"torch\.clamp\(variance.*min=self\.eps\)", True),
            ("标准差clamp", r"safe_stdev\s*=\s*torch\.clamp\(self\.stdev", True),
        ],
        "exp_main.py": [
            ("AdamW优化器", r"optim\.AdamW", True),
            ("log-sum-exp", r"log-sum-exp", True),
            ("诊断函数", r"_diagnose_model_state", True),
            ("NaN检查", r"_check_nan_inf", True),
        ],
    }
    
    total_files = 0
    passed_files = 0
    total_checks = 0
    passed_checks = 0
    
    for filename, checks in all_checks.items():
        filepath = os.path.join(base_path, "layers" if filename in ["Constraint.py", "Cluster.py", "InitializeD.py", "RevIN.py"] else "exp" if filename == "exp_main.py" else "", filename)
        
        print(f"\n📄 检查 {filename}...")
        print("-" * 70)
        
        total_files += 1
        file_passed, results = check_file_fix(filepath, checks)
        
        if isinstance(results, str):
            print(f"  ❌ {results}")
            continue
        
        for result in results:
            print(f"  {result['message']}")
            total_checks += 1
            if result['passed']:
                passed_checks += 1
        
        if file_passed:
            passed_files += 1
            print(f"  ✅ {filename} 所有检查通过")
        else:
            print(f"  ❌ {filename} 部分检查未通过")
    
    # 总结
    print("\n" + "="*70)
    print("📊 验证总结")
    print("="*70)
    print(f"文件: {passed_files}/{total_files} 通过")
    print(f"检查项: {passed_checks}/{total_checks} 通过")
    
    if passed_files == total_files and passed_checks == total_checks:
        print("\n🎉 所有修复已正确应用！可以安全开始训练。")
        print("\n下一步:")
        print("  1. 运行诊断: python diagnose_nan.py")
        print("  2. 开始训练: bash scripts/etth1.sh")
        return 0
    else:
        print("\n⚠️  部分修复未正确应用，请检查上述失败项。")
        print("\n建议:")
        print("  1. 重新应用修复")
        print("  2. 检查文件权限")
        print("  3. 确认使用正确的代码版本")
        return 1

if __name__ == "__main__":
    sys.exit(main())
