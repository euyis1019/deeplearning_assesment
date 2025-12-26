#!/usr/bin/env python
"""
验证 Kaggle 提交文件格式

Usage:
    python validate_submission.py submissions/submission.csv
"""

import sys
import pandas as pd
import os


def validate_submission(submission_path: str, test_csv_path: str = 'data/test.csv'):
    """验证提交文件格式"""
    
    print("=" * 60)
    print("🔍 Validating Submission File")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(submission_path):
        print(f"❌ Error: Submission file not found: {submission_path}")
        return False
    
    if not os.path.exists(test_csv_path):
        print(f"⚠️  Warning: Test CSV not found: {test_csv_path}")
        print("   Skipping ID matching check")
        test_df = None
    else:
        test_df = pd.read_csv(test_csv_path)
    
    # 读取提交文件
    try:
        submission_df = pd.read_csv(submission_path)
    except Exception as e:
        print(f"❌ Error: Failed to read CSV file: {e}")
        return False
    
    # 检查列名
    print("\n📋 Checking columns...")
    expected_columns = ['id', 'target']
    if list(submission_df.columns) != expected_columns:
        print(f"❌ Error: Column names don't match!")
        print(f"   Expected: {expected_columns}")
        print(f"   Got: {list(submission_df.columns)}")
        return False
    print("   ✅ Column names are correct")
    
    # 检查数据类型
    print("\n🔢 Checking data types...")
    if submission_df['id'].dtype not in [int, 'int64']:
        print(f"⚠️  Warning: 'id' column is not integer type: {submission_df['id'].dtype}")
    else:
        print("   ✅ 'id' column type is correct")
    
    if submission_df['target'].dtype not in [int, 'int64']:
        print(f"⚠️  Warning: 'target' column is not integer type: {submission_df['target'].dtype}")
        print("   Attempting to convert...")
        try:
            submission_df['target'] = submission_df['target'].astype(int)
            print("   ✅ Converted to integer")
        except:
            print("   ❌ Failed to convert to integer")
            return False
    else:
        print("   ✅ 'target' column type is correct")
    
    # 检查行数
    print("\n📊 Checking row count...")
    if test_df is not None:
        expected_rows = len(test_df)
        actual_rows = len(submission_df)
        if actual_rows != expected_rows:
            print(f"❌ Error: Row count mismatch!")
            print(f"   Expected: {expected_rows} (test set size)")
            print(f"   Got: {actual_rows}")
            return False
        print(f"   ✅ Row count matches: {actual_rows}")
    else:
        print(f"   📝 Submission has {len(submission_df)} rows")
    
    # 检查 ID 匹配
    if test_df is not None:
        print("\n🔗 Checking ID matching...")
        test_ids = set(test_df['id'])
        submission_ids = set(submission_df['id'])
        
        missing_ids = test_ids - submission_ids
        extra_ids = submission_ids - test_ids
        
        if missing_ids:
            print(f"❌ Error: Missing IDs in submission: {len(missing_ids)}")
            print(f"   First 10 missing IDs: {list(missing_ids)[:10]}")
            return False
        
        if extra_ids:
            print(f"⚠️  Warning: Extra IDs in submission: {len(extra_ids)}")
            print(f"   First 10 extra IDs: {list(extra_ids)[:10]}")
        else:
            print("   ✅ All IDs match")
    
    # 检查目标值范围
    print("\n🎯 Checking target values...")
    unique_values = submission_df['target'].unique()
    if not all(val in [0, 1] for val in unique_values):
        invalid_values = [v for v in unique_values if v not in [0, 1]]
        print(f"❌ Error: Invalid target values found!")
        print(f"   Invalid values: {invalid_values}")
        print(f"   Target values must be 0 or 1")
        return False
    print(f"   ✅ Target values are valid: {unique_values}")
    
    # 检查 NaN 值
    print("\n🔍 Checking for NaN values...")
    nan_count = submission_df.isna().sum().sum()
    if nan_count > 0:
        print(f"❌ Error: Found {nan_count} NaN values!")
        print(submission_df.isna().sum())
        return False
    print("   ✅ No NaN values found")
    
    # 显示统计信息
    print("\n📈 Submission Statistics:")
    print(f"   Total predictions: {len(submission_df)}")
    print(f"   Predicted as disaster (1): {submission_df['target'].sum()}")
    print(f"   Predicted as not disaster (0): {len(submission_df) - submission_df['target'].sum()}")
    disaster_ratio = submission_df['target'].mean()
    print(f"   Disaster ratio: {disaster_ratio:.2%}")
    
    # 检查分布是否合理
    if disaster_ratio < 0.01 or disaster_ratio > 0.99:
        print(f"\n⚠️  Warning: Disaster ratio is very extreme ({disaster_ratio:.2%})")
        print("   This might indicate a problem with the model")
    
    print("\n" + "=" * 60)
    print("✅ Submission file is valid!")
    print("=" * 60)
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <submission_file> [test_file]")
        print("\nExample:")
        print("  python validate_submission.py submissions/submission.csv")
        print("  python validate_submission.py submissions/submission.csv data/test.csv")
        sys.exit(1)
    
    submission_path = sys.argv[1]
    test_path = sys.argv[2] if len(sys.argv) > 2 else 'data/test.csv'
    
    is_valid = validate_submission(submission_path, test_path)
    
    if not is_valid:
        print("\n❌ Validation failed. Please fix the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Ready to submit to Kaggle!")
        print(f"\nTo submit via API:")
        print(f"  kaggle competitions submit -c nlp-getting-started -f {submission_path} -m \"Your message\"")


if __name__ == '__main__':
    main()





