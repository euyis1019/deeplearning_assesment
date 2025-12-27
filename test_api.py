#!/usr/bin/env python3
"""
测试 DeepWisdom API 连接
"""

import requests
import json

# API 配置
API_URL = "https://newapi.deepwisdom.ai/v1/chat/completions"
API_KEY = "sk-DUuJeAxX6fwNViL2JHRIHj4SI9OEcDlr4TMctin2DLxvuFY8"
MODEL = "deepseek-v3"

def test_api():
    """测试 API 连接"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "你好，请简单介绍一下你自己。"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    print("=" * 60)
    print("测试 DeepWisdom API")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print(f"Model: {MODEL}")
    print(f"Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\n发送请求...")
    print("-" * 60)
    
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        print("\n响应内容:")
        print("-" * 60)
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 提取回复内容
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {}).get("content", "")
                print("\n" + "=" * 60)
                print("AI 回复:")
                print("=" * 60)
                print(message)
            
            # 显示使用情况
            if "usage" in result:
                print("\n" + "=" * 60)
                print("Token 使用情况:")
                print("=" * 60)
                print(json.dumps(result["usage"], indent=2, ensure_ascii=False))
            
            print("\n✅ API 测试成功！")
            return True
        else:
            print(f"❌ API 请求失败")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 连接错误，请检查网络连接")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {str(e)}")
        return False
    except json.JSONDecodeError:
        print(f"❌ 响应不是有效的 JSON")
        print(f"原始响应: {response.text}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")
        return False

if __name__ == "__main__":
    test_api()

