import subprocess
import sys

def test_tiktoken_import():
    # main.py 파일을 실행하여 출력 확인
    result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
    
    # tiktoken이 출력에 포함되어 있는지 확인
    assert "tiktoken" in result.stdout