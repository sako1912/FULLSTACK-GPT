import subprocess
import sys
import importlib

def test_tiktoken_import():
    # main.py 파일을 실행하여 출력 확인
    result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
    # tiktoken이 출력에 포함되어 있는지 확인
    assert "tiktoken" in result.stdout

def test_direct_tiktoken_import():
    # tiktoken을 직접 임포트 시도
    try:
        importlib.import_module('tiktoken')
        assert True
    except ImportError:
        assert False, "tiktoken 모듈을 임포트할 수 없습니다."