from pathlib import Path

def get_project_root() -> Path:
    """프로젝트 루트 디렉토리를 반환"""
    # pyproject.toml, .git, setup.py 등의 마커 파일 찾기
    current = Path(__file__).resolve().parent
    markers = ['pyproject.toml', '.git', 'setup.py', 'requirements.txt']
    
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    # 마커를 찾지 못한 경우 현재 파일의 부모 디렉토리 반환
    return current

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"