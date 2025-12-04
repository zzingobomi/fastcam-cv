import torch
from lightglue import LightGlue, SuperPoint

print("=" * 50)
print("LightGlue Import Test")
print("=" * 50)

# 1. 기본 import 확인
print("\n✓ Import successful!")
print(f"LightGlue: {LightGlue}")
print(f"SuperPoint: {SuperPoint}")

# 2. 모델 초기화 테스트
print("\n" + "=" * 50)
print("Model Initialization Test")
print("=" * 50)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

try:    
    # SuperPoint 특징점 추출기 초기화
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    print("✓ SuperPoint initialized successfully")
    
    # LightGlue 매칭기 초기화
    matcher = LightGlue(features='superpoint').eval().to(device)
    print("✓ LightGlue initialized successfully")
    
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    exit(1)

# 3. 간단한 더미 데이터로 실행 테스트
print("\n" + "=" * 50)
print("Forward Pass Test")
print("=" * 50)

try:
    # 더미 이미지 생성 (배치 크기 1, 그레이스케일, 320x240)
    dummy_image0 = torch.randn(1, 1, 240, 320).to(device)
    dummy_image1 = torch.randn(1, 1, 240, 320).to(device)
    
    # 특징점 추출
    with torch.no_grad():
        feats0 = extractor.extract(dummy_image0)
        feats1 = extractor.extract(dummy_image1)
        
        print(f"✓ Feature extraction successful")
        print(f"  - Image 0: {feats0['keypoints'].shape[1]} keypoints detected")
        print(f"  - Image 1: {feats1['keypoints'].shape[1]} keypoints detected")
        
        # 매칭 수행
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        
        # matches는 리스트일 수도 있고 텐서일 수도 있음
        matches = matches01['matches']
        if isinstance(matches, list):
            matches = matches[0]  # 배치의 첫 번째 요소
        
        num_matches = matches.shape[0]  # matches는 [N, 2] 형태
        print(f"✓ Matching successful")
        print(f"  - {num_matches} matches found")
        
        # 매칭 결과 상세 정보
        print(f"\nMatch details:")
        print(f"  - matches shape: {matches.shape}")
        print(f"  - matches type: {type(matches)}")
        if 'matching_scores0' in matches01:
            scores = matches01['matching_scores0']
            if isinstance(scores, list):
                scores = scores[0]
            print(f"  - confidence scores available: {scores.shape}")
        
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 50)
print("🎉 All tests passed! LightGlue is working correctly.")
print("=" * 50)