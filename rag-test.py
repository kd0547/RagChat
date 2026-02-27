import os
os.environ['USERPROFILE'] = 'C:/AI'
os.environ['HOME'] = 'C:/AI'
import fitz  # PyMuPDF (PDF를 이미지로 변환해주는 라이브러리)
import paddle
import cv2

print("Device available:", paddle.device.get_device())

from paddleocr import PPStructureV3
from pathlib import Path

# 1. 꼬였던 파라미터 다 빼고 순수 기본 상태로 호출합니다. (이제 안 꼬이니까요!)
pipeline = PPStructureV3(
    lang="korean",

    use_doc_orientation_classify=False,  # 자동 회전 끄기
    use_doc_unwarping=False,  # 구겨짐 자동 펴기 끄기
   # 이제 밀림 현상이 없으니, 박스 팽창률은 기본으로 돌려놓습니다.
    layout_unclip_ratio=1.1,
    # 텍스트 인식 해상도는 짱짱하게 유지

    text_det_limit_side_len=2048
)

pdf_path = "./Dtx400 Desktop Meeting.pdf"
image_path = "./test_page_0.jpg"

# 2. PDF의 첫 페이지를 고해상도 이미지(JPG)로 강제 변환합니다.
print("PDF를 이미지로 변환 중...")
doc = fitz.open(pdf_path)
page = doc[0]  # 첫 번째 페이지
pix = page.get_pixmap(dpi=200) # 고해상도 렌더링
pix.save(image_path)
doc.close()

# 3. 파이프라인에 PDF 파일 대신 '변환된 JPG 이미지'를 넣습니다!
print("이미지로 OCR 분석 시작...")
output = pipeline.predict(input=pdf_path)

# 결과 출력
print(output)

import re
import numpy as np

# --- [전체 시각화 코드] ---
for page_res in output:
    page_idx = page_res.get('page_index', 0)

    # 1. 모델이 읽어들인 전체 원본 이미지 가져오기
    # (cv2로 그리기 위해 메모리 상의 배열을 복사합니다)
    full_image = page_res['doc_preprocessor_res']['input_img'].copy()

    clean_image = page_res['doc_preprocessor_res']['input_img'].copy()

    # 2. 결과물 전체를 문자열로 바꾸고, 정규식(Regex)으로 bbox 좌표만 싹 다 추출!
    # 로그의 "bbox: [112, 118, 1088, 139]" 패턴을 모두 찾아냅니다.
    bbox_pattern = re.compile(r"bbox:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")
    bboxes = bbox_pattern.findall(str(page_res))

    crop_dir = f"./page_{page_idx}_crops"
    if not os.path.exists(crop_dir):
        os.mkdir(crop_dir)

    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)
        padding_bottom = 2
        y2_new = y2 + padding_bottom
        cropped_img = clean_image[y1:y2_new,x1:x2]
        crop_save_path = os.path.join(crop_dir, f"crop_{i}.jpg")
        cv2.imwrite(crop_save_path,cropped_img)

    # 3. 추출한 좌표들을 돌면서 원본 이미지 위에 빨간색 네모 그리기
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)
        # BGR 기준 (0, 0, 255)는 빨간색, 두께는 2
        cv2.rectangle(full_image, (x1, y1), (x2, y2), (0, 0, 255), 2)


    # 4. 빨간 박스가 쳐진 전체 이미지를 폴더에 저장
    save_file = f"./page_{page_idx}_all_bboxes.jpg"
    cv2.imwrite(str(save_file), full_image)

    print(f"🎉 {page_idx}번째 페이지 시각화 완료! 총 {len(bboxes)}개의 박스를 그렸습니다.")
    print(f"👉 폴더를 확인해 보세요: {save_file}")