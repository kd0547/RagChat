import os
import numpy as np
from splitter import string_to_markdown_table
os.environ['USERPROFILE'] = 'C:/AI'
os.environ['HOME'] = 'C:/AI'
import fitz  # PyMuPDF
from paddleocr import PaddleOCRVL
import fitz
import cv2
pipeline = PaddleOCRVL()

file_url = './docs/docs1.pdf'
doc = fitz.open(file_url)
print(f"총 {len(doc)}페이지 분석 시작 (PyMuPDF 방식)...")
image_list = [] # 여기에 이미지들을 먼저 다 담습니다.


#이미지 추출
for i in range(len(doc)):
    page = doc.load_page(i)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    # RGBA인 경우 RGB로 변환
    if pix.n == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    image_list.append(img_array)

import time
total_start_time = time.time()
# 추론 시작
for i, image in enumerate(image_list):
    page_start_time = time.time()

    output_generator = pipeline.predict(input= image)
    # 페이지별 종료 시간 및 소요 계산
    page_end_time = time.time()
    page_duration = page_end_time - page_start_time


    print(f"--- {i + 1}페이지 분석 완료 ({page_duration:.2f}초 소요) ---")

    # res.save_to_json(save_path="output")  ## Save the current image's structured result in JSON format
    # res.save_to_markdown(save_path="output")  ## Save the current image's result in Markdown format

    for res in output_generator:

        res_data = res.json
        parsing_res_list = res_data['res']['parsing_res_list']

        # print(parsing_res_list)

        full_content = ""

        for data in parsing_res_list:
            # 2. 각 조건마다 += 연산자를 사용하여 내용을 덧붙입니다.
            if data['block_label'] == 'paragraph_title':
                # 제목은 ##를 붙여서 구분
                full_content += f"\n## {data['block_content']}\n"

            elif data['block_label'] == 'table':
                # 테이블 변환 함수 호출 후 결과 추가
                full_content += string_to_markdown_table(data['block_content']) + "\n"

            elif data['block_label'] == 'text':
                # 일반 텍스트 추가
                full_content += data['block_content'] + "\n"

            elif data['block_label'] == 'chart':
                img_base = './imgs/img_in_chart_box'
                bbox_str = "_".join(map(str, data['block_bbox']))
                img_path = f"{img_base}_{bbox_str}.jpg"
                full_content += f"\n<image>{img_path}</image>\n"
            elif data['block_label'] == 'image':
                img_base = './imgs/img_in_image_box'
                bbox_str = "_".join(map(str, data['block_bbox']))
                img_path = f"{img_base}_{bbox_str}.jpg"
                full_content += f"\n<image>{img_path}</image>\n"
            else:
                full_content += f"{data['block_label']}:{data['block_content']}" + "\n"
                # full_content += f"{data}" + "\n"

        print(full_content)

total_end_time = time.time()
total_duration = total_end_time - total_start_time
avg_duration = total_duration / len(image_list)

print("=" * 50)
print(f"✅ 전체 분석 완료!")
print(f"⏱️ 총 소요 시간: {total_duration:.2f}초")
print(f"📊 페이지당 평균 시간: {avg_duration:.2f}초")
print("=" * 50)







