import pdfplumber
import pymupdf
import pandas as pd
import os

file_path = r"C:\Users\김동욱\Downloads\user_guild.pdf"
output_dir = r"./extracted_images"

# 이미지 저장할 폴더 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"[{file_path}] 처리 시작...\n")

# ==========================================
# 1. 텍스트 및 테이블 추출 (pdfplumber 사용)
# ==========================================
print(">>> 1. 텍스트 및 테이블 추출 중...")
with pdfplumber.open(file_path) as pdf:
    page = pdf.pages[0]  # 첫 번째 페이지만 대상

    # [A] 텍스트 추출
    text = page.extract_text()
    print("\n[추출된 텍스트 내용 일부]:")
    print(text[:200] + "..." if text else "텍스트 없음")  # 너무 기니까 앞부분만 출력

    # [B] 테이블 추출
    tables = page.extract_tables()  # extract_table()은 하나만, tables()는 모두 다

    print(f"\n[발견된 테이블 개수]: {len(tables)}개")

    for i, table in enumerate(tables):
        # 데이터프레임 변환 (None 값 처리 및 헤더 설정)
        if table:
            # 헤더가 있는 경우와 없는 경우를 구분해서 처리하는 것이 좋습니다.
            # 여기서는 첫 번째 행을 헤더로 가정합니다.
            df = pd.DataFrame(table[1:], columns=table[0])
            print(f"\n--- 테이블 {i + 1} ---")
            print(df)
            # 필요하면 csv 저장: df.to_csv(f"table_{i}.csv", index=False)

# ==========================================
# 2. 이미지 추출 (PyMuPDF 사용)
# ==========================================
print("\n>>> 2. 이미지 추출 중...")
doc = pymupdf.open(file_path)
page = doc[0]  # 첫 번째 페이지
image_list = page.get_images(full=True)

print(f"[발견된 이미지 개수]: {len(image_list)}개")

for i, img in enumerate(image_list):
    xref = img[0]  # 이미지의 참조 ID
    base_image = doc.extract_image(xref)
    image_bytes = base_image["image"]  # 이미지 바이너리 데이터
    image_ext = base_image["ext"]  # 확장자 (png, jpeg 등)

    #Ollama에 질문   

    # 파일로 저장
    image_filename = f"image_{i + 1}.{image_ext}"
    save_path = os.path.join(output_dir, image_filename)

    with open(save_path, "wb") as f:
        f.write(image_bytes)

    print(f"  - 저장 완료: {save_path}")

print("\n모든 처리가 완료되었습니다.")