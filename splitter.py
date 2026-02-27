import re

from langchain_text_splitters import MarkdownHeaderTextSplitter
from bs4 import BeautifulSoup

def _convert_html_to_md(html_str):
    soup = BeautifulSoup(html_str,'html.parser')
    rows = soup.find_all('tr')
    if not rows:
        return ""

    md_rows = []

    for i, row in enumerate(rows):
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td','th'])]

        # 빈 행 방지
        if not cells: continue

        md_rows.append(f"| {' | '.join(cells)} |")

        if i == 0:
            md_rows.append(f"| {' | '.join(['---'] * len(cells))} |")
    return "\n" + "\n".join(md_rows) + "\n"

def string_to_markdown_table(input_data):
    html_str = input_data.group(0) if hasattr(input_data, 'group') else input_data
    return _convert_html_to_md(html_str)

def test():
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ()
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    file_url = './output/embedded-images-tables_0.md'

    with open(file_url, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    docs = markdown_splitter.split_text(markdown_content)
    print(f"총 {len(docs)}개의 청크로 분할되었습니다.")

    table_pattern = r'<table.*?>.*?</table>'

    for doc in docs:
        doc.page_content = re.sub(table_pattern,
                                  string_to_markdown_table,
                                  doc.page_content,
                                  flags=re.DOTALL)
        print(doc.page_content)


