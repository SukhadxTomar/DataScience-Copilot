import pymupdf

doc = pymupdf.open("DataScience_Copilot_Interview_Prep.pdf")
text = "\n".join(page.get_text() for page in doc)
with open("_pdf_extract.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(f"pages={doc.page_count} chars={len(text)}")
