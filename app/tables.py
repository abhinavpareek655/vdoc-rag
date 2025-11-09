# app/tables.py
import os
import uuid
import pdfplumber
import pandas as pd
from typing import List, Dict

TABLES_DIR = os.environ.get('VDOCRAG_TABLES_DIR', '/tmp/vdoc_tables')
os.makedirs(TABLES_DIR, exist_ok=True)

def extract_tables_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract tables using pdfplumber and save each as CSV. Returns a list of metadata dicts:
    [{ 'csv_path': str, 'page': int, 'table_index': int, 'summary_text': str }]
    """
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for tidx, table in enumerate(tables):
                # Convert to DataFrame
                try:
                    df = pd.DataFrame(table[1:], columns=table[0]) if len(table) > 1 else pd.DataFrame(table)
                except Exception:
                    df = pd.DataFrame(table)

                fname = f"table_{uuid.uuid4().hex}_p{pno}_t{tidx}.csv"
                csv_path = os.path.join(TABLES_DIR, fname)
                # Save CSV
                try:
                    df.to_csv(csv_path, index=False)
                except Exception:
                    df.to_csv(csv_path, index=False, encoding='utf-8', errors='ignore')

                # Get table bbox (approximate)
                try:
                    # Each table has a bounding box in page._objects['rects'] or use the table extractor
                    table_bbox = page.find_tables()[tidx - 1].bbox  # (x0, top, x1, bottom)
                except Exception:
                    table_bbox = None

                # create a short textual summary: columns and first N rows
                cols = list(df.columns) if len(df.columns) > 0 else []
                top_rows = df.head(5).to_dict(orient='records')
                summary = f"Table (page {pno}) with columns: {cols}. First rows: {top_rows}"

                results.append({
                    'csv_path': csv_path,
                    'page': pno,
                    'table_index': tidx,
                    'summary_text': summary,
                    'rows': len(df),
                    'bbox': table_bbox
                })
    return results
