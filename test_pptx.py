from unstructured.partition.pptx import partition_pptx
from unstructured.chunking.title import chunk_by_title
import sys

if len(sys.argv) < 2:
    print("Usage: python test_pptx.py <path_to_pptx>")
    sys.exit(1)

file_path = sys.argv[1]
try:
    elements = partition_pptx(filename=file_path)
    chunks = chunk_by_title(elements)
    
    for i, chunk in enumerate(chunks, 1):
        page_num = chunk.metadata.page_number
        print(f"Chunk {i}: extracted page_number={page_num}, text={repr(chunk.text[:50])}...")
except Exception as e:
    print(f"Error: {e}")
