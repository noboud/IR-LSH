from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="cuda")

vectors_file  = 'washington_data/washington_vectors2.npz'
bin_vectors_file  = 'washington_data/washington_vectors_bin2.npz'

chunk_size = 5000
chunk_num = 0

np.savez(vectors_file, vectors=np.empty((0,768), dtype=np.float32))
np.savez(bin_vectors_file, vectors=np.empty((0,768), dtype=np.uint8))

def append_data(file, vecs):
    comb_vecs = np.array([], dtype=object)
    with np.load(file, allow_pickle=True) as data:
        if 'vectors' in data:
            comb_vecs = np.concatenate((data['vectors'], vecs), axis=0)
        else:
            comb_vecs = vecs
    np.savez(file, vectors=comb_vecs)

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def clean_content(content):
    if content['type'] and content['type'] == 'sanitized_html':
        return clean_html(str(content['content']))
    else:
        return str(content['content'])
    

for chunk in pd.read_json('data/TREC_Washington_Post_collection.jl', lines=True, chunksize=chunk_size):
    chunk_num += 1
    print("Processing chunk #", chunk_num)
    
    chunk_content = []
    for index, row in chunk.iterrows():
        contents = row.get("contents") or []
        
        single_content = '. '.join([
            clean_content(content) for content in contents
            if isinstance(content, dict) and "content" in content and content["content"] is not None
        ])
        
        chunk_content.append(single_content)
    
    vectors = model.encode(chunk_content, show_progress_bar=True)

    # Binary vectors
    vectors_bin = np.where(vectors > 0.5, 1, 0)
        
    append_data(vectors_file, vectors)
    append_data(bin_vectors_file, vectors_bin.astype(np.uint8))
    
    print("Processed rows till #", chunk_num*chunk_size)