import chromadb

client = chromadb.PersistentClient(path='ocr_text_db')
collection = client.get_or_create_collection('text_docs')

print(f'Total documents: {collection.count()}')

if collection.count() > 0:
    docs = collection.get()
    print(f'\nFirst 5 document IDs:')
    for doc_id in docs['ids'][:5]:
        print(f'  - {doc_id}')
    
    print(f'\nDocument types:')
    types = {}
    for meta in docs['metadatas']:
        label = meta['label']
        types[label] = types.get(label, 0) + 1
    
    for doc_type, count in types.items():
        print(f'  - {doc_type}: {count}')
else:
    print('\nDatabase is EMPTY!')
    print('You need to click "Full Rebuild" to index documents.')
