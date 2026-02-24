import chromadb

class DBManager:
    def __init__(self, path='data/chromadb_data'):
        self.client = chromadb.PersistentClient(path=path)

    def get_collection(self, collection_name):
        return self.client.get_or_create_collection(name=collection_name)
