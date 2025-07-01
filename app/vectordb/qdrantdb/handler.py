from qdrant_client import QdrantClient, models
import yaml
from loguru import logger
import time
import flatdict
import ast
from app.base.base_vectordb import BaseVectorDB


class QdrantDataBase(BaseVectorDB):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
    ):
        logger.info("initializing Qdrant with configs")
        self.client = None
        self.embedding_function = None
        self.host = host
        self.port = port
        self.collections = [
            "schema_store",
            "documentation_store",
            "samples_store",
            "cache_store",
        ]

    def connect(self):
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            self.embedding_function = self.load_embeddings_function()

            for collection in self.collections:
                if not self.client.collection_exists(collection_name=collection):
                    self.client.create_collection(
                        collection_name=collection,
                        vectors_config=models.VectorParams(
                            size=len(self.embedding_function(["some text"])[0]),
                            distance=models.Distance.COSINE,
                        ),
                    )
            logger.info(f"Connected to QdrantDB")
            return None
        except Exception as e:
            logger.critical(f"Failed connecting QdrantDB: {e}")
            return str(e)

    def clear_collection(self, config_id):
        for collection in self.collections:
            self.client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="config_id",
                                match=models.MatchValue(value=config_id),
                            )
                        ]
                    )
                ),
            )

    def health_check(self):
        try:
            status = self.client.get_collections()
            return status
        except Exception as e:
            logger.critical(f"Qdrant health check failed: {e}")
            return str(e)

    def load_yaml_data(self, yaml_path):
        with open(yaml_path, "r") as stream:
            data_loaded = yaml.safe_load(stream)
        start_time = time.time()
        for i in range(len(data_loaded)):
            self._add_to_store(
                collection="documentation_store",
                document=data_loaded[i]["description"],
                metadata=data_loaded[i]["metadata"],
                idx=i,
            )
        end_time = time.time()
        response_time = end_time - start_time
        logger.info(f"vector db insertion time -> yaml loading : {response_time}")

    def _convert_lists_to_strings(self, d):
        for key, value in d.items():
            if isinstance(value, list):
                d[key] = str(value)
            elif isinstance(value, dict):
                self._convert_lists_to_strings(value)
        return d

    def prepare_data(
        self, datasource_name, chunked_document, chunked_schema, queries, config_id
    ):
        logger.info("Inserting into Qdrant vector store")
        logger.info(f"datasource_name:{datasource_name}")
        start_time = time.time()
        if chunked_document:
            for i, doc in enumerate(chunked_document):
                self._add_to_store(
                    collection="documentation_store",
                    document=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "datasource": datasource_name,
                        "config_id": config_id,
                    },
                    idx=i,
                )
        if chunked_schema:
            for i, doc in enumerate(chunked_schema):
                self._add_to_store(
                    collection="schema_store",
                    document=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "datasource": datasource_name,
                        "config_id": config_id,
                    },
                    idx=i,
                )
        if queries:
            for j, doc in enumerate(queries):
                doc = self._convert_lists_to_strings(doc)
                doc = flatdict.FlatDict(doc, delimiter=".")
                self._add_to_store(
                    collection="samples_store",
                    document=doc["description"],
                    metadata={
                        **dict(doc["metadata"]),
                        "datasource": datasource_name,
                        "config_id": config_id,
                    },
                    idx=j,
                )
                self._add_to_store(
                    collection="cache_store",
                    document=doc["description"],
                    metadata={
                        **dict(doc["metadata"]),
                        "datasource": datasource_name,
                        "config_id": config_id,
                    },
                    idx=j,
                )
        logger.info("Created Qdrant vector store for the source documents")
        end_time = time.time()
        response_time = end_time - start_time
        logger.info(f"vector db insertion time -> source docs : {response_time}")

    def _add_to_store(self, collection, document, metadata, idx):
        vector = self.embedding_function([document])[0]
        point = models.PointStruct(id=idx, vector=vector, payload=metadata)
        self.client.upsert(collection_name=collection, points=[point])

    def update_cache(self, document, metadata):
        try:
            vector = self.embedding_function([document])[0]
            idx = int(time.time() * 1000)
            point = models.PointStruct(id=idx, vector=vector, payload=metadata)
            self.client.upsert(collection_name="cache_store", points=[point])
            logger.info("cache updated successfully")
        except Exception as e:
            logger.info(f"error updating cache: {e}")

    def _convert_strings_to_lists(self, d):
        for key, value in d.items():
            if isinstance(value, str) and "[" in value and "]" in value:
                d[key] = ast.literal_eval(value)
            elif isinstance(value, dict):
                self._convert_strings_to_lists(value)
        return d

    def unflatten_dict(self, flat_dict):
        unflat_dict = {}
        for key, value in flat_dict.items():
            parts = key.split(".")
            d = unflat_dict
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
        unflat_dict = self._convert_strings_to_lists(unflat_dict)
        return unflat_dict

    async def _find_similar(self, datasource, query, collection, sample_count=3):
        vector = self.embedding_function([query])[0]
        res = self.client.query_points(
            collection_name=collection,
            query=vector,
            limit=sample_count,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="datasource", match=models.MatchValue(value=datasource[0])
                    )
                ]
            ),
        )
        output = []
        for hit in res:
            output.append(
                {
                    "document": hit.payload.get("document", ""),
                    "id": hit.id,
                    "metadatas": self.unflatten_dict(hit.payload),
                    "score": hit.score,
                }
            )
        return output

    def update_store(self, ids=None, metadatas=None, documents=None):
        if ids is None:
            ids = int(time.time() * 1000)
        vector = self.embedding_function([documents])[0]
        point = models.PointStruct(id=ids, vector=vector, payload=metadatas)
        self.client.upsert(collection_name="samples_store", points=[point])

    def update_weights(self, results, increment_value=1):
        if "weights" in results["metadatas"]:
            results["metadatas"]["weights"] += increment_value
        else:
            results["metadatas"]["weights"] = increment_value
        self.update_store(results["id"], results["metadatas"], results["document"])

    def _find_by_id(self, id_d, collection):
        res = self.client.retrieve(collection_name=collection, ids=[id_d])
        output = []
        for hit in res:
            output.append(
                {
                    "document": hit.payload.get("document", ""),
                    "id": hit.id,
                    "metadatas": self.unflatten_dict(hit.payload),
                }
            )
        if output:
            self.update_weights(output[0])
        return output

    async def find_similar_documentation(self, datasource, query, count):
        return await self._find_similar(datasource, query, "documentation_store", count)

    async def find_similar_schema(self, datasource, query, count):
        return await self._find_similar(datasource, query, "schema_store", count)

    async def find_samples_by_id(self, id):
        return await self._find_by_id(id, "samples_store")

    async def find_similar_cache(self, datasource, query, count=3):
        return await self._find_similar(datasource, query, "samples_store", count)
