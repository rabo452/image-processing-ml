from pymilvus import MilvusClient

vectorDBClient = MilvusClient()
# vectorDBClient.delete('image_search', filter='id > 0')
# print(vectorDBClient.query('image_search', output_fields=['count(*)']))