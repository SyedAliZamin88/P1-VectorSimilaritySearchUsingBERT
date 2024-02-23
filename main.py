import os
import pandas as pd
import mysql.connector
import openai
import json
import qdrant_client
import numpy as np
import wget
import random
import warnings

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models as rest
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer('all-MiniLM-L6-v2')

##using local computer memory as temporary storage
#qdrant = QdrantClient(":memory:")
#local
qdrant = QdrantClient("http://qdrant-instance-url:6333")

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# MySQL database connection details
db_config = {
    "host": os.getenv("host"),
    "user": os.getenv("user"),
    "password": os.getenv("password"),
    "database": os.getenv("database"),
}
connection = mysql.connector.connect(**db_config)
cursor = connection.cursor()

def db_connection_status():
        if connection.is_connected():
            print("Connected to the database")
            return True
        else:
            print("DB Connection failed")
            return False

db_connection_status()

def check_api_key(api_key):
    query = "SELECT EXISTS (SELECT 1 FROM DB.TABLE WHERE api_key = %s)"
    cursor.execute(query, (api_key,))
    result = cursor.fetchone()
    if result[0] == 1:
        print("API is Valid")
        return True
    else:
        print("API is not  valid")
        return False

#function to create embeddings of the query and search similarity from the Vector Store
def query_qdrant(query, current_collection_name, vector_name='vector_embeds', top_k=5):
    qdrant = QdrantClient("localhost", port=6333)
    # Creates embedding vector from user query
    embedded_query =encoder.encode(query)
    query_results = qdrant.search(
        collection_name=current_collection_name,
        query_vector=(
            vector_name, embedded_query
        ),
        limit=top_k,
    )
    return query_results


app = FastAPI()

@app.get("/test")
async def healthcheck():
    db_status=db_connection_status()
    if db_status is False:
        return {"Success":False, "status":"DB Connection failed"}
    return {"Success":True, "Status":"200 OK Connected to DB and FastAPI is working"}


@app.get("/embeddings")
async def create_embeddings(request: Request):
    headers = request.headers
    api_key = headers.get('Authorization')
    print("Entered API_Key: ",api_key)
    status_api_key_exists=check_api_key(api_key)
    if status_api_key_exists==False:
        return {"Success": False, "Message": "Invalid API Key"}

    if status_api_key_exists==True:
            print(f"Status is True")
            query = "SELECT products_url FROM db.table WHERE api_key = %s;"
            cursor = connection.cursor()
            cursor.execute(query, (api_key,))
            result = cursor.fetchone()
            csv_file_url = result[0]
            url=csv_file_url
            print(url)
            selected_columns=['col1','col2', 'col3','col4','col5','col6']
            print(f"selected columns: {selected_columns}")
            df1=df[selected_columns]
            df1=df1.head(10)
            #print(df1)
            df1['vector_embeds']=None
            df1['vector_id']=range(len(df1))
            print("Length of CSV files is: ",range(len(df1)))

            ## Creating Embeddings using BERT
            # Generate embeddings for the specified column

            print("Creating Embeddings using BERT")
            df1['col1andcol2']=df1['col1'] + ' ' + df1['col2'].astype(str)
            print("DataFrame before Embeddings",df1)
            embeddings = df1['col1andcol2'].apply(lambda text: encoder.encode(text))
            print("DataFrame after Embeddings",df1)
            df1['vector_embeds']=embeddings
            print("Embeddings stored in DF")


            #Indexing
            client = QdrantClient("localhost", port=6333,timeout=7200)
            #client = QdrantClient("localhost", port=6333)
            current_collection_name=api_key
            print("Client:", client)
            client.recreate_collection(collection_name=current_collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.DOT),)
            print("Collection Created for testing")
            vector_size = len(df1['vector_embeds'][0])
            print("The size of vector is :",vector_size)

            #again recreating collection
            print("Creating collection")
            client.recreate_collection(
            collection_name=current_collection_name,
            vectors_config={
                'title_tags': rest.VectorParams(
                    distance=rest.Distance.COSINE,
                    size=vector_size,
                ),
                }
            )
            print("Collecction Created")

            print(df1['col1andcol2'])
            print("Embeddings in array form",df1['vector_embeds'])
            # Convert NumPy arrays to lists
            df1['vector_embeds'] = df1['vector_embeds'].apply(lambda arr: arr.tolist())
            print("Embeddings in list form",df1['vector_embeds'])

            print("Storing embeddings in the Qdrant VS")
            client.upsert( collection_name=current_collection_name,
                points=[
                    rest.PointStruct(
                        id=k,
                        vector={
                            'col1andcol2': v['vector_embeds'],
                        },
                        payload=v.to_dict(),
                    )
                    for k, v in df1.iterrows()
                ],
            )
            print("Embeddings added in the collection")
            # Check the collection size to make sure all the points have been stored
            client.count(collection_name=current_collection_name)
            return {"success":True, "status":"Embeddings created"}


##Creating Conversation API
@app.get("/conversations")
async def read_conversation(request: Request,query: str = None):
    headers = request.headers
    api_key = headers.get('Authorization')
    print("API KEY entered by brand is:",api_key)
    classifier=-1
    api_status=check_api_key(api_key)
    if api_status is False:
        print("API key is not available in DB")
        return {"Status":"Unauthorized", "message":"This Brand/API is not available"}
    else:
        
        print("Sending query for similarity search to Qdrant")
        current_collection_name=api_key
        print("The user query is:",query)
        print("Searching from the collection:",current_collection_name)
        #query_results = query_qdrant(query, current_collection_name)
        print("Trying to create query embeddings and then searching from qdrant")
        qdrant = QdrantClient("http://qdrant-instance-url:6333")
        # Search for similar items in the Qdrant collection
            #############################################################

          ##function to create embeddings of the query and search similiarity from the VS
          search_results=query_qdrant(query, current_collection_name, vector_name='title_tags', top_k=4)

            #############################################################

            query_results=search_results
            #print(query_results)

            ID=''
            Title=''
            Price=''
            Product_link=''

            products_list = []

            for i, products in enumerate(query_results):
                # Extract data from the current product
                ID = products.payload["ID"]
                Title = products.payload["title"]
                Price = products.payload["price"]
                Product_link = products.payload["link"]

                product_data = {
                    "ID": ID,
                    "Title": Title,
                    "Price": Price,
                    "Product_link": Product_link
                }
                products_list.append(product_data)
            #final_response=query_llm(query)
            #return final_response,products_list
            return products_list




################################################################
##Creating Conversation API
@app.get("/InteractiveConversations")
async def read_conversation(request: Request,query: str = None):
    headers = request.headers
    api_key = headers.get('Authorization')
    print("API KEY entered by brand is:",api_key)
    classifier=-1
    api_status=check_api_key(api_key)
    if api_status is False:
        print("API key is not available in DB")
        return {"Status":"Unauthorized", "message":"This Brand/API is not available"}
    else:
        recommendation_allow_value = check_recommendation_status(api_key)
        print(recommendation_allow_value)
        if recommendation_allow_value==False:
            print("API key is availbe but Recommendation status is 0")
            return {"Status":"Unauthorized", "message":"Recommendation status of this brand is deactivated"}
        else:
        #now use classifier
            classifier,reply=fixed_reply(query)
        if classifier==0:
            print("Bot Reply:",reply)
            return {"Status":"Authorized", "message":reply}
        else:
            classifier=-1
            print("This is not a greeting message by the. Now calling input classifier")
            print("Sending query for smilarity seach to Qdrant")
            current_collection_name=api_key
            print("The user query is:",query)
            #Below is the query for OpenAI qdrant
            print("Searching from the collection:",current_collection_name)
            #query_results = query_qdrant(query, current_collection_name)
            print("Trying to created query embeddings and then searching from qdrant")
            qdrant = QdrantClient("http://qdrant-instance-url:6333")
            # Search for similar items in the Qdrant collection
            #############################################################

            ##function to create embeddings of the query and search similiarity from the VS
            search_results=query_qdrant(query, current_collection_name, vector_name='title_tags', top_k=4)

            #############################################################

            query_results=search_results
            print(query_results)

            ID=''
            Title=''
            Price=''
            Product_link=''

            products_list = []

            for i, products in enumerate(query_results):
                # Extract data from the current product
                ID = products.payload["ID"]
                Title = products.payload["title"]
                Price = products.payload["price"]
                Product_link = products.payload["link"]

                product_data = {
                    "ID": ID,
                    "Title": Title,
                    "Price": Price,
                    "Product_link": Product_link
                }
                products_list.append(product_data)
            final_response=query_llm(query)
            return final_response,products_list
            return products_list
