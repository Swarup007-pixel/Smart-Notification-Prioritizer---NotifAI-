from dotenv import load_dotenv
import os
import motor.motor_asyncio

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
mongoDB = client[DB_NAME]