import motor.motor_asyncio
from bson import ObjectId
import os
import certifi
import ssl
from dotenv import load_dotenv

# 2. Load the environment variables
load_dotenv()

class Database:
    client: motor.motor_asyncio.AsyncIOMotorClient = None
    db = None

    def connect(self):
        """Connect to MongoDB"""
        mongo_url = os.getenv("MONGO_URL")
        self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_url, tlsCAFile=certifi.where())
        self.db = self.client.attendance_system
        print(f"Connected to MongoDB at {mongo_url}")

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("Closed MongoDB connection")

    async def get_student(self, student_id: str):
        return await self.db.students.find_one({"student_id": student_id})

    async def save_student(self, student_data: dict):
        # Check if exists to update or insert
        await self.db.students.replace_one(
            {"student_id": student_data["student_id"]},
            student_data,
            upsert=True
        )

    async def delete_student(self, student_id: str):
        await self.db.students.delete_one({"student_id": student_id})
        # Also delete associated attendance records? Maybe keep them for history.
        # For now, let's keep attendance records.

    async def get_all_students(self):
        cursor = self.db.students.find({})
        students = await cursor.to_list(length=None)
        return students

    async def save_attendance(self, record: dict):
        await self.db.attendance.insert_one(record)

    async def get_attendance(self):
        cursor = self.db.attendance.find({})
        records = await cursor.to_list(length=None)
        return records

    async def clear_attendance(self):
        await self.db.attendance.delete_many({})

db = Database()

def fix_id(doc):
    """Helper to convert ObjectId to string for JSON serialization"""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc
