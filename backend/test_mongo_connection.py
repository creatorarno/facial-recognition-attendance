import asyncio
from database import db

async def test_connection():
    try:
        print("Testing MongoDB connection...")
        db.connect()
        # Try a simple command
        info = await db.client.server_info()
        print(f"Connected to MongoDB version: {info.get('version')}")
        db.close()
        print("Connection test passed!")
    except Exception as e:
        print(f"Connection test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
