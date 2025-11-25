import asyncio
from database import db

async def check_data():
    try:
        db.connect()
        students = await db.get_all_students()
        attendance = await db.get_attendance()
        
        print(f"Students count: {len(students)}")
        print(f"Attendance count: {len(attendance)}")
        
        for s in students:
            print(f"Student: {s.get('name')} ({s.get('student_id')})")
            
        for a in attendance:
            print(f"Attendance: {a.get('name')} at {a.get('timestamp')}")
            
        db.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_data())
