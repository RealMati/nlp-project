"""
Create sample university database with realistic data.

This script generates:
- Students table (30 students)
- Professors table (8 professors)
- Courses table (15 courses)
- Enrollments table (60+ enrollments)
"""

import sqlite3
from pathlib import Path
from typing import List, Tuple
import random


# Sample data
FIRST_NAMES = [
    'Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'Ethan', 'Sophia', 'Mason',
    'Isabella', 'William', 'Mia', 'James', 'Charlotte', 'Benjamin', 'Amelia',
    'Lucas', 'Harper', 'Henry', 'Evelyn', 'Alexander', 'Abigail', 'Michael',
    'Emily', 'Daniel', 'Elizabeth', 'Matthew', 'Sofia', 'Jackson', 'Avery', 'David'
]

LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
    'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
    'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
    'Lee', 'Thompson', 'White', 'Harris', 'Clark', 'Lewis', 'Robinson', 'Walker'
]

MAJORS = [
    'Computer Science',
    'Electrical Engineering',
    'Mechanical Engineering',
    'Biology',
    'Chemistry',
    'Physics',
    'Mathematics',
    'English',
    'History',
    'Psychology',
    'Economics',
    'Business Administration'
]

DEPARTMENTS = [
    'Computer Science',
    'Engineering',
    'Natural Sciences',
    'Mathematics',
    'Humanities',
    'Social Sciences',
    'Business'
]

COURSES_DATA = [
    ('Introduction to Programming', 3, 'Computer Science'),
    ('Data Structures and Algorithms', 4, 'Computer Science'),
    ('Database Systems', 3, 'Computer Science'),
    ('Machine Learning', 3, 'Computer Science'),
    ('Calculus I', 4, 'Mathematics'),
    ('Calculus II', 4, 'Mathematics'),
    ('Linear Algebra', 3, 'Mathematics'),
    ('General Chemistry', 4, 'Natural Sciences'),
    ('Organic Chemistry', 4, 'Natural Sciences'),
    ('General Physics', 4, 'Natural Sciences'),
    ('World History', 3, 'Humanities'),
    ('English Literature', 3, 'Humanities'),
    ('Microeconomics', 3, 'Social Sciences'),
    ('Psychology 101', 3, 'Social Sciences'),
    ('Business Strategy', 3, 'Business')
]

GRADES = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F']
SEMESTERS = ['Fall 2023', 'Spring 2024', 'Fall 2024', 'Spring 2025']


def create_database(db_path: Path) -> None:
    """Create the university database with schema and sample data."""

    # Remove existing database
    if db_path.exists():
        db_path.unlink()

    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Create tables
        print("Creating tables...")

        # Professors table (must be created first due to foreign key)
        cursor.execute("""
            CREATE TABLE professors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary REAL NOT NULL,
                hire_date TEXT,
                email TEXT
            )
        """)

        # Students table
        cursor.execute("""
            CREATE TABLE students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                gpa REAL CHECK(gpa >= 0.0 AND gpa <= 4.0),
                major TEXT,
                enrollment_year INTEGER,
                email TEXT
            )
        """)

        # Courses table
        cursor.execute("""
            CREATE TABLE courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_name TEXT NOT NULL,
                credits INTEGER NOT NULL CHECK(credits > 0),
                department TEXT,
                professor_id INTEGER,
                FOREIGN KEY (professor_id) REFERENCES professors(id)
            )
        """)

        # Enrollments table
        cursor.execute("""
            CREATE TABLE enrollments (
                student_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                grade TEXT,
                semester TEXT NOT NULL,
                enrollment_date TEXT,
                PRIMARY KEY (student_id, course_id, semester),
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
                FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
            )
        """)

        # Create indexes for better query performance
        cursor.execute("CREATE INDEX idx_students_major ON students(major)")
        cursor.execute("CREATE INDEX idx_students_gpa ON students(gpa)")
        cursor.execute("CREATE INDEX idx_courses_department ON courses(department)")
        cursor.execute("CREATE INDEX idx_enrollments_student ON enrollments(student_id)")
        cursor.execute("CREATE INDEX idx_enrollments_course ON enrollments(course_id)")

        print("Tables created successfully.")

        # Insert sample data
        print("\nInserting sample data...")

        # Insert professors
        professors_data = [
            ('Dr. Sarah Johnson', 'Computer Science', 95000, '2018-08-15', 'sjohnson@university.edu'),
            ('Dr. Michael Chen', 'Computer Science', 92000, '2019-01-10', 'mchen@university.edu'),
            ('Dr. Emily Rodriguez', 'Mathematics', 88000, '2017-09-01', 'erodriguez@university.edu'),
            ('Dr. Robert Anderson', 'Natural Sciences', 90000, '2016-08-20', 'randerson@university.edu'),
            ('Dr. Jennifer Lee', 'Natural Sciences', 87000, '2020-01-15', 'jlee@university.edu'),
            ('Dr. David Thompson', 'Humanities', 85000, '2015-09-01', 'dthompson@university.edu'),
            ('Dr. Maria Garcia', 'Social Sciences', 86000, '2018-08-25', 'mgarcia@university.edu'),
            ('Dr. James Wilson', 'Business', 98000, '2017-01-10', 'jwilson@university.edu')
        ]

        cursor.executemany(
            "INSERT INTO professors (name, department, salary, hire_date, email) VALUES (?, ?, ?, ?, ?)",
            professors_data
        )
        print(f"Inserted {len(professors_data)} professors.")

        # Insert students
        students_data = []
        for i in range(30):
            first_name = random.choice(FIRST_NAMES)
            last_name = random.choice(LAST_NAMES)
            name = f"{first_name} {last_name}"
            gpa = round(random.uniform(2.0, 4.0), 2)
            major = random.choice(MAJORS)
            year = random.choice([2021, 2022, 2023, 2024])
            email = f"{first_name.lower()}.{last_name.lower()}@student.edu"

            students_data.append((name, gpa, major, year, email))

        cursor.executemany(
            "INSERT INTO students (name, gpa, major, enrollment_year, email) VALUES (?, ?, ?, ?, ?)",
            students_data
        )
        print(f"Inserted {len(students_data)} students.")

        # Insert courses with professor assignments
        # Map departments to professor IDs
        dept_to_prof = {
            'Computer Science': [1, 2],
            'Mathematics': [3],
            'Natural Sciences': [4, 5],
            'Humanities': [6],
            'Social Sciences': [7],
            'Business': [8]
        }

        courses_data = []
        for course_name, credits, department in COURSES_DATA:
            prof_ids = dept_to_prof.get(department, [1])
            prof_id = random.choice(prof_ids)
            courses_data.append((course_name, credits, department, prof_id))

        cursor.executemany(
            "INSERT INTO courses (course_name, credits, department, professor_id) VALUES (?, ?, ?, ?)",
            courses_data
        )
        print(f"Inserted {len(courses_data)} courses.")

        # Insert enrollments (each student takes 2-4 courses per semester)
        enrollments_data = []
        for student_id in range(1, 31):  # 30 students
            # Random number of semesters (1-3)
            num_semesters = random.randint(1, 3)
            semesters_taken = random.sample(SEMESTERS, num_semesters)

            for semester in semesters_taken:
                # Each student takes 2-4 courses per semester
                num_courses = random.randint(2, 4)
                course_ids = random.sample(range(1, 16), num_courses)

                for course_id in course_ids:
                    # 80% chance of having a grade (20% in progress)
                    if random.random() < 0.8:
                        # Grade distribution (weighted towards B/C)
                        grade = random.choices(
                            GRADES,
                            weights=[10, 15, 20, 25, 15, 8, 5, 1, 0.5, 0.5],
                            k=1
                        )[0]
                    else:
                        grade = None  # In progress

                    # Generate enrollment date
                    if semester.startswith('Fall'):
                        enrollment_date = f"{semester.split()[1]}-08-25"
                    else:  # Spring
                        enrollment_date = f"{semester.split()[1]}-01-15"

                    enrollments_data.append((student_id, course_id, grade, semester, enrollment_date))

        cursor.executemany(
            "INSERT INTO enrollments (student_id, course_id, grade, semester, enrollment_date) VALUES (?, ?, ?, ?, ?)",
            enrollments_data
        )
        print(f"Inserted {len(enrollments_data)} enrollments.")

        conn.commit()
        print("\nDatabase created successfully!")

        # Print statistics
        print("\n" + "=" * 50)
        print("DATABASE STATISTICS")
        print("=" * 50)

        cursor.execute("SELECT COUNT(*) FROM students")
        print(f"Total Students: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM professors")
        print(f"Total Professors: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM courses")
        print(f"Total Courses: {cursor.fetchone()[0]}")

        cursor.execute("SELECT COUNT(*) FROM enrollments")
        print(f"Total Enrollments: {cursor.fetchone()[0]}")

        print("\nSample queries you can try:")
        print("1. SELECT * FROM students WHERE gpa > 3.5")
        print("2. SELECT c.course_name, p.name as professor FROM courses c JOIN professors p ON c.professor_id = p.id")
        print("3. SELECT s.name, AVG(CASE e.grade WHEN 'A' THEN 4.0 WHEN 'B' THEN 3.0 WHEN 'C' THEN 2.0 ELSE 1.0 END) as avg_gpa FROM students s JOIN enrollments e ON s.id = e.student_id WHERE e.grade IS NOT NULL GROUP BY s.id, s.name")
        print("4. SELECT major, COUNT(*) as student_count FROM students GROUP BY major ORDER BY student_count DESC")

    except Exception as e:
        print(f"Error creating database: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    # Create database in data/databases directory
    db_path = Path(__file__).parent.parent / "data" / "databases" / "sample_university.db"
    create_database(db_path)
    print(f"\nDatabase created at: {db_path.absolute()}")
