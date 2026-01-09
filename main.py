from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from typing import List, Optional

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="CBC Lesson Plan Generator", version="1.0")

# ============== CORS CONFIGURATION ==============
# Allow all origins from Vercel (or specify your exact domain)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://ai-lesson-planner-beta.vercel.app",
    "https://*.vercel.app",  # Allow all Vercel preview deployments
]

# Get FRONTEND_URL from environment variable if set
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For debugging - allows ALL origins (change to origins list for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Request model
class LessonPlanRequest(BaseModel):
    school: str
    subject: str
    class_name: str
    grade: int
    term: int
    date: str
    start_time: str
    end_time: str
    teacher_name: str
    teacher_tsc_number: str
    boys: int
    girls: int
    strand: str
    sub_strand: str

# Load lesson plan template
def load_lesson_template():
    try:
        with open('lesson_plan_template.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "lessonPlan": {
                "administrativeDetails": {},
                "curriculumAlignment": {},
                "learningOutcomes": [],
                "guidingQuestion": "",
                "learningResources": [],
                "lessonFlow": {
                    "introduction": {},
                    "development": [],
                    "conclusion": {}
                }
            }
        }

# Load curriculum
def load_curriculum(subject: str):
    try:
        filename = f"{subject.lower()}_curriculum.json"
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Curriculum file for {subject} not found. Please ensure {filename} exists."
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON in {filename}"
        )

# Extract curriculum content for specific strand and sub-strand
def extract_curriculum_content(curriculum, strand_name, sub_strand_name):
    """Find and extract specific curriculum content"""
    for strand in curriculum.get("strands", []):
        if strand_name.lower() in strand.get("name", "").lower():
            for sub_strand in strand.get("sub_strands", []):
                if sub_strand_name.lower() in sub_strand.get("name", "").lower():
                    return {
                        "strand": strand.get("name"),
                        "sub_strand": sub_strand.get("name"),
                        "topics": sub_strand.get("topics", []),
                        "learning_outcomes": sub_strand.get("specific_learning_outcomes", []),
                        "key_concepts": sub_strand.get("key_concepts", ""),
                        "key_inquiry_questions": sub_strand.get("key_inquiry_questions", []),
                        "suggested_experiences": sub_strand.get("suggested_learning_experiences", []),
                        "core_competencies": sub_strand.get("core_competencies", []),
                        "values": sub_strand.get("values", [])
                    }
    
    return {
        "strand": strand_name,
        "sub_strand": sub_strand_name,
        "topics": [],
        "learning_outcomes": [],
        "key_concepts": "",
        "key_inquiry_questions": [],
        "suggested_experiences": [],
        "core_competencies": [],
        "values": []
    }

# Generate lesson plan
def generate_lesson_plan(request: LessonPlanRequest):
    template = load_lesson_template()
    curriculum = load_curriculum(request.subject)
    
    curriculum_content = extract_curriculum_content(
        curriculum,
        request.strand,
        request.sub_strand
    )
    
    total_students = request.boys + request.girls
    
    topics_str = "\n- " + "\n- ".join(curriculum_content["topics"]) if curriculum_content["topics"] else "None specified"
    outcomes_str = "\n- " + "\n- ".join(curriculum_content["learning_outcomes"]) if curriculum_content["learning_outcomes"] else "None specified"
    experiences_str = "\n- " + "\n- ".join(curriculum_content["suggested_experiences"][:3]) if curriculum_content["suggested_experiences"] else "None specified"
    
    prompt = f"""
You are an expert {request.subject} teacher in Kenya creating a CBC-aligned lesson plan.

**LESSON PLAN TEMPLATE TO FOLLOW:**
{json.dumps(template, indent=2)}

**ADMINISTRATIVE DETAILS TO USE:**
- School: {request.school}
- Subject: {request.subject}
- Class: {request.class_name}
- Grade: {request.grade}
- Term: {request.term}
- Date: {request.date}
- Time: {request.start_time} - {request.end_time}
- Duration: 40 minutes
- Teacher: {request.teacher_name}
- TSC Number: {request.teacher_tsc_number}
- Boys: {request.boys}, Girls: {request.girls}, Total: {total_students}

**CURRICULUM FOCUS:**
Strand: {curriculum_content["strand"]}
Sub-strand: {curriculum_content["sub_strand"]}

**CURRICULUM CONTENT FROM CBC SYLLABUS:**

Topics to Cover:{topics_str}

Learning Outcomes:{outcomes_str}

Key Concepts: {curriculum_content["key_concepts"]}

Suggested Learning Experiences:{experiences_str}

Core Competencies to Develop:
{chr(10).join(f"- {comp}" for comp in curriculum_content["core_competencies"][:2])}

Values to Integrate:
{chr(10).join(f"- {val}" for val in curriculum_content["values"][:2])}

**INSTRUCTIONS:**
1. Follow the EXACT JSON structure from the template
2. Fill in all administrative details accurately
3. Create 3-4 specific, measurable learning outcomes (with ids "a", "b", "c", "d") based on the curriculum outcomes
4. Develop a thought-provoking guiding question from the key inquiry questions
5. List 3-5 practical learning resources appropriate for the topic
6. Design an engaging 5-minute introduction activity
7. Create 3 progressive development steps (each 10-12 minutes):
   - Step 1: Observation/exploration activity
   - Step 2: Hands-on practice/modeling activity
   - Step 3: Application/presentation activity
8. Design a 5-minute conclusion for summary and reflection
9. Incorporate the core competencies and values throughout activities
10. Make activities practical, engaging, and appropriate for Grade {request.grade}
11. Reference digital literacy, collaboration, and teamwork in activities
12. Ensure the lesson can be completed in 40 minutes

Return ONLY valid JSON matching the template structure exactly. No additional text or markdown.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Kenyan CBC teacher who creates detailed, engaging, standards-aligned lesson plans in JSON format. You understand the Kenyan education system and create culturally relevant lessons."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        lesson_plan = json.loads(response.choices[0].message.content)
        return lesson_plan
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# ============== API ENDPOINTS ==============

@app.get("/")
def read_root():
    return {
        "message": "CBC Lesson Plan Generator API",
        "status": "running",
        "version": "1.0",
        "cors_enabled": True,
        "endpoints": {
            "health": "/health",
            "generate": "/generate-lesson-plan",
            "strands": "/strands/{subject}",
            "sub_strands": "/sub-strands/{subject}/{strand}"
        }
    }

@app.post("/generate-lesson-plan")
async def create_lesson_plan(request: LessonPlanRequest):
    """Generate a CBC-aligned lesson plan based on curriculum content"""
    try:
        lesson_plan = generate_lesson_plan(request)
        return {
            "success": True,
            "message": "Lesson plan generated successfully",
            "lesson_plan": lesson_plan
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating lesson plan: {str(e)}")

@app.get("/strands/{subject}")
def get_strands(subject: str):
    """Get all available strands for a subject"""
    try:
        curriculum = load_curriculum(subject)
        strands = [
            {
                "id": strand.get("id"),
                "name": strand.get("name"),
                "code": strand.get("code")
            }
            for strand in curriculum.get("strands", [])
        ]
        return {
            "success": True,
            "subject": subject,
            "strands": strands
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sub-strands/{subject}/{strand}")
def get_sub_strands(subject: str, strand: str):
    """Get sub-strands for a specific strand"""
    try:
        curriculum = load_curriculum(subject)
        
        for strand_obj in curriculum.get("strands", []):
            if strand.lower() in strand_obj.get("name", "").lower():
                sub_strands = [
                    {
                        "id": ss.get("id"),
                        "name": ss.get("name"),
                        "topics": ss.get("topics", [])
                    }
                    for ss in strand_obj.get("sub_strands", [])
                ]
                return {
                    "success": True,
                    "strand": strand_obj.get("name"),
                    "sub_strands": sub_strands
                }
        
        return {
            "success": False,
            "message": f"Strand '{strand}' not found"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/curriculum/{subject}")
def get_full_curriculum(subject: str):
    """Get the complete curriculum for a subject"""
    try:
        curriculum = load_curriculum(subject)
        return {
            "success": True,
            "curriculum": curriculum
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Check API and OpenAI connection health"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "status": "error",
                "message": "OpenAI API key not found in .env file",
                "api": "running",
                "openai": "not configured"
            }
        
        files_exist = {
            "biology_curriculum.json": os.path.exists("biology_curriculum.json"),
            "lesson_plan_template.json": os.path.exists("lesson_plan_template.json")
        }
        
        return {
            "status": "healthy",
            "message": "API is running and OpenAI key is configured",
            "api": "running",
            "openai": "configured",
            "files": files_exist
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)