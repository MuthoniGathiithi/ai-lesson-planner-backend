from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from typing import List, Optional, Dict, Any
from rapidfuzz import fuzz, process

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="CBC Lesson Plan Generator", version="2.0")

# ============== CORS CONFIGURATION ==============
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://ai-lesson-planner-beta.vercel.app",
    "https://*.vercel.app",
]

frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============== REQUEST MODELS ==============
class LessonPlanRequest(BaseModel):
    school: str
    subject: str  # NO DEFAULT - user must provide
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

# ============== HELPER FUNCTIONS ==============

def load_lesson_template():
    """Load the lesson plan template JSON"""
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

def find_best_match(query: str, options: List[str], threshold: int = 70) -> Optional[str]:
    """
    Find the best matching string from options using fuzzy matching.
    Handles typos like "geogrphy" -> "geography"
    
    Args:
        query: The string to match (e.g., "geogrphy")
        options: List of possible matches (e.g., ["biology", "geography", "mathematics"])
        threshold: Minimum similarity score (0-100)
    
    Returns:
        Best matching string or None
    """
    if not query or not options:
        return None
    
    # Use fuzzy matching to find best match
    result = process.extractOne(query, options, scorer=fuzz.ratio)
    
    if result and result[1] >= threshold:
        print(f"✅ Fuzzy match: '{query}' → '{result[0]}' (similarity: {result[1]}%)")
        return result[0]
    
    print(f"⚠️ No good match for '{query}' (best score: {result[1] if result else 0}%)")
    return None

def load_curriculum(subject: str) -> Optional[Dict[str, Any]]:
    """
    Load curriculum JSON file for a subject with smart name matching.
    Handles typos in subject names (e.g., "Geogrphy" -> "Geography")
    Returns None if file doesn't exist.
    """
    # First try exact match
    try:
        filename = f"{subject.lower()}_curriculum.json"
        with open(filename, 'r', encoding='utf-8') as f:
            curriculum = json.load(f)
            print(f"✅ Successfully loaded curriculum file: {filename}")
            return curriculum
    except FileNotFoundError:
        # Try fuzzy matching with existing curriculum files
        print(f"⚠️ Curriculum file '{subject.lower()}_curriculum.json' not found. Trying fuzzy match...")
        
        # Get all curriculum files in directory
        curriculum_files = [f for f in os.listdir('.') if f.endswith('_curriculum.json')]
        
        if curriculum_files:
            # Extract subject names from filenames
            available_subjects = [f.replace('_curriculum.json', '') for f in curriculum_files]
            
            # Try fuzzy matching
            best_match = find_best_match(subject.lower(), available_subjects, threshold=75)
            
            if best_match:
                try:
                    filename = f"{best_match}_curriculum.json"
                    with open(filename, 'r', encoding='utf-8') as f:
                        curriculum = json.load(f)
                        print(f"✅ Fuzzy matched '{subject}' to '{best_match}' and loaded {filename}")
                        return curriculum
                except Exception as e:
                    print(f"❌ Error loading fuzzy matched file: {str(e)}")
        
        print(f"⚠️ No curriculum file found for '{subject}'. AI will use general knowledge.")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in curriculum file: {str(e)}")
        return None

def extract_curriculum_content(
    curriculum: Optional[Dict[str, Any]], 
    strand_name: str, 
    sub_strand_name: str
) -> Dict[str, Any]:
    """
    Extract specific curriculum content with intelligent fuzzy matching.
    Handles typos in strand and sub-strand names.
    
    Args:
        curriculum: The loaded curriculum JSON (can be None)
        strand_name: User-provided strand name (may have typos)
        sub_strand_name: User-provided sub-strand name (may have typos)
    
    Returns:
        Dictionary with curriculum content or empty structure
    """
    # Default empty structure
    empty_structure = {
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
    
    # If no curriculum file exists, return empty structure
    if curriculum is None:
        print(f"ℹ️ No curriculum data available. Using strand: '{strand_name}', sub-strand: '{sub_strand_name}'")
        return empty_structure
    
    strands = curriculum.get("strands", [])
    
    if not strands:
        print("⚠️ No strands found in curriculum")
        return empty_structure
    
    # Get all strand names for fuzzy matching
    strand_names = [strand.get("name", "") for strand in strands if strand.get("name")]
    
    if not strand_names:
        print("⚠️ No valid strand names in curriculum")
        return empty_structure
    
    # Find best matching strand (handles typos)
    best_strand_match = find_best_match(strand_name, strand_names, threshold=70)
    
    if not best_strand_match:
        print(f"⚠️ No good match found for strand '{strand_name}'. Available: {strand_names}")
        return empty_structure
    
    # Find the matched strand object
    matched_strand = None
    for strand in strands:
        if strand.get("name", "") == best_strand_match:
            matched_strand = strand
            break
    
    if not matched_strand:
        print(f"⚠️ Could not find strand object for '{best_strand_match}'")
        return empty_structure
    
    # Get all sub-strand names for fuzzy matching
    sub_strands = matched_strand.get("sub_strands", [])
    sub_strand_names = [ss.get("name", "") for ss in sub_strands if ss.get("name")]
    
    if not sub_strand_names:
        print(f"⚠️ No sub-strands found in strand '{best_strand_match}'")
        return {
            "strand": best_strand_match,
            "sub_strand": sub_strand_name,
            "topics": [],
            "learning_outcomes": [],
            "key_concepts": "",
            "key_inquiry_questions": [],
            "suggested_experiences": [],
            "core_competencies": [],
            "values": []
        }
    
    # Find best matching sub-strand (handles typos)
    best_substrand_match = find_best_match(sub_strand_name, sub_strand_names, threshold=70)
    
    if not best_substrand_match:
        print(f"⚠️ No good match for sub-strand '{sub_strand_name}' in '{best_strand_match}'. Available: {sub_strand_names}")
        return {
            "strand": best_strand_match,
            "sub_strand": sub_strand_name,
            "topics": [],
            "learning_outcomes": [],
            "key_concepts": "",
            "key_inquiry_questions": [],
            "suggested_experiences": [],
            "core_competencies": [],
            "values": []
        }
    
    # Find the matched sub-strand object
    matched_sub_strand = None
    for ss in sub_strands:
        if ss.get("name", "") == best_substrand_match:
            matched_sub_strand = ss
            break
    
    if not matched_sub_strand:
        print(f"⚠️ Could not find sub-strand object for '{best_substrand_match}'")
        return {
            "strand": best_strand_match,
            "sub_strand": sub_strand_name,
            "topics": [],
            "learning_outcomes": [],
            "key_concepts": "",
            "key_inquiry_questions": [],
            "suggested_experiences": [],
            "core_competencies": [],
            "values": []
        }
    
    # Successfully found curriculum content
    print(f"✅ Found curriculum content: Strand='{best_strand_match}', Sub-strand='{best_substrand_match}'")
    
    return {
        "strand": best_strand_match,
        "sub_strand": best_substrand_match,
        "topics": matched_sub_strand.get("topics", []),
        "learning_outcomes": matched_sub_strand.get("specific_learning_outcomes", []),
        "key_concepts": matched_sub_strand.get("key_concepts", ""),
        "key_inquiry_questions": matched_sub_strand.get("key_inquiry_questions", []),
        "suggested_experiences": matched_sub_strand.get("suggested_learning_experiences", []),
        "core_competencies": matched_sub_strand.get("core_competencies", []),
        "values": matched_sub_strand.get("values", [])
    }

def generate_lesson_plan(request: LessonPlanRequest):
    """
    Generate a CBC-aligned lesson plan using OpenAI.
    Intelligently handles missing curriculum files and typos.
    """
    template = load_lesson_template()
    curriculum = load_curriculum(request.subject)
    
    curriculum_content = extract_curriculum_content(
        curriculum,
        request.strand,
        request.sub_strand
    )
    
    total_students = request.boys + request.girls
    
    # Check if we have actual curriculum data
    has_curriculum_data = (
        curriculum is not None and 
        len(curriculum_content.get("topics", [])) > 0
    )
    
    # Prepare curriculum content strings for the prompt
    if has_curriculum_data:
        topics_str = "\n- " + "\n- ".join(curriculum_content["topics"])
        outcomes_str = "\n- " + "\n- ".join(curriculum_content["learning_outcomes"])
        experiences_str = "\n- " + "\n- ".join(curriculum_content["suggested_experiences"][:3]) if curriculum_content["suggested_experiences"] else "Use your expertise to design appropriate learning experiences"
        competencies_str = "\n".join(f"- {comp}" for comp in curriculum_content["core_competencies"][:2]) if curriculum_content["core_competencies"] else "- Communication and collaboration\n- Critical thinking and problem solving"
        values_str = "\n".join(f"- {val}" for val in curriculum_content["values"][:2]) if curriculum_content["values"] else "- Responsibility\n- Respect"
    else:
        topics_str = "No specific topics provided - use your expertise in this subject area"
        outcomes_str = "Generate appropriate learning outcomes for this strand and sub-strand"
        experiences_str = "Design engaging, age-appropriate learning experiences"
        competencies_str = "- Communication and collaboration\n- Critical thinking and problem solving\n- Creativity and imagination\n- Digital literacy"
        values_str = "- Responsibility\n- Respect\n- Unity\n- Peace"
  

    prompt = f"""


You are a Kenyan secondary school teacher preparing a detailed CBC lesson plan for official school inspection.

VERY IMPORTANT RULES:
- Follow the lesson plan structure EXACTLY as provided
- Do NOT change the order of sections
- Do NOT rename headings
- Do NOT add or remove fields
- Use simple, teacher-centered Kenyan lesson plan language
- Write DETAILED, COMPREHENSIVE descriptions (3-5 sentences minimum per section)
- Each development step should be at least 4-5 sentences with specific activities

Use phrases commonly found in teachers' lesson plan records such as:
- "Learners brainstorm..."
- "Learners discuss..."
- "Learners work in groups..."
- "Learners are guided to..."

LESSON PLAN TEMPLATE (DO NOT ALTER):
{json.dumps(template, indent=2)}

ADMINISTRATIVE DETAILS (FILL EXACTLY):
School: {request.school}
Subject: {request.subject}
Year: {request.date.split('-')[2]}
Term: {request.term}
Date: {request.date}
Time: {request.start_time} - {request.end_time}
Grade: {request.grade}
Roll:
- Boys: {request.boys}
- Girls: {request.girls}
- Total: {total_students}

TEACHER DETAILS:
Name: {request.teacher_name}
TSC Number: {request.teacher_tsc_number}

CURRICULUM DETAILS:
Strand: {curriculum_content["strand"]}
Sub-strand: {curriculum_content["sub_strand"]}

LESSON LEARNING OUTCOMES:
Write THREE comprehensive outcomes.
Begin each with an action verb (analyze, evaluate, create, apply, etc.)
Follow this statement exactly:
"By the end of the lesson, the learner should be able to:"
a) [Write a detailed, specific outcome with context - 1-2 sentences]
b) [Write a detailed, specific outcome with context - 1-2 sentences]
c) [Write a detailed, specific outcome with context - 1-2 sentences]

KEY INQUIRY QUESTION:
Write ONE thought-provoking inquiry question that encourages critical thinking about the sub-strand.
Make it open-ended and engaging for Grade {request.grade} learners.

LEARNING RESOURCES:
List 5-7 specific, practical classroom learning resources with details where helpful.
Examples: "Chart showing the water cycle", "Set of 30 calculators", "Biology textbook pages 45-52"

LESSON FLOW GUIDELINES - WRITE DETAILED DESCRIPTIONS:

INTRODUCTION (Write 4-6 sentences):
Describe a detailed opening activity where learners:
- Recall and share prior knowledge
- Connect to real-life experiences
- Get curious about the lesson topic
Include specific questions the teacher might ask and expected student responses.
Describe the classroom interaction in detail.

DEVELOPMENT (Each step should be 5-7 sentences):

Step 1 - Teacher-Led Exploration:
Describe in detail:
- Exactly what the teacher does (demonstrates, explains, shows)
- Specific examples or content covered
- How learners observe, take notes, or participate
- Questions asked and discussions held
- Materials or resources used
Make this vivid and specific to {request.subject}.

Step 2 - Guided Practice:
Describe in detail:
- The specific activity learners do (solve problems, conduct experiments, practice skills)
- How they're organized (pairs, groups of 4, individually)
- Exactly what they're working on (specific problems, questions, or tasks)
- How the teacher circulates and supports
- Examples of learner-teacher interactions
Be subject-specific for {request.subject}.

Step 3 - Independent Application & Presentation:
Describe in detail:
- What learners create, solve, or present
- How they share their work (presentations, gallery walk, peer review)
- Specific assessment questions or tasks
- How peers provide feedback
- Teacher's role in facilitating
Make this practical for a Grade {request.grade} {request.subject} class.

CONCLUSION (Write 4-6 sentences):
Describe in detail how:
- Learners summarize the key lesson points
- The teacher guides reflection on learning outcomes
- Connections are made to real life or future lessons
- Assessment of understanding occurs
- Learners are given homework or extension tasks (if appropriate)

TIME ALLOCATION:
- Introduction: 8-10 minutes
- Development: 25-30 minutes total (8-10 mins per step)
- Conclusion: 5-7 minutes
Total: 40 minutes

CRITICAL REQUIREMENT:
Every section must have SUBSTANTIAL DETAIL. Think of this as a lesson plan that another teacher could pick up and teach from.
Each description should paint a clear picture of what's happening in the classroom.

Write descriptions that are:
✅ Detailed (4-7 sentences per section)
✅ Specific to the subject and grade level
✅ Actionable (another teacher could follow them)
✅ Realistic for a Kenyan classroom
❌ NOT brief one-liners
❌ NOT vague generalities
❌ NOT overly academic language

FINAL OUTPUT RULE:
Return ONLY valid JSON.
No explanations.
No markdown.
No extra text.

"""

    try:
        print(f"🤖 Generating lesson plan for {request.subject} - Grade {request.grade}")
        print(f"📚 Strand: {curriculum_content['strand']}, Sub-strand: {curriculum_content['sub_strand']}")
        print(f"📊 Using curriculum data: {has_curriculum_data}")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Kenyan CBC teacher who creates detailed, engaging, standards-aligned lesson plans in JSON format. You understand the Kenyan education system, cultural context, and create culturally relevant, practical lessons. You are intelligent enough to interpret unclear or misspelled strand/sub-strand names and create appropriate content."
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
        print("✅ Lesson plan generated successfully")
        return lesson_plan
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        print(f"❌ OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# ============== API ENDPOINTS ==============

@app.get("/")
def read_root():
    return {
        "message": "CBC Lesson Plan Generator API - Smart Edition",
        "status": "running",
        "version": "2.0",
        "features": [
            "Fuzzy matching for subject names (handles typos)",
            "Fuzzy matching for strands and sub-strands",
            "Graceful handling of missing curriculum files",
            "AI-powered fallback for missing data",
            "No default subjects - fully customizable"
        ],
        "cors_enabled": True,
        "endpoints": {
            "health": "/health",
            "generate": "/generate-lesson-plan",
            "strands": "/strands/{subject}",
            "sub_strands": "/sub-strands/{subject}/{strand}",
            "curriculum": "/curriculum/{subject}"
        }
    }

@app.post("/generate-lesson-plan")
async def create_lesson_plan(request: LessonPlanRequest):
    """
    Generate a CBC-aligned lesson plan based on curriculum content.
    Intelligently handles typos and missing curriculum files.
    """
    try:
        print(f"\n{'='*60}")
        print(f"📝 New lesson plan request:")
        print(f"   Subject: {request.subject}")
        print(f"   Grade: {request.grade}")
        print(f"   Strand: {request.strand}")
        print(f"   Sub-strand: {request.sub_strand}")
        print(f"{'='*60}\n")
        
        lesson_plan = generate_lesson_plan(request)
        
        return {
            "success": True,
            "message": "Lesson plan generated successfully",
            "lesson_plan": lesson_plan
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating lesson plan: {str(e)}")

@app.get("/strands/{subject}")
def get_strands(subject: str):
    """
    Get all available strands for a subject.
    Uses fuzzy matching to handle typos in subject name.
    """
    try:
        curriculum = load_curriculum(subject)
        
        if curriculum is None:
            return {
                "success": False,
                "message": f"No curriculum file found for subject '{subject}'. Available curriculum files should be named like 'biology_curriculum.json'",
                "subject": subject,
                "strands": []
            }
        
        strands = [
            {
                "id": strand.get("id"),
                "name": strand.get("name"),
                "code": strand.get("code", "")
            }
            for strand in curriculum.get("strands", [])
        ]
        
        return {
            "success": True,
            "subject": subject,
            "strands": strands,
            "count": len(strands)
        }
    except Exception as e:
        print(f"❌ Error in get_strands: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sub-strands/{subject}/{strand}")
def get_sub_strands(subject: str, strand: str):
    """
    Get sub-strands for a specific strand.
    Uses fuzzy matching for both subject and strand names.
    """
    try:
        curriculum = load_curriculum(subject)
        
        if curriculum is None:
            return {
                "success": False,
                "message": f"No curriculum file found for subject '{subject}'",
                "strand": strand,
                "sub_strands": []
            }
        
        # Get all strand names for fuzzy matching
        strands = curriculum.get("strands", [])
        strand_names = [s.get("name", "") for s in strands if s.get("name")]
        
        # Find best match for strand
        best_strand_match = find_best_match(strand, strand_names, threshold=70)
        
        if not best_strand_match:
            return {
                "success": False,
                "message": f"Strand '{strand}' not found in {subject} curriculum",
                "available_strands": strand_names,
                "sub_strands": []
            }
        
        # Find the strand object
        for strand_obj in strands:
            if strand_obj.get("name", "") == best_strand_match:
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
                    "strand": best_strand_match,
                    "original_query": strand,
                    "sub_strands": sub_strands,
                    "count": len(sub_strands)
                }
        
        return {
            "success": False,
            "message": f"Strand '{strand}' not found",
            "sub_strands": []
        }
        
    except Exception as e:
        print(f"❌ Error in get_sub_strands: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/curriculum/{subject}")
def get_full_curriculum(subject: str):
    """
    Get the complete curriculum for a subject.
    Uses fuzzy matching to handle typos in subject name.
    """
    try:
        curriculum = load_curriculum(subject)
        
        if curriculum is None:
            return {
                "success": False,
                "message": f"No curriculum file found for subject '{subject}'",
                "curriculum": None
            }
        
        return {
            "success": True,
            "subject": subject,
            "curriculum": curriculum
        }
    except Exception as e:
        print(f"❌ Error in get_full_curriculum: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """
    Check API health and list available curriculum files.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return {
                "status": "error",
                "message": "OpenAI API key not found in environment variables",
                "api": "running",
                "openai": "not configured"
            }
        
        # Find all curriculum files
        curriculum_files = [f for f in os.listdir('.') if f.endswith('_curriculum.json')]
        available_subjects = [f.replace('_curriculum.json', '').title() for f in curriculum_files]
        
        files_exist = {
            "lesson_plan_template.json": os.path.exists("lesson_plan_template.json"),
            "curriculum_files": curriculum_files
        }
        
        return {
            "status": "healthy",
            "message": "API is running with smart features enabled",
            "api": "running",
            "openai": "configured",
            "files": files_exist,
            "available_subjects": available_subjects,
            "features": {
                "fuzzy_matching": "enabled",
                "typo_handling": "enabled",
                "missing_curriculum_fallback": "enabled"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "api": "running with errors"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Starting CBC Lesson Plan Generator API on port {port}")
    print(f"📚 Smart features enabled: Fuzzy matching, Typo handling, Missing curriculum fallback")
    uvicorn.run(app, host="0.0.0.0", port=port)