from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from typing import List, Optional, Dict, Any
from rapidfuzz import fuzz, process
import time

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
subject: str # NO DEFAULT - user must provide
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
t0_total = time.perf_counter()

t0 = time.perf_counter()
template = load_lesson_template()
t_template = time.perf_counter() - t0

t0 = time.perf_counter()
curriculum = load_curriculum(request.subject)
t_curriculum_load = time.perf_counter() - t0

t0 = time.perf_counter()
curriculum_content = extract_curriculum_content(
curriculum,
request.strand,
request.sub_strand
)
t_curriculum_extract = time.perf_counter() - t0
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

t0 = time.perf_counter()
prompt = f"""

You are a Kenyan secondary school teacher for grade 10 preparing a COMPREHENSIVE, DETAILED CBC lesson plan for official school inspection.

CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. Follow the lesson plan structure EXACTLY as provided
2. Do NOT change section names or order
3. Write detailed, specific descriptions with exact word counts as specified
4. Use authentic Kenyan teacher language
5. Be SPECIFIC - include actual examples, questions, and activities

LESSON PLAN TEMPLATE (FOLLOW EXACTLY):
{json.dumps(template, indent=2)}

ADMINISTRATIVE DETAILS (FILL THESE EXACTLY AS GIVEN):
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

CURRICULUM DETAILS (USE THESE EXACTLY):
Strand: {curriculum_content["strand"]}
Sub-strand: {curriculum_content["sub_strand"]}

LESSON LEARNING OUTCOMES:
Write THREE detailed, measurable outcomes.
Start each with an action verb (analyze, evaluate, create, apply, demonstrate, design, construct).
Each outcome MUST be EXACTLY 20 WORDS - no more, no less.

Format:
"By the end of the lesson, the learner should be able to:"
a) [Detailed outcome - EXACTLY 20 WORDS]
b) [Detailed outcome - EXACTLY 20 WORDS]
c) [Detailed outcome - EXACTLY 20 WORDS]

KEY INQUIRY QUESTION:
Write ONE thought-provoking, open-ended question that:
- Encourages critical thinking
- Relates directly to the sub-strand: {curriculum_content["sub_strand"]}
- Is appropriate for Grade {request.grade} studying {request.subject}
- Has no simple yes/no answer
- Is MAXIMUM 10 WORDS (can be shorter, but not longer)

LEARNING RESOURCES:
List 4-6 specific, practical resources. Be detailed:

LESSON FLOW - WRITE DETAILED DESCRIPTIONS:

INTRODUCTION (MAXIMUM 10 WORDS):
Write a detailed narrative describing:
- The exact opening activity (be specific about what happens)
- Specific questions the teacher asks (write out 2-3 actual questions)
- How learners respond and what they say
- How this connects to their prior knowledge or real life
- How curiosity is built about today's topic

Keep it focused and concise - MAXIMUM 10 WORDS TOTAL.

DEVELOPMENT - VARYING LENGTHS (3-5 STEPS):

You can create 3, 4, or 5 development steps depending on what makes sense for this {request.subject} lesson.
Each step should be approximately 20 WORDS.

For EACH step (whether you do 3, 4, or 5 steps), write approximately 20 WORDS covering:
- Exactly what happens in this step (specific actions)
- The specific content/concepts/activities involved
- Materials or resources used
- How learners are organized (groups, pairs, individual)
- Specific examples, questions, or tasks
- How the teacher supports learning
- What learners produce or demonstrate
- Assessment or feedback methods

Make each step practical and realistic for a Kenyan Grade {request.grade} {request.subject} classroom.

Step 1: [~20 words - Teacher-Led Exploration or first major activity]
Step 2: [~20 words - Guided Practice or second major activity]
Step 3: [~20 words - Independent Work/Application or third major activity]
[Step 4: Optional - ~20 words if needed]
[Step 5: Optional - ~20 words if needed]

CONCLUSION (15 WORDS MAXIMUM):
Write a detailed narrative covering:
- How learners summarize the main points (specific process)
- Key questions asked to review learning outcomes (write out 2-3 questions)
- How learners reflect on what they learned
- Real-life applications or connections made
- Brief assessment activity (exit ticket, quick quiz, reflection)
- Homework or extension task assigned (be specific)

Keep focused and purposeful - MAXIMUM 15 WORDS.

TIME ALLOCATION:
- Introduction: 8-10 minutes
- Development: 25-30 minutes total (distributed across your 3-5 steps)
- Conclusion: 5-7 minutes
Total: 40 minutes

QUALITY STANDARDS - YOUR OUTPUT MUST MEET THESE:
✅ Each Learning Outcome: EXACTLY 20 WORDS
✅ Key Inquiry Question: MAXIMUM 10 WORDS
✅ Introduction: MAXIMUM 10 WORDS
✅ Each Development Step: APPROXIMATELY 20 WORDS (3-5 steps total)
✅ Conclusion: MAXIMUM 15 WORDS
✅ Resources: 4-6 items with details
✅ Subject-specific content for {request.subject}
✅ Realistic for Kenyan classroom context
✅ Written in teacher's narrative voice
✅ Includes specific examples, questions, and activities

❌ NEVER exceed the word limits specified
❌ NEVER skip specific examples or actual questions
❌ NEVER use overly academic language
❌ NEVER make it generic - it MUST be specific to {request.subject}

FINAL OUTPUT:
Return ONLY valid JSON matching the template structure.
NO markdown code blocks.
NO explanations.
NO preamble.
Just pure JSON.



========================================
CRITICAL JSON OUTPUT REQUIREMENTS:
========================================
⚠️ The "subject" field MUST contain: "{request.subject}"
⚠️ The "strand" field MUST contain: "{curriculum_content["strand"]}"
⚠️ The "subStrand" field MUST contain: "{curriculum_content["sub_strand"]}"
⚠️ DO NOT leave these fields empty
⚠️ DO NOT use placeholder values
⚠️ DO NOT change these to different subjects/strands

QUALITY STANDARDS - YOUR OUTPUT MUST MEET THESE:
✅ Subject field: "{request.subject}" (EXACT MATCH REQUIRED)
✅ Strand field: "{curriculum_content["strand"]}" (EXACT MATCH REQUIRED)
✅ SubStrand field: "{curriculum_content["sub_strand"]}" (EXACT MATCH REQUIRED)
✅ Each Learning Outcome: EXACTLY 20 WORDS, relevant to {request.subject}
✅ Key Inquiry Question: MAXIMUM 10 WORDS, about {curriculum_content["sub_strand"]}
✅ Introduction: MAXIMUM 10 WORDS, specific to {request.subject}
✅ Each Development Step: APPROXIMATELY 20 WORDS (3-5 steps total), teaching {curriculum_content["sub_strand"]}
✅ Conclusion: MAXIMUM 15 WORDS, reviewing {curriculum_content["sub_strand"]}
✅ Resources: 3-5 items relevant to {request.subject}
✅ All content specific to {request.subject}, not generic

FINAL OUTPUT:
Return ONLY valid JSON matching the template structure.
NO markdown code blocks.
NO explanations.
NO preamble.
Just pure JSON.

"""

t_prompt_build = time.perf_counter() - t0





try:
print(f"🤖 Generating lesson plan for {request.subject} - Grade {request.grade}")
print(f"📚 Strand: {curriculum_content['strand']}, Sub-strand: {curriculum_content['sub_strand']}")
print(f"📊 Using curriculum data: {has_curriculum_data}")

t0 = time.perf_counter()
response = client.chat.completions.create(
model="gpt-4o-mini",
messages=[
{
"role": "system",
"content": "You are an expert Kenyan CBC teacher for grade 10 who creates detailed, engaging, standards-aligned lesson plans in JSON format. You understand the Kenyan education system for grade 10 , cultural context, and create culturally relevant, practical lessons. You are intelligent enough to interpret unclear or misspelled strand/sub-strand names and create appropriate content."
},
{
"role": "user",
"content": prompt
}
],
temperature=0.7,
response_format={"type": "json_object"}
)
t_openai = time.perf_counter() - t0

t0 = time.perf_counter()
lesson_plan = json.loads(response.choices[0].message.content)
t_json_parse = time.perf_counter() - t0

t_total = time.perf_counter() - t0_total
print(
"⏱️ Timing(ms) | "
f"template={t_template*1000:.0f} "
f"curriculum_load={t_curriculum_load*1000:.0f} "
f"curriculum_extract={t_curriculum_extract*1000:.0f} "
f"prompt_build={t_prompt_build*1000:.0f} "
f"openai={t_openai*1000:.0f} "
f"json_parse={t_json_parse*1000:.0f} "
f"total={t_total*1000:.0f}"
)
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
t_req_start = time.perf_counter()
print(f"\n{'='*60}")
print(f"📝 New lesson plan request:")
print(f" Subject: {request.subject}")
print(f" Grade: {request.grade}")
print(f" Strand: {request.strand}")
print(f" Sub-strand: {request.sub_strand}")
print(f"{'='*60}\n")
lesson_plan = generate_lesson_plan(request)

t_req_total = time.perf_counter() - t_req_start
print(f"⏱️ Request total(ms): {t_req_total*1000:.0f}")
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
