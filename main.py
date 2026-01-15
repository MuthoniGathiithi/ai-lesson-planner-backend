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
app = FastAPI(title="CBC Lesson Plan Generator", version="2.1")

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

# ============== SUBJECT-SPECIFIC TERMINOLOGY ==============
SUBJECT_TERMINOLOGY = {
    "mathematics": {
        "action_verbs": [
            "calculate", "solve", "compute", "derive", "prove", "simplify", 
            "factorize", "graph", "plot", "measure", "estimate", "construct",
            "verify", "determine", "formulate", "apply"
        ],
        "key_terms": [
            "equation", "formula", "theorem", "proof", "variable", "constant",
            "function", "algorithm", "pattern", "relationship", "properties",
            "operations", "expressions", "inequalities", "coordinates", "ratio",
            "proportion", "percentage", "angle", "area", "volume", "perimeter"
        ],
        "language_style": "Use precise mathematical language with exact terminology. Include specific formulas, calculations, and step-by-step problem-solving approaches.",
        "example_outcomes": [
            "Calculate the area of composite shapes using appropriate formulas and show all working steps clearly",
            "Solve linear equations by applying inverse operations and verify solutions through substitution method"
        ]
    },
    "biology": {
        "action_verbs": [
            "observe", "classify", "identify", "dissect", "examine", "investigate",
            "analyze", "compare", "differentiate", "describe", "explain biological processes",
            "demonstrate", "relate", "predict", "hypothesize"
        ],
        "key_terms": [
            "organism", "cell", "tissue", "organ", "system", "species", "adaptation",
            "evolution", "ecosystem", "photosynthesis", "respiration", "metabolism",
            "reproduction", "genetics", "DNA", "protein", "enzyme", "membrane",
            "mitosis", "meiosis", "homeostasis", "biodiversity"
        ],
        "language_style": "Use scientific terminology with biological processes and systems. Focus on observation, experimentation, and understanding of living organisms.",
        "example_outcomes": [
            "Classify organisms into their taxonomic groups based on observable characteristics and structural features",
            "Explain the process of photosynthesis showing how plants convert light energy into chemical energy"
        ]
    },
    "chemistry": {
        "action_verbs": [
            "react", "combine", "separate", "test", "analyze", "synthesize",
            "balance equations", "calculate molarity", "observe reactions", "measure",
            "determine", "identify compounds", "predict products", "neutralize"
        ],
        "key_terms": [
            "atom", "molecule", "element", "compound", "reaction", "bond", "ion",
            "acid", "base", "pH", "solution", "solvent", "solute", "catalyst",
            "oxidation", "reduction", "mole", "molarity", "titration", "precipitation",
            "equilibrium", "energy", "exothermic", "endothermic"
        ],
        "language_style": "Use chemical terminology with focus on reactions, equations, and laboratory procedures. Include safety considerations and practical applications.",
        "example_outcomes": [
            "Balance chemical equations by applying the law of conservation of mass and verify atom counts",
            "Determine the pH of solutions using indicators and explain the relationship between hydrogen ion concentration"
        ]
    },
    "physics": {
        "action_verbs": [
            "measure", "calculate forces", "determine", "apply laws", "analyze motion",
            "investigate", "demonstrate", "explain phenomena", "compute", "predict",
            "verify", "experiment", "observe patterns", "derive formulas"
        ],
        "key_terms": [
            "force", "energy", "work", "power", "momentum", "velocity", "acceleration",
            "mass", "weight", "friction", "gravity", "pressure", "density", "motion",
            "electricity", "magnetism", "wave", "frequency", "wavelength", "circuit",
            "resistance", "current", "voltage", "heat", "temperature"
        ],
        "language_style": "Use precise physical terminology with focus on laws, principles, and quantitative relationships. Include mathematical calculations and SI units.",
        "example_outcomes": [
            "Calculate velocity and acceleration of moving objects by applying equations of motion with proper SI units",
            "Determine the work done by a force using the formula and explain energy transformations in the system"
        ]
    },
    "geography": {
        "action_verbs": [
            "locate", "map", "identify regions", "analyze patterns", "describe landscapes",
            "interpret", "compare", "explain processes", "investigate", "observe",
            "measure", "collect data", "draw maps", "evaluate", "assess"
        ],
        "key_terms": [
            "landform", "climate", "ecosystem", "region", "location", "migration",
            "urbanization", "population", "resources", "agriculture", "industry",
            "environment", "weathering", "erosion", "deposition", "plate tectonics",
            "latitude", "longitude", "altitude", "vegetation", "settlement"
        ],
        "language_style": "Use geographical terminology focusing on spatial relationships, environmental processes, and human-environment interactions.",
        "example_outcomes": [
            "Identify major landforms on topographic maps using contour lines and explain their formation processes clearly",
            "Analyze population distribution patterns in Kenya and explain factors influencing settlement in different regions"
        ]
    },
    "history": {
        "action_verbs": [
            "trace", "describe events", "analyze causes", "evaluate impact", "compare periods",
            "explain significance", "identify", "sequence", "interpret sources", "investigate",
            "examine", "relate", "assess", "discuss", "contextualize"
        ],
        "key_terms": [
            "era", "period", "civilization", "empire", "colonization", "independence",
            "revolution", "constitution", "governance", "society", "culture", "trade",
            "migration", "conflict", "treaty", "reform", "movement", "nationalism",
            "heritage", "archaeology", "chronology", "primary source", "secondary source"
        ],
        "language_style": "Use historical terminology with focus on chronology, causation, and significance. Include dates, events, and historical figures.",
        "example_outcomes": [
            "Trace the development of trade routes in pre-colonial Kenya and explain their economic and social impact",
            "Analyze the causes and effects of the Mau Mau uprising showing its significance in Kenya's independence struggle"
        ]
    },
    "kiswahili": {
        "action_verbs": [
            "kusoma", "kuandika", "kueleza", "kufafanua", "kutambua", "kutumia",
            "kujadili", "kulinganisha", "kuchambua", "kuainisha", "kusimulia",
            "kutafsiri", "kuhutubia", "kuimba", "kucheza"
        ],
        "key_terms": [
            "sarufi", "nomino", "vitenzi", "vivumishi", "vihusishi", "tungo",
            "sentensi", "aya", "insha", "utunzi", "msemo", "methali", "vitendawili",
            "hadithi", "mashairi", "maana", "matumizi", "muktadha", "lugha",
            "mazungumzo", "hotuba", "maandishi"
        ],
        "language_style": "Tumia lugha ya Kiswahili sanifu. Zingatia sarufi sahihi, matumizi ya maneno, na ufahamu wa muktadha. Jumuisha mifano ya vitendawili, methali, na misemo.",
        "example_outcomes": [
            "Kutambua na kueleza aina mbalimbali za nomino katika sentensi zilizoandikwa kwa Kiswahili sanifu",
            "Kuandika insha ya kusimulia kwa kutumia lugha nzuri na kuzingatia mpangilio sahihi wa matukio"
        ],
        "language": "kiswahili"
    },
    "english": {
        "action_verbs": [
            "read", "write", "analyze", "interpret", "compose", "evaluate",
            "discuss", "compare", "summarize", "identify literary devices", "express",
            "present", "argue", "persuade", "narrate", "describe"
        ],
        "key_terms": [
            "grammar", "syntax", "vocabulary", "comprehension", "composition", "essay",
            "paragraph", "sentence", "phrase", "tense", "speech", "dialogue",
            "narrative", "poetry", "metaphor", "simile", "theme", "plot", "character",
            "tone", "style", "genre", "punctuation"
        ],
        "language_style": "Use proper English language terminology with focus on grammar, composition, and literary analysis. Include examples of correct usage.",
        "example_outcomes": [
            "Compose a well-structured essay using appropriate vocabulary and demonstrating correct grammar and punctuation throughout",
            "Analyze literary devices in a given text and explain their effect on meaning and reader engagement"
        ]
    },
    "business": {
        "action_verbs": [
            "calculate profit", "analyze markets", "develop strategies", "evaluate performance",
            "prepare accounts", "determine costs", "forecast", "budget", "market",
            "negotiate", "manage resources", "record transactions", "assess"
        ],
        "key_terms": [
            "profit", "loss", "revenue", "expenses", "capital", "assets", "liabilities",
            "market", "demand", "supply", "price", "competition", "entrepreneur",
            "investment", "budget", "accounting", "finance", "management", "marketing",
            "production", "trade", "economy", "business plan"
        ],
        "language_style": "Use business and commercial terminology with focus on practical applications, calculations, and real-world scenarios.",
        "example_outcomes": [
            "Calculate gross and net profit for a business using correct formulas and interpret financial performance indicators",
            "Develop a simple marketing strategy for a product by analyzing target market and identifying promotional methods"
        ]
    },
    "agriculture": {
        "action_verbs": [
            "plant", "cultivate", "harvest", "identify crops", "practice", "demonstrate techniques",
            "prepare soil", "control pests", "manage", "apply", "select", "maintain",
            "conserve", "improve", "assess crop health", "determine"
        ],
        "key_terms": [
            "crop", "livestock", "soil", "fertilizer", "irrigation", "pest", "disease",
            "harvest", "planting", "cultivation", "farm", "agriculture", "rotation",
            "organic matter", "nutrients", "yield", "breed", "pasture", "fodder",
            "conservation", "sustainability", "tool", "implement"
        ],
        "language_style": "Use agricultural terminology with practical focus on farming practices, crop management, and livestock care.",
        "example_outcomes": [
            "Demonstrate proper land preparation techniques and explain how soil structure affects crop growth and productivity",
            "Identify common crop pests and diseases by their symptoms and recommend appropriate organic control methods"
        ]
    },
    "computer": {
        "action_verbs": [
            "code", "program", "debug", "design", "create", "develop applications",
            "analyze algorithms", "implement", "test", "troubleshoot", "configure",
            "install", "operate", "process data", "network", "secure systems"
        ],
        "key_terms": [
            "algorithm", "program", "code", "software", "hardware", "data", "input",
            "output", "processing", "storage", "network", "internet", "database",
            "variable", "loop", "function", "syntax", "debugging", "application",
            "operating system", "file", "folder", "cybersecurity"
        ],
        "language_style": "Use computing terminology with focus on logical thinking, problem-solving, and practical ICT skills.",
        "example_outcomes": [
            "Write simple algorithms using pseudocode and flowcharts to solve logical problems with clear step-by-step instructions",
            "Create a basic program using appropriate programming constructs and test it to ensure correct functionality"
        ]
    },
    "home science": {
        "action_verbs": [
            "prepare", "cook", "sew", "demonstrate", "practice hygiene", "maintain",
            "manage", "budget", "select", "evaluate nutrition", "design", "create",
            "apply", "identify", "care for", "preserve"
        ],
        "key_terms": [
            "nutrition", "diet", "health", "hygiene", "cooking", "recipe", "ingredients",
            "meal planning", "food preservation", "sewing", "fabric", "pattern",
            "stitching", "household management", "budgeting", "childcare", "family",
            "cleanliness", "safety", "balanced diet", "nutrients"
        ],
        "language_style": "Use home economics terminology with focus on practical life skills, nutrition, and household management.",
        "example_outcomes": [
            "Prepare a balanced meal by selecting appropriate ingredients and applying proper cooking methods for nutrition retention",
            "Demonstrate basic sewing techniques to construct a simple garment using correct stitching and finishing methods"
        ]
    },
    "cre": {
        "action_verbs": [
            "explain teachings", "relate", "apply values", "discuss", "identify",
            "interpret", "demonstrate", "practice", "reflect", "compare",
            "describe", "analyze", "evaluate", "live by principles"
        ],
        "key_terms": [
            "faith", "prayer", "worship", "Bible", "scripture", "commandment",
            "covenant", "prophet", "salvation", "grace", "sin", "forgiveness",
            "love", "compassion", "justice", "moral", "ethics", "values",
            "Christian living", "church", "ministry", "testimony"
        ],
        "language_style": "Use religious and moral terminology with focus on biblical teachings, values, and Christian living.",
        "example_outcomes": [
            "Explain the Ten Commandments and demonstrate how to apply them in daily life situations showing moral understanding",
            "Discuss the importance of forgiveness in Christian teaching and relate it to personal relationships and community harmony"
        ]
    },
    "ire": {
        "action_verbs": [
            "recite", "explain teachings", "practice", "apply", "demonstrate",
            "discuss", "identify", "interpret", "relate", "observe",
            "memorize", "reflect", "compare", "describe Islamic principles"
        ],
        "key_terms": [
            "Quran", "Hadith", "Sunnah", "Iman", "Islam", "Tawheed", "Salah",
            "Zakah", "Sawm", "Hajj", "Prophet Muhammad (PBUH)", "Allah", "mosque",
            "faith", "worship", "prayer", "fasting", "charity", "pilgrimage",
            "Muslim", "Islamic values", "morals", "ethics"
        ],
        "language_style": "Use Islamic terminology with focus on Quranic teachings, Hadith, and Islamic values and practices.",
        "example_outcomes": [
            "Recite selected Surahs correctly and explain their meanings showing understanding of Quranic teachings and messages",
            "Explain the five pillars of Islam and demonstrate how Muslims practice them in daily life situations"
        ]
    }
}

def get_subject_guidance(subject: str) -> Dict[str, Any]:
    """
    Get subject-specific terminology and guidance.
    Uses fuzzy matching to handle variations in subject names.
    """
    subject_lower = subject.lower().strip()
    
    # Try exact match first
    if subject_lower in SUBJECT_TERMINOLOGY:
        return SUBJECT_TERMINOLOGY[subject_lower]
    
    # Try fuzzy matching
    available_subjects = list(SUBJECT_TERMINOLOGY.keys())
    best_match = find_best_match(subject_lower, available_subjects, threshold=75)
    
    if best_match:
        print(f"📚 Matched '{subject}' to subject terminology: '{best_match}'")
        return SUBJECT_TERMINOLOGY[best_match]
    
    # Default generic guidance
    print(f"⚠️ No specific terminology for '{subject}', using generic guidance")
    return {
        "action_verbs": [
            "analyze", "evaluate", "create", "apply", "demonstrate", "explain",
            "investigate", "compare", "develop", "interpret", "construct"
        ],
        "key_terms": [],
        "language_style": "Use clear, precise academic language appropriate for the subject matter.",
        "example_outcomes": []
    }

# ============== REQUEST MODELS ==============
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
    """
    if not query or not options:
        return None
    result = process.extractOne(query, options, scorer=fuzz.ratio)
    if result and result[1] >= threshold:
        print(f"✅ Fuzzy match: '{query}' → '{result[0]}' (similarity: {result[1]}%)")
        return result[0]
    print(f"⚠️ No good match for '{query}' (best score: {result[1] if result else 0}%)")
    return None

def load_curriculum(subject: str) -> Optional[Dict[str, Any]]:
    """
    Load curriculum JSON file for a subject with smart name matching.
    """
    try:
        filename = f"{subject.lower()}_curriculum.json"
        with open(filename, 'r', encoding='utf-8') as f:
            curriculum = json.load(f)
        print(f"✅ Successfully loaded curriculum file: {filename}")
        return curriculum
    except FileNotFoundError:
        print(f"⚠️ Curriculum file '{subject.lower()}_curriculum.json' not found. Trying fuzzy match...")
        curriculum_files = [f for f in os.listdir('.') if f.endswith('_curriculum.json')]
        if curriculum_files:
            available_subjects = [f.replace('_curriculum.json', '') for f in curriculum_files]
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
    """
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
    
    if curriculum is None:
        print(f"ℹ️ No curriculum data available. Using strand: '{strand_name}', sub-strand: '{sub_strand_name}'")
        return empty_structure
    
    strands = curriculum.get("strands", [])
    if not strands:
        return empty_structure
    
    strand_names = [strand.get("name", "") for strand in strands if strand.get("name")]
    best_strand_match = find_best_match(strand_name, strand_names, threshold=70)
    
    if not best_strand_match:
        return empty_structure
    
    matched_strand = None
    for strand in strands:
        if strand.get("name", "") == best_strand_match:
            matched_strand = strand
            break
    
    if not matched_strand:
        return empty_structure
    
    sub_strands = matched_strand.get("sub_strands", [])
    sub_strand_names = [ss.get("name", "") for ss in sub_strands if ss.get("name")]
    best_substrand_match = find_best_match(sub_strand_name, sub_strand_names, threshold=70)
    
    if not best_substrand_match:
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
    
    matched_sub_strand = None
    for ss in sub_strands:
        if ss.get("name", "") == best_substrand_match:
            matched_sub_strand = ss
            break
    
    if not matched_sub_strand:
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
    Generate a CBC-aligned lesson plan using OpenAI with subject-specific terminology.
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
    
    # Get subject-specific guidance
    t0 = time.perf_counter()
    subject_guidance = get_subject_guidance(request.subject)
    t_subject_guidance = time.perf_counter() - t0
    
    total_students = request.boys + request.girls
    has_curriculum_data = (
        curriculum is not None and
        len(curriculum_content.get("topics", [])) > 0
    )
    
    # Determine if Kiswahili
    is_kiswahili = subject_guidance.get("language") == "kiswahili"
    
    # FIX ISSUE #2: Use CORRECTED subject name from fuzzy matching
    # Get the actual corrected subject name
    corrected_subject = request.subject
    if curriculum:
        # If curriculum was loaded via fuzzy matching, get the corrected name
        curriculum_files = [f for f in os.listdir('.') if f.endswith('_curriculum.json')]
        if curriculum_files:
            available_subjects = [f.replace('_curriculum.json', '') for f in curriculum_files]
            fuzzy_match = find_best_match(request.subject.lower(), available_subjects, threshold=75)
            if fuzzy_match:
                # Capitalize properly (first letter of each word)
                corrected_subject = fuzzy_match.title()
                print(f"✅ Using corrected subject name: '{request.subject}' → '{corrected_subject}'")
    
    # Also correct strand and substrand names from curriculum matching
    corrected_strand = curriculum_content.get("strand", request.strand)
    corrected_substrand = curriculum_content.get("sub_strand", request.sub_strand)
    
    # Prepare curriculum content strings
    if has_curriculum_data:
        topics_str = "\n- " + "\n- ".join(curriculum_content["topics"])
        outcomes_str = "\n- " + "\n- ".join(curriculum_content["learning_outcomes"])
        experiences_str = "\n- " + "\n- ".join(curriculum_content["suggested_experiences"][:3]) if curriculum_content["suggested_experiences"] else "Use your expertise to design appropriate learning experiences"
    else:
        topics_str = "No specific topics provided - use your expertise in this subject area"
        outcomes_str = "Generate appropriate learning outcomes for this strand and sub-strand"
        experiences_str = "Design engaging, age-appropriate learning experiences"

    # Build subject-specific guidance
    action_verbs_str = ", ".join(subject_guidance["action_verbs"][:10])
    key_terms_str = ", ".join(subject_guidance["key_terms"][:15]) if subject_guidance["key_terms"] else "subject-appropriate terms"
    
    example_outcomes_section = ""
    if subject_guidance.get("example_outcomes"):
        example_outcomes_section = f"""
EXAMPLE OUTCOMES FOR {request.subject.upper()}:
{chr(10).join(f"- {outcome}" for outcome in subject_guidance["example_outcomes"])}
"""

    t0 = time.perf_counter()
    
    # Build language-specific instructions
    if is_kiswahili:
        language_instruction = f"""
🌍 MUHIMU SANA: SOMO HILI NI LA KISWAHILI - ANDIKA YOTE KWA KISWAHILI SANIFU!

MAELEKEZO MAHUSUSI YA KISWAHILI:
- Andika YOTE kwa Kiswahili sanifu (HAPANA Kiingereza)
- Tumia maneno ya Kiswahili sahihi na sarufi nzuri
- Anza matokeo ya kujifunza kwa vitenzi vya Kiswahili: {action_verbs_str}
- Tumia istilahi za Kiswahili: {key_terms_str}
- Jumuisha mifano ya methali, misemo, au vitendawili pale inapofaa
- Andika maswali kwa Kiswahili
- Eleza shughuli za darasa kwa Kiswahili
- Hakikisha mpango WOTE wa somo uko kwa Kiswahili

TAFSIRI ZA VICHWA VYA HABARI (TRANSLATIONS FOR SECTION HEADERS):
- "Administrative Details" → "MAELEZO YA KIUTAWALA"
- "School" → "Shule"
- "Subject" → "Somo"
- "Year" → "Mwaka"
- "Term" → "Muhula"
- "Date" → "Tarehe"
- "Time" → "Muda"
- "Grade" → "Darasa"
- "Roll" → "Orodha ya Wanafunzi"
- "Boys" → "Wavulana"
- "Girls" → "Wasichana"
- "Total" → "Jumla"
- "Teacher Details" → "MAELEZO YA MWALIMU"
- "Name" → "Jina"
- "TSC Number" → "Namba ya TSC"
- "Strand" → "MSTARI"
- "Sub-strand" → "MSTARI MDOGO"
- "Lesson Learning Outcomes" → "MATOKEO YA KUJIFUNZA"
- "By the end of the lesson, the learner should be able to:" → "Mwishoni mwa somo, mwanafunzi aweze:"
- "Key Inquiry Question" → "SWALI KUU LA UCHUNGUZI"
- "Learning Resources" → "VIFAA VYA KUJIFUNZIA"
- "Lesson Flow" → "MTIRIRIKO WA SOMO"
- "Introduction" → "Utangulizi"
- "Development" → "Maendeleo"
- "Conclusion" → "Hitimisho"
- "Step" → "Hatua"

MUHIMU: Tumia tafsiri hizi KATIKA JSON OUTPUT. Vichwa vyote viwe kwa Kiswahili!

{subject_guidance["language_style"]}
{example_outcomes_section}
"""
    else:
        language_instruction = f"""
SUBJECT-SPECIFIC TERMINOLOGY FOR {corrected_subject.upper()}:

REQUIRED ACTION VERBS (Use these in learning outcomes):
{action_verbs_str}

KEY TERMINOLOGY TO INCORPORATE:
{key_terms_str}

LANGUAGE STYLE:
{subject_guidance["language_style"]}
{example_outcomes_section}

CRITICAL: All learning outcomes MUST use action verbs from the list above.
Make the language specific to {corrected_subject}, not generic.
"""

    prompt = f"""
You are a Kenyan secondary school teacher for grade {request.grade} preparing a COMPREHENSIVE, DETAILED CBC lesson plan.

{language_instruction}

CRITICAL INSTRUCTIONS:
1. Follow the lesson plan structure EXACTLY
2. Use SUBJECT-SPECIFIC terminology from the guidance above
3. Write detailed descriptions with exact word counts as specified
4. {"WRITE EVERYTHING IN KISWAHILI SANIFU - Including ALL field names, section headers, and content" if is_kiswahili else f"Use {corrected_subject}-specific language throughout"}

LESSON PLAN TEMPLATE:
{json.dumps(template, indent=2)}

ADMINISTRATIVE DETAILS {"(MAELEZO YA KIUTAWALA - Write field names in Kiswahili!)" if is_kiswahili else ""}:
{"Shule" if is_kiswahili else "School"}: {request.school}
{"Somo" if is_kiswahili else "Subject"}: {corrected_subject}
{"Mwaka" if is_kiswahili else "Year"}: {request.date.split('-')[0]}
{"Muhula" if is_kiswahili else "Term"}: {request.term}
{"Tarehe" if is_kiswahili else "Date"}: {request.date}
{"Muda" if is_kiswahili else "Time"}: {request.start_time} - {request.end_time}
{"Darasa" if is_kiswahili else "Grade"}: {request.grade}
{"Orodha ya Wanafunzi" if is_kiswahili else "Roll"}: {"Wavulana" if is_kiswahili else "Boys"}: {request.boys}, {"Wasichana" if is_kiswahili else "Girls"}: {request.girls}, {"Jumla" if is_kiswahili else "Total"}: {total_students}

TEACHER DETAILS {"(MAELEZO YA MWALIMU)" if is_kiswahili else ""}:
{"Jina" if is_kiswahili else "Name"}: {request.teacher_name}
{"Namba ya TSC" if is_kiswahili else "TSC Number"}: {request.teacher_tsc_number}

CURRICULUM DETAILS {"(Tumia majina yaliyosahihishwa!)" if is_kiswahili else "(Use corrected names!)"}:
{"Mstari" if is_kiswahili else "Strand"}: {corrected_strand}
{"Mstari Mdogo" if is_kiswahili else "Sub-strand"}: {corrected_substrand}

LESSON LEARNING OUTCOMES {"(MATOKEO YA KUJIFUNZA)" if is_kiswahili else ""}:
Write THREE detailed, measurable outcomes.
{"Start with Kiswahili action verbs: " + action_verbs_str if is_kiswahili else f"Start with {corrected_subject}-specific action verbs: " + action_verbs_str}
Each outcome MUST be EXACTLY 20 WORDS.

Format:
{"Mwishoni mwa somo, mwanafunzi aweze:" if is_kiswahili else "By the end of the lesson, the learner should be able to:"}
a) [20 WORDS {"- Kiswahili" if is_kiswahili else f"- {corrected_subject} terminology"}]
b) [20 WORDS {"- Kiswahili" if is_kiswahili else f"- {corrected_subject} terminology"}]
c) [20 WORDS {"- Kiswahili" if is_kiswahili else f"- {corrected_subject} terminology"}]

KEY INQUIRY QUESTION {"(SWALI KUU LA UCHUNGUZI)" if is_kiswahili else ""} (MAXIMUM 10 WORDS):
{"Kwa Kiswahili, " if is_kiswahili else f"Using {corrected_subject} terminology, "}write a thought-provoking question about {corrected_substrand}.

LEARNING RESOURCES {"(VIFAA VYA KUJIFUNZIA)" if is_kiswahili else ""} (4-6 items):
List specific resources {"kwa Kiswahili" if is_kiswahili else f"relevant to {corrected_subject}"}.

LESSON FLOW {"(MTIRIRIKO WA SOMO)" if is_kiswahili else ""}:

{"UTANGULIZI" if is_kiswahili else "INTRODUCTION"} (MAXIMUM 10 WORDS):
Describe the opening activity{"kwa Kiswahili" if is_kiswahili else f" using {corrected_subject} context"}.

{"MAENDELEO" if is_kiswahili else "DEVELOPMENT"} (3-5 {"HATUA" if is_kiswahili else "STEPS"}, ~20 WORDS EACH):
Describe each {"hatua" if is_kiswahili else "step"}{"kwa Kiswahili" if is_kiswahili else f" with {corrected_subject}-specific activities"}.

{"HITIMISHO" if is_kiswahili else "CONCLUSION"} (MAXIMUM 15 WORDS):
Describe how learners summarize{"kwa Kiswahili" if is_kiswahili else f" using {corrected_subject} concepts"}.

JSON OUTPUT REQUIREMENTS:
⚠️ Subject field MUST contain: "{corrected_subject}" (NOT "{request.subject}")
⚠️ Strand field MUST contain: "{corrected_strand}" (NOT "{request.strand}")
⚠️ SubStrand field MUST contain: "{corrected_substrand}" (NOT "{request.sub_strand}")
{"⚠️ ALL SECTION HEADERS AND FIELD NAMES IN KISWAHILI!" if is_kiswahili else f"⚠️ Use {corrected_subject.upper()} terminology!"}
{"⚠️ Example: Use 'Shule' not 'School', 'Tarehe' not 'Date', 'Mwalimu' not 'Teacher'" if is_kiswahili else ""}

QUALITY STANDARDS:
✅ Subject: "{corrected_subject}" (corrected from "{request.subject}")
✅ Strand: "{corrected_strand}" (corrected from "{request.strand}")
✅ SubStrand: "{corrected_substrand}" (corrected from "{request.sub_strand}")
{"✅ ALL field labels in Kiswahili (Shule, Somo, Tarehe, Muda, etc.)" if is_kiswahili else ""}
{"✅ Section headers in Kiswahili (MAELEZO YA KIUTAWALA, MATOKEO YA KUJIFUNZA, etc.)" if is_kiswahili else ""}
{"✅ ENTIRE CONTENT in Kiswahili sanifu" if is_kiswahili else f"✅ {corrected_subject}-specific terminology throughout"}

Return ONLY valid JSON. No markdown. No explanations.
"""

    t_prompt_build = time.perf_counter() - t0

    try:
        print(f"🤖 Generating lesson plan for {corrected_subject} - Grade {request.grade}")
        print(f"   Original input: Subject='{request.subject}', Strand='{request.strand}', Sub-strand='{request.sub_strand}'")
        print(f"   Corrected to: Subject='{corrected_subject}', Strand='{corrected_strand}', Sub-strand='{corrected_substrand}'")
        print(f"🎯 Subject-specific guidance: Active")
        if is_kiswahili:
            print(f"🌍 Language: KISWAHILI (Full translation enabled)")

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert Kenyan CBC teacher who creates detailed lesson plans with subject-specific terminology.{' For Kiswahili lessons, you write EVERYTHING in Kiswahili sanifu including ALL field names (Shule, Somo, Tarehe, Muda, etc.), section headers (MAELEZO YA KIUTAWALA, MATOKEO YA KUJIFUNZA, etc.), and all content.' if is_kiswahili else ''}"
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
        print(f"⏱️ Total: {t_total*1000:.0f}ms")
        print("✅ Lesson plan generated successfully")
        return lesson_plan
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== API ENDPOINTS ==============
@app.get("/")
def read_root():
    return {
        "message": "CBC Lesson Plan Generator - Enhanced",
        "version": "2.1",
        "features": [
            "Subject-specific terminology",
            "Kiswahili lesson support",
            "Fuzzy matching for typos"
        ],
        "supported_subjects": list(SUBJECT_TERMINOLOGY.keys())
    }

@app.post("/generate-lesson-plan")
async def create_lesson_plan(request: LessonPlanRequest):
    try:
        lesson_plan = generate_lesson_plan(request)
        return {
            "success": True,
            "message": "Lesson plan generated with subject-specific terminology",
            "lesson_plan": lesson_plan
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    curriculum_files = [f for f in os.listdir('.') if f.endswith('_curriculum.json')]
    return {
        "status": "healthy",
        "features": {
            "subject_terminology": "enabled",
            "kiswahili_support": "enabled",
            "fuzzy_matching": "enabled"
        },
        "supported_subjects": list(SUBJECT_TERMINOLOGY.keys()),
        "curriculum_files": curriculum_files
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 CBC Lesson Plan Generator (Enhanced) on port {port}")
    print(f"📚 Supported subjects: {', '.join(SUBJECT_TERMINOLOGY.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=port)