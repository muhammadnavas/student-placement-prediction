import config
from model import load_models

def generate_recommendations(student_data: dict) -> dict:
    """
    Analyzes student metrics against placed cohort benchmarks,
    incorporates Random Forest feature importance weights, and produces
    prioritized, actionable recommendations and skill gap analysis.
    """
    models = load_models()
    feat_importances = models.get("Feature Importance", {
        "cgpa": 0.268,
        "aptitude_score": 0.189,
        "communication_score": 0.165,
        "technical_score": 0.150,
        "projects": 0.082,
        "backlogs": 0.056,
        "internships": 0.046,
        "certifications": 0.043
    })
    
    benchmarks = config.PLACED_BENCHMARKS
    gaps = {}
    recommendation_items = []
    
    # Extract student values
    cgpa = float(student_data.get("cgpa", 0.0))
    aptitude = float(student_data.get("aptitude_score", 0.0))
    comm = float(student_data.get("communication_score", 0.0))
    tech = float(student_data.get("technical_score", 0.0))
    internships = int(student_data.get("internships", 0))
    projects = int(student_data.get("projects", 0))
    certs = int(student_data.get("certifications", 0))
    backlogs = int(student_data.get("backlogs", 0))
    
    # 1. Backlogs Assessment (High critical priority if > 0)
    if backlogs > 0:
        gap_score = backlogs * 1.5 * feat_importances.get("backlogs", 0.056) * 100
        gaps["backlogs"] = {
            "feature": "Active Backlogs",
            "current": backlogs,
            "target": 0,
            "gap": backlogs,
            "severity": "CRITICAL",
            "weighted_impact": round(gap_score, 2)
        }
        recommendation_items.append({
            "category": "Academic Eligibility",
            "priority": "P0 - Urgent",
            "icon": "⚠️",
            "title": f"Clear {backlogs} Active Backlog(s)",
            "action": "Most tier-1 & tier-2 campus recruiters enforce strict 0-backlog eligibility. Prioritize semester makeup exams and consult departmental remedial tutoring immediately."
        })
        
    # 2. CGPA Assessment
    cgpa_diff = benchmarks["cgpa"] - cgpa
    if cgpa_diff > 0:
        gap_score = (cgpa_diff / 10.0) * feat_importances.get("cgpa", 0.268) * 100
        gaps["cgpa"] = {
            "feature": "CGPA",
            "current": cgpa,
            "target": benchmarks["cgpa"],
            "gap": round(cgpa_diff, 2),
            "severity": "High" if cgpa < 7.0 else "Medium",
            "weighted_impact": round(gap_score, 2)
        }
        if cgpa < 7.0:
            recommendation_items.append({
                "category": "Academic Performance",
                "priority": "P1 - High",
                "icon": "📈",
                "title": "Boost Upcoming Semester GPA (Target > 7.5)",
                "action": "A CGPA below 7.0 locks out ~40% of standard IT & analytics recruitment drives. Maximize internal assessments and target 8.0+ in forthcoming semester modules."
            })
            
    # 3. Technical & Programming Skills
    tech_diff = benchmarks["technical_score"] - tech
    if tech_diff > 0:
        gap_score = (tech_diff / 100.0) * feat_importances.get("technical_score", 0.150) * 100
        gaps["technical_score"] = {
            "feature": "Technical / Coding Score",
            "current": tech,
            "target": benchmarks["technical_score"],
            "gap": round(tech_diff, 1),
            "severity": "High" if tech < 65 else "Medium",
            "weighted_impact": round(gap_score, 2)
        }
        recommendation_items.append({
            "category": "Technical Mastery",
            "priority": "P1 - High" if tech < 65 else "P2 - Medium",
            "icon": "💻",
            "title": "Sharpen Core Data Structures & Coding Problem-Solving",
            "action": f"Current technical score is {tech}/100 (Placed average: {benchmarks['technical_score']}). Practice 2-3 medium coding problems daily on LeetCode/HackerRank in Python/Java, emphasizing Arrays, Strings, Trees, and Dynamic Programming."
        })

    # 4. Aptitude Score
    apt_diff = benchmarks["aptitude_score"] - aptitude
    if apt_diff > 0:
        gap_score = (apt_diff / 100.0) * feat_importances.get("aptitude_score", 0.189) * 100
        gaps["aptitude_score"] = {
            "feature": "Aptitude Score",
            "current": aptitude,
            "target": benchmarks["aptitude_score"],
            "gap": round(apt_diff, 1),
            "severity": "High" if aptitude < 65 else "Medium",
            "weighted_impact": round(gap_score, 2)
        }
        recommendation_items.append({
            "category": "Aptitude & Reasoning",
            "priority": "P1 - High" if aptitude < 65 else "P2 - Medium",
            "icon": "🧠",
            "title": "Timed Quantitative & Logical Aptitude Drills",
            "action": f"Aptitude is the second largest driver of placement success (18.9% importance). Take weekly full-length speed tests on platforms like IndiaBIX, Pariksha, or PrepInsta covering Speed Maths, Probability, and Syllogisms."
        })

    # 5. Communication Score
    comm_diff = benchmarks["communication_score"] - comm
    if comm_diff > 0:
        gap_score = (comm_diff / 100.0) * feat_importances.get("communication_score", 0.165) * 100
        gaps["communication_score"] = {
            "feature": "Communication Score",
            "current": comm,
            "target": benchmarks["communication_score"],
            "gap": round(comm_diff, 1),
            "severity": "High" if comm < 65 else "Medium",
            "weighted_impact": round(gap_score, 2)
        }
        recommendation_items.append({
            "category": "Soft Skills & Communication",
            "priority": "P1 - High" if comm < 65 else "P2 - Medium",
            "icon": "🗣️",
            "title": "Participate in Mock Interviews & Group Discussions",
            "action": f"Communication score is {comm}/100. Engage in weekly placement cell mock interviews, practice the STAR method for behavioral answers, and record elevator pitches to build confident delivery."
        })

    # 6. Projects
    if projects < 2:
        proj_diff = benchmarks["projects"] - projects
        gap_score = (proj_diff / 5.0) * feat_importances.get("projects", 0.082) * 100
        gaps["projects"] = {
            "feature": "Academic/Personal Projects",
            "current": projects,
            "target": 3,
            "gap": 3 - projects,
            "severity": "Medium",
            "weighted_impact": round(gap_score, 2)
        }
        recommendation_items.append({
            "category": "Project Portfolio",
            "priority": "P2 - Medium",
            "icon": "🚀",
            "title": "Build & Deploy 1-2 End-to-End Projects",
            "action": "Develop a production-ready application (e.g., Full-Stack web app or ML pipeline with deployed API) on GitHub with clear README, live demo link, and modular architecture."
        })

    # 7. Internships
    if internships < 1:
        gap_score = 1.0 * feat_importances.get("internships", 0.046) * 100
        gaps["internships"] = {
            "feature": "Internships",
            "current": internships,
            "target": 1,
            "gap": 1,
            "severity": "Medium",
            "weighted_impact": round(gap_score, 2)
        }
        recommendation_items.append({
            "category": "Industry Exposure",
            "priority": "P2 - Medium",
            "icon": "💼",
            "title": "Secure an Industry Internship / Virtual Experience",
            "action": "Target a 4-8 week internship via Internshala, LinkedIn, or complete virtual experience programs (Forage / TCS iON) to showcase real-world project delivery."
        })

    # 8. Certifications
    if certs < 1:
        gap_score = 1.0 * feat_importances.get("certifications", 0.043) * 100
        gaps["certifications"] = {
            "feature": "Relevant Certifications",
            "current": certs,
            "target": 2,
            "gap": 2 - certs,
            "severity": "Low",
            "weighted_impact": round(gap_score, 2)
        }
        recommendation_items.append({
            "category": "Skill Credentialing",
            "priority": "P3 - Low",
            "icon": "📜",
            "title": "Earn Recognized Domain Certifications",
            "action": "Complete a verified certification in Cloud (AWS Certified Cloud Practitioner), AI/ML (Coursera DeepLearning.AI), or Python/Java to strengthen resume screening ranking."
        })
        
    # If student is already very strong across all areas
    if len(recommendation_items) == 0:
        recommendation_items.append({
            "category": "Elite Preparation",
            "priority": "P1 - Optimal",
            "icon": "⭐",
            "title": "Target Dream / Super-Dream Recruiter Profiles",
            "action": "Your profile exceeds placed cohort averages across all dimensions. Focus on advanced system design, competitive programming, and domain-specific architecture interviews."
        })
        
    return {
        "weak_areas_count": len(gaps),
        "gap_analysis": gaps,
        "recommendations": recommendation_items,
        "benchmarks": benchmarks
    }
