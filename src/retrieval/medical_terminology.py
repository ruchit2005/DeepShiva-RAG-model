"""
Medical terminology mapping for Western to Ayurvedic terms.
Helps bridge the gap between common patient language and Ayurvedic texts.
"""

MEDICAL_TERM_MAP = {
    # Digestive Issues
    "stomach ache": ["Atisara", "Grahani", "Ajirna", "abdominal pain", "gastric disorder"],
    "stomach pain": ["Atisara", "Grahani", "Ajirna", "abdominal pain", "gastric disorder"],
    "diarrhea": ["Atisara", "loose motions", "frequent stools"],
    "constipation": ["Vibandha", "Anaha", "difficult defecation"],
    "indigestion": ["Ajirna", "digestive disorder", "Mandagni"],
    "bloating": ["Adhmana", "Anaha", "abdominal distension"],
    "acidity": ["Amlapitta", "acid reflux", "heartburn"],
    
    # Fever and General
    "fever": ["Jwara", "pyrexia", "elevated temperature"],
    "cold": ["Pratishyaya", "Kasa", "nasal discharge"],
    "cough": ["Kasa", "respiratory disorder"],
    
    # Pain
    "headache": ["Shiroroga", "Shirahshula", "head pain"],
    "joint pain": ["Sandhivata", "Amavata", "arthritis"],
    "back pain": ["Katigraha", "Katishula", "lumbar pain"],
    
    # Skin Issues
    "skin rash": ["Kustha", "Vicharchika", "dermatological disorder"],
    "itching": ["Kandu", "pruritus", "skin irritation"],
    
    # Hemorrhoids
    "piles": ["Arsha", "hemorrhoids"],
    "anal fissure": ["Parikartika", "fissure-in-ano"],
    
    # Women's Health
    "menstrual pain": ["Kashtartava", "dysmenorrhoea"],
    "excessive bleeding": ["Raktapradar", "menorrhagia"],
    
    # Neurological
    "epilepsy": ["Apasmara", "seizure disorder"],
    "paralysis": ["Pakshaghata", "hemiplegia"],
    
    # Urinary
    "urinary problems": ["Mutraghata", "urinary disorder"],
    "kidney stones": ["Ashmari", "urinary calculi"],
    
    # Respiratory
    "asthma": ["Tamaka Shwasa", "breathing difficulty"],
    "breathlessness": ["Shwasa", "dyspnea"],
    
    # Liver
    "jaundice": ["Kamala", "Panduroga", "liver disorder"],
    "liver problems": ["Yakrit Vikara", "hepatic disorder"],
    
    # Diabetes
    "diabetes": ["Prameha", "Madhumeha", "high blood sugar"],
    "excessive thirst": ["Trishna", "polydipsia"],
    "frequent urination": ["Prabhuta Mutrata", "polyuria"],
}


def expand_query_with_ayurvedic_terms(query: str) -> str:
    """
    Expand a Western medical query with Ayurvedic equivalent terms.
    
    Args:
        query: Original query in Western medical terminology
    
    Returns:
        Expanded query including Ayurvedic terms
    """
    query_lower = query.lower()
    expanded_terms = [query]
    
    # Check if any mapped term appears in the query
    for western_term, ayurvedic_terms in MEDICAL_TERM_MAP.items():
        if western_term in query_lower:
            expanded_terms.extend(ayurvedic_terms)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in expanded_terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique_terms.append(term)
    
    return " ".join(unique_terms)


def get_ayurvedic_synonyms(western_term: str) -> list[str]:
    """
    Get Ayurvedic synonyms for a Western medical term.
    
    Args:
        western_term: Western medical terminology
    
    Returns:
        List of Ayurvedic equivalent terms
    """
    return MEDICAL_TERM_MAP.get(western_term.lower(), [])
