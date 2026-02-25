class ReportGenerator:
    def __init__(self):
        self.reports = {
            'akiec': {
                "name": "Actinic Keratoses and Intraepithelial Carcinoma",
                "description": "Precancerous skin growths caused by prolonged UV exposure, which can develop into squamous cell carcinoma.",
                "causes": "Long-term sun exposure and tanning bed use.",
                "symptoms": "Dry, scaly, rough skin patches, sometimes itchy or burning.",
                "risk_factors": "Fair skin, blond/red hair, older age, history of severe sunburns.",
                "medications": "Topical creams (e.g., fluorouracil 5-FU, imiquimod, diclofenac).",
                "advanced_treatments": "Cryotherapy (freezing), photodynamic therapy, surgical excision, or laser resurfacing.",
                "recovery": "Most heal within 2-4 weeks post-treatment, but continuous monitoring is required due to precancerous nature."
            },
            'bcc': {
                "name": "Basal Cell Carcinoma",
                "description": "The most common and least dangerous form of skin cancer, arising from the basal cells in the epidermis.",
                "causes": "Intense or cumulative UV damage from the sun.",
                "symptoms": "Pearly or waxy bump, a flat, flesh-colored or brown scar-like lesion, or a bleeding/scabbing sore that heals and returns.",
                "risk_factors": "Fair skin, chronic sun exposure, radiation therapy, family history.",
                "medications": "Topical chemotherapy (5-FU) or immune response modifiers (imiquimod) for superficial cases.",
                "advanced_treatments": "Mohs micrographic surgery, excisional surgery, radiation therapy, laser therapy.",
                "recovery": "Surgical recovery is usually 1-3 weeks. Excellent prognosis but requires annual skin exams."
            },
            'bkl': {
                "name": "Benign Keratosis-like Lesions",
                "description": "Non-cancerous skin growths, including seborrheic keratoses and solar lentigines. Harmless but can resemble melanoma.",
                "causes": "Age and cumulative sun exposure.",
                "symptoms": "Slightly elevated, waxy, scaly patches that look 'pasted on'.",
                "risk_factors": "Age (common in adults over 50), genetics.",
                "medications": "None required unless symptomatic (can be treated with urea-based creams).",
                "advanced_treatments": "Cryosurgery, electrosurgery, or curettage for cosmetic removal.",
                "recovery": "Healing usually takes 1-2 weeks if removed; normally requires no treatment."
            },
            'df': {
                "name": "Dermatofibroma",
                "description": "Common, benign fibrous nodules that typically occur on the lower legs.",
                "causes": "Exact cause unknown; may be a reaction to minor trauma like an insect bite or thorn puncture.",
                "symptoms": "Small, firm, red to brown bumps that dimple when pinched.",
                "risk_factors": "More common in women and young adults.",
                "medications": "Usually none required.",
                "advanced_treatments": "Surgical excision if they become painful, change in appearance, or for cosmetic reasons.",
                "recovery": "Immediate if no surgery; localized scar if surgically removed."
            },
            'nv': {
                "name": "Melanocytic Nevi",
                "description": "Common moles composed of melanocytes (pigment-producing cells). They are benign.",
                "causes": "Genetics and sun exposure during early childhood.",
                "symptoms": "Symmetric, even-colored brown, black or tan spots on the skin.",
                "risk_factors": "Fair skin, family history of many moles, frequent sun exposure.",
                "medications": "None.",
                "advanced_treatments": "Surgical removal and biopsy only if they display atypical features (ABCDEs of melanoma).",
                "recovery": "No treatment needed for benign nevi."
            },
            'vasc': {
                "name": "Vascular Lesions",
                "description": "A category of benign skin lesions formed by abnormally large or numerous blood vessels (e.g., cherry angiomas).",
                "causes": "Genetics, pregnancy, advancing age, and sometimes liver damage.",
                "symptoms": "Red or purple spots or patches on the skin.",
                "risk_factors": "Aging and genetics.",
                "medications": "None required for most asymptomatic lesions.",
                "advanced_treatments": "Pulsed dye laser therapy or electrocautery if removal is desired.",
                "recovery": "Quick recovery post-laser treatment (typically a few days to a week)."
            },
            'mel': {
                "name": "Melanoma",
                "description": "The most serious type of skin cancer, developing from melanocytes. Highly aggressive if not caught early.",
                "causes": "Severe, blistering sunburns; genetic mutations; excessive UV radiation.",
                "symptoms": "A new, unusual growth or a change in an existing mole (Asymmetry, Border irregularity, Color changes, Diameter > 6mm, Evolving).",
                "risk_factors": "Fair skin, history of sunburns, multiple atypical moles, family history, weakened immune system.",
                "medications": "Targeted therapies (e.g., BRAF inhibitors), Immunotherapy (e.g., Keytruda, Opdivo).",
                "advanced_treatments": "Wide surgical excision, lymph node biopsy, radiation therapy, and systemic treatments.",
                "recovery": "Highly variable based on cancer stage. Early stages have a 99% localized survival rate; late stages require extensive long-term oncology care."
            }
        }

    def generate(self, short_name):
        return self.reports.get(short_name, {
            "name": "Unknown Condition",
            "description": "No data available.",
            "causes": "N/A", "symptoms": "N/A",
            "risk_factors": "N/A", "medications": "N/A",
            "advanced_treatments": "N/A", "recovery": "N/A"
        })
