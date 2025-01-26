reviews = [
	"The food was tasty, the hamburger was delicious, the sauce had a wonderful aftertaste. The pilav was little dry for me, although the pinac de laut complimented nicely with the dryness i suppose.",
	"The food was flavorful, but the portions were much smaller than expected. The desserts were delightful, though, and the variety was impressive. The vegetarian dishes were well-received, but the meat options could have been more seasoned",
	"The food selection was extensive, offering a wide variety of options, but the quality didn’t match the quantity. Some dishes were bland, especially the pasta and salads, while others, like the steak, were overcooked. The dessert was the highlight, though—delicious and rich."
]

synonym_map = {
        # Food-Related
        "food quality": ["food", "dish", "meal", "cuisine", "flavor", "taste", "seasoned", 
                        "vegetarian", "meat", "dessert", "spicy", "bland", "presentation", 
                        "freshness", "ingredients", "cooked", "raw", "burnt", "soggy"],
        
        # Service & Staff
        "service": ["service", "staff", "waiters", "server", "personnel", "attentiveness", 
                "friendliness", "professionalism", "rude", "ignored", "unhelpful"],
        
        # Management & Administration
        "management": ["management", "admin", "leadership", "policies", "supervisor", 
                    "coordination", "efficiency", "communication", "responsiveness", 
                    "training", "staff turnover", "scheduling", "complaints handling", 
                    "decision-making", "transparency", "organization", "accountability", 
                    "bureaucracy", "paperwork", "hierarchy", "conflict resolution"],
        
        # Accommodation & Facilities
        "accommodation": ["room", "suite", "bed", "linens", "bathroom", "amenities", 
                        "facilities", "wi-fi", "aircon", "heating", "parking", "elevator", 
                        "noise", "view", "balcony", "space", "furniture", "cleanliness", 
                        "maintenance", "safety", "security", "lighting", "ventilation", 
                        "housekeeping", "check-in", "check-out", "luggage storage", 
                        "accessibility", "repairs", "pest control", "smell", "dampness"],
        
        # App/Technology
        "app_usage": ["app", "application", "interface", "navigation", "usability", 
                    "performance", "crash", "loading", "speed", "features", "user experience", 
                    "updates", "notifications", "bugs", "errors", "registration", "login", 
                    "security", "payment", "search", "filters", "recommendations", 
                    "customer support", "offline mode", "data usage", "compatibility", 
                    "tutorials", "responsive", "sync", "download", "upload", "freeze"],
        
        # General Business Aspects
        "pricing": ["price", "cost", "value", "bill", "pricing", "overpriced", "discounts", 
                "refunds", "charges", "hidden fees", "subscription"],
        
        "waiting": ["wait time", "waiting", "delay", "queue", "line", "reservation", 
                "hold time", "ETA", "latency", "response time"],
        
        # Location/Environment
        "ambiance": ["ambiance", "atmosphere", "decor", "environment", "vibe", "music", 
                    "lighting", "crowd", "privacy", "temperature", "layout", "spaciousness"],
        
        # Special Cases
        "special_requests": ["allergy", "dietary", "customization", "substitution", 
                            "accessibility", "wheelchair", "child-friendly", "pet-friendly"]
    }