"""
LumaForge Category Prompts - 16 Specialized Categories with 110+ Templates
v1.1 Release - Advanced Image Generation Categories and Enhancements
"""

CATEGORY_PROMPTS = {
    "creative_art": {
        "Digital Art": "digital art, high-resolution concept art, professional digital illustration, trending on artstation, vibrant colors, detailed brushwork, cinematic lighting",
        "Concept Art": "concept art, detailed environment design, professional concept artist work, cinematic rendering, atmospheric lighting, trending on artstation",
        "Fantasy": "fantasy illustration, magical atmosphere, detailed magical creatures, enchanted landscapes, mystical lighting, professional fantasy art",
        "Sci-Fi": "sci-fi concept art, futuristic technology, cyberpunk aesthetics, high-tech environments, neon lighting, dystopian atmosphere",
        "Surreal": "surreal art, dreamlike atmosphere, impossible geometry, floating elements, ethereal lighting, abstract surrealism, professional digital art",
        "Abstract": "abstract art, geometric shapes, color field painting, contemporary art, professional abstract painting, modern art gallery",
        "Matte Painting": "matte painting, landscape artwork, digital matte, cinematic environment, professional concept art, detailed background painting"
    },
    "characters": {
        "Anime": "anime character, beautiful anime girl, anime art style, high quality anime, trending anime, detailed eyes, professional anime illustration",
        "Realistic": "realistic character portrait, photorealistic digital art, realistic human face, professional portrait painting, detailed facial features",
        "Cartoon": "cartoon character, professional cartoon illustration, vibrant cartoon style, fun character design, digital cartoon art",
        "Game Character": "video game character, game art style, character design for RPG, game character concept art, professional game asset",
        "Superhero": "superhero character, superhero concept art, dynamic pose, comic book style, professional superhero illustration",
        "Medieval": "medieval character, fantasy warrior, historical character, medieval armor, professional fantasy character design",
        "Cyberpunk": "cyberpunk character, sci-fi character design, futuristic outfit, neon aesthetic, cyberpunk fashion, detailed digital art",
        "Pixel Art": "pixel art character, retro gaming style, 8-bit art, pixel perfect, professional pixel art"
    },
    "landscapes": {
        "Mountains": "mountain landscape, snow-capped peaks, majestic mountains, alpine scenery, landscape photography, cinematic lighting",
        "Forests": "forest landscape, dense woodland, enchanted forest, old growth trees, forest atmosphere, detailed foliage, professional landscape",
        "Beaches": "beach landscape, ocean view, tropical beach, sunset beach, sandy shores, waves, professional landscape photography",
        "Waterfalls": "waterfall landscape, cascading water, powerful waterfall, misty atmosphere, nature landscape, flowing water effects",
        "Desert": "desert landscape, sand dunes, arid environment, desert sunset, vast landscape, detailed sand textures",
        "Snow Landscape": "snow landscape, winter scenery, frozen landscape, snow-covered mountains, winter atmosphere, icy terrain",
        "Space": "space landscape, cosmic environment, nebula, stars, planetary landscape, astronomy, space art, sci-fi environment",
        "Underwater": "underwater landscape, ocean depths, coral reef, marine life, underwater scenery, bioluminescent creatures"
    },
    "architecture": {
        "Modern": "modern architecture, contemporary building design, glass and steel, sleek design, modern office building, urban architecture",
        "Futuristic": "futuristic architecture, sci-fi building, advanced technology, floating structures, neon lighting, cyberpunk architecture",
        "Ancient": "ancient architecture, historical building, classical design, temple, ruins, historical landmark, ancient civilization",
        "Interior Design": "interior design, modern room, architectural interior, professional interior, detailed room design, furniture arrangement",
        "Luxury Architecture": "luxury architecture, high-end building, elegant design, premium materials, sophisticated architecture, luxury real estate",
        "Office Space": "office architecture, modern office space, corporate building, professional workspace, sleek office design",
        "Smart Building": "smart building, intelligent architecture, technology integrated building, futuristic office, connected spaces",
        "Castles": "castle architecture, medieval fortress, grand castle, fantasy castle, stone architecture, imposing structure"
    },
    "vehicles": {
        "Sports Cars": "sports car, high-performance vehicle, sleek car design, racing car, luxury sports car, detailed car model, automotive photography",
        "Luxury Vehicles": "luxury car, premium automobile, elegant vehicle design, high-end car, luxury sedan, professional car photography",
        "Motorcycles": "motorcycle, sport bike, detailed bike design, motorcycle design, professional motorcycle photography, sleek two-wheeler",
        "Aircraft": "aircraft, airplane design, flying plane, jet aircraft, commercial airplane, professional aircraft illustration",
        "Spacecraft": "spacecraft, alien ship, futuristic spacecraft, space vehicle, sci-fi spaceship, detailed space vehicle design",
        "Ships": "ship, ocean vessel, detailed ship design, maritime vessel, sailing ship, naval architecture, professional ship illustration",
        "Military Vehicles": "military vehicle, combat vehicle, tank, armored vehicle, military technology, warfare equipment"
    },
    "products": {
        "Product Mockups": "product mockup, professional product photography, product showcase, clean background, detailed product shot",
        "Furniture": "furniture design, modern furniture, detailed furniture piece, interior furniture, professional furniture photography",
        "Shoes": "shoe design, sneaker design, professional shoe photography, detailed footwear, stylish shoes, product photography",
        "Watches": "watch design, luxury watch, detailed timepiece, watch product shot, elegant watch design, professional photography",
        "Electronics": "electronic device, tech product, professional electronics photography, detailed gadget, technology product",
        "Perfume": "perfume bottle design, fragrance product, luxury perfume, elegant bottle design, professional product photography",
        "Packaging Design": "product packaging, package design, box design, professional packaging, retail packaging, detailed box",
        "Cosmetics": "cosmetics product, beauty product, makeup packaging, luxury cosmetics, professional beauty photography"
    },
    "marketing": {
        "Posters": "movie poster, professional poster design, eye-catching poster, marketing poster, detailed poster art",
        "Flyers": "flyer design, promotional flyer, professional flyer, marketing flyer, eye-catching design",
        "Social Media": "social media content, Instagram post, social media graphic, eye-catching design, digital marketing content",
        "Thumbnails": "YouTube thumbnail, video thumbnail, eye-catching thumbnail, professional thumbnail design",
        "Book Covers": "book cover design, professional book cover, eye-catching cover, literary design, detailed book art",
        "Magazine Covers": "magazine cover, professional magazine design, eye-catching cover, editorial design, glossy magazine",
        "Banners": "banner design, web banner, marketing banner, eye-catching banner, professional graphic design",
        "Advertisements": "advertisement, promotional ad, marketing advertisement, professional ad design, eye-catching advertisement"
    },
    "food": {
        "Dishes": "food photography, professional dish photo, mouth-watering food, detailed food shot, restaurant quality food",
        "Desserts": "dessert photography, beautiful dessert, cake design, pastry design, professional food photography",
        "Beverages": "beverage photography, drink photography, professional drink shot, refreshing beverage, detailed drink",
        "Cakes": "cake design, professional cake, detailed cake decoration, beautiful cake, pastry art, intricate cake design",
        "Fast Food": "fast food, burger, pizza, professional food photography, appetizing fast food, detailed food shot",
        "Gourmet Cuisine": "gourmet food, fine dining, professional culinary photography, upscale food, exquisite plating",
        "Recipes": "food recipe photo, cooking process, food preparation, professional cooking photography, detailed ingredients"
    },
    "fashion": {
        "Clothing": "fashion clothing, garment design, professional clothing photography, stylish outfit, detailed clothing",
        "Dresses": "dress design, fashion dress, beautiful dress, detailed dress, professional fashion photography, elegant gown",
        "Jackets": "jacket design, fashionable jacket, detailed jacket, professional jacket photography, stylish outerwear",
        "Sneakers": "sneaker design, shoe design, fashion sneakers, detailed footwear, professional shoe photography, stylish sneakers",
        "Jewelry": "jewelry design, elegant jewelry, luxury jewelry, detailed jewelry, professional jewelry photography",
        "Accessories": "fashion accessories, stylish accessories, detailed accessory design, professional accessories photography",
        "Runway": "fashion runway, model on runway, fashion show, professional fashion photography, haute couture, detailed clothing"
    },
    "gaming": {
        "Game Icons": "game icon, professional game icon design, detailed icon, game UI element, pixel perfect icon",
        "Game UI": "game user interface, professional UI design, detailed UI elements, game interface, modern game UI",
        "Game Backgrounds": "game background, detailed environment art, game scenery, professional background design, atmospheric game art",
        "NPCs": "game NPC character, detailed character design, game character art, character concept art, professional game character",
        "Game Weapons": "game weapon, detailed weapon design, fantasy weapon, game weapon asset, professional weapon design",
        "Visual Effects": "game visual effects, particle effects, spell effects, professional VFX design, detailed effects",
        "Game Inventory": "game inventory, RPG inventory, detailed inventory items, game assets, professional item design"
    },
    "animals": {
        "Pets": "pet photography, cute pet, detailed animal portrait, professional pet photo, adorable animal",
        "Wildlife": "wildlife photography, wild animal, detailed wildlife, nature photography, professional wildlife shot",
        "Birds": "bird illustration, detailed bird design, bird art, ornithology art, professional bird illustration",
        "Marine Life": "marine animal, ocean creature, underwater animal, fish, detailed sea creature, professional marine art",
        "Fantasy Animals": "fantasy creature, magical animal, fantastical beast, detailed creature design, professional fantasy art",
        "Dragons": "dragon illustration, detailed dragon design, dragon art, fantasy dragon, professional dragon artwork",
        "Mythical Creatures": "mythical creature, legendary beast, fantasy creature, detailed mythical being, professional creature design",
        "Endangered Species": "endangered animal, conservation photography, rare animal, detailed wildlife, nature documentary style"
    },
    "events": {
        "Weddings": "wedding photography, bride and groom, wedding ceremony, detailed wedding scene, professional wedding photo",
        "Birthdays": "birthday celebration, birthday party, festive atmosphere, celebration scene, detailed party design",
        "Festivals": "festival scene, cultural festival, festive gathering, celebration, detailed festival artwork",
        "Holidays": "holiday celebration, festive holiday scene, holiday atmosphere, detailed holiday design",
        "Parties": "party scene, celebration event, detailed party setting, festive gathering, professional event photography"
    },
    "business": {
        "Infographics": "infographic design, professional infographic, data visualization, detailed infographic, modern design",
        "Presentations": "presentation design, slide design, professional presentation, detailed design elements, corporate design",
        "Dashboards": "dashboard design, data dashboard, professional UI design, detailed dashboard layout, modern interface",
        "Banners": "business banner, corporate banner, professional banner design, detailed business graphics, marketing banner",
        "Branding": "brand design, company branding, professional branding, detailed brand identity, corporate logo design"
    },
    "education": {
        "Scientific": "scientific illustration, detailed scientific drawing, educational diagram, professional scientific art, accurate representation",
        "Biology": "biological illustration, detailed biology diagram, educational biology, professional scientific illustration, anatomical accuracy",
        "History": "historical illustration, detailed historical scene, educational history art, period-accurate design, professional historical art",
        "Geography": "geographical illustration, map design, detailed geography art, educational geography, professional cartography",
        "Medical": "medical illustration, detailed medical diagram, anatomical illustration, professional medical art, scientific accuracy"
    }
}

def get_category_prompts(category: str, subcategory: str) -> str:
    """Get prompt template for a specific category and subcategory."""
    if category in CATEGORY_PROMPTS:
        if subcategory in CATEGORY_PROMPTS[category]:
            return CATEGORY_PROMPTS[category][subcategory]
    return ""

def get_all_categories() -> list:
    """Get list of all available categories."""
    return list(CATEGORY_PROMPTS.keys())

def get_subcategories(category: str) -> list:
    """Get list of subcategories for a specific category."""
    if category in CATEGORY_PROMPTS:
        return list(CATEGORY_PROMPTS[category].keys())
    return []
