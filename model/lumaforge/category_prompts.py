"""
Optimized prompt templates for 16 specialized generation categories.
Each category has subcategories with specific styling and quality hints.
"""

CATEGORY_PROMPTS = {
    # 🎨 CREATIVE ART (7 subcategories)
    "creative_art": {
        "label": "Creative Art",
        "icon": "🎨",
        "subcategories": {
            "digital": {
                "name": "Digital Art",
                "enhancement": ", digital painting, vibrant colors, modern art style, high detail, professional digital artist work"
            },
            "concept": {
                "name": "Concept Art",
                "enhancement": ", concept art, cinematic lighting, dramatic composition, professional concept artist, detailed environment design"
            },
            "fantasy": {
                "name": "Fantasy Art",
                "enhancement": ", fantasy art, magical effects, mystical atmosphere, epic landscape, intricate details, vibrant magical colors"
            },
            "scifi": {
                "name": "Sci-Fi Art",
                "enhancement": ", sci-fi art, futuristic design, neon colors, technological aesthetic, advanced civilization, detailed mechanics"
            },
            "surreal": {
                "name": "Surreal Art",
                "enhancement": ", surreal art, dreamlike atmosphere, impossible geometry, hyper-realistic detail, mysterious mood, artistic vision"
            },
            "abstract": {
                "name": "Abstract Art",
                "enhancement": ", abstract art, geometric shapes, bold colors, modern artistic expression, creative composition, artistic vision"
            },
            "matte": {
                "name": "Matte Painting",
                "enhancement": ", matte painting, cinematic environment, detailed landscape, atmospheric perspective, professional studio quality"
            }
        }
    },
    
    # 👤 CHARACTER GENERATION (8 subcategories)
    "character": {
        "label": "Character Generation",
        "icon": "👤",
        "subcategories": {
            "anime": {
                "name": "Anime Characters",
                "enhancement": ", anime character, cel-shaded style, expressive eyes, vibrant colors, anime key visual, high quality anime art"
            },
            "realistic": {
                "name": "Realistic Humans",
                "enhancement": ", photorealistic character portrait, realistic skin texture, natural lighting, professional photography, detailed facial features"
            },
            "cartoon": {
                "name": "Cartoon Characters",
                "enhancement": ", cartoon character, fun art style, bright colors, exaggerated features, cheerful expression, children's book illustration"
            },
            "game": {
                "name": "Game Characters",
                "enhancement": ", game character design, RPG style, detailed armor/clothing, video game aesthetic, character sheet quality"
            },
            "superhero": {
                "name": "Superheroes",
                "enhancement": ", superhero character, muscular physique, epic pose, dramatic lighting, cinematic superhero movie style, iconic costume"
            },
            "medieval": {
                "name": "Medieval Characters",
                "enhancement": ", medieval character, historical armor, medieval clothing, fantasy setting, epic fantasy art style"
            },
            "cyberpunk": {
                "name": "Cyberpunk Characters",
                "enhancement": ", cyberpunk character, neon aesthetic, futuristic outfit, high-tech augmentations, moody lighting, dystopian future"
            },
            "pixel": {
                "name": "Pixel Art Characters",
                "enhancement": ", pixel art character, retro video game style, 8-bit or 16-bit aesthetic, limited color palette, nostalgic gaming"
            }
        }
    },
    
    # 🌄 LANDSCAPES & NATURE (8 subcategories)
    "landscape": {
        "label": "Landscapes & Nature",
        "icon": "🌄",
        "subcategories": {
            "mountains": {
                "name": "Mountains",
                "enhancement": ", majestic mountain landscape, dramatic peaks, scenic vista, golden hour lighting, photorealistic nature photography"
            },
            "forests": {
                "name": "Forests",
                "enhancement": ", enchanted forest, lush greenery, sunlight through trees, mystical atmosphere, detailed foliage, serene nature"
            },
            "beaches": {
                "name": "Beaches",
                "enhancement": ", beautiful beach landscape, sandy shore, ocean waves, sunset or sunrise, tropical paradise, photorealistic"
            },
            "waterfalls": {
                "name": "Waterfalls",
                "enhancement": ", majestic waterfall, flowing water, mist effect, lush surroundings, dramatic landscape, nature photography"
            },
            "desert": {
                "name": "Desert Scenes",
                "enhancement": ", desert landscape, golden sand dunes, dramatic sky, vast terrain, heat shimmer, cinematic desert"
            },
            "snow": {
                "name": "Snow Landscapes",
                "enhancement": ", snowy landscape, winter wonderland, frosted trees, pristine snow, cold blue lighting, peaceful winter scene"
            },
            "space": {
                "name": "Space Scenes",
                "enhancement": ", space landscape, distant planets, nebula, stars, cosmic atmosphere, sci-fi space environment, detailed space art"
            },
            "underwater": {
                "name": "Underwater Worlds",
                "enhancement": ", underwater scene, marine life, coral reef, bioluminescence, deep ocean, atmospheric underwater lighting"
            }
        }
    },
    
    # 🏙️ ARCHITECTURE (8 subcategories)
    "architecture": {
        "label": "Architecture",
        "icon": "🏙️",
        "subcategories": {
            "modern": {
                "name": "Modern Buildings",
                "enhancement": ", modern architecture, sleek design, contemporary building, glass and steel, minimalist aesthetic, professional architectural render"
            },
            "futuristic": {
                "name": "Futuristic Cities",
                "enhancement": ", futuristic city, advanced architecture, flying vehicles, neon lights, cyberpunk cityscape, sci-fi metropolis"
            },
            "ancient": {
                "name": "Ancient Temples",
                "enhancement": ", ancient temple architecture, historical building, intricate carvings, grand structure, archaeological site, mystical atmosphere"
            },
            "interior": {
                "name": "Interior Design",
                "enhancement": ", interior design, luxurious room, modern furnishings, professional interior styling, architectural photography, elegant space"
            },
            "luxury": {
                "name": "Luxury Homes",
                "enhancement": ", luxury mansion, high-end real estate, elegant architecture, premium materials, scenic views, architectural photography"
            },
            "office": {
                "name": "Office Spaces",
                "enhancement": ", modern office space, professional environment, contemporary design, productive workspace, architectural render"
            },
            "smart": {
                "name": "Smart Homes",
                "enhancement": ", smart home technology, futuristic interior, automated systems visible, modern tech-integrated living space"
            },
            "castle": {
                "name": "Fantasy Castles",
                "enhancement": ", fantasy castle architecture, magical fortress, ornate design, epic scale, fairytale aesthetic, detailed stonework"
            }
        }
    },
    
    # 🚗 VEHICLES (7 subcategories)
    "vehicles": {
        "label": "Vehicles",
        "icon": "🚗",
        "subcategories": {
            "sports": {
                "name": "Sports Cars",
                "enhancement": ", sports car, high-performance vehicle, sleek design, detailed engineering, professional product photography"
            },
            "luxury": {
                "name": "Luxury Cars",
                "enhancement": ", luxury car, premium vehicle, elegant design, prestige automobile, professional photography, detailed finish"
            },
            "motorcycle": {
                "name": "Motorcycles",
                "enhancement": ", motorcycle, detailed bike design, high-speed aesthetic, professional product shot, shiny chrome details"
            },
            "aircraft": {
                "name": "Aircraft",
                "enhancement": ", aircraft design, detailed airplane, aerial vehicle, professional technical render, engineering detail"
            },
            "spacecraft": {
                "name": "Spacecraft",
                "enhancement": ", spacecraft design, sci-fi spaceship, futuristic vehicle, detailed exterior, cosmic background, space vessel"
            },
            "ships": {
                "name": "Ships",
                "enhancement": ", ship design, maritime vessel, detailed naval architecture, ocean setting, realistic water"
            },
            "military": {
                "name": "Military Vehicles",
                "enhancement": ", military vehicle, tactical design, combat equipment, detailed mechanical engineering, professional render"
            }
        }
    },
    
    # 🛍️ PRODUCT DESIGN (8 subcategories)
    "products": {
        "label": "Product Design",
        "icon": "🛍️",
        "subcategories": {
            "mockups": {
                "name": "Product Mockups",
                "enhancement": ", professional product mockup, clean white background, studio lighting, detailed product, commercial photography"
            },
            "furniture": {
                "name": "Furniture",
                "enhancement": ", furniture design, interior aesthetic, detailed craftsmanship, professional product photography, beautiful material"
            },
            "shoes": {
                "name": "Shoes",
                "enhancement": ", shoe design, detailed footwear, professional product shot, clean background, fashion product photography"
            },
            "watches": {
                "name": "Watches",
                "enhancement": ", watch design, luxury timepiece, detailed craftsmanship, professional close-up photography, shiny metallic finish"
            },
            "electronics": {
                "name": "Electronics",
                "enhancement": ", electronic device, tech product, detailed design, professional product render, clean aesthetic"
            },
            "perfume": {
                "name": "Perfume Bottles",
                "enhancement": ", perfume bottle, luxury fragrance, elegant glass design, professional product photography, reflective surface"
            },
            "packaging": {
                "name": "Packaging Design",
                "enhancement": ", package design, product packaging, professional layout, eye-catching graphics, retail ready"
            },
            "cosmetics": {
                "name": "Cosmetic Products",
                "enhancement": ", cosmetic product, beauty packaging, luxury cosmetics, professional product photography, attractive presentation"
            }
        }
    },
    
    # 📢 MARKETING & BRANDING (8 subcategories)
    "marketing": {
        "label": "Marketing & Branding",
        "icon": "📢",
        "subcategories": {
            "posters": {
                "name": "Posters",
                "enhancement": ", movie poster design, bold typography, eye-catching layout, professional marketing material, title-safe composition"
            },
            "flyers": {
                "name": "Flyers",
                "enhancement": ", flyer design, marketing material, professional layout, striking visuals, promotional design"
            },
            "social": {
                "name": "Social Media Creatives",
                "enhancement": ", social media content, engaging visuals, trending aesthetic, digital marketing, eye-catching design"
            },
            "thumbnails": {
                "name": "YouTube Thumbnails",
                "enhancement": ", YouTube thumbnail design, clickable layout, bold text, vibrant colors, high contrast, engagement optimized"
            },
            "bookcovers": {
                "name": "Book Covers",
                "enhancement": ", book cover design, professional typography, compelling visuals, literary aesthetic, commercial book cover"
            },
            "magazines": {
                "name": "Magazine Covers",
                "enhancement": ", magazine cover design, professional layout, striking imagery, editorial aesthetic, publishing quality"
            },
            "banners": {
                "name": "Event Banners",
                "enhancement": ", event banner design, promotional graphics, bold composition, marketing material, professional event branding"
            },
            "ads": {
                "name": "Business Ads",
                "enhancement": ", advertisement design, promotional content, marketing material, professional advertising, compelling visuals"
            }
        }
    },
    
    # 🍔 FOOD (7 subcategories)
    "food": {
        "label": "Food",
        "icon": "🍔",
        "subcategories": {
            "dishes": {
                "name": "Restaurant Dishes",
                "enhancement": ", restaurant dish, gourmet food photography, appetizing presentation, professional food styling, culinary art"
            },
            "desserts": {
                "name": "Desserts",
                "enhancement": ", dessert photography, delicious cake or pastry, mouth-watering presentation, professional food styling, appetizing colors"
            },
            "beverages": {
                "name": "Beverages",
                "enhancement": ", beverage photography, refreshing drink, professional product shot, condensation on glass, appetizing presentation"
            },
            "cakes": {
                "name": "Cakes",
                "enhancement": ", cake design, decorated cake, professional bakery photography, tempting presentation, detailed frosting"
            },
            "fastfood": {
                "name": "Fast Food",
                "enhancement": ", fast food photography, appetizing burger or pizza, professional food styling, commercial food photography"
            },
            "gourmet": {
                "name": "Gourmet Meals",
                "enhancement": ", gourmet cuisine, fine dining presentation, professional food photography, exquisite plating, culinary masterpiece"
            },
            "recipes": {
                "name": "Recipe Images",
                "enhancement": ", recipe photography, finished dish, professional food styling, cooking ingredient presentation, appetizing composition"
            }
        }
    },
    
    # 👕 FASHION (7 subcategories)
    "fashion": {
        "label": "Fashion",
        "icon": "👕",
        "subcategories": {
            "clothing": {
                "name": "Clothing",
                "enhancement": ", fashion clothing, apparel design, professional fashion photography, clean background, detailed fabric texture"
            },
            "dresses": {
                "name": "Dresses",
                "enhancement": ", dress design, fashion garment, elegant styling, professional runway photography, detailed fabric and design"
            },
            "jackets": {
                "name": "Jackets",
                "enhancement": ", jacket design, outerwear fashion, detailed styling, professional product photography, quality fabric showcase"
            },
            "sneakers": {
                "name": "Sneakers",
                "enhancement": ", sneaker design, athletic footwear, detailed shoe design, professional product shot, fashion footwear"
            },
            "jewelry": {
                "name": "Jewelry",
                "enhancement": ", jewelry design, luxury accessories, detailed precious metal work, professional jewelry photography, shiny finish"
            },
            "accessories": {
                "name": "Accessories",
                "enhancement": ", fashion accessories, detailed design, professional product photography, clean aesthetic, luxury presentation"
            },
            "runway": {
                "name": "Runway Concepts",
                "enhancement": ", runway fashion show, model on catwalk, fashion design, professional photography, editorial fashion"
            }
        }
    },
    
    # 🎮 GAMING (7 subcategories)
    "gaming": {
        "label": "Gaming Assets",
        "icon": "🎮",
        "subcategories": {
            "icons": {
                "name": "Game Icons",
                "enhancement": ", game icon design, simple vector art, clear symbolic design, gaming aesthetic, professional icon set"
            },
            "ui": {
                "name": "UI Assets",
                "enhancement": ", game UI design, user interface elements, gaming aesthetic, clean design, professional game art"
            },
            "backgrounds": {
                "name": "Backgrounds",
                "enhancement": ", game background art, environment design, detailed scenery, gaming aesthetic, layered parallax background"
            },
            "npcs": {
                "name": "NPC Characters",
                "enhancement": ", game NPC character design, non-player character, gaming art style, detailed character sprite, game asset"
            },
            "weapons": {
                "name": "Weapons",
                "enhancement": ", game weapon design, fantasy or sci-fi weapon, detailed design, gaming aesthetic, professional game asset"
            },
            "effects": {
                "name": "Magic Effects",
                "enhancement": ", magical effect, spell animation frame, visual effect, particle effects, glowing energy, game FX art"
            },
            "inventory": {
                "name": "Inventory Items",
                "enhancement": ", game inventory item, loot design, detailed item sprite, gaming aesthetic, professional game asset"
            }
        }
    },
    
    # 🐶 ANIMALS (8 subcategories)
    "animals": {
        "label": "Animals",
        "icon": "🐶",
        "subcategories": {
            "pets": {
                "name": "Pets",
                "enhancement": ", pet portrait, animal photography, cute expression, professional photography, detailed fur texture"
            },
            "wildlife": {
                "name": "Wildlife",
                "enhancement": ", wildlife photography, wild animal in natural habitat, detailed fur, professional nature photography"
            },
            "birds": {
                "name": "Birds",
                "enhancement": ", bird illustration, detailed bird species, feather detail, nature photography, beautiful plumage"
            },
            "marine": {
                "name": "Marine Animals",
                "enhancement": ", marine creature, underwater animal, ocean life, detailed scales or skin, underwater photography"
            },
            "fantasy": {
                "name": "Fantasy Creatures",
                "enhancement": ", fantasy creature design, mythical beast, imaginative design, detailed creature art, fantastical illustration"
            },
            "dragons": {
                "name": "Dragons",
                "enhancement": ", dragon design, detailed dragon illustration, scales and wings, fantasy art, mythical creature"
            },
            "mythical": {
                "name": "Mythical Beasts",
                "enhancement": ", mythical creature, legendary beast, fantasy illustration, detailed design, epic creature art"
            }
        }
    },
    
    # 🎉 EVENTS (5 subcategories)
    "events": {
        "label": "Events",
        "icon": "🎉",
        "subcategories": {
            "weddings": {
                "name": "Wedding Invitations",
                "enhancement": ", wedding invitation design, elegant aesthetic, romantic colors, professional layout, luxury card design"
            },
            "birthdays": {
                "name": "Birthday Posters",
                "enhancement": ", birthday poster design, celebratory aesthetic, vibrant colors, fun typography, party celebration design"
            },
            "festivals": {
                "name": "Festival Artwork",
                "enhancement": ", festival art, celebration design, colorful aesthetic, event marketing material, festive atmosphere"
            },
            "holidays": {
                "name": "Holiday Cards",
                "enhancement": ", holiday card design, seasonal aesthetic, festive colors, professional greeting card, warm holiday feel"
            },
            "parties": {
                "name": "Party Decorations",
                "enhancement": ", party decoration design, celebration aesthetic, colorful graphics, festive elements, event planning design"
            }
        }
    },
    
    # 🏢 BUSINESS (5 subcategories)
    "business": {
        "label": "Business",
        "icon": "🏢",
        "subcategories": {
            "infographics": {
                "name": "Infographics",
                "enhancement": ", infographic design, data visualization, clean layout, professional business graphic, informative design"
            },
            "presentations": {
                "name": "Presentation Graphics",
                "enhancement": ", presentation slide design, business graphics, professional layout, corporate aesthetic, educational visual"
            },
            "dashboards": {
                "name": "Dashboard Illustrations",
                "enhancement": ", dashboard design, data visualization interface, modern UI, analytics display, professional business software"
            },
            "banners": {
                "name": "Corporate Banners",
                "enhancement": ", corporate banner design, business marketing, professional layout, company branding, promotional material"
            },
            "branding": {
                "name": "Startup Branding",
                "enhancement": ", startup branding, logo design, brand identity, modern aesthetic, professional corporate identity"
            }
        }
    },
    
    # 📚 EDUCATION (5 subcategories)
    "education": {
        "label": "Education",
        "icon": "📚",
        "subcategories": {
            "scientific": {
                "name": "Scientific Illustrations",
                "enhancement": ", scientific illustration, detailed diagram, educational visual, anatomical accuracy, professional scientific art"
            },
            "biology": {
                "name": "Biology Diagrams",
                "enhancement": ", biology diagram, cell illustration, biological system, educational scientific visual, detailed anatomy"
            },
            "history": {
                "name": "Historical Art",
                "enhancement": ", historical illustration, period-accurate design, educational historical visual, museum quality art"
            },
            "geography": {
                "name": "Geography Maps",
                "enhancement": ", geographic map, cartographic design, detailed terrain, educational geography visual, atlas quality"
            },
            "medical": {
                "name": "Medical Illustrations",
                "enhancement": ", medical illustration, anatomical accuracy, educational healthcare visual, professional medical diagram"
            }
        }
    }
}

def get_category_enhancement(category: str, subcategory: str = None) -> str:
    """
    Get prompt enhancement for a specific category and optional subcategory.
    
    Args:
        category: Category name (e.g., 'creative_art', 'character')
        subcategory: Optional subcategory name
    
    Returns:
        Enhancement prompt string
    """
    if category not in CATEGORY_PROMPTS:
        return ""
    
    cat_data = CATEGORY_PROMPTS[category]
    
    if not subcategory or subcategory not in cat_data["subcategories"]:
        # Return first subcategory enhancement as default
        first_sub = list(cat_data["subcategories"].values())[0]
        return first_sub["enhancement"]
    
    return cat_data["subcategories"][subcategory]["enhancement"]

def get_all_categories() -> dict:
    """Return all available categories with their subcategories."""
    return {k: v["label"] for k, v in CATEGORY_PROMPTS.items()}

def get_subcategories(category: str) -> dict:
    """Get all subcategories for a category."""
    if category not in CATEGORY_PROMPTS:
        return {}
    return {k: v["name"] for k, v in CATEGORY_PROMPTS[category]["subcategories"].items()}
