import os
import json
import urllib.request
import time
import random
from PIL import Image
from lumaforge.ollama_client import OllamaClient

class DatasetCurator:
    def __init__(self, data_dir="data/creative_dataset", ollama_client=None):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.metadata_path = os.path.join(data_dir, "metadata.jsonl")
        self.ollama = ollama_client or OllamaClient()
        
        # A much larger, diverse set of high-quality creative images (90 items)
        # representing different creative poster styles, characters, lighting, and environments.
        # A much larger, diverse set of high-quality creative images (90 items)
        # representing different creative poster styles, characters, lighting, and environments.
        self.source_images = [
            # Cyberpunk
            {"url": "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=512&h=512&fit=crop", "category": "Cyberpunk", "default_caption": "Abstract digital painting with flowing neon pink and cyan ribbons on dark background."},
            {"url": "https://images.unsplash.com/photo-1607604276583-eef5d076aa5f?w=512&h=512&fit=crop", "category": "Cyberpunk", "default_caption": "Digital anime art illustration of futuristic city skyline with glowing neon signs at night."},
            {"url": "https://images.unsplash.com/photo-1542751371-adc38448a05e?w=512&h=512&fit=crop", "category": "Cyberpunk", "default_caption": "Gaming setup with glowing RGB keyboard, mouse, and neon futuristic background."},
            {"url": "https://images.unsplash.com/photo-1578894381163-e72c17f2d45f?w=512&h=512&fit=crop", "category": "Cyberpunk", "default_caption": "Dystopian cyberpunk city street at night with glowing purple and cyan neon signs, rain reflections."},
            {"url": "https://images.unsplash.com/photo-1548345680-f5475ea5df84?w=512&h=512&fit=crop", "category": "Cyberpunk", "default_caption": "Futuristic alleyway with high-tech wires, industrial pipes, and glowing green holographic interfaces."},
            {"url": "https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=512&h=512&fit=crop", "category": "Cyberpunk", "default_caption": "Cyberpunk character with glowing augmented eyes looking at a virtual interface screen."},
            {"url": "https://images.unsplash.com/photo-1562157873-818bc0726f68?w=512&h=512&fit=crop", "category": "Cyberpunk", "default_caption": "Neon lit garage with a futuristic cyberpunk sports car parked under glowing purple lights."},
            {"url": "https://images.unsplash.com/photo-1504051771394-dd2e66b2e08f?w=512&h=512&fit=crop", "category": "Cyberpunk", "default_caption": "Crowded night market in a futuristic metropolis, glowing paper lanterns mixed with neon signs."},
            {"url": "https://images.unsplash.com/photo-1549490349-8643362247b5?w=512&h=512&fit=crop", "category": "Cyberpunk", "default_caption": "Detail shot of cybernetic hand with glowing yellow circuit lines resting on a metallic table."},
            {"url": "https://images.unsplash.com/photo-1535223289827-42f1e9919769?w=512&h=512&fit=crop", "category": "Cyberpunk", "default_caption": "Intricate server room with racks of computers emitting blinking blue and red indicator lights."},

            # Sci-Fi
            {"url": "https://images.unsplash.com/photo-1508739773434-c26b3d09e071?w=512&h=512&fit=crop", "category": "Sci-Fi", "default_caption": "An astronaut standing on a remote desert planet looking at a massive red moon rise."},
            {"url": "https://images.unsplash.com/photo-1541185933-ef5d8ed016c2?w=512&h=512&fit=crop", "category": "Sci-Fi", "default_caption": "A retro-futuristic rocket ship landing on Mars with volcanic dust and hazy red skies."},
            {"url": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=512&h=512&fit=crop", "category": "Sci-Fi", "default_caption": "Abstract view of planet Earth from deep space with glowing blue atmosphere and visible continents."},
            {"url": "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=512&h=512&fit=crop", "category": "Sci-Fi", "default_caption": "A satellite solar array catching bright sunlight in outer space with stars in the background."},
            {"url": "https://images.unsplash.com/photo-1516339901601-2e1d62dc0c45?w=512&h=512&fit=crop", "category": "Sci-Fi", "default_caption": "Mysterious deep blue space nebula cloud with shining stars and cosmic dust."},
            {"url": "https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?w=512&h=512&fit=crop", "category": "Sci-Fi", "default_caption": "A majestic spiral galaxy swirling in empty space with clusters of bright stars and dark dust lanes."},
            {"url": "https://images.unsplash.com/photo-1614728894747-a83421e2b9c9?w=512&h=512&fit=crop", "category": "Sci-Fi", "default_caption": "A space shuttle preparing to launch on a pad under a dramatic twilight sky."},
            {"url": "https://images.unsplash.com/photo-1581822261290-991b38693d1b?w=512&h=512&fit=crop", "category": "Sci-Fi", "default_caption": "Retro rocket illustration on landing pad under deep starry skies."},
            {"url": "https://images.unsplash.com/photo-1502134249126-9f3755a50d78?w=512&h=512&fit=crop", "category": "Sci-Fi", "default_caption": "A colossal space station floating in orbit above a rings-encircled gas giant planet."},
            {"url": "https://images.unsplash.com/photo-1518364538800-6bcb3f25da49?w=512&h=512&fit=crop", "category": "Sci-Fi", "default_caption": "Vintage rocket capsule floating in a swirling asteroid belt."},

            # Movie Poster
            {"url": "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=512&h=512&fit=crop", "category": "Movie Poster", "default_caption": "Dramatic fantasy movie poster concept of a dark medieval castle standing on a high cliff."},
            {"url": "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=512&h=512&fit=crop", "category": "Movie Poster", "default_caption": "Cinematic film poster style scene of movie theater seats and projector beam cutting through haze."},
            {"url": "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=512&h=512&fit=crop", "category": "Movie Poster", "default_caption": "Cinematic silhouette of a person standing at the end of a dark theater aisle with screen glow."},
            {"url": "https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?w=512&h=512&fit=crop", "category": "Movie Poster", "default_caption": "Vintage cinema building exterior at night with a bright glowing sign board and retro lamps."},
            {"url": "https://images.unsplash.com/photo-1440404653325-ab127d49abc1?w=512&h=512&fit=crop", "category": "Movie Poster", "default_caption": "Cinematic film strip frames in warm studio glowing bokeh background."},
            {"url": "https://images.unsplash.com/photo-1505686994434-e3cc5abf1330?w=512&h=512&fit=crop", "category": "Movie Poster", "default_caption": "Cinematic film strip frames glowing with soft golden warm lights on a black background."},
            {"url": "https://images.unsplash.com/photo-1585647347483-22b66260dfff?w=512&h=512&fit=crop", "category": "Movie Poster", "default_caption": "Suspense movie poster style scene of a foggy wooden pier leading into a dark calm lake."},
            {"url": "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?w=512&h=512&fit=crop", "category": "Movie Poster", "default_caption": "Abstract conceptual film poster of red and blue light beams intersecting on dark textured background."},
            {"url": "https://images.unsplash.com/photo-1485846234645-a62644f84728?w=512&h=512&fit=crop", "category": "Movie Poster", "default_caption": "Cinematic shot of film director chair on set with studio lights blurred in the background."},
            {"url": "https://images.unsplash.com/photo-1478720568477-152d9b164e26?w=512&h=512&fit=crop", "category": "Movie Poster", "default_caption": "Dramatic movie poster layout of a winding mountain road at dusk under deep indigo skies."},

            # Typography
            {"url": "https://images.unsplash.com/photo-1561070791-2526d30994b5?w=512&h=512&fit=crop", "category": "Typography", "default_caption": "Bold neon pink sign showing the word 'HELLO' on brick wall."},
            {"url": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=512&h=512&fit=crop", "category": "Typography", "default_caption": "Vintage theater marquee letters spelling out 'CINEMA' in glowing incandescent lightbulbs."},
            {"url": "https://images.unsplash.com/photo-1563245372-f21724e3856d?w=512&h=512&fit=crop", "category": "Typography", "default_caption": "Minimalist graphic poster with big clean sans-serif typography text overlay on abstract shapes."},
            {"url": "https://images.unsplash.com/photo-1580136579312-94651dfd596d?w=512&h=512&fit=crop", "category": "Typography", "default_caption": "Abstract typographic poster with overlapping letters in black and white geometric layout."},
            {"url": "https://images.unsplash.com/photo-1525253086316-d0c936c814f8?w=512&h=512&fit=crop", "category": "Typography", "default_caption": "Clean minimalist branding poster with elegant serif text on a warm beige background."},
            {"url": "https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?w=512&h=512&fit=crop", "category": "Typography", "default_caption": "Close up of old retro wooden printing press letters spelling out 'ART' in ink on paper."},
            {"url": "https://images.unsplash.com/photo-1513542789411-b6a5d4f31634?w=512&h=512&fit=crop", "category": "Typography", "default_caption": "Cursive neon light letters showing a creative phrase in soft cyan glowing on dark wall."},
            {"url": "https://images.unsplash.com/photo-1516962215378-7fa2e137ae93?w=512&h=512&fit=crop", "category": "Typography", "default_caption": "Retro style typography poster with bold 3D letters spelling 'CREATIVE' in warm gradient colors."},
            {"url": "https://images.unsplash.com/photo-1582213782179-e0d53f98f2ca?w=512&h=512&fit=crop", "category": "Typography", "default_caption": "Symmetric abstract vector graphic poster with bold sans-serif text aligned in columns."},
            {"url": "https://images.unsplash.com/photo-1504198453319-5ce911bafcde?w=512&h=512&fit=crop", "category": "Typography", "default_caption": "Chalkboard vintage lettering on a black wall, rustic sign design."},

            # Stylized Portrait
            {"url": "https://images.unsplash.com/photo-1579783902614-a3fb3927b6a5?w=512&h=512&fit=crop", "category": "Stylized Portrait", "default_caption": "Classic oil painting portrait of a young woman with warm dramatic light and floral details."},
            {"url": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=512&h=512&fit=crop", "category": "Stylized Portrait", "default_caption": "Cinematic close-up portrait of a woman in low studio lighting, warm volumetric gradients."},
            {"url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=512&h=512&fit=crop", "category": "Stylized Portrait", "default_caption": "Black and white studio portrait of a man with high contrast key lighting, shadow depth."},
            {"url": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=512&h=512&fit=crop", "category": "Stylized Portrait", "default_caption": "Polished studio headshot portrait of a woman looking forward, soft natural light."},
            {"url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=512&h=512&fit=crop", "category": "Stylized Portrait", "default_caption": "Creative portrait of a smiling woman with colorful pink and yellow rim lighting in studio."},
            {"url": "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=512&h=512&fit=crop", "category": "Stylized Portrait", "default_caption": "Cinematic portrait of a man in low light with dramatic side lighting and dark background."},
            {"url": "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=512&h=512&fit=crop", "category": "Stylized Portrait", "default_caption": "Artistic portrait of a model with dramatic lighting and warm shadows, soft focus look."},
            {"url": "https://images.unsplash.com/photo-1554151228-14d9def656e4?w=512&h=512&fit=crop", "category": "Stylized Portrait", "default_caption": "Close-up artistic portrait of a woman with vibrant makeup and soft lighting on grey backing."},
            {"url": "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=512&h=512&fit=crop", "category": "Stylized Portrait", "default_caption": "Studio portrait headshot of a smiling model in soft lighting."},
            {"url": "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?w=512&h=512&fit=crop", "category": "Stylized Portrait", "default_caption": "Studio portrait of a model with colorful neon blue and magenta key lights, high contrast."},

            # Character Art
            {"url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=512&h=512&fit=crop", "category": "Character Art", "default_caption": "High detail studio photography portrait of a model with colored neon rim lighting."},
            {"url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512&h=512&fit=crop", "category": "Character Art", "default_caption": "Close up headshot of a smiling man with textured grey background, dramatic key lighting."},
            {"url": "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?w=512&h=512&fit=crop", "category": "Character Art", "default_caption": "Isolated studio character portrait of a creator with warm rim lights on grey backing."},
            {"url": "https://images.unsplash.com/photo-1566753323558-f4e0952af115?w=512&h=512&fit=crop", "category": "Character Art", "default_caption": "Studio character sheet pose of a young model wearing modern athletic apparel, white background."},
            {"url": "https://images.unsplash.com/photo-1501196354995-cbb51c65aaea?w=512&h=512&fit=crop", "category": "Character Art", "default_caption": "Polished portrait of a character with dramatic side lighting showing fine facial details."},
            {"url": "https://images.unsplash.com/photo-1580489944761-15a19d654956?w=512&h=512&fit=crop", "category": "Character Art", "default_caption": "Studio portrait of a model wearing high-tech glasses with colorful light reflections."},
            {"url": "https://images.unsplash.com/photo-1618336753974-aae8e04506aa?w=512&h=512&fit=crop", "category": "Character Art", "default_caption": "Digital painting concept art of a fantasy character with silver armor and glowing elements."},
            {"url": "https://images.unsplash.com/photo-1560250097-0b93528c311a?w=512&h=512&fit=crop", "category": "Character Art", "default_caption": "Polished studio headshot of a confident executive character."},
            {"url": "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=512&h=512&fit=crop", "category": "Character Art", "default_caption": "Business character posing confidently in studio, corporate headshot layout."},
            {"url": "https://images.unsplash.com/photo-1500048993953-d23a436266cf?w=512&h=512&fit=crop", "category": "Character Art", "default_caption": "Candid portrait headshot of a smiling programmer character."},

            # Fantasy Landscape
            {"url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=512&h=512&fit=crop", "category": "Fantasy Landscape", "default_caption": "Cinematic wide oil painting of a deep valley with rivers running through green mountains."},
            {"url": "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=512&h=512&fit=crop", "category": "Fantasy Landscape", "default_caption": "Mysterious fantasy forest environment with glowing mist, tall ancient mossy trees."},
            {"url": "https://images.unsplash.com/photo-1447752875215-b2761acb3c5d?w=512&h=512&fit=crop", "category": "Fantasy Landscape", "default_caption": "Sun rays breaking through a dense forest canopy, glowing light rays on forest floor."},
            {"url": "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=512&h=512&fit=crop", "category": "Fantasy Landscape", "default_caption": "Epic mountain landscape at sunset with warm orange glowing clouds and deep purple shadow peaks."},
            {"url": "https://images.unsplash.com/photo-1501785888041-af3ef285b470?w=512&h=512&fit=crop", "category": "Fantasy Landscape", "default_caption": "Peaceful lake scene surrounded by high alpine peaks, reflection of mountains on water surface."},
            {"url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=512&h=512&fit=crop", "category": "Fantasy Landscape", "default_caption": "Beautiful tropical beach with golden sand, palm tree silhouettes, and glowing pink sunset."},
            {"url": "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=512&h=512&fit=crop", "category": "Fantasy Landscape", "default_caption": "Beautiful lush green valley under massive fluffy clouds."},
            {"url": "https://images.unsplash.com/photo-1518495973542-4542c06a5843?w=512&h=512&fit=crop", "category": "Fantasy Landscape", "default_caption": "Magical sunlight filtering through ancient oak trees, soft mist rising from forest floor."},
            {"url": "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?w=512&h=512&fit=crop", "category": "Fantasy Landscape", "default_caption": "Epic deep canyon landscape with a winding blue river under a bright golden sky."},
            {"url": "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=512&h=512&fit=crop", "category": "Fantasy Landscape", "default_caption": "Mystic waterfalls cascading down a vertical mossy cliff inside a hidden jungle cavern."},

            # Vector Art
            {"url": "https://images.unsplash.com/photo-1531403009284-440f080d1e12?w=512&h=512&fit=crop", "category": "Vector Art", "default_caption": "Modern vector illustration of graphic design workspace with monitors, colorful flat icons."},
            {"url": "https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?w=512&h=512&fit=crop", "category": "Vector Art", "default_caption": "Abstract glassmorphism vector background with frosted glass cards and vibrant mesh gradients."},
            {"url": "https://images.unsplash.com/photo-1618005198143-d5660460c55f?w=512&h=512&fit=crop", "category": "Vector Art", "default_caption": "Minimalist vector landscape illustration with flat geometric mountains and warm terracotta sun."},
            {"url": "https://images.unsplash.com/photo-1558655146-d09347e92766?w=512&h=512&fit=crop", "category": "Vector Art", "default_caption": "Cute colorful flat vector icon sheet of office elements, rounded corners, clean lines."},
            {"url": "https://images.unsplash.com/photo-1626785774573-4b799315345d?w=512&h=512&fit=crop", "category": "Vector Art", "default_caption": "Elegant abstract vector pattern background of thin gold geometric lines on dark navy blue background."},
            {"url": "https://images.unsplash.com/photo-1634017839464-5c339ebe3cb4?w=512&h=512&fit=crop", "category": "Vector Art", "default_caption": "Colorful 3D abstract vector rendering of geometric spheres and rings, neon mesh shading."},
            {"url": "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?w=512&h=512&fit=crop", "category": "Vector Art", "default_caption": "Modern flat vector banner design with colorful abstract fluid shapes overlapping."},
            {"url": "https://images.unsplash.com/photo-1558591710-4b4a1ae0f04d?w=512&h=512&fit=crop", "category": "Vector Art", "default_caption": "Abstract vector pattern of clean black geometric lines on a bright white background."},
            {"url": "https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=512&h=512&fit=crop", "category": "Vector Art", "default_caption": "Vibrant abstract vector mesh gradient background with organic flowing shapes, blue and purple."},
            {"url": "https://images.unsplash.com/photo-1557672172-298e090bd0f1?w=512&h=512&fit=crop", "category": "Vector Art", "default_caption": "Colorful abstract fluid painting mesh background."},

            # Product Mockup
            {"url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=512&h=512&fit=crop", "category": "Product Mockup", "default_caption": "Clean product mockup photograph of a white smart watch sitting on a grey concrete pedestal."},
            {"url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=512&h=512&fit=crop", "category": "Product Mockup", "default_caption": "Studio overhead photo of black wireless headphones resting on warm wooden table surface."},
            {"url": "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?w=512&h=512&fit=crop", "category": "Product Mockup", "default_caption": "Retro style film camera product photo with high contrast lighting, black textured backdrop."},
            {"url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=512&h=512&fit=crop", "category": "Product Mockup", "default_caption": "Vibrant product shot of red athletic running shoe sitting on glowing yellow studio background."},
            {"url": "https://images.unsplash.com/photo-1583394838336-acd977736f90?w=512&h=512&fit=crop", "category": "Product Mockup", "default_caption": "Minimalist aesthetic product display pedestal with dry palm leaf shadows on neutral beige wall."},
            {"url": "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=512&h=512&fit=crop", "category": "Product Mockup", "default_caption": "Clean product photo of elegant retro sunglasses sitting on a white stone tile, soft key light."},
            {"url": "https://images.unsplash.com/photo-1560343090-f0409e92791a?w=512&h=512&fit=crop", "category": "Product Mockup", "default_caption": "Studio shot of a single brown leather shoe resting on a stark white background."},
            {"url": "https://images.unsplash.com/photo-1586495777744-4413f21062fa?w=512&h=512&fit=crop", "category": "Product Mockup", "default_caption": "Elegant cosmetic product bottle mock up resting on a concrete slab, volumetric lighting."},
            {"url": "https://images.unsplash.com/photo-1608231387042-66d1773070a5?w=512&h=512&fit=crop", "category": "Product Mockup", "default_caption": "Minimalist mock up of white ceramic coffee mug sitting on a soft grey surface, side shadows."},
            {"url": "https://images.unsplash.com/photo-1524678606370-a47ad25cb82a?w=512&h=512&fit=crop", "category": "Product Mockup", "default_caption": "Overhead studio shot of modern white wireless earbuds inside charging case, yellow background."}
        ]

    def download_and_curate(self, limit=90, use_ollama_captioning=True) -> int:
        """
        Downloads target images, filters low-quality / duplicates,
        captions them, and writes the metadata in Hugging Face format.
        """
        print(f"[DatasetCurator] Starting curation and download process for {limit} items...")
        os.makedirs(self.images_dir, exist_ok=True)
        
        curated_count = 0
        metadata_records = []
        
        # Deduplication tracker (using simple image hash similarity)
        hashes = []
        
        # Limit cannot exceed available source images
        actual_limit = min(limit, len(self.source_images))
        selected_sources = self.source_images[:actual_limit]
        
        for idx, src in enumerate(selected_sources):
            img_filename = f"image_{idx:03d}.jpg"
            img_path = os.path.join(self.images_dir, img_filename)
            
            print(f" -> Curating item {idx+1}/{len(selected_sources)} ({src['category']})...")
            
            # Download image
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                req = urllib.request.Request(src["url"], headers=headers)
                with urllib.request.urlopen(req, timeout=10) as response:
                    with open(img_path, 'wb') as f:
                        f.write(response.read())
            except Exception as e:
                print(f"    [Warning] Failed to download {src['url']}: {e}")
                # Fallback: create a solid color image if offline
                self._create_fallback_image(img_path, idx)
                
            # Perform Curation Step 1: Quality Checks
            if not self._check_image_validity(img_path):
                print(f"    [Curation Filter] Image {img_filename} failed quality checks. Skipping.")
                if os.path.exists(img_path):
                    os.remove(img_path)
                continue
                
            # Perform Curation Step 2: Deduplication Checks (Overfitting Control)
            img = Image.open(img_path)
            img_hash = self._calculate_average_hash(img)
            
            is_duplicate = False
            for existing_hash in hashes:
                similarity = self._hash_similarity(img_hash, existing_hash)
                if similarity > 0.82: # More strict threshold for near-duplicates
                    is_duplicate = True
                    break
                    
            if is_duplicate:
                print(f"    [Curation Filter] Image {img_filename} identified as a near-duplicate. Skipping.")
                img.close()
                os.remove(img_path)
                continue
                
            hashes.append(img_hash)
            img.close()
            
            # Perform Curation Step 3: Caption Curation
            caption = src["default_caption"]
            if use_ollama_captioning and self.ollama.check_connection():
                # Yield to let local Ollama run (simulate a slight gap to prevent hammering the local server)
                refined_caption = self._generate_ollama_caption(src["category"], caption)
                if refined_caption:
                    caption = refined_caption
                    
            relative_img_path = os.path.join("images", img_filename)
            metadata_records.append({
                "file_name": relative_img_path,
                "text": caption,
                "category": src["category"]
            })
            
            curated_count += 1
            
        # Write metadata.jsonl file
        with open(self.metadata_path, "w") as f:
            for record in metadata_records:
                f.write(json.dumps(record) + "\n")
                
        print(f"[DatasetCurator] Successfully curated {curated_count} images in '{self.data_dir}'")
        return curated_count

    def _create_fallback_image(self, path: str, idx: int):
        """Creates a solid color image as an offline fallback."""
        colors = [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50), (200, 50, 200), (50, 200, 200)]
        color = colors[idx % len(colors)]
        img = Image.new("RGB", (512, 512), color)
        img.save(path)
        img.close()

    def _check_image_validity(self, path: str) -> bool:
        """Verifies if image is valid, uncorrupted, and meets size requirements."""
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                w, h = img.size
                if w < 256 or h < 256: # Filter low-res images
                    return False
            return True
        except Exception:
            return False

    def _calculate_average_hash(self, image: Image) -> str:
        """Calculates a simple 64-bit average hash for deduplication."""
        img_gray = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        pixels = list(img_gray.getdata())
        avg = sum(pixels) / 64
        bits = "".join(["1" if pixel > avg else "0" for pixel in pixels])
        return bits

    def _hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculates Hamming similarity between two binary hashes."""
        matching_bits = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matching_bits / 64.0

    def _generate_ollama_caption(self, category: str, default_caption: str) -> str:
        """Uses Ollama to refine or expand image captions into detailed training prompts."""
        prompt = (
            "You are a professional image caption curator. Your job is to refine an image description "
            "to make it extremely detailed, clear, and rich for text-to-image model training. "
            "Describe subjects, colors, layout, style, and lighting. Do not use conversational text or preambles. "
            f"Image Category: {category}\n"
            f"Base description: {default_caption}\n\n"
            "Refined Caption:"
        )
        
        data = {
            "model": self.ollama.model,
            "prompt": prompt,
            "stream": False
        }
        res = self.ollama._call_api("/api/generate", data)
        if res:
            refined = res.get("response", "").strip().strip('"')
            if len(refined) > 10:
                return refined
        return None
