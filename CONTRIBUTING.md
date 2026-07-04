# Contributing to LumaForge

First off, thank you for considering contributing to LumaForge! It's people like you who make LumaForge such a great AI image generation platform.

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change. 

Please note we have a [Code of Conduct](file:///Users/sujithputta/Projects/LumaForge/CODE_OF_CONDUCT.md), please follow it in all your interactions with the project.

---

## 🛠️ Development Setup

LumaForge consists of two major subsystems:
1. **AI Engine Backend**: Written in Python using FastAPI, PyTorch, and Stable Diffusion.
2. **Spatial UI Web Playground**: Written in TypeScript using Next.js, Tailwind CSS, and Bun.

### Prerequisites
- **macOS** with Apple Silicon (M1/M2/M3)
- **Python 3.10+**
- **Node.js 18+** & **Bun**
- **Ollama** installed and running locally with the `llama3.2:1b` model.

### Setting Up the Backend
1. Navigate to the `model` folder:
   ```bash
   cd model
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI development server:
   ```bash
   python3 app.py
   ```

### Setting Up the Frontend
1. Navigate to the `web` folder:
   ```bash
   cd web
   ```
2. Install dependencies:
   ```bash
   bun install
   ```
3. Run the development server:
   ```bash
   bun run dev
   ```

---

## 🧪 Testing Your Changes

Before submitting your contribution, please verify that your changes do not break any existing functionality:

1. **Verify Post-Processing / Shaders**:
   Run the dedicated enhancement tests from the root directory:
   ```bash
   python3 test_enhancements.py
   ```
   Ensure all latencies and operations (background removal, sketches, bilateral filters) pass successfully.

2. **Verify Performance & Latencies**:
   Run the pipeline benchmarks in mock mode to verify CLI operations:
   ```bash
   python3 main.py benchmark --mock
   ```

---

## 📥 Pull Request Process

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add relevant test cases or updates to `test_enhancements.py`.
3. Update the `README.md` or other documentation if you are introducing new features or options.
4. Ensure your code follows the style guidelines:
   - **Backend**: Clean Python code (PEP 8 compliant, type hints preferred).
   - **Frontend**: Clean TypeScript, reusable components, and responsive Tailwind CSS layout.
5. Submit a pull request. Once submitted, project maintainers will review your changes and provide feedback.
