# 📤 Files to Upload to GitHub

## ✅ MUST UPLOAD (Required for deployment)

### Core Application Files
- [ ] `src/app.py`
- [ ] `src/data_loader.py`
- [ ] `src/train.py`
- [ ] `src/model_utils.py`
- [ ] `src/predict_api.py`
- [ ] `src/__init__.py`

### Test Files (Optional but recommended)
- [ ] `src/tests/test_data_loader.py`
- [ ] `src/tests/test_train.py`
- [ ] `src/tests/__init__.py`

### Data Files (CRITICAL!)
- [ ] `data/energy_dataset.csv` (5.98 MB)
- [ ] `data/weather_features.csv` (19 MB)
- [ ] `data/README.md`

### Configuration Files
- [ ] `requirements.txt`
- [ ] `packages.txt`
- [ ] `.gitignore`
- [ ] `.streamlit/config.toml`

### Documentation
- [ ] `README.md`
- [ ] `SETUP_INSTRUCTIONS.md`
- [ ] `DEPLOYMENT_GUIDE.md`
- [ ] `DEPLOYMENT_READY.md`
- [ ] `UPLOAD_CHECKLIST.md` (this file)

### Other Files
- [ ] `notebooks/EDA.ipynb`
- [ ] `assets/logo.png`

## ⚠️ OPTIONAL (Pre-trained model - saves deployment time)
- [ ] `src/model.pkl` (if under 100MB)
- [ ] `src/metrics.json`
- [ ] `src/feature_importance.csv`

## ❌ DO NOT UPLOAD

- ❌ `.venv/` folder (virtual environment)
- ❌ `__pycache__/` folders (Python cache)
- ❌ `*.pyc` files (compiled Python)
- ❌ `.DS_Store` (Mac files)
- ❌ `.vscode/` (VS Code settings)

## 📦 Total Upload Size

Estimated size: ~30-40 MB (with data files)

## 🚀 Quick Upload Steps

### Method 1: GitHub Desktop
1. Install from https://desktop.github.com/
2. Sign in with GitHub
3. Add local repository
4. Select all files above
5. Commit: "Initial deployment"
6. Publish repository

### Method 2: Web Upload
1. Create new repo on GitHub
2. Drag & drop all folders:
   - `src/`
   - `data/`
   - `notebooks/`
   - `assets/`
   - `.streamlit/`
3. Add all root files (README.md, requirements.txt, etc.)
4. Commit changes

## ✅ Verification

After upload, verify on GitHub:
1. Check `src/` has all Python files
2. Check `data/` has both CSV files
3. Check `requirements.txt` is present
4. Check `.streamlit/config.toml` is present

## 🎯 Next: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. New app → Select your repository
4. Main file: `src/app.py`
5. Deploy!

---

**Current Status**: ✅ All files ready!
**Total Files**: ~25 files
**Total Size**: ~30-40 MB
**GitHub Limit**: 100 MB per file ✅
