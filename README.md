# MultiModFuseNeT
The BMCC corpus link: https://data.mendeley.com/datasets/htpk9y2pwf/2
Article Link: https://www.sciencedirect.com/science/article/pii/S095070512501130X?dgcid=coauthor


## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.25+
- CUDA-capable GPU (recommended)
- OpenCV (cv2)
- Pandas, NumPy, scikit-learn
- CLIP (for image-text similarity scoring)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mrhossain/MultiModFuseNeT
cd MultiModFuseNeT
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision
pip install transformers
pip install pandas numpy scikit-learn opencv-python
pip install clip
pip install openai  # For GPT-based experiments
```

