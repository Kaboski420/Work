# Server Deployment Guide

## Server Access
- **Host**: ml1@148.251.90.81
- **Directory**: /root/app
- **Password**: Zt%23fz&23Rfhwu

## Quick Start

### 1. Transfer Code to Server

From your local machine:
```bash
# Option 1: Using scp (if you have SSH key setup)
scp -r /Users/hamzashaheen/Downloads/Virility_Prediction ml1@148.251.90.81:/root/app/

# Option 2: Using rsync
rsync -avz /Users/hamzashaheen/Downloads/Virility_Prediction ml1@148.251.90.81:/root/app/
```

Or manually upload via SFTP/FTP client.

### 2. SSH into Server

```bash
ssh ml1@148.251.90.81
# Password: Zt%23fz&23Rfhwu
```

### 3. Setup Environment

```bash
cd /root/app/Virility_Prediction
bash scripts/server_setup.sh
```

This will:
- Install all dependencies
- Create necessary directories (models/, data/, audit_logs/)

### 4. Train Model on Full ClickHouse Data

**Option A: Train on sample (10,000 rows) - Recommended for first run**
```bash
cd /root/app/Virility_Prediction
PYTHONPATH=. python3 scripts/train_on_full_clickhouse.py --limit 10000 --top-percentile 20
```

**Option B: Train on full dataset (all rows)**
```bash
cd /root/app/Virility_Prediction
PYTHONPATH=. python3 scripts/train_on_full_clickhouse.py --top-percentile 20
```

**Parameters:**
- `--limit`: Number of rows to use (None = all rows)
- `--top-percentile`: Top X% to label as viral (default: 20%)
- `--table`: ClickHouse table name (default: 'ig_post')
- `--model-path`: Where to save model (default: './models/virality_model.pkl')

**Expected output:**
- Model will be saved to `./models/virality_model.pkl`
- Training time: ~5-30 minutes depending on dataset size

### 5. Score All Posts

**Option A: Score sample (10,000 rows)**
```bash
cd /root/app/Virility_Prediction
PYTHONPATH=. python3 scripts/score_full_clickhouse.py --limit 10000
```

**Option B: Score full dataset**
```bash
cd /root/app/Virility_Prediction
PYTHONPATH=. python3 scripts/score_full_clickhouse.py
```

**Parameters:**
- `--limit`: Number of rows to score (None = all rows)
- `--table`: ClickHouse table name (default: 'ig_post')
- `--output`: Custom output CSV path (auto-generated if not specified)
- `--batch-size`: Batch size for async processing (default: 10)

**Output:**
- CSV file saved to `data/clickhouse_imports/ig_post_*_scored.csv`
- Contains all scoring results with virality scores, probabilities, classifications

### 6. Start API Server (Optional)

```bash
cd /root/app/Virility_Prediction
./start_api.sh
```

Or manually:
```bash
cd /root/app/Virility_Prediction
PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API will be available at: `http://148.251.90.81:8000`
- Docs: `http://148.251.90.81:8000/docs`
- Health: `http://148.251.90.81:8000/health`

## Monitoring

### Check Model Status
```bash
ls -lh /root/app/Virility_Prediction/models/virality_model.pkl
```

### View Scoring Results
```bash
# List all scored CSVs
ls -lh /root/app/Virility_Prediction/data/clickhouse_imports/*_scored.csv

# View latest results summary
python3 -c "import pandas as pd; df = pd.read_csv('data/clickhouse_imports/ig_post_*_scored.csv'); print(df[['virality_score', 'viral_classification', 'prediction_source']].describe())"
```

## Troubleshooting

### Model Not Loading
- Check if model file exists: `ls -lh models/virality_model.pkl`
- Check if scikit-learn is installed: `python3 -c "import sklearn; print(sklearn.__version__)"`
- Re-train if needed: `PYTHONPATH=. python3 scripts/train_on_full_clickhouse.py --limit 1000`

### ClickHouse Connection Issues
- Verify credentials in scripts match your ClickHouse setup
- Test connection: `python3 -c "import clickhouse_connect; client = clickhouse_connect.create_client(host='clickhouse.bragmant.noooo.art', port=443, username='dev2', password='...', database='crawler', secure=True); print(client.ping())"`

### Memory Issues
- Use `--limit` to process in batches
- Reduce `--batch-size` in scoring script
- Monitor memory: `free -h`

## Performance Tips

1. **Start Small**: Test with `--limit 1000` first
2. **Batch Processing**: Use `--limit` to process in chunks
3. **Parallel Processing**: Adjust `--batch-size` based on server resources
4. **Model Caching**: Once trained, model is cached and reused

## Expected Results

After training and scoring, you should see:
- **Model file**: `./models/virality_model.pkl` (typically 300KB - 5MB)
- **Scored CSV**: Contains columns:
  - `virality_score` (0-100)
  - `virality_probability` (0-1)
  - `viral_classification` (yes/no)
  - `prediction_source` (ml_model or rule_based)
  - `virality_dynamics` (deep/broad/hybrid)
  - Plus all original post data

## Next Steps

1. Analyze results: Load CSV in pandas/Jupyter
2. Fine-tune model: Adjust `--top-percentile` for different label distributions
3. Production deployment: Use trained model in API endpoints
4. Continuous training: Re-train periodically with new data

